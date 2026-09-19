"""Numerically stable lognormal fragility-curve fitting.

The primary binary model is

P(Y=1 | x) = Phi((log(x) - mu) / sigma),

where x is a positive intensity measure, exp(mu) is the median capacity,
and sigma is the logarithmic dispersion. The fit uses every binary
observation directly; binning is only a plotting diagnostic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from scipy.optimize import minimize
from scipy.special import log_ndtr, ndtr
from scipy.stats import norm


ArrayLike = np.ndarray | Iterable[float]


def _as_1d_finite(values: ArrayLike, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=float)
    if array.ndim != 1 or array.size == 0:
        raise ValueError(f"{name} must be a non-empty one-dimensional array.")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} must contain only finite values.")
    return array


def lognormal_failure_probability(
    intensity: ArrayLike | float,
    mu: float,
    sigma: float,
) -> np.ndarray:
    """Evaluate a binary lognormal fragility curve."""

    intensity_array = np.asarray(intensity, dtype=float)
    if not np.isfinite(mu):
        raise ValueError("mu must be finite.")
    if not np.isfinite(sigma) or sigma <= 0.0:
        raise ValueError("sigma must be finite and positive.")
    if not np.all(np.isfinite(intensity_array)) or np.any(intensity_array <= 0.0):
        raise ValueError("intensity must contain only finite positive values.")
    return ndtr((np.log(intensity_array) - mu) / sigma)


def _negative_log_likelihood_gradient(
    parameters: np.ndarray,
    log_intensity: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> tuple[float, np.ndarray]:
    """Return stable Bernoulli NLL and its exact gradient."""

    mu, log_sigma = parameters
    sigma = np.exp(log_sigma)
    z = (log_intensity - mu) / sigma
    nll = -np.sum(
        weights
        * (outcomes * log_ndtr(z) + (1.0 - outcomes) * log_ndtr(-z))
    )

    log_density = -0.5 * z**2 - 0.5 * np.log(2.0 * np.pi)
    derivative_z = weights * (
        (1.0 - outcomes) * np.exp(log_density - log_ndtr(-z))
        - outcomes * np.exp(log_density - log_ndtr(z))
    )
    gradient = np.array(
        [
            -np.sum(derivative_z) / sigma,
            -np.sum(derivative_z * z),
        ],
        dtype=float,
    )

    if regularization:
        nll += 0.5 * regularization * float(parameters @ parameters)
        gradient += regularization * parameters
    return float(nll), gradient


def _hessian_from_gradient(
    parameters: np.ndarray,
    log_intensity: np.ndarray,
    outcomes: np.ndarray,
    weights: np.ndarray,
    regularization: float,
) -> np.ndarray:
    steps = 1.0e-4 * np.maximum(1.0, np.abs(parameters))
    hessian = np.empty((2, 2), dtype=float)
    for column, step in enumerate(steps):
        offset = np.zeros(2, dtype=float)
        offset[column] = step
        _, gradient_plus = _negative_log_likelihood_gradient(
            parameters + offset,
            log_intensity,
            outcomes,
            weights,
            regularization,
        )
        _, gradient_minus = _negative_log_likelihood_gradient(
            parameters - offset,
            log_intensity,
            outcomes,
            weights,
            regularization,
        )
        hessian[:, column] = (gradient_plus - gradient_minus) / (2.0 * step)
    return 0.5 * (hessian + hessian.T)


@dataclass(frozen=True)
class LognormalFragilityFit:
    """Fitted curve, uncertainty estimates, and prediction helpers."""

    mu: float
    sigma: float
    covariance: np.ndarray
    negative_log_likelihood: float
    n_observations: int
    confidence_level: float
    converged: bool
    message: str

    @property
    def median_capacity(self) -> float:
        return float(np.exp(self.mu))

    @property
    def aic(self) -> float:
        return 4.0 + 2.0 * self.negative_log_likelihood

    def predict(self, intensity: ArrayLike | float) -> np.ndarray:
        return lognormal_failure_probability(intensity, self.mu, self.sigma)

    def parameter_confidence_intervals(
        self, confidence_level: float | None = None
    ) -> dict[str, tuple[float, float]]:
        """Return intervals for median capacity and logarithmic dispersion."""

        level = self.confidence_level if confidence_level is None else confidence_level
        if not 0.0 < level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one.")
        critical = norm.ppf(0.5 + level / 2.0)
        standard_errors = np.sqrt(np.maximum(np.diag(self.covariance), 0.0))
        mu_limits = self.mu + np.array([-1.0, 1.0]) * critical * standard_errors[0]
        log_sigma = np.log(self.sigma)
        sigma_limits = (
            log_sigma + np.array([-1.0, 1.0]) * critical * standard_errors[1]
        )
        return {
            "median_capacity": tuple(np.exp(mu_limits)),
            "sigma": tuple(np.exp(sigma_limits)),
        }

    def confidence_band(
        self,
        intensity: ArrayLike,
        confidence_level: float | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Pointwise delta-method confidence band for fitted probability."""

        values = _as_1d_finite(intensity, "intensity")
        if np.any(values <= 0.0):
            raise ValueError("intensity must be positive.")
        level = self.confidence_level if confidence_level is None else confidence_level
        if not 0.0 < level < 1.0:
            raise ValueError("confidence_level must lie strictly between zero and one.")

        z = (np.log(values) - self.mu) / self.sigma
        probability = ndtr(z)
        density = np.exp(-0.5 * z**2) / np.sqrt(2.0 * np.pi)
        gradient = np.column_stack((-density / self.sigma, -density * z))
        variance = np.einsum(
            "ij,jk,ik->i", gradient, self.covariance, gradient
        )
        standard_error = np.sqrt(np.maximum(variance, 0.0))
        critical = norm.ppf(0.5 + level / 2.0)
        lower = np.clip(probability - critical * standard_error, 0.0, 1.0)
        upper = np.clip(probability + critical * standard_error, 0.0, 1.0)
        return probability, lower, upper


def fit_lognormal_fragility(
    intensity: ArrayLike,
    failed: ArrayLike,
    *,
    weights: ArrayLike | None = None,
    sigma_bounds: tuple[float, float] = (0.02, 5.0),
    intensity_bounds: tuple[float, float] | None = None,
    regularization: float = 0.0,
    confidence_level: float = 0.95,
    max_iterations: int = 2_000,
) -> LognormalFragilityFit:
    """Fit binary lognormal fragility by maximum likelihood.

    Parameters are optimized as (mu, log(sigma)). Using log_ndtr in the
    likelihood avoids taking the logarithm of probabilities rounded to zero
    in the distribution tails.
    """

    values = _as_1d_finite(intensity, "intensity")
    outcomes = _as_1d_finite(failed, "failed")
    if values.shape != outcomes.shape:
        raise ValueError("intensity and failed must have the same shape.")
    if np.any(values <= 0.0):
        raise ValueError("intensity must be positive.")
    if not np.all(np.isin(outcomes, (0.0, 1.0))):
        raise ValueError("failed must contain only zero and one.")
    if np.unique(outcomes).size != 2:
        raise ValueError("A fragility fit requires failures and non-failures.")
    if not 0.0 < confidence_level < 1.0:
        raise ValueError("confidence_level must lie strictly between zero and one.")
    if regularization < 0.0 or not np.isfinite(regularization):
        raise ValueError("regularization must be finite and non-negative.")

    if weights is None:
        observation_weights = np.ones_like(values)
    else:
        observation_weights = _as_1d_finite(weights, "weights")
        if observation_weights.shape != values.shape:
            raise ValueError("weights and intensity must have the same shape.")
        if np.any(observation_weights < 0.0) or not np.any(observation_weights > 0.0):
            raise ValueError("weights must be non-negative with at least one positive value.")

    sigma_min, sigma_max = map(float, sigma_bounds)
    if not 0.0 < sigma_min < sigma_max:
        raise ValueError("sigma_bounds must be positive and increasing.")
    if intensity_bounds is None:
        mu_bounds = (np.log(values.min()) - 5.0, np.log(values.max()) + 5.0)
    else:
        lower, upper = map(float, intensity_bounds)
        if not 0.0 < lower < upper:
            raise ValueError("intensity_bounds must be positive and increasing.")
        mu_bounds = (np.log(lower), np.log(upper))

    log_intensity = np.log(values)
    initial = np.array([np.median(log_intensity), np.log(0.45)], dtype=float)
    result = minimize(
        _negative_log_likelihood_gradient,
        initial,
        args=(
            log_intensity,
            outcomes,
            observation_weights,
            float(regularization),
        ),
        method="L-BFGS-B",
        jac=True,
        bounds=[mu_bounds, (np.log(sigma_min), np.log(sigma_max))],
        options={"maxiter": int(max_iterations), "ftol": 1.0e-12},
    )
    if not result.success:
        raise RuntimeError(f"Fragility optimisation failed: {result.message}")

    hessian = _hessian_from_gradient(
        result.x,
        log_intensity,
        outcomes,
        observation_weights,
        float(regularization),
    )
    covariance = np.linalg.pinv(hessian, hermitian=True)
    covariance = 0.5 * (covariance + covariance.T)
    return LognormalFragilityFit(
        mu=float(result.x[0]),
        sigma=float(np.exp(result.x[1])),
        covariance=covariance,
        negative_log_likelihood=float(result.fun),
        n_observations=values.size,
        confidence_level=float(confidence_level),
        converged=bool(result.success),
        message=str(result.message),
    )


# Backwards-compatible names from the original module.
lognorm_cdf_vectorized = lognormal_failure_probability
predict_exceed_prob = lognormal_failure_probability


def fit_binary_lognormal_mle(
    im: ArrayLike,
    exceed: ArrayLike,
    reg_lambda: float = 0.0,
) -> dict[str, object]:
    """Compatibility wrapper around fit_lognormal_fragility."""

    fit = fit_lognormal_fragility(im, exceed, regularization=reg_lambda)
    return {
        "mu": fit.mu,
        "sigma": fit.sigma,
        "nll": fit.negative_log_likelihood,
        "covariance": fit.covariance,
        "fit": fit,
    }


def fit_fragility_mle(
    im: ArrayLike,
    damage: ArrayLike,
    levels: ArrayLike,
    weights: ArrayLike | None = None,
    **kwargs: object,
) -> dict[str, object]:
    """Fit one cumulative exceedance curve per increasing damage level."""

    damage_array = _as_1d_finite(damage, "damage")
    levels_array = _as_1d_finite(levels, "levels")
    if np.any(np.diff(levels_array) <= 0.0):
        raise ValueError("levels must be strictly increasing.")
    allowed = {
        key: value
        for key, value in kwargs.items()
        if key
        in {
            "sigma_bounds",
            "intensity_bounds",
            "regularization",
            "confidence_level",
            "max_iterations",
        }
    }
    fits = [
        fit_lognormal_fragility(
            im,
            damage_array >= level,
            weights=weights,
            **allowed,
        )
        for level in levels_array
    ]
    return {
        "success": True,
        "mu": np.array([fit.mu for fit in fits]),
        "sigma": np.array([fit.sigma for fit in fits]),
        "nll": float(sum(fit.negative_log_likelihood for fit in fits)),
        "levels": levels_array,
        "fits": fits,
    }


def predict_state_probs(
    im: ArrayLike,
    mu: ArrayLike,
    sigma: ArrayLike,
) -> np.ndarray:
    """Convert ordered cumulative exceedance curves to state probabilities."""

    values = _as_1d_finite(im, "im")
    mus = _as_1d_finite(mu, "mu")
    sigmas = _as_1d_finite(sigma, "sigma")
    if mus.shape != sigmas.shape:
        raise ValueError("mu and sigma must have the same shape.")
    exceedance = np.vstack(
        [
            lognormal_failure_probability(values, location, dispersion)
            for location, dispersion in zip(mus, sigmas)
        ]
    )
    exceedance = np.minimum.accumulate(exceedance, axis=0)
    boundaries = np.vstack(
        [np.ones(values.size), exceedance, np.zeros(values.size)]
    )
    return np.clip(boundaries[:-1] - boundaries[1:], 0.0, 1.0)


__all__ = [
    "LognormalFragilityFit",
    "fit_lognormal_fragility",
    "lognormal_failure_probability",
    "fit_binary_lognormal_mle",
    "fit_fragility_mle",
    "lognorm_cdf_vectorized",
    "predict_exceed_prob",
    "predict_state_probs",
]
