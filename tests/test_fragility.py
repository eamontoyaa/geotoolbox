import numpy as np
import pytest

from geotoolbox.fragility import (
    fit_lognormal_fragility,
    lognormal_failure_probability,
)


def test_binary_fit_recovers_synthetic_curve():
    rng = np.random.default_rng(42)
    true_mu = np.log(80.0)
    true_sigma = 0.40
    intensity = np.exp(rng.uniform(np.log(10.0), np.log(250.0), 5_000))
    probability = lognormal_failure_probability(intensity, true_mu, true_sigma)
    failed = rng.binomial(1, probability)

    fit = fit_lognormal_fragility(intensity, failed)

    assert fit.median_capacity == pytest.approx(80.0, rel=0.08)
    assert fit.sigma == pytest.approx(true_sigma, rel=0.12)
    assert fit.covariance.shape == (2, 2)
    assert np.all(np.isfinite(fit.covariance))


def test_confidence_band_is_ordered_and_bounded():
    intensity = np.geomspace(10.0, 200.0, 100)
    failed = (intensity > 75.0).astype(int)
    fit = fit_lognormal_fragility(intensity, failed)

    probability, lower, upper = fit.confidence_band(np.array([20.0, 75.0, 150.0]))

    assert np.all((0.0 <= lower) & (lower <= probability))
    assert np.all((probability <= upper) & (upper <= 1.0))


def test_fit_requires_both_outcome_classes():
    with pytest.raises(ValueError, match="failures and non-failures"):
        fit_lognormal_fragility([10.0, 20.0, 30.0], [0, 0, 0])
