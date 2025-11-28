# fragility_lognormal.py
import numpy as np
from math import log, sqrt, erfc
from dataclasses import dataclass
from typing import Optional, Tuple, Dict
from scipy.optimize import minimize

# try:
#     from numba import njit
#     njit_available = True
# except Exception:
#     def njit(*args, **kwargs): 
#         def wrap(f): return f
#         return wrap
#     njit_available = False

# --------- Utilidades numéricas (robustas y numba-friendly) ---------

def _softplus(x: np.ndarray) -> np.ndarray:
    # estable: log(1+exp(x))
    out = np.empty_like(x)
    # corte para evitar overflow en exp
    big = x > 20.0
    small = ~big
    out[big] = x[big]          # ~ lineal para x grande
    out[small] = np.log1p(np.exp(x[small]))
    return out

def _ndtr(z: np.ndarray) -> np.ndarray:
    """
    CDF de Normal estándar robusta y numba-friendly:
    Phi(z) = 0.5 * erfc(-z / sqrt(2))
    """
    inv_s2 = 1.0 / sqrt(2.0)
    return 0.5 * erfc(-z * inv_s2)

def lognorm_cdf_vectorized(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """
    CDF lognormal F(x; mu, sigma) = Phi((ln x - mu)/sigma).
    - Maneja x<=0 devolviendo 0.
    - Evita under/overflow en colas con erfc.
    """
    x = np.asarray(x, dtype=float)
    out = np.zeros_like(x)
    mask = x > 0.0
    if sigma <= 0:
        # Evitar sigma no positiva (devolver escalón en la mediana)
        out[mask] = (np.log(x[mask]) >= mu).astype(float)
        return out
    z = (np.log(x[mask]) - mu) / sigma
    out[mask] = _ndtr(z)
    return out

# --------- Modelo y verosimilitud multi-estado ---------

@dataclass
class FragilitySpec:
    levels: np.ndarray            # (K,) umbrales de daño (ascendentes)
    shared_sigma: bool = True     # True: una sigma; False: sigma_k por estado
    reg_lambda: float = 0.0       # penalización L2 (para separación)
    clip_eps: float = 1e-12       # evita log(0)
    max_iter: int = 2000
    tol: float = 1e-9

def _unpack_params(theta: np.ndarray, K: int, shared_sigma: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Re-parametrización sin restricciones:
      mu_1 = a
      mu_2 = a + softplus(d1)
      mu_3 = a + softplus(d1) + softplus(d2) ...
      sigma  = softplus(s)  (o sigma_k = softplus(s_k))
    """
    a = theta[0]
    deltas = theta[1:K]                   # K-1
    mus = np.empty(K, dtype=float)
    mus[0] = a
    if K > 1:
        mus[1:] = a + np.cumsum(_softplus(deltas))

    if shared_sigma:
        sigma = np.array([_softplus(np.array([theta[K]]) )[0]])
        sigmas = np.full(K, sigma[0], dtype=float)
        next_idx = K + 1
    else:
        s = theta[K:K+K]                  # K
        sigmas = _softplus(s)
        next_idx = K + K

    return mus, sigmas

def _nll_lognormal(params_u: np.ndarray,
                   im: np.ndarray,
                   X_states: np.ndarray,
                   spec: FragilitySpec,
                   weights: Optional[np.ndarray] = None) -> float:
    """
    NLL para K estados (exclusivos y exhaustivos).
    X_states: shape (K, N)  -> 1 si obs i pertenece a estado k, 0 si no.
    """
    K, N = X_states.shape
    mus, sigmas = _unpack_params(params_u, K, spec.shared_sigma)

    # F_k = P(D >= DS_k) con DS_{K+1} = +inf, F_{K+1} = 0; F_0 = 1
    F = np.empty((K+2, N), dtype=float)
    F[0, :] = 1.0
    for k in range(1, K+1):
        F[k, :] = lognorm_cdf_vectorized(im, mus[k-1], sigmas[k-1])
    F[K+1, :] = 0.0

    # p_k = P(DS_k) = F_k - F_{k+1}
    P = F[:-1, :] - F[1:, :]
    P = np.clip(P, spec.clip_eps, 1.0 - spec.clip_eps)

    if weights is None:
        ll = np.sum(X_states * np.log(P))
    else:
        w = weights.reshape(1, -1)
        ll = np.sum(w * X_states * np.log(P))

    # Penalización L2 para suavizar separaciones
    if spec.reg_lambda > 0.0:
        reg = spec.reg_lambda * (np.sum(mus**2) + np.sum(sigmas**2))
    else:
        reg = 0.0

    return -(ll - 0.5 * reg)

# --------- Ajuste por MV ---------

def _default_init_params(im: np.ndarray, X_states: np.ndarray,
                         shared_sigma: bool) -> np.ndarray:
    """Inicializa mu_k con percentiles crecientes de ln(IM); sigma(s) desde su std."""
    ln_im = np.log(np.clip(im, 1e-12, None))
    K, _ = X_states.shape

    # Percentiles por estado (distribuidos entre 35..65 para evitar extremos)
    ps = np.linspace(35, 65, K)
    mus0 = np.percentile(ln_im, ps)

    if shared_sigma:
        sigmas0 = np.array([max(ln_im.std(ddof=1), 0.1)])
        theta0 = np.concatenate(([mus0[0]], np.diff(mus0), sigmas0))
    else:
        sigmas0 = np.full(K, max(ln_im.std(ddof=1)*0.8, 0.1))
        theta0 = np.concatenate(([mus0[0]], np.diff(mus0), sigmas0))
    return theta0

def fit_fragility_mle(im: np.ndarray,
                      damage: np.ndarray,
                      levels: np.ndarray,
                      weights: Optional[np.ndarray] = None,
                      shared_sigma: bool = True,
                      reg_lambda: float = 0.0,
                      x0: Optional[np.ndarray] = None,
                      max_iter: int = 2000,
                      tol: float = 1e-9) -> Dict[str, np.ndarray]:
    """
    Ajusta curvas de fragilidad lognormal (K estados) por MV.
    - im      : (N,) medida de intensidad (IM)
    - damage  : (N,) medida de daño/ respuesta continua
    - levels  : (K,) umbrales crecientes que definen estados exclusivos:
                DS_k = [levels[k-1], levels[k]) con level[0] = DS1, ..., DS_K = [levels[K-1], +inf)
    - weights : (N,) opcional
    """
    levels = np.asarray(levels, dtype=float)
    assert np.all(np.diff(levels) > 0), "levels debe ser creciente"

    # Construye matriz de estados (K, N)
    K = len(levels)
    N = len(damage)
    X = np.zeros((K, N), dtype=float)
    low = np.hstack(([ -np.inf ], levels[:-1]))
    high = np.hstack((levels, [ np.inf ]))
    for k in range(K):
        X[k, :] = (damage >= low[k]) & (damage < high[k])

    spec = FragilitySpec(levels=levels, shared_sigma=shared_sigma,
                         reg_lambda=reg_lambda, max_iter=max_iter, tol=tol)

    if x0 is None:
        x0 = _default_init_params(im, X, shared_sigma)

    # Optimización sin restricciones (L-BFGS-B)
    fun = lambda th: _nll_lognormal(th, im, X, spec, weights)
    res = minimize(fun, x0, method="L-BFGS-B",
                   options=dict(maxiter=spec.max_iter, ftol=spec.tol))

    if not res.success:
        raise RuntimeError(f"Optimización fallida: {res.message}")

    mus, sigmas = _unpack_params(res.x, K, shared_sigma)
    return {
        "success": res.success,
        "message": res.message,
        "mu": mus,
        "sigma": sigmas,
        "theta_opt": res.x,
        "nll": res.fun,
        "spec": spec,
    }

# --------- Caso binario (excedencia de un umbral) ---------

def fit_binary_lognormal_mle(im: np.ndarray,
                             exceed: np.ndarray,
                             reg_lambda: float = 0.0) -> Dict[str, float]:
    """
    Ajusta P(exceed=1 | IM) = Phi((ln IM - mu)/sigma)
    exceed: {0,1}
    """
    im = np.asarray(im, float)
    y = np.asarray(exceed, float)
    spec = FragilitySpec(levels=np.array([0.0]), shared_sigma=True,
                         reg_lambda=reg_lambda)

    # Construimos X con dos “estados” implícitos: {0,1}
    X = np.vstack([1.0 - y, y])  # (2, N), pero el modelo usa K=1 estado de “daño”

    # Para adaptar nll anterior: simulamos K=1 con P(DS1) = F - 0
    # y X_states = y (solo el estado de daño)
    def nll_theta(th):
        mu = th[0]
        sigma = _softplus(np.array([th[1]]) )[0]
        F = lognorm_cdf_vectorized(im, mu, sigma)
        F = np.clip(F, spec.clip_eps, 1.0 - spec.clip_eps)
        ll = np.sum(y * np.log(F) + (1.0 - y) * np.log(1.0 - F))
        reg = 0.5 * reg_lambda * (mu**2 + sigma**2)
        return -(ll - reg)

    # Inicialización
    ln_im = np.log(np.clip(im, 1e-12, None))
    mu0 = np.median(ln_im)
    s0 = max(ln_im.std(ddof=1), 0.1)
    x0 = np.array([mu0, log(np.expm1(s0))])   # inversa pobre de softplus

    res = minimize(nll_theta, x0, method="L-BFGS-B", options=dict(maxiter=2000))
    if not res.success:
        raise RuntimeError(f"Optimización fallida: {res.message}")

    mu = res.x[0]
    sigma = _softplus(np.array([res.x[1]]) )[0]
    return {"mu": float(mu), "sigma": float(sigma), "nll": float(res.fun)}

# --------- Predicción / utilidades ---------

def predict_exceed_prob(im: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Devuelve P(exceed=1|IM) con la forma lognormal binaria."""
    return lognorm_cdf_vectorized(im, mu, sigma)

def predict_state_probs(im: np.ndarray, mu: np.ndarray, sigma: np.ndarray) -> np.ndarray:
    """Devuelve P(DS_k) para K estados."""
    K = len(mu)
    N = len(im)
    F = np.empty((K+2, N))
    F[0] = 1.0
    for k in range(1, K+1):
        F[k] = lognorm_cdf_vectorized(im, mu[k-1], sigma[k-1])
    F[K+1] = 0.0
    P = F[:-1] - F[1:]
    return np.clip(P, 1e-12, 1.0 - 1e-12)
