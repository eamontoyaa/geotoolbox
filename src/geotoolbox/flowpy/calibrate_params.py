import os
import math
import numpy as np
import geopandas as gpd
import rasterio
from rasterio.features import rasterize
# from shapely.geometry import box
from typing import Dict, Tuple, List, Optional

# ============ UTILIDADES ============

def rasterize_footprint_gdf(
    footprint_gdf: gpd.GeoDataFrame,
    out_shape: Tuple[int, int],
    transform,
    dem_crs
) -> np.ndarray:
    """
    Rasteriza el footprint (valor 1 dentro, 0 fuera) a la grilla del DEM.
    Reproyecta si es necesario. Devuelve np.uint8 (0/1).
    """
    if footprint_gdf.crs != dem_crs:
        footprint_gdf = footprint_gdf.to_crs(dem_crs)

    # Unir geometrías para evitar huecos de borde
    geom = footprint_gdf.unary_union
    if geom.is_empty:
        raise ValueError("El GeoDataFrame de footprint no tiene geometrías válidas.")

    mask = rasterize(
        [(geom, 1)],
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="uint8",
        all_touched=True,  # útil para footprints delgados/irregulares
    )
    return mask


def load_observed_mask(
    *,
    footprint_gdf: Optional[gpd.GeoDataFrame],
    observed_raster_path: Optional[str],
    dem_shape: Tuple[int, int],
    dem_transform,
    dem_crs
) -> np.ndarray:
    """
    Crea/lee el mask observado (0/1) alineado a la grilla del DEM.
    Puedes pasar footprint_gdf (vector) o observed_raster_path (raster binario ya alineado).
    """
    if footprint_gdf is not None:
        obs = rasterize_footprint_gdf(footprint_gdf, dem_shape, dem_transform, dem_crs)
        return obs.astype(bool)

    if observed_raster_path is not None:
        with rasterio.open(observed_raster_path) as src:
            obs = src.read(1)
            # Verificar alineación básica
            if src.transform != dem_transform or src.crs != dem_crs or src.shape != dem_shape:
                raise ValueError("El raster observado no está alineado (shape/transform/CRS) con el DEM.")
            # Suponer binario (0/1 o 0/255). Considerar >0 como True
            return (obs > 0)

    raise ValueError("Debes proveer footprint_gdf o observed_raster_path.")


def binarize_prediction(z_delta: np.ndarray, nan_mask: np.ndarray, threshold: float = 0.0) -> np.ndarray:
    """
    Binariza la predicción: 1 donde z_delta > threshold, 0 en caso contrario. Aplica nan_mask.
    """
    pred = (z_delta > threshold)
    if nan_mask is not None:
        pred = pred & (~nan_mask)
    return pred


def compute_metrics(pred: np.ndarray, obs: np.ndarray) -> Dict[str, float]:
    """
    Métricas de solape y diagnóstico.
    """
    # Asegurar booleanos
    pred = pred.astype(bool)
    obs = obs.astype(bool)

    TP = np.logical_and(pred, obs).sum()
    FP = np.logical_and(pred, ~obs).sum()
    FN = np.logical_and(~pred, obs).sum()
    TN = np.logical_and(~pred, ~obs).sum()

    eps = 1e-12
    iou = TP / (TP + FP + FN + eps)
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    f1 = 2 * precision * recall / (precision + recall + eps)

    obs_area = obs.sum() + eps
    overreach = FP / obs_area      # cuánto “se pasa” respecto al footprint
    underreach = FN / obs_area     # cuánto “le falta” cubrir
    Tversky = TP / (TP + 0.75 * FP + 0.25 * FN + eps)  # Tversky = TP / (TP + α·FP + β·FN) α grande ⇒ castiga más el sobre-alcance
    return {
        "IoU": float(iou),
        "F1": float(f1),
        "precision": float(precision),
        "recall": float(recall),
        "overreach": float(overreach),
        "underreach": float(underreach),
        "Tversky": float(Tversky),
        "TP": int(TP),
        "FP": int(FP),
        "FN": int(FN),
        "TN": int(TN),
    }


# ============ OPTIMIZADOR ============

def optimize_flowpy_random_search(
    *,
    n_iter: int,
    param_space: Dict[str, Tuple[float, float]],
    run_flowpy_fn,              # inyecta la función run_flowpy
    working_dir: str,
    dem_path: str,
    release_path: str,
    save_dir=False,
    dem_transform=None,
    dem_crs=None,
    dem_shape: Tuple[int, int] = None,
    nan_mask: np.ndarray = None,
    observed_mask: np.ndarray = None,
    zdelta_threshold: float = 0.0,
    seed: int = 42,
    loss_weights: Dict[str, float] = None,
    early_stop_iou: float = 0.95
) -> Dict:
    """
    Búsqueda aleatoria simple sobre:
      - alpha (float)
      - exponent (float)
      - flux_threshold (float)
      - max_z (float)

    loss = (1 - IoU) + w_over*overreach + w_under*underreach
    Devolvemos el mejor set + historial.
    """
    rng = np.random.default_rng(seed)

    # Ponderaciones por defecto: penaliza más el faltar (FN) que el sobre-alcance (FP)
    if loss_weights is None:
        loss_weights = {"overreach": 0.75, "underreach": 0.25}
    print(loss_weights)
    # print("RELOADED 2 calibrate_params.py")
    def sample_param(name):
        lo, hi = param_space[name]
        return rng.uniform(lo, hi)

    history: List[Dict] = []
    best = {"loss": float("inf"), "params": None, "metrics": None}

    for k in range(n_iter):
        alpha = sample_param("alpha")
        exponent = sample_param("exponent")
        flux_threshold = sample_param("flux_threshold")
        max_z = sample_param("max_z")

        # Ejecutar FlowPy
        try:
            flux, z_delta, fp_ta, sl_ta, cell_counts, z_delta_sum, backcalc = run_flowpy_fn(
                alpha=alpha,
                exponent=exponent,
                working_dir=working_dir,
                dem_path=dem_path,
                release_path=release_path,
                flux_threshold=flux_threshold,
                max_z=max_z,
                save_dir=save_dir,
            )
        except Exception as e:
            # Registra intento fallido con pérdida alta
            history.append({
                "params": {"alpha": alpha, "exponent": exponent, "flux_threshold": flux_threshold, "max_z": max_z},
                "error": repr(e),
                "loss": 1e9,
            })
            continue

        pred = binarize_prediction(z_delta, nan_mask, threshold=zdelta_threshold)
        metrics = compute_metrics(pred, observed_mask)

        # # Función de pérdida: prioriza maximizar IoU y reducir faltantes
        loss = (1.0 - metrics["IoU"]) \
               + loss_weights["overreach"] * metrics["overreach"] \
               + loss_weights["underreach"] * metrics["underreach"]
        # loss = (1.0 - metrics["Tversky"])

        trial = {
            "k": k + 1,
            "params": {"alpha": float(alpha), "exponent": float(exponent),
                       "flux_threshold": float(flux_threshold), "max_z": float(max_z)},
            "metrics": metrics,
            "loss": float(loss),
        }
        history.append(trial)

        if loss < best["loss"]:
            best = {"loss": float(loss), "params": trial["params"], "metrics": metrics}

        # Early stop si el solape ya es muy alto
        if metrics["IoU"] >= early_stop_iou:
            break

    return {"best": best, "history": history}


def optimize_flowpy_bayes_optional(
    *,
    param_space: Dict[str, Tuple[float, float]],
    **kwargs
) -> Optional[Dict]:
    """
    Versión Bayesiana (skopt) — opcional si tienes instalado scikit-optimize.
    """
    try:
        from skopt import gp_minimize
        from skopt.space import Real
    except Exception:
        return None

    run_flowpy_fn = kwargs["run_flowpy_fn"]
    working_dir = kwargs["working_dir"]
    dem_path = kwargs["dem_path"]
    release_path = kwargs["release_path"]
    save_dir = kwargs.get("save_dir", False)
    nan_mask = kwargs["nan_mask"]
    observed_mask = kwargs["observed_mask"]
    zdelta_threshold = kwargs.get("zdelta_threshold", 0.0)
    loss_weights = kwargs.get("loss_weights", {"overreach": 0.25, "underreach": 0.50})

    space = [
        Real(*param_space["alpha"], name="alpha"),
        Real(*param_space["exponent"], name="exponent"),
        Real(*param_space["flux_threshold"], name="flux_threshold"),
        Real(*param_space["max_z"], name="max_z"),
    ]

    def objective(x):
        alpha, exponent, flux_threshold, max_z = x
        try:
            flux, z_delta, *_ = run_flowpy_fn(
                alpha=alpha,
                exponent=exponent,
                working_dir=working_dir,
                dem_path=dem_path,
                release_path=release_path,
                flux_threshold=flux_threshold,
                max_z=max_z,
                save_dir=save_dir,
            )
            pred = binarize_prediction(z_delta, nan_mask, threshold=zdelta_threshold)
            m = compute_metrics(pred, observed_mask)
            loss = (1.0 - m["IoU"]) \
                   + loss_weights["overreach"] * m["overreach"] \
                   + loss_weights["underreach"] * m["underreach"]
            return float(loss)
        except Exception:
            return 1e9

    res = gp_minimize(
        objective,
        space,
        n_calls=40,
        n_initial_points=12,
        acq_func="EI",
        random_state=42,
    )

    best = {"alpha": float(res.x[0]), "exponent": float(res.x[1]),
            "flux_threshold": float(res.x[2]), "max_z": float(res.x[3])}
    return {"best_params": best, "best_loss": float(res.fun)}


# ============ “WRAPPER” DE USO RÁPIDO ============

def calibrate_flowpy(
    *,
    run_flowpy_fn,                  # pasa aquí tu run_flowpy
    working_dir: str,
    dem_path: str,
    release_path: str,
    # Geometría/CRS del DEM (ya los tienes de tu código)
    transform_DEM,
    crs_DEM,
    DEM: np.ndarray,
    nodata_DEM,
    # Observación: usa uno u otro
    footprint_gdf: Optional[gpd.GeoDataFrame] = None,
    observed_raster_path: Optional[str] = None,
    # Espacio de parámetros (ajusta rangos a tu caso)
    param_space: Dict[str, Tuple[float, float]] = None,
    # Config
    n_iter: int = 30,
    seed: int = 42,
    save_dir: bool = False,
    zdelta_threshold: float = 0.0,
    loss_weights: Dict[str, float] = None,
    try_bayes: bool = False,
) -> Dict:
    """
    Orquesta todo: construye máscara observada, hace búsqueda aleatoria (y opcional Bayes),
    y devuelve el mejor set y el historial.
    """
    if param_space is None:
        param_space = {
            "alpha": (10.0, 45.0),          # grados (según tu caso)
            "exponent": (3.0, 9.0),         # adimensional
            "flux_threshold": (1e-5, 1e-3), # m2/s aprox (ajusta)
            "max_z": (2.0, 50.0),           # m (ajusta)
        }

    dem_shape = DEM.shape
    nan_mask = (DEM == nodata_DEM)

    observed_mask = load_observed_mask(
        footprint_gdf=footprint_gdf,
        observed_raster_path=observed_raster_path,
        dem_shape=dem_shape,
        dem_transform=transform_DEM,
        dem_crs=crs_DEM,
    )

    rs = optimize_flowpy_random_search(
        n_iter=n_iter,
        param_space=param_space,
        run_flowpy_fn=run_flowpy_fn,
        working_dir=working_dir,
        dem_path=dem_path,
        release_path=release_path,
        save_dir=save_dir,
        dem_transform=transform_DEM,
        dem_crs=crs_DEM,
        dem_shape=dem_shape,
        nan_mask=nan_mask,
        observed_mask=observed_mask,
        zdelta_threshold=zdelta_threshold,
        seed=seed,
        loss_weights=loss_weights,
    )

    result = {"random_search": rs}

    if try_bayes:
        bayes_res = optimize_flowpy_bayes_optional(
            param_space=param_space,
            run_flowpy_fn=run_flowpy_fn,
            working_dir=working_dir,
            dem_path=dem_path,
            release_path=release_path,
            save_dir=save_dir,
            nan_mask=nan_mask,
            observed_mask=observed_mask,
            zdelta_threshold=zdelta_threshold,
            loss_weights=loss_weights,
        )
        result["bayes"] = bayes_res

    return result
