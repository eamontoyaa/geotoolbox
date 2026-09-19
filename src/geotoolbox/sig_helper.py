import numpy as np
import geopandas as gpd
import matplotlib as mpl
import rasterio
from rasterio import features
from rasterio.enums import Resampling
from rasterio.windows import from_bounds
from rasterio.transform import Affine, array_bounds
from rasterio.warp import reproject

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence, Tuple, Union


@dataclass(frozen=True)
class RasterLayer:
    """A single raster band and the grid metadata required to interpret it."""

    data: np.ma.MaskedArray
    transform: Affine
    crs: Any
    nodata: float | int | None
    profile: dict[str, Any]
    name: str | None = None

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return array_bounds(*self.shape, self.transform)

    @property
    def resolution(self) -> tuple[float, float]:
        return abs(self.transform.a), abs(self.transform.e)


def grids_match(first: RasterLayer, second: RasterLayer, *, atol: float = 1e-9) -> bool:
    """Return whether two layers use the same shape, CRS, and affine grid."""

    return (
        first.shape == second.shape
        and first.crs == second.crs
        and np.allclose(tuple(first.transform), tuple(second.transform), atol=atol, rtol=0)
    )


def read_raster_layer(
    file_path: Union[str, Path],
    *,
    reference: RasterLayer | None = None,
    resampling: Resampling | str = Resampling.bilinear,
    dtype: str | np.dtype = "float64",
    name: str | None = None,
) -> RasterLayer:
    """Read one band and optionally align it to a reference grid.

    Alignment uses the reference shape, transform, and CRS. Continuous fields
    should use bilinear resampling; categorical rasters must use ``nearest``.
    Source masks, explicit nodata values, NaNs, and infinities become a single
    NumPy mask in the returned layer.
    """

    if isinstance(resampling, str):
        try:
            resampling = Resampling[resampling]
        except KeyError as exc:
            raise ValueError(f"Unknown resampling method: {resampling!r}.") from exc

    path = Path(file_path) if not str(file_path).startswith(("http://", "https://")) else file_path
    target_dtype = np.dtype(dtype)
    with rasterio.open(path) as src:
        source = src.read(1, masked=True).astype(target_dtype)
        source_data = np.ma.filled(source, np.nan)
        source_data[~np.isfinite(source_data)] = np.nan
        profile = src.profile.copy()
        native = RasterLayer(
            data=np.ma.masked_invalid(source_data),
            transform=src.transform,
            crs=src.crs,
            nodata=src.nodata,
            profile=profile,
            name=name or Path(str(file_path)).stem,
        )

        if reference is None or grids_match(native, reference):
            return native
        if src.crs is None or reference.crs is None:
            raise ValueError(
                "Both rasters need a CRS before a grid alignment can be performed."
            )

        destination = np.full(reference.shape, np.nan, dtype=target_dtype)
        reproject(
            source=source_data,
            destination=destination,
            src_transform=src.transform,
            src_crs=src.crs,
            src_nodata=np.nan,
            dst_transform=reference.transform,
            dst_crs=reference.crs,
            dst_nodata=np.nan,
            resampling=resampling,
            init_dest_nodata=True,
        )

    aligned_profile = profile.copy()
    aligned_profile.update(
        height=reference.shape[0],
        width=reference.shape[1],
        transform=reference.transform,
        crs=reference.crs,
        dtype=target_dtype.name,
        nodata=np.nan,
    )
    return RasterLayer(
        data=np.ma.masked_invalid(destination),
        transform=reference.transform,
        crs=reference.crs,
        nodata=np.nan,
        profile=aligned_profile,
        name=name or Path(str(file_path)).stem,
    )


def align_raster_stack(
    rasters: Mapping[str, Union[str, Path]],
    *,
    reference: str | RasterLayer | None = None,
    categorical: Sequence[str] = (),
) -> tuple[dict[str, RasterLayer], np.ndarray]:
    """Load raster paths on one grid and return their common valid-data mask.

    Parameters
    ----------
    rasters
        Mapping from stable layer names to raster paths. Mapping order is
        preserved and the first layer is the default reference.
    reference
        Layer name or previously loaded :class:`RasterLayer` defining the grid.
    categorical
        Names to align with nearest-neighbour resampling. All remaining layers
        use bilinear resampling.
    """

    if not rasters:
        raise ValueError("rasters must contain at least one named path.")
    unknown = set(categorical).difference(rasters)
    if unknown:
        raise KeyError(f"Unknown categorical raster names: {sorted(unknown)}")

    if isinstance(reference, RasterLayer):
        reference_layer = reference
    else:
        reference_name = reference or next(iter(rasters))
        if reference_name not in rasters:
            raise KeyError(f"Unknown reference raster: {reference_name!r}.")
        reference_layer = read_raster_layer(
            rasters[reference_name], name=reference_name
        )

    layers: dict[str, RasterLayer] = {}
    for layer_name, path in rasters.items():
        method = Resampling.nearest if layer_name in categorical else Resampling.bilinear
        layers[layer_name] = read_raster_layer(
            path,
            reference=reference_layer,
            resampling=method,
            name=layer_name,
        )

    valid = np.ones(reference_layer.shape, dtype=bool)
    for layer in layers.values():
        valid &= ~np.ma.getmaskarray(layer.data)
        valid &= np.isfinite(np.ma.getdata(layer.data))
    return layers, valid


def write_raster_like(
    file_path: Union[str, Path],
    data: np.ndarray | np.ma.MaskedArray,
    reference: RasterLayer,
    *,
    nodata: float | int = -9999.0,
    dtype: str | np.dtype = "float32",
    compress: str = "deflate",
) -> str:
    """Write a 2-D result while preserving a reference layer's exact grid."""

    output = np.ma.masked_invalid(np.ma.asarray(data))
    if output.shape != reference.shape:
        raise ValueError(
            f"data has shape {output.shape}; expected reference shape {reference.shape}."
        )
    target_dtype = np.dtype(dtype)
    profile = reference.profile.copy()
    profile.update(
        driver="GTiff",
        height=reference.shape[0],
        width=reference.shape[1],
        count=1,
        dtype=target_dtype.name,
        crs=reference.crs,
        transform=reference.transform,
        nodata=nodata,
        compress=compress,
    )
    destination = Path(file_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(destination, "w", **profile) as dst:
        dst.write(output.filled(nodata).astype(target_dtype), 1)
    return str(destination)


def load_raster(file_path, masked=True):
    """
    Load a raster file (e.g., DEM or other geospatial raster) and extract its key metadata.

    Parameters:
        file_path (str): Path to the raster file (e.g., a GeoTIFF).

    Returns:
        tuple:
            - data (np.ndarray): 2D array of raster values (first band).
            - transform (Affine): Affine transformation mapping pixel coordinates to spatial coordinates.
            - bounds (BoundingBox): Bounding box of the raster in spatial coordinates.
            - crs (CRS): Coordinate Reference System of the raster.
            - nodata (float or int or None): Value used to represent no-data pixels, if defined.
    """
    with rasterio.open(file_path) as src:
        data = src.read(1, masked=masked)  # Read the first band, optionally masked
        transform = src.transform
        bounds = src.bounds
        crs = src.crs
        nodata = src.nodata
        # affine = src.affine
    return data, transform, bounds, crs, nodata


def save_raster(
    file_path,
    array,
    crs,
    transform,
    format="tif",
    *,
    nodata=None,
    compress="deflate",
):
    """
    Save a NumPy array as a GeoTIFF using rasterio.

    Parameters:
        file_path (str): Path to the raster file (e.g., a GeoTIFF)..
        array (np.ndarray): 2D array to be saved.
        crs (str or CRS): Coordinate reference system.
        transform (Affine): Affine transform for the raster.
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    array = np.asarray(array)
    if array.ndim != 2:
        raise ValueError("array must be two-dimensional.")
    driver = "GTiff" if format.lower() in {"tif", "tiff", "gtiff"} else "AAIGrid"
    creation_options = {"compress": compress} if driver == "GTiff" and compress else {}
    with rasterio.open(
        file_path,
        "w",
        driver=driver,
        height=array.shape[0],
        width=array.shape[1],
        count=1,
        dtype=array.dtype,
        crs=crs,
        transform=transform,
        nodata=nodata,
        **creation_options,
    ) as dst:
        dst.write(array, 1)


def convert_tif_to_asc(tif_path, asc_path, nodata_value=-9999, dtype=float):
    with rasterio.open(tif_path) as src:
        data = src.read(1, masked=True).astype(dtype)
        transform = src.transform
        data = data.filled(nodata_value)

        ncols, nrows = src.width, src.height
        xllcorner = transform.c
        yllcorner = (
            transform.f + transform.e * nrows
        )  # rasterio uses top-left as origin
        cellsize = transform.a

    asc_path = Path(asc_path)
    asc_path.parent.mkdir(parents=True, exist_ok=True)
    with asc_path.open("w", encoding="utf-8") as f:
        f.write(f"ncols         {ncols}\n")
        f.write(f"nrows         {nrows}\n")
        f.write(f"xllcorner     {xllcorner:.3f}\n")
        f.write(f"yllcorner     {yllcorner:.3f}\n")
        f.write(f"cellsize      {cellsize}\n")
        f.write(f"NODATA_value  {nodata_value}\n")

        for row in data:
            f.write(" ".join(map(str, row)) + "\n")


def get_aspect(dem, cellsize=1):
    """Calculate the azimuth of the slope dip direction from the North.

    Positive values are clockwise from the North. The azimuth is returned in
    [°], ranging from 0 to 360.

    Parameters
    ----------
    dem : (m, n) array
        2D array with the digital elevation model (DEM) of the area.
    cellsize : int, optional
        Cell size of the raster representing the DEM. Its default value is 1.

    Returns
    -------
    azimuth : (m, n) array
        Spatial distribution of the azimuth of the slope dip direction.
    """
    dzdy, dzdx = np.gradient(dem, cellsize)
    # dzdx *= -1  # Adequate sign for the x component
    # return np.degrees(0.5 * np.pi - np.arctan2(dzdy, dzdx)) % 360
    return np.degrees(np.arctan2(dzdy, dzdx) - 0.5 * np.pi) % 360


def get_slope(dem, cellsize=1):
    """Calculate the slope (or inclination) of the terrain.

    Parameters
    ----------
    dem : (m, n) array
        2D array with the digital elevation model (DEM) of the area.
    cellsize : int, optional
        Cell size of the raster representing the DEM. Units must be consistent
        with the units of ``dem``. Its default value is 1.

    Returns
    -------
    slope : (m, n) array
        Spatial distribution of the slope, in degrees.
    """
    dzdy, dzdx = np.gradient(dem, cellsize)
    dzdx *= -1  # Adequate sign for the x component
    # return np.degrees(np.arctan(np.sqrt(dzdx**2 + dzdy**2)))
    return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))
    # dx = sobel(dem, axis=1) / (8 * cellsize)
    # dy = sobel(dem, axis=0) / (8 * cellsize)
    # slope_rad = np.arctan(np.hypot(dx, dy))


def get_hillshade(dem, cellsize=1, azimuth=315, altitude=30):
    """Generate a hillshade image from the DEM.

    Parameters
    ----------
    dem : (m, n) array
        2D array with the digital elevation model (DEM) of the area
    cellsize : int, optional
        Cell size of the raster representing the DEM. Units must be consistent
        with the units of ``dem``. Its default value is 1.
    azimuth : int or float, optional
        Azimuth of the sun, in [°]. Its default value is 315.
    altitude : int or float, optional
        Angle from the horizontal plane to the sun, in degrees, indicating the
        altitude of the sun. Its default value is 30.

    Returns
    -------
    hillshade : (m, n) array
        2D array with the hillshade raster.
    """
    if not 0.0 <= altitude <= 90.0:
        raise ValueError("altitude must be between 0 and 90 degrees.")
    azimuth_r = np.radians(float(azimuth) % 360.0)
    altitude_r = np.radians(altitude)
    slope_r = np.radians(get_slope(dem, cellsize))
    aspect_r = np.radians(get_aspect(dem, cellsize))
    reflectance = (
        np.sin(altitude_r) * np.cos(slope_r)
        + np.cos(altitude_r)
        * np.sin(slope_r)
        * np.cos(azimuth_r - aspect_r)
    )
    return 255.0 * np.clip(reflectance, 0.0, 1.0)


def get_multidir_hillshade(dem, cellsize=1.0, azimuths=None, altitude=45, gamma=1.0):
    """
    Generate a multidirectional hillshade from a DEM using multiple azimuths.

    Parameters:
    - dem: 2D numpy array of elevation
    - cellsize: spatial resolution of DEM (assumes square pixels)
    - azimuths: list of azimuths (in degrees) to simulate light from
    - altitude: sun elevation angle in degrees
    - plot: whether to show the resulting hillshade

    Returns:
    - hillshade: 2D numpy array of multidirectional hillshade
    """
    if azimuths is None:
        # 8 compass directions
        azimuths = [45, 135, 225, 315]

    if len(azimuths) == 0:
        raise ValueError("azimuths must contain at least one direction.")
    if not np.isfinite(gamma) or gamma <= 0.0:
        raise ValueError("gamma must be finite and positive.")
    hillshades = [
        get_hillshade(dem, cellsize, azimuth, altitude) / 255.0
        for azimuth in azimuths
    ]
    hillshade = np.mean(hillshades, axis=0)
    return 255.0 * np.power(np.clip(hillshade, 0.0, 1.0), 1.0 / gamma)


# def vectorize(data, nodata, transform, crs, name="value"):
#     feats_gen = features.shapes(
#         data,
#         mask=data != nodata,
#         transform=transform,
#         connectivity=8,
#     )
#     feats = [
#         {"geometry": geom, "properties": {name: val}} for geom, val in list(feats_gen)
#     ]

#     # parse to geopandas for plotting / writing to file
#     gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
#     gdf[name] = gdf[name].astype(data.dtype)
#     return gdf


def vectorize(
    data,
    nodata,
    transform,
    crs,
    name="value",
    *,
    connectivity=8,
    dissolve=False,
):
    """Convert valid raster regions to a GeoDataFrame.

    Masked arrays, an undefined nodata value, and NaN nodata values are all
    handled explicitly. Set dissolve=True to return one (possibly multipart)
    feature per raster value, which is appropriate for categorical unit IDs.
    """

    if connectivity not in {4, 8}:
        raise ValueError("connectivity must be either 4 or 8.")
    if not isinstance(name, str) or not name:
        raise ValueError("name must be a non-empty string.")

    values = np.asarray(np.ma.getdata(data))
    if values.ndim != 2:
        raise ValueError("data must be a two-dimensional array.")
    mask = ~np.ma.getmaskarray(data)
    if nodata is not None:
        is_nan_nodata = np.isscalar(nodata) and bool(np.asarray(np.isnan(nodata)))
        mask &= ~np.isnan(values) if is_nan_nodata else values != nodata
    if np.issubdtype(values.dtype, np.floating):
        mask &= np.isfinite(values)

    if not np.any(mask):
        return gpd.GeoDataFrame(
            {name: np.array([], dtype=values.dtype)},
            geometry=gpd.GeoSeries([], crs=crs),
            crs=crs,
        )

    supported = values.dtype in {
        np.dtype("int16"),
        np.dtype("int32"),
        np.dtype("uint8"),
        np.dtype("uint16"),
        np.dtype("float32"),
        np.dtype("float64"),
    }
    shape_values = values if supported else values.astype(np.float64)

    feats_gen = features.shapes(
        shape_values,
        mask=mask,
        transform=transform,
        connectivity=connectivity,
    )

    feats = [{"geometry": geom, "properties": {name: val}} for geom, val in feats_gen]

    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    gdf[name] = gdf[name].astype(values.dtype)
    gdf = gdf.loc[~gdf.geometry.is_empty & gdf.geometry.notna()].copy()
    if dissolve and not gdf.empty:
        gdf = gdf.dissolve(by=name, as_index=False, sort=True)
    return gdf.reset_index(drop=True)


def make_discrete_legend(
    n_classes: int,
    cmap: mpl.colors.Colormap,
    labels: list[str] | None = None,
    label_prefix: str = "Type",
    markersize: int = 10,
    alpha: float = 1.0,
):
    if labels is None:
        labels = [f"{label_prefix} {i+1}" for i in range(n_classes)]
    if n_classes <= 0:
        raise ValueError("n_classes must be positive.")
    if len(labels) != n_classes:
        raise ValueError("labels must contain exactly n_classes entries.")
    colours = cmap((np.arange(n_classes) + 0.5) / n_classes)
    handles = [
        mpl.lines.Line2D(
            [],
            [],
            marker="s",
            color="w",
            markerfacecolor=col,
            markersize=markersize,
            label=lab,
            alpha=alpha,
        )
        for col, lab in zip(colours, labels)
    ]
    return handles


def crop_raster_by_bbox(
    src_path: Union[str, Path],
    dst_path: Union[str, Path],
    corner1: Tuple[float, float],   # (x1, y1)
    corner2: Tuple[float, float],   # (x2, y2)
    allow_outside: bool = False,                 # True -> allow bbox partly outside (fill with nodata)
):
    """
    Extract a rectangular subset from `src_path` and save to `dst_path`.

    - corner1/corner2 are opposite corners of your desired box (order doesn't matter).
    - If bbox_crs is provided and differs from the raster CRS, the bbox is reprojected.
    - If allow_outside=True, areas outside the source are filled with nodata.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)

    with rasterio.open(src_path) as src:
        x1, y1 = corner1
        x2, y2 = corner2

        # Normalize bbox (min/max in the bbox CRS)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])
        if xmin == xmax or ymin == ymax:
            raise ValueError("The crop bounding box must have positive area.")
        if not allow_outside and (
            xmin < src.bounds.left
            or xmax > src.bounds.right
            or ymin < src.bounds.bottom
            or ymax > src.bounds.top
        ):
            raise ValueError(
                "The crop bounding box extends beyond the raster; "
                "set allow_outside=True to pad it."
            )

        # Compute window
        window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        # Round to full pixels
        window = window.round_offsets().round_lengths()

        # Read data and build new transform
        fill_value = src.nodata if src.nodata is not None else 0
        data = src.read(
            window=window,
            boundless=allow_outside,
            fill_value=fill_value,
        )
        new_transform: Affine = rasterio.windows.transform(window, src.transform)

        profile = src.profile.copy()
        profile.update(
            {
                "height": data.shape[1],
                "width": data.shape[2],
                "transform": new_transform,
                "driver": "GTiff",
            }
        )

        dst_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

    return str(dst_path)
