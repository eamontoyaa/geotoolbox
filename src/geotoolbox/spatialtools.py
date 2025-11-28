import numpy as np
import geopandas as gpd
import matplotlib as mpl
import rasterio
from rasterio import features
from rasterio.windows import from_bounds
from rasterio.transform import Affine

from pathlib import Path
from typing import Tuple, Union


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


def save_raster(file_path, array, crs, transform, format="tif"):
    """
    Save a NumPy array as a GeoTIFF using rasterio.

    Parameters:
        file_path (str): Path to the raster file (e.g., a GeoTIFF)..
        array (np.ndarray): 2D array to be saved.
        crs (str or CRS): Coordinate reference system.
        transform (Affine): Affine transform for the raster.
    """
    driver = "GTiff" if format == "tif" else "AAIGrid"
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
    ) as dst:
        dst.write(array, 1)


def convert_tif_to_asc(tif_path, asc_path, nodata_value=-9999, dtype=float):
    with rasterio.open(tif_path) as src:
        data = src.read(1)
        transform = src.transform

        # Use nodata from tif if not explicitly provided
        nodata_org = src.nodata
        data[data == nodata_org] = nodata_value

        # Convert to the specified data type
        data = data.astype(dtype)

        ncols, nrows = src.width, src.height
        xllcorner = transform.c
        yllcorner = (
            transform.f + transform.e * nrows
        )  # rasterio uses top-left as origin
        cellsize = transform.a

    with open(asc_path, "w") as f:
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
    azimuth_r = np.radians(azimuth)
    altitude_r = np.radians(altitude)
    slope_r = np.radians(get_slope(dem, cellsize))
    azimuth_r = np.radians(get_aspect(dem, cellsize))
    # Lambertian reflectance
    term1 = (
        np.cos(azimuth_r - azimuth_r)
        * np.sin(slope_r)
        * np.sin(0.5 * np.pi - altitude_r)
    )
    term2 = np.cos(slope_r) * np.cos(0.5 * np.pi - altitude_r)
    reflectance = term1 + term2
    ptp = np.nanmax(reflectance) - np.nanmin(reflectance)
    return 255 * (reflectance - np.nanmin(reflectance)) / ptp


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

    # Compute slope and aspect
    slope_rad = np.radians(get_slope(dem, cellsize=cellsize))
    aspect_rad = np.radians(get_aspect(dem, cellsize=cellsize))

    # Convert altitude to radians
    zenith_rad = np.radians(90 - altitude)

    # Accumulate hillshades from each direction
    hs_combined = np.zeros_like(dem, dtype=np.float32)

    for azimuth_deg in azimuths:
        az_rad = np.radians(360.0 - azimuth_deg + 90.0)
        az_rad = np.mod(az_rad, 2 * np.pi)

        shaded = np.cos(zenith_rad) * np.cos(slope_rad) + np.sin(zenith_rad) * np.sin(
            slope_rad
        ) * np.cos(az_rad - aspect_rad)
        hs = np.clip(shaded, 0, 1)
        hs_combined += hs

    hillshade = hs_combined / len(azimuths)

    return hillshade * gamma


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


def vectorize(data, nodata, transform, crs, name="value"):
    # Handle NaN and numeric nodata appropriately
    if np.isnan(nodata):
        mask = ~np.isnan(data)  # mask is True where data is valid
    else:
        mask = data != nodata

    feats_gen = features.shapes(
        data,
        mask=mask,
        transform=transform,
        connectivity=8,
    )

    feats = [{"geometry": geom, "properties": {name: val}} for geom, val in feats_gen]

    gdf = gpd.GeoDataFrame.from_features(feats, crs=crs)
    gdf[name] = gdf[name].astype(data.dtype)
    # gdf = gdf[gdf.is_valid & ~gdf.is_empty]
    # gdf = gdf[gdf.is_valid]
    return gdf


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
    colours = cmap(range(n_classes))
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
        src_crs = src.crs
        x1, y1 = corner1
        x2, y2 = corner2

        # Normalize bbox (min/max in the bbox CRS)
        xmin, xmax = sorted([x1, x2])
        ymin, ymax = sorted([y1, y2])

        # Compute window
        window = from_bounds(xmin, ymin, xmax, ymax, transform=src.transform)
        # Round to full pixels
        window = window.round_offsets().round_lengths()

        # Read data and build new transform
        data = src.read(window=window, boundless=allow_outside, fill_value=src.nodata)
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

        # Write out
        with rasterio.open(dst_path, "w", **profile) as dst:
            dst.write(data)

    return str(dst_path)