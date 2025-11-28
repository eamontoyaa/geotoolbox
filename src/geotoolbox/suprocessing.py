from scipy.ndimage import distance_transform_edt, label
import numpy as np


def merge_small_su(su, nodata=0, min_area_pixels=1000):
    """
    Merges small slope units patches into the nearest larger basin.

    Parameters:
        su (np.ndarray): Input raster with slope units IDs.
        nodata (int or float): Value representing NoData in the raster.
        min_area_pixels (int): Minimum size (in pixels) for a SU to be retained as is.

    Returns:
        np.ndarray: Modified raster where small su are merged into nearest larger ones.
    """
    # Identify unique basin areas excluding nodata
    unique, counts = np.unique(su[su != nodata], return_counts=True)
    # print(unique.dtype, counts.dtype)
    basin_areas = dict(zip(unique, counts))

    # Identify small and large su
    large_su = [k for k, v in basin_areas.items() if v >= min_area_pixels]
    small_mask = np.isin(su, list(set(basin_areas) - set(large_su)))
    large_mask = np.isin(su, large_su)

    # Compute distance to nearest large basin pixel
    _, indices = distance_transform_edt(~large_mask, return_indices=True)

    # Reassign small basin pixels to nearest large basin
    corrected_su = su.copy()
    rows, cols = np.where(small_mask)
    for r, c in zip(rows, cols):
        nearest_r, nearest_c = indices[0, r, c], indices[1, r, c]
        corrected_su[r, c] = su[nearest_r, nearest_c]

    return corrected_su


def relabel_su(su, nodata=0):
    # Find unique valid basin IDs
    unique_vals = np.unique(su[su != nodata])

    # Create a mapping from original values to sequential ones
    val_map = {old_val: new_val for new_val, old_val in enumerate(unique_vals)}

    # Create the relabeled array
    relabeled_su = np.full_like(su, fill_value=nodata)

    # Apply the mapping
    for old_val, new_val in val_map.items():
        relabeled_su[su == old_val] = new_val

    return relabeled_su
