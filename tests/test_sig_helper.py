import numpy as np
import rasterio
from affine import Affine

from geotoolbox.sig_helper import (
    align_raster_stack,
    get_hillshade,
    grids_match,
    read_raster_layer,
    vectorize,
    write_raster_like,
)


def test_vectorize_supports_nodata_and_dissolves_unit_ids():
    values = np.array(
        [
            [1, 1, 255],
            [255, 255, 255],
            [1, 2, 2],
        ],
        dtype=np.uint8,
    )

    units = vectorize(
        values,
        nodata=255,
        transform=Affine.translation(0, 3) @ Affine.scale(1, -1),
        crs="EPSG:3116",
        name="su_id",
        connectivity=4,
        dissolve=True,
    )

    assert units["su_id"].tolist() == [1, 2]
    assert len(units) == 2
    assert units.geometry.is_valid.all()


def test_vectorize_respects_mask_when_nodata_is_none():
    values = np.ma.array(
        [[1, 1], [2, 2]],
        mask=[[False, True], [False, False]],
        dtype=np.int16,
    )
    features = vectorize(
        values,
        nodata=None,
        transform=Affine.identity(),
        crs=None,
        connectivity=4,
    )

    assert features.geometry.area.sum() == 3.0


def test_hillshade_uses_requested_azimuth():
    eastward_rise = np.tile(np.arange(5.0), (5, 1))

    from_east = get_hillshade(eastward_rise, azimuth=90)
    from_west = get_hillshade(eastward_rise, azimuth=270)

    assert not np.allclose(from_east, from_west)


def _write_test_raster(path, data, transform, nodata=-9999.0):
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=data.shape[0],
        width=data.shape[1],
        count=1,
        dtype=data.dtype,
        crs="EPSG:3116",
        transform=transform,
        nodata=nodata,
    ) as dst:
        dst.write(data, 1)


def test_align_raster_stack_uses_reference_grid_and_common_mask(tmp_path):
    reference_path = tmp_path / "reference.tif"
    fine_path = tmp_path / "classes.tif"
    _write_test_raster(
        reference_path,
        np.array([[1.0, 2.0], [3.0, -9999.0]], dtype="float32"),
        Affine.translation(0, 2) @ Affine.scale(1, -1),
    )
    _write_test_raster(
        fine_path,
        np.arange(16, dtype="float32").reshape(4, 4),
        Affine.translation(0, 2) @ Affine.scale(0.5, -0.5),
    )

    layers, valid = align_raster_stack(
        {"reference": reference_path, "classes": fine_path},
        categorical=["classes"],
    )

    assert grids_match(layers["reference"], layers["classes"])
    assert layers["classes"].shape == (2, 2)
    assert valid.tolist() == [[True, True], [True, False]]


def test_write_raster_like_preserves_grid_and_masks_invalid_values(tmp_path):
    reference_path = tmp_path / "reference.tif"
    output_path = tmp_path / "result.tif"
    transform = Affine.translation(10, 20) @ Affine.scale(2, -2)
    _write_test_raster(
        reference_path,
        np.ones((2, 3), dtype="float32"),
        transform,
    )
    reference = read_raster_layer(reference_path)

    write_raster_like(
        output_path,
        np.array([[1.0, np.nan, 3.0], [4.0, 5.0, 6.0]]),
        reference,
    )

    result = read_raster_layer(output_path)
    assert grids_match(reference, result)
    assert np.ma.getmaskarray(result.data)[0, 1]
