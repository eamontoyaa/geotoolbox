# geotoolbox

A collection of tools for geospatial, geotechnical, and statistical data processing and modelling.

## Motivation

**geotoolbox** aims to provide a unified and user-friendly toolkit for professionals and researchers working with geospatial and geotechnical data. It streamlines common workflows for data processing, analysis, and modelling in Earth sciences and engineering applications.

## Installation

You can install `geotoolbox` using pip:

```bash
pip install geotoolbox
```

For development or to install from source:

```bash
git clone https://github.com/eamontoyaa/geotoolbox.git
cd geotoolbox
pip install -e .
```

## Usage

Import the package using the recommended alias:

```python
import geotoolbox as gtb

# Check the version
print(gtb.__version__)
```

### Lognormal fragility curves

Binary fragility curves are fitted directly to unbinned outcomes with a stable
maximum-likelihood implementation:

```python
from geotoolbox.fragility import fit_lognormal_fragility

fit = fit_lognormal_fragility(intensity, failed)
probability = fit.predict(intensity_grid)
probability, lower, upper = fit.confidence_band(intensity_grid)
print(fit.median_capacity, fit.sigma)
```

The fitted model is `P(F | x) = Phi((log(x) - mu) / sigma)`. The result also
contains the observed-information covariance, parameter confidence intervals,
negative log-likelihood, and AIC.

### Raster categories to vectors

`geotoolbox.sig_helper.vectorize` accepts masked arrays and numeric, NaN, or
undefined nodata values. Use `dissolve=True` to obtain one multipart feature per
categorical raster value.

### Aligning raster inputs

Use one explicitly selected raster as the computational grid before combining
cell values. Continuous fields are resampled bilinearly, categorical fields by
nearest neighbour, and the returned Boolean mask is valid in every layer:

```python
from geotoolbox.sig_helper import align_raster_stack, write_raster_like

layers, valid = align_raster_stack(
    {
        "elevation": "dem.tif",
        "slope": "slope.tif",
        "units": "geology.tif",
    },
    reference="elevation",
    categorical=["units"],
)

elevation = layers["elevation"].data.filled(float("nan"))
units = layers["units"].data.filled(float("nan"))
result = elevation.copy()  # Replace with the actual spatial model.
result[~valid] = float("nan")
write_raster_like("result.tif", result, layers["elevation"])
```

`align_raster_stack` checks CRS information, aligns extent, origin, resolution,
and array shape, and propagates source nodata/NaN masks. Do not use bilinear or
cubic resampling for class identifiers, inventories, or other categorical data.

## Contributing

Contributions are welcome! Please feel free to submit issues or pull requests.

### Collaborators

- **Prof. Daniel F. Ruiz** – EAFIT University

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
