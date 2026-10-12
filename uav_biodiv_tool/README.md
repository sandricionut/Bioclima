# uav-biodiv

## v0.9.1 API contract

The synchronized v0.9.1 build supports AOI-aware raster **and point-cloud** fusion.
`UAVMonitoringPipeline.run_point_cloud_fusion(...)` accepts `aoi`, `aoi_crs`, and
`registration_scope`; `run_multimodal_point_cloud_fusion(...)` forwards these options.
Headerless TXT point clouds require explicit `field_names`, and the demo separates the
complete TXT schema from the subset of fields transferred to the LiDAR reference.

The Bahna demo uses `gdown` to download `Bahna_demo.zip` and resolves the demo files
recursively, ignoring `__MACOSX` metadata directories.


`uav_biodiv` is a sensor-agnostic Python toolkit developed in the BioClima project for **multimodal UAV co-registration, analysis-ready data fusion, and EBV/ECV-oriented processing**.

The package keeps physical variables separate and traceable. Fusion means geometric co-registration, harmonisation and attribute-preserving integration; it does **not** silently mix thermal, RGB, multispectral, LiDAR or other measurements into unitless synthetic values.

## Core design

The core contains no manufacturer presets, satellite-style band naming conventions, fixed wavelengths, or hard-coded point-cloud attribute names. All channel names, units, wavelengths, acquisition dates, sensor metadata and point dimensions are supplied by the user or project configuration.

Three processing branches are supported:

1. **Point cloud → point cloud** geometric registration and attribute transfer.
2. **Raster → raster** multimodal co-registration and variable-preserving stacking.
3. **Point cloud → raster metrics → raster fusion** on a common analysis grid.

## Version 0.9.1 highlights

Version 0.9.1 makes the schema of headerless TXT/CSV/XYZ point clouds explicit and reproducible.

- `text_options["field_names"]` is required when the text file has no header.
- `field_names` must contain exactly one name for every column in the text file.
- Every streamed data row is validated against that column count; mismatches raise a clear error instead of being silently accepted.
- `field_names` may also be supplied for files with a header to rename all columns while preserving their order.
- If coordinate fields are not named `x`, `y`, `z` (case-insensitive), `coordinate_fields` explicitly maps the user-defined names to X/Y/Z.
- Unnamed fallback fields such as `field1`/`value_1` are no longer created for headerless point clouds.
- The resolved schema is stored in point-cloud metadata.

Version 0.9.0 added an optional **Area of Interest (AOI)** to raster, LiDAR-raster and point-cloud fusion without requiring users to permanently clip their source datasets first.

- An AOI can be supplied as a bounding box, vector dataset, GeoJSON, Shapely geometry or GeoPandas object.
- Raster outputs are cropped to the AOI bounding grid and polygon AOIs are masked outside the geometry.
- LiDAR raster metrics are computed only for the AOI processing grid; source points outside the AOI are ignored.
- Point-cloud fusion writes only reference points inside the AOI and only spools source points needed near that AOI.
- **AOI does not change co-registration by default.** `registration_scope="full"` estimates alignment from the complete available overlap and applies that transform to the AOI output.
- `registration_scope="aoi"` is available when the scientific objective is explicitly a local alignment. In that mode the AOI participates in registration and the resulting transform should be interpreted as local to that AOI.
- AOI definition, CRS, bounds, masking and registration scope are recorded in metadata/QA sidecars.

The streaming LAS/LAZ and numeric TXT/CSV/XYZ point-cloud support introduced in v0.8.0 is retained. All GeoTIFF outputs continue to use `BIGTIFF=YES`.

## Installation

The project intentionally uses one installation command:

```bash
python -m pip install -e .
```

The repository keeps a small focused test set for the current point-cloud and AOI functionality. Run it with:

```bash
python -m pytest -v
```

Build a wheel and source distribution with:

```bash
python -m build
```

The package requires Python 3.10 or newer.

## Area of Interest (AOI)

Using an AOI is recommended when only a subset of a large acquisition is needed. The source data do not need to be manually clipped first, so provenance is retained and the same inputs can be reused for different study areas.

The important distinction is between **fusion extent** and **registration extent**:

```text
source datasets
      │
      ├── registration_scope="full"  (default)
      │       alignment estimated from full available overlap
      │
      └── AOI
              output cropped/masked to study area
```

This default makes the estimated alignment independent of the chosen AOI. It is the recommended mode for products that should share one consistent geometric correction.

Use `registration_scope="aoi"` only when a local transform is scientifically justified, for example when residual geometric distortion changes spatially and the final analysis is intentionally restricted to one local area. In that mode changing the AOI can change the estimated transform.

### Raster AOI

```python
result = pipeline.run_multimodal_raster_fusion(
    raster_sources={
        "channel_1": ("source_a.tif", 1),
        "channel_2": ("source_b.tif", 1),
    },
    source_groups={
        "geometry_a": {"keys": ["channel_1"]},
        "geometry_b": {"keys": ["channel_2"]},
    },
    reference_group="geometry_a",
    output_path="analysis_ready_aoi.tif",
    aoi=(500000, 4600000, 500500, 4600500),
    aoi_crs="EPSG:32634",
    registration_scope="full",   # AOI does not influence alignment
)
```

A polygon/vector AOI is also supported:

```python
result = pipeline.run_multimodal_raster_fusion(
    ...,
    aoi="study_area.gpkg",
    registration_scope="full",
)
```

Raster outputs are snapped to the original analysis grid. Polygon areas outside the AOI are written as NoData rather than creating a new arbitrary pixel alignment.

### Point-cloud AOI

```python
result = pipeline.run_multimodal_point_cloud_fusion(
    reference_path="reference_tiles",
    source_clouds={
        "source_variable": {
            "path": "source_points.txt",
            "crs": "EPSG:32634",
            "text_options": {
                "has_header": False,
                "field_names": ["x", "y", "z", "value"],
            },
            "attributes": {
                "value": {"dimension": "value", "unit": "user_defined"}
            },
        }
    },
    output_path="analysis_ready_aoi.laz",
    aoi="study_area.gpkg",
    registration_scope="full",
)
```

For point-cloud attribute transfer, reference points are retained only inside the AOI. Source points in a small buffer around the AOI are retained internally so reference points near the boundary can still find valid nearest neighbours.

Supported AOI inputs are:

- `(minx, miny, maxx, maxy)` bounding box;
- vector path readable by GeoPandas;
- GeoJSON geometry/Feature/FeatureCollection;
- Shapely geometry;
- GeoPandas GeoSeries/GeoDataFrame.

For geometries without an embedded CRS, provide `aoi_crs`. The package does not guess coordinate reference systems.

## Quick start: TXT point cloud + LAS/LAZ reference

Assume a numeric text file exported from photogrammetric processing contains:

```text
X Y Z temperature
```

or, without a header, four numeric columns in that order. The column meaning is declared explicitly:

```python
from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline

cfg = UAVProjectConfig(
    project_name="BioClima",
    base_data_dir=r"D:\data\outputs",
    crs="EPSG:32634",
)
cfg.set_processing_limits(mode="auto")

pipeline = UAVMonitoringPipeline(cfg)

result = pipeline.run_multimodal_point_cloud_fusion(
    reference_path=r"D:\data\reference.laz",
    source_clouds={
        "thermal_source": {
            "path": r"D:\data\thermal_points.txt",
            "crs": "EPSG:32634",
            "text_options": {
                "delimiter": "whitespace",
                "has_header": False,
                "field_names": ["x", "y", "z", "temperature"],
            },
            "attributes": {
                "temperature": {
                    "dimension": "temperature",
                    "unit": "degC",
                }
            },
        }
    },
    output_path=r"D:\data\outputs\analysis_ready.laz",
    transfer_max_distance=0.50,
)
```

The output reference geometry remains the reference point cloud. The source variable is transferred after geometric registration; its physical unit is preserved.

## TXT files with a header

For a header such as:

```text
X,Y,Z,Temperature
```

use the header directly:

```python
"text_options": {
    "delimiter": ",",
    "has_header": True,
}
```

To rename those four source columns while preserving their order:

```python
"text_options": {
    "delimiter": ",",
    "has_header": True,
    "field_names": ["x", "y", "z", "temperature_surface"],
}
```

If the user-defined field names do not use x/y/z, declare which fields carry the coordinates:

```python
"text_options": {
    "delimiter": "whitespace",
    "has_header": False,
    "field_names": ["easting", "northing", "elevation", "temperature"],
    "coordinate_fields": {
        "x": "easting",
        "y": "northing",
        "z": "elevation",
    },
}
```

`delimiter="auto"` and `has_header="auto"` are available, but a headerless file still requires `field_names`. Explicit schemas are recommended for reproducible scientific workflows.

TXT/CSV/XYZ files do not normally store a CRS. Supply it through the source specification or project configuration. The package does not guess a CRS.

## LAS/LAZ source attributes

If the source variable is already stored in LAS/LAZ, reference the actual point dimension or Extra Bytes name:

```python
result = pipeline.run_multimodal_point_cloud_fusion(
    reference_path=r"D:\data\reference.laz",
    source_clouds={
        "source_value": {
            "path": r"D:\data\source.laz",
            "attributes": {
                "temperature": {
                    "dimension": "Temperature",
                    "unit": "degC",
                }
            },
        }
    },
    output_path=r"D:\data\outputs\analysis_ready.laz",
)
```

Inspect a LAS/LAZ file before configuring attributes:

```python
print(pipeline.ingestion.inspect_point_cloud(r"D:\data\source.laz"))
```

No LAS dimension name is assumed by the core.

## Use original LiDAR tiles directly as the reference

Point-cloud fusion can use a directory containing original LAS/LAZ reference tiles:

```python
result = pipeline.run_multimodal_point_cloud_fusion(
    reference_path=r"D:\data\lidar_tiles",
    source_clouds={
        "thermal_source": {
            "path": r"D:\data\thermal_points.txt",
            "text_options": {
                "has_header": False,
                "field_names": ["x", "y", "z", "temperature"],
            },
            "attributes": {
                "temperature": {"dimension": "temperature", "unit": "degC"}
            },
        }
    },
    output_path=r"D:\data\outputs\analysis_ready.laz",
    transfer_max_distance=0.50,
)
```

The reference tiles are streamed sequentially into the fused output. They are **not first merged into an intermediate LAS/LAZ**. If all reference tiles have the same LAS point format and dimensions, those original reference records are preserved in the output and the new fused variables are appended as Extra Bytes.

## Point-cloud fusion method

The point-cloud workflow is:

```text
reference geometry
      │
      ├──────── source cloud A
      ├──────── source cloud B
      └──────── ...
               │
        geometric registration
               │
         ICP / GICP transform
               │
     bounded nearest-neighbour transfer
               │
               ▼
 analysis-ready reference geometry
 + preserved reference dimensions
 + transferred source variables
 + transfer-distance QA
```

Registration is performed on bounded samples rather than loading complete large clouds. For the scalable streaming path, attribute transfer currently uses nearest-neighbour transfer (`transfer_k=1`) with an explicit `transfer_max_distance`.

The output includes QA dimensions such as the nearest transfer distance and a matched/not-matched flag for each source. These are useful for filtering values that are too far from the source geometry.

## Point-cloud metadata

For every fused point-cloud output the package writes a metadata sidecar containing, when supplied:

- project/site/campaign identification;
- reference file(s);
- source file and format;
- source CRS for text clouds;
- text-column schema;
- physical variable name and unit;
- optional wavelength information supplied by the user;
- acquisition/provenance metadata supplied by the user;
- registration transformation and QA;
- output LAS/LAZ dimensions;
- statement that cross-sensor numeric values were not mixed or unit-converted.

The package never invents missing wavelength or sensor metadata.

## Multimodal raster co-registration and analysis-ready fusion

Raster fusion retains the v0.7 conservative georeferencing-first policy. The default is:

```python
registration_transform="auto"
```

Model selection follows:

```text
identity → translation → similarity → affine
```

The simplest model that passes the geometric and residual QA is retained. If the existing georeferencing is already adequate, no fine transform is applied. Optional product-level `rms`, `rms_x` and `rms_y` can constrain physically plausible corrections.

The raster workflow preserves every source variable in its own output band. Co-registration uses temporary multimodal representations for geometry estimation, while the original physical raster values are resampled and written unchanged in meaning/unit.

The raster output is accompanied by:

- `*.metadata.json` — band/variable provenance and physical metadata;
- `*.registration.json` — geometric transformation and registration QA;
- QA/report sidecars produced by the pipeline.

All package-created GeoTIFFs use `BIGTIFF=YES`.

## LiDAR tile collections for raster metrics

Raster metrics can also be computed directly from original LAS/LAZ tiles:

```python
result = pipeline.run_lidar_raster_fusion(
    point_cloud_path=r"D:\data\lidar_tiles",
    raster_sources={"reference": (reference_tif, 1)},
    reference_key="reference",
    point_metrics=("point_count", "point_density", "z_mean", "z_std", "z_p95"),
    output_path=output_tif,
)
```

No merged LAS is required. Points from all selected files are mapped into the same analysis grid before cell metrics are calculated. Input selection is explicit: pass one LAS/LAZ file, a directory containing the desired LAS/LAZ tiles, a glob, or an explicit list of files. The core never excludes files according to filename patterns.

## Hardware-adaptive processing

The normal configuration is:

```python
cfg.set_processing_limits(mode="auto")
```

The package detects CPU and available RAM, keeps a reserve for the operating system, and derives bounded worker counts, point chunks, raster tiles and in-flight task limits.

Inspect the current plan with:

```python
print(cfg.processing_limits.resource_plan())
```

Manual limits are available when reproducible infrastructure constraints are required:

```python
cfg.set_processing_limits(
    mode="manual",
    max_ram_gb=16,
    max_workers=4,
    raster_tile_size=512,
)
```

CPU-heavy independent work uses process-based parallelism. Large point clouds are streamed and partitioned rather than loaded completely into RAM.

## Examples

The `examples/` directory contains executable templates:

- `01_configure_project.py` — generic project configuration;
- `02_inspect_point_cloud.py` — inspect available point dimensions;
- `03_pointcloud_registration_and_fusion.py` — generic point-cloud fusion;
- `04_raster_fusion.py` — generic raster fusion;
- `05_point_metrics_raster_fusion.py` — point metrics + raster fusion;
- `06_multimodal_raster_analysis_ready.py` — multimodal raster co-registration;
- `07_lidar_tiles_collection.py` — raster metrics from original LiDAR tiles;
- `08_pointcloud_fusion_txt_source.py` — TXT physical variable → LAS/LAZ reference;
- `09_pointcloud_fusion_las_source.py` — LAS Extra Bytes variable → reference;
- `10_bahna_june2026_current_data.py` — project-specific raster/LiDAR workflow example;
- `11_pointcloud_fusion_lidar_tiles_txt.py` — original LiDAR tiles + TXT source cloud;
- `12_aoi_fusion.py` — AOI-limited raster and point-cloud fusion with global-vs-local registration scope.

## Repository layout

```text
uav-biodiv/
├── .github/workflows/package.yml
├── examples/
├── src/uav_biodiv/
├── tests/
├── CHANGELOG.md
├── CITATION.cff
├── CONTRIBUTING.md
├── LICENSE
├── MANIFEST.in
├── pyproject.toml
└── README.md
```

## GitHub workflow

The included GitHub Actions workflow tests Python 3.10–3.12 with the same single installation command used locally:

```bash
python -m pip install -e .
```

Tagged releases or manual workflow runs also build the wheel and source distribution.

## Scientific use and current scope

`uav_biodiv` is designed to produce traceable, analysis-ready multisensor datasets for subsequent biodiversity/climate analysis. PCA, machine-learning feature fusion and deep fusion are intentionally separate analytical stages; they do not replace the physically interpretable co-registered dataset.

Current point-cloud streaming fusion transfers nearest attributes to the reference geometry. More complex transfer kernels can be added later, but any future method should preserve source units, provenance and transfer uncertainty.

## License

MIT License. See `LICENSE`.
