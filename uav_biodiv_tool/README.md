# uav-biodiv

`uav_biodiv` is a sensor-agnostic Python toolkit developed in the BioClima project for **multimodal UAV co-registration, analysis-ready data fusion, and EBV/ECV-oriented processing**.

The package keeps physical variables separate and traceable. Fusion means geometric co-registration, harmonisation and attribute-preserving integration; it does **not** silently mix thermal, RGB, multispectral, LiDAR or other measurements into unitless synthetic values.

## Core design

The core contains no manufacturer presets, satellite-style band naming conventions, fixed wavelengths, or hard-coded point-cloud attribute names. All channel names, units, wavelengths, acquisition dates, sensor metadata and point dimensions are supplied by the user or project configuration.

Three processing branches are supported:

1. **Point cloud → point cloud** geometric registration and attribute transfer.
2. **Raster → raster** multimodal co-registration and variable-preserving stacking.
3. **Point cloud → raster metrics → raster fusion** on a common analysis grid.

## Version 0.8.0 highlights

Version 0.8.0 adds a streaming point-cloud ingestion/fusion layer for **LAS/LAZ and numeric TXT/CSV/XYZ** files.

- A thermal or other photogrammetric point cloud can be used directly from a TXT export; manual TXT → LAS conversion is not required.
- TXT coordinate and attribute columns are declared explicitly.
- TXT data are streamed in bounded chunks and participate in the same geometric registration and nearest-neighbour attribute transfer used for LAS/LAZ.
- A directory of original LAS/LAZ reference tiles can be used directly for point-cloud fusion, avoiding a potentially problematic externally merged reference cloud.
- The fused analysis-ready point cloud is written as LAS/LAZ. User variables are stored as LAS Extra Bytes dimensions together with transfer-distance and matched/not-matched QA dimensions.
- CRS, variable units, provenance, registration information and transfer QA are written to sidecar metadata/QA files.
- Raster outputs continue to use `BIGTIFF=YES` to avoid the classic TIFF 4 GiB limit.

## Installation

The project intentionally uses one installation command:

```bash
python -m pip install -e .
```

The repository keeps only the focused v0.8.0 tests (TXT/LAS point-cloud input and fusion). Run them with:

```bash
python -m pytest -v
```

Build a wheel and source distribution with:

```bash
python -m build
```

The package requires Python 3.10 or newer.

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
                "columns": {
                    "x": 0,
                    "y": 1,
                    "z": 2,
                    "temperature": 3,
                },
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

use:

```python
"text_options": {
    "delimiter": ",",
    "has_header": True,
    "columns": {
        "x": "X",
        "y": "Y",
        "z": "Z",
        "temperature": "Temperature",
    },
}
```

`delimiter="auto"` and `has_header="auto"` are also available, but explicit schemas are recommended for reproducible scientific workflows.

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
                "columns": {"x": 0, "y": 1, "z": 2, "temperature": 3},
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
- `11_pointcloud_fusion_lidar_tiles_txt.py` — original LiDAR tiles + TXT source cloud.

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
