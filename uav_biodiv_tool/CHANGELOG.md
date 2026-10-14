# Changelog

## 0.9.3
- Finalized optional LST support in `run_forest_multisensor_example()` and `run_forest_dji_m3m_lidar_example()` via `lst_path`, `lst_band`, and `lst_reference`.
- Added `lst_anomaly.tif`, LST anomaly statistics, ECV text-report output, and JSON metrics to the forest workflow.
- Synchronized the Bahna forest demo with the core LST API.
- Updated the Bahna fusion demo to use a single-line `tqdm` elapsed-time progress display.
- Added `tqdm` as a regular dependency so the project still installs with one command.
- Preserved all existing fusion APIs and defaults.

## 0.9.2
- Added `examples/demo_bahna/forest_ebv_ecv_demo.py` covering all documented forest EBV/ECV outputs from the Bahna demo data.
- Fixed derived single-band GeoTIFF writing so non-finite values are converted to declared nodata.
- Fixed forest-health nodata handling and derived tiled-GeoTIFF block metadata for small rasters.

Additive sensor-preset and EBV/ECV workflow release. Existing 0.9.1 fusion APIs and defaults are preserved.

- Added `sensors/multispectral.py` with an optional DJI Mavic 3 Multispectral preset.
- Added DJI Zenmuse L2 metadata preset to `sensors/lidar.py` without guessing LAS/LAZ dimension names.
- Added generic `load_multispectral(..., sensor=...)` and compatibility wrapper `load_dji_m3m_multispectral(...)`.
- Added generic `compute_multispectral_indices(..., sensor=...)` and compatibility wrapper `compute_dji_m3m_indices(...)`.
- Added `compute_vegetation_indices()` as a semantic alias of the generic `compute_indices()` implementation.
- Added `run_forest_multisensor_example(..., sensor=..., lidar_sensor=...)`.
- Added compatibility wrapper `run_forest_dji_m3m_lidar_example(...)`.
- Forest example outputs can include CHM, NDVI/GNDVI/NDRE, forest-health mask, ecosystem extent, canopy structure metrics, Random Forest habitat classification, validation summary and machine-readable metrics.
- Manufacturer presets remain outside the processing core and can be replaced by explicit user mappings.

## 0.9.1

- Synchronized point-cloud AOI API (`aoi`, `aoi_crs`, `registration_scope`).
- Generic TXT schemas and point-cloud attribute transfer.
- Multimodal raster registration/fusion and LiDAR raster metrics.
- BigTIFF output and bounded-memory processing.
