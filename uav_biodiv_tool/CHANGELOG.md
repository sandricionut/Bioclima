
## 0.9.1 — synchronized corrected build

- AOI arguments (`aoi`, `aoi_crs`, `registration_scope`) are available in point-cloud fusion as well as raster fusion.
- Headerless TXT schemas require explicit `field_names`.
- Bahna demo uses `gdown`, resolves ZIP contents recursively, and uses the actual `demo_multispectral.txt` filename.
- Demo TXT schemas include source normals but transfer only scientific variables by default.

# Changelog

## 0.9.1

- Added explicit `field_names` for numeric TXT/CSV/XYZ point clouds.
- Headerless text point clouds now require one user-defined field name per column.
- Added `coordinate_fields` so arbitrary user names can be mapped explicitly to X/Y/Z.
- `field_names` can override/rename columns in text files that already contain a header.
- Added streaming validation that every data row has the declared number of columns.
- Removed unnamed fallback attributes for headerless text inputs.
- Recorded field names, coordinate mapping, source header and column count in text-schema metadata.

## 0.9.0

- Added optional AOI-limited fusion for raster, LiDAR-raster and point-cloud workflows.
- AOIs can be supplied as bbox, vector path, GeoJSON, Shapely geometry or GeoPandas object.
- Raster outputs preserve the source analysis-grid alignment, crop to the AOI bounds and mask polygon exteriors to NoData.
- Added `registration_scope="full"` (default) so AOI selection does not alter the estimated co-registration transform.
- Added explicit `registration_scope="aoi"` for local registration when scientifically justified; the polygon AOI participates in registration QA/estimation.
- Point-cloud AOI output retains only reference points inside the AOI while retaining source points in a transfer-distance buffer for boundary matches.
- LiDAR rasterization filters point input by AOI and records AOI provenance in processing metadata.
- AOI definition, CRS, bounds, mask status and registration scope are recorded in metadata/QA sidecars.
- Retains v0.8.0 direct LAS/LAZ and TXT/CSV/XYZ point-cloud fusion, LAS tile collections, hardware-adaptive processing and `BIGTIFF=YES` raster outputs.


## 0.8.0

- Point-cloud source selection is explicit: one LAS/LAZ file, multiple files, a tile directory, or a glob. No filename-based exclusion rules are used by the core. — 2026-10-12

### Added

- Streaming numeric TXT/CSV/XYZ point-cloud reader with explicit column schemas.
- Direct point-cloud fusion between LAS/LAZ and TXT/CSV/XYZ sources.
- Direct LAS/LAZ tile-directory support as reference geometry for point-cloud fusion.
- Generic LAS/LAZ analysis-ready output with arbitrary transferred variables stored as Extra Bytes.
- Source-format, text-schema and transfer QA metadata in point-cloud sidecars.
- Concrete TXT and LAS point-cloud fusion examples.
- GitHub-ready repository documentation, citation, contribution and license files.

### Preserved

- Conservative multimodal raster co-registration with automatic model selection.
- Hardware-adaptive bounded-memory processing.
- Direct LiDAR tile collections for raster metric computation.
- `BIGTIFF=YES` for package-created GeoTIFFs.

## 0.7.2

- Direct LAS/LAZ tile collections for LiDAR raster metrics.

## 0.7.1

- Forced BigTIFF output for package-created GeoTIFFs.

## 0.7.0

- Conservative automatic raster registration model selection: identity, translation, similarity, affine.
