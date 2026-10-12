# Changelog

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
