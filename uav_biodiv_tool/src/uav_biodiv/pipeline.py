from __future__ import annotations

from concurrent.futures import wait, FIRST_COMPLETED
from pathlib import Path
from typing import Any, Iterable, Mapping
import json
import logging

import numpy as np

from .config import UAVProjectConfig
from .ingestion import DataIngestionHarmonization, PointCloudData
from .fusion import MultiSensorFusion, RegistrationResult, stream_reference_attribute_fusion
from .validation import ValidationReporting
from .ebv_ecv import EBVECVDerivation
from .ai import AIAnalytics
from .sensors.lidar import rasterize_lidar_metrics_files
from .execution import process_pool, window_to_tuple
from .raster_registration import (
    estimate_multimodal_registrations,
    read_registered_raster_window,
)
from .metadata import sidecar_path, write_json
from .aoi import (
    AOISelection, normalize_aoi, clip_profile_to_aoi, aoi_mask_for_window,
    point_mask, buffered_geometry, transform_for_cropped_grid,
)

LOGGER = logging.getLogger(__name__)


RasterSource = str | Path | tuple[str | Path, int]
PointCloudSource = str | Path | Mapping[str, Any]
PointCloudReference = str | Path | Iterable[str | Path]


def _resolve_reference_point_paths(value: PointCloudReference) -> list[Path]:
    """Resolve one reference cloud or a LAS/LAZ tile collection.

    A directory is interpreted as a collection of original LAS/LAZ tiles.
    Text point clouds must be supplied as explicit files because a directory of
    arbitrary text files has no reliable point-cloud semantics.
    """
    if isinstance(value, (str, Path)):
        path = Path(value)
        if path.is_dir():
            found = sorted(
                [p for p in path.iterdir() if p.is_file() and p.suffix.lower() in {".las", ".laz"}],
                key=lambda x: str(x).lower(),
            )
            if not found:
                raise ValueError(f"No LAS/LAZ reference tiles found in {path}")
            return found
        if not path.is_file():
            raise FileNotFoundError(f"Point-cloud reference does not exist: {path}")
        return [path]
    paths = [Path(x) for x in value]
    if not paths:
        raise ValueError("Point-cloud reference collection is empty")
    missing = [str(x) for x in paths if not x.is_file()]
    if missing:
        raise FileNotFoundError(f"Point-cloud reference files do not exist: {missing}")
    return paths


def _point_input_crs(path: str | Path, explicit_crs: str | None, project_crs: str | None) -> str | None:
    """Use embedded CRS for LAS/LAZ; use explicit/project CRS for text clouds."""
    suffix = Path(path).suffix.lower()
    if explicit_crs is not None:
        return str(explicit_crs)
    if suffix in {".txt", ".csv", ".xyz", ".ply", ".npz"}:
        return project_crs
    return None


def _process_raster_window_task(task: Mapping[str, Any]):
    """Process one raster tile in a separate process.

    Only serializable metadata is passed to workers; datasets are opened inside
    each process and closed before the result is returned.
    """
    from rasterio.windows import Window

    specs = task["specs"]
    profile = task["profile"]
    col_off, row_off, width, height = task["window"]
    window = Window(col_off, row_off, width, height)
    ingestion = DataIngestionHarmonization(target_crs=None)
    fusion = MultiSensorFusion()
    nodata = float(task["nodata"])
    fine_transforms = task.get("fine_transforms") or {}
    if fine_transforms:
        original = read_registered_raster_window(
            specs,
            profile,
            window,
            transformations=fine_transforms,
            resampling=task["resampling"],
            nodata=nodata,
        )
    else:
        original = ingestion.read_aligned_raster_window(
            specs,
            profile,
            window,
            resampling=task["resampling"],
            nodata=nodata,
        )
    derived: dict[str, np.ndarray] = {}
    definitions = task.get("definitions") or {}
    if definitions:
        derived.update(fusion.compute_indices(original, definitions, nodata=nodata))
    for key in task.get("anomaly_keys", ()):
        arr = np.asarray(original[key], dtype=np.float32).copy()
        invalid = ~np.isfinite(arr) | np.isclose(arr, nodata)
        arr = arr - float(task["anomaly_refs"][key])
        arr[invalid] = nodata
        derived[f"{key}_anomaly"] = arr

    features: dict[str, np.ndarray] = {}
    if task.get("include_original", True):
        features.update(original)
    features.update(derived)
    band_names = list(task["band_names"])
    cube = np.stack(
        [
            np.where(np.isfinite(features[name]), features[name], nodata).astype(np.float32)
            for name in band_names
        ],
        axis=0,
    )
    aoi_geojson = task.get("aoi_geojson")
    if aoi_geojson is not None:
        inside = aoi_mask_for_window(aoi_geojson, profile, window)
        if inside is not None:
            cube[:, ~inside] = nodata
    return task["window"], cube



class _StreamingStats:
    def __init__(self, names: Iterable[str], nodata: float) -> None:
        self.nodata = float(nodata)
        self.data = {
            str(name): {
                "count": 0,
                "sum": 0.0,
                "sumsq": 0.0,
                "min": float("inf"),
                "max": float("-inf"),
            }
            for name in names
        }

    def update(self, name: str, array: np.ndarray) -> None:
        a = np.asarray(array, dtype=np.float64)
        valid = np.isfinite(a) & ~np.isclose(a, self.nodata)
        if not np.any(valid):
            return
        x = a[valid]
        d = self.data[name]
        d["count"] += int(x.size)
        d["sum"] += float(np.sum(x, dtype=np.float64))
        d["sumsq"] += float(np.sum(x * x, dtype=np.float64))
        d["min"] = min(d["min"], float(np.min(x)))
        d["max"] = max(d["max"], float(np.max(x)))

    def result(self) -> dict[str, dict[str, float | int]]:
        out: dict[str, dict[str, float | int]] = {}
        for name, d in self.data.items():
            n = int(d["count"])
            if n == 0:
                out[name] = {
                    "count": 0,
                    "min": float("nan"),
                    "max": float("nan"),
                    "mean": float("nan"),
                    "std": float("nan"),
                }
                continue
            mean = float(d["sum"] / n)
            var = max(float(d["sumsq"] / n - mean * mean), 0.0)
            out[name] = {
                "count": n,
                "min": float(d["min"]),
                "max": float(d["max"]),
                "mean": mean,
                "std": float(np.sqrt(var)),
            }
        return out


class UAVMonitoringPipeline:
    """High-level orchestration for generic multi-sensor UAV workflows.

    The core package has no manufacturer-specific sensor or band definitions.
    All sensor metadata, band names and point-cloud dimensions come from the
    project configuration or the method call.

    Three representations remain explicitly separated:

    1. point-cloud -> point-cloud fusion;
    2. raster -> raster fusion;
    3. point cloud -> raster metrics -> raster fusion.
    """

    def __init__(self, config: UAVProjectConfig) -> None:
        self.config = config
        self.ingestion = DataIngestionHarmonization(
            target_crs=config.crs,
            processing_limits=config.processing_limits,
        )
        self.fusion = MultiSensorFusion()
        self.ebv_ecv = EBVECVDerivation()
        self.ai = AIAnalytics()
        self.validation = ValidationReporting()

    # ------------------------------------------------------------------
    # Case 1: point-cloud -> point-cloud fusion
    # ------------------------------------------------------------------
    def run_point_cloud_fusion(
        self,
        *,
        reference_path: PointCloudReference,
        source_clouds: Mapping[str, PointCloudSource],
        reference_name: str = "reference",
        reference_crs: str | None = None,
        reference_text_options: Mapping[str, Any] | None = None,
        output_path: str | Path | None = None,
        fusion_mode: str = "reference_attributes",
        registration_method: str = "icp",
        voxel_size: float = 0.25,
        max_correspondence_distance: float | None = 1.0,
        transfer_max_distance: float | None = 0.50,
        transfer_k: int = 1,
        transfer_weighting: str = "idw",
        registration_max_points: int = 250_000,
        initial_transforms: Mapping[str, np.ndarray] | None = None,
        registration_only: bool = False,
        aoi: Any | None = None,
        aoi_crs: str | None = None,
        registration_scope: str = "full",
    ) -> dict[str, Any]:
        """Register and optionally fuse arbitrary point-cloud sources.

        Registration is always performed on streaming samples and therefore
        does not require full clouds in RAM. Full point-domain fusion is
        guarded by the configured RAM budget: if a cloud cannot be safely
        materialised, the method raises ``MemoryError`` before the operating
        system starts paging. Use ``registration_only=True`` to obtain safe
        transforms for very large datasets.

        A source may carry one or many numeric attributes. Use ``attributes``
        as ``{output_variable: point_dimension}`` (or a mapping with a
        ``dimension`` field and physical metadata). Configured sensor channels
        can also be selected with ``bands``. Registration is geometric; source
        variables are transferred without cross-variable arithmetic or unit
        conversion. Actual LAS/LAZ dimension names are never guessed.

        ``aoi`` limits the fused output. By default ``registration_scope="full"``
        estimates the geometric transform from the full available point-cloud
        samples, so changing AOI does not change alignment. Use
        ``registration_scope="aoi"`` only when a local transform is explicitly
        intended; then registration samples are restricted to the AOI.
        """
        if not source_clouds:
            raise ValueError("At least one source point cloud is required")
        registration_scope = str(registration_scope).lower().strip()
        if registration_scope not in {"full", "aoi"}:
            raise ValueError("registration_scope must be 'full' or 'aoi'")
        initial_transforms = dict(initial_transforms or {})
        reference_paths = _resolve_reference_point_paths(reference_path)

        first_ref = reference_paths[0]
        first_ref_crs = _point_input_crs(first_ref, reference_crs, self.config.crs)
        first_ref_info = self.ingestion.inspect_point_cloud(
            first_ref,
            crs=first_ref_crs,
            text_options=reference_text_options if first_ref.suffix.lower() in {".txt", ".csv", ".xyz"} else None,
        )
        resolved_reference_crs = reference_crs or first_ref_info.get("crs") or self.config.crs
        aoi_selection = normalize_aoi(aoi, target_crs=resolved_reference_crs, aoi_crs=aoi_crs) if aoi is not None else None
        if registration_scope == "aoi" and aoi_selection is None:
            raise ValueError("registration_scope='aoi' requires an AOI")

        sample_limit = int(registration_max_points) * (4 if registration_scope == "aoi" else 1)
        reference_sample = self._sample_reference_point_cloud(
            reference_paths,
            max_points=sample_limit,
            source=reference_name,
            reference_crs=reference_crs,
            text_options=reference_text_options,
        )
        self._check_point_cloud_crs(reference_sample, reference_name)
        if registration_scope == "aoi":
            reference_sample = self._subset_point_cloud_to_aoi(
                reference_sample, aoi_selection, max_points=registration_max_points
            )

        registrations: dict[str, RegistrationResult] = {}
        specs: dict[str, dict[str, Any]] = {}
        for key, raw_spec in source_clouds.items():
            spec = self._normalize_point_source_spec(key, raw_spec)
            specs[key] = spec
            source_crs = _point_input_crs(spec["path"], spec.get("crs"), self.config.crs)
            moving_sample = self.ingestion.sample_point_cloud(
                spec["path"],
                max_points=sample_limit,
                source=str(spec.get("source_name") or key),
                crs=source_crs,
                attributes=[],
                text_options=spec.get("text_options"),
            )
            self._check_point_cloud_crs(moving_sample, key)
            if registration_scope == "aoi":
                registration_buffer = (
                    float(max_correspondence_distance)
                    if max_correspondence_distance is not None
                    else max(float(voxel_size) * 4.0, 0.0)
                )
                moving_aoi = AOISelection(
                    geometry=buffered_geometry(aoi_selection, registration_buffer),
                    crs=aoi_selection.crs,
                    source=f"{aoi_selection.source}:registration_buffer",
                    input_crs=aoi_selection.input_crs,
                )
                moving_sample = self._subset_point_cloud_to_aoi(
                    moving_sample, moving_aoi, max_points=registration_max_points
                )
            _, reg = self.fusion.register_point_cloud(
                reference_sample,
                moving_sample,
                method=registration_method,
                voxel_size=voxel_size,
                max_correspondence_distance=max_correspondence_distance,
                max_registration_points=registration_max_points,
                initial_transform=spec.get("initial_transform", initial_transforms.get(key)),
            )
            registrations[key] = reg

        if registration_only:
            qa = self.validation.registration_report(registrations, {})
            qa["aoi"] = (
                {**aoi_selection.to_metadata(), "registration_scope": registration_scope}
                if aoi_selection is not None else {"enabled": False}
            )
            report = self.validation.summary_report(point_cloud=qa)
            qa_path, report_path = self._write_reports(output_path, qa, report)
            return {
                "fused_cloud": None,
                "registrations": registrations,
                "transfer_stats": {},
                "qa": qa,
                "report": report,
                "output_path": None,
                "qa_path": str(qa_path) if qa_path else None,
                "report_path": str(report_path) if report_path else None,
            }

        mode = fusion_mode.lower()
        # Out-of-core nearest-neighbour transfer for LAS/LAZ and numeric text
        # point clouds. A LAS/LAZ tile collection can be used directly as the
        # reference geometry; no merged reference cloud is required.
        streamable_suffixes = {".las", ".laz", ".txt", ".csv", ".xyz"}
        all_streamable = all(p.suffix.lower() in streamable_suffixes for p in reference_paths) and all(
            Path(spec["path"]).suffix.lower() in streamable_suffixes for spec in specs.values()
        )
        output_is_las = output_path is not None and Path(output_path).suffix.lower() in {".las", ".laz"}
        if (
            mode == "reference_attributes"
            and all_streamable
            and output_is_las
            and transfer_k == 1
            and transfer_max_distance is not None
        ):
            stream_specs: dict[str, dict[str, Any]] = {}
            point_variable_metadata: dict[str, dict[str, Any]] = {}
            for key, spec in specs.items():
                attributes, variable_meta, _ = self._resolve_point_attribute_specs(key, spec)
                source_crs = _point_input_crs(spec["path"], spec.get("crs"), self.config.crs)
                source_info = self.ingestion.inspect_point_cloud(
                    spec["path"], crs=source_crs, text_options=spec.get("text_options")
                )
                dims = source_info.get("dimensions", [])
                if source_info.get("text_schema") is not None:
                    spec["resolved_text_schema"] = source_info.get("text_schema")
                lut = {str(x).lower(): str(x) for x in dims}
                resolved: dict[str, str] = {}
                for output_name, requested in attributes.items():
                    actual = lut.get(str(requested).lower())
                    if actual is None:
                        raise KeyError(
                            f"Configured attribute {requested!r} for source {key!r} is missing. "
                            f"Available dimensions: {dims}"
                        )
                    resolved[output_name] = actual
                    point_variable_metadata[output_name] = dict(variable_meta.get(output_name, {}))
                stream_specs[key] = {
                    "path": spec["path"],
                    "attributes": resolved,
                    "quality_prefix": str(spec.get("quality_prefix") or key),
                    "transformation": registrations[key].transformation,
                    "crs": source_crs,
                    "text_options": spec.get("text_options"),
                }
            stream_meta = stream_reference_attribute_fusion(
                [str(p) for p in reference_paths],
                stream_specs,
                str(output_path),
                ingestion=self.ingestion,
                processing_limits=self.config.processing_limits,
                max_distance=float(transfer_max_distance),
                reference_crs=reference_crs or self.config.crs,
                reference_text_options=reference_text_options,
                aoi=aoi_selection,
            )
            transfer_stats = stream_meta["transfer_stats"]
            qa = self.validation.registration_report(registrations, transfer_stats)
            qa["streaming"] = stream_meta
            qa["aoi"] = (
                {**aoi_selection.to_metadata(), "registration_scope": registration_scope}
                if aoi_selection is not None else {"enabled": False}
            )
            report = self.validation.summary_report(point_cloud=qa)
            qa_path, report_path = self._write_reports(output_path, qa, report)
            metadata_path = self._write_point_cloud_metadata(
                output_path=output_path,
                reference_path=reference_paths,
                specs=specs,
                registrations=registrations,
                variable_metadata=point_variable_metadata,
                output_dimensions=stream_meta.get("output_dimensions", {}),
                fusion_mode=mode,
                aoi=aoi_selection,
                registration_scope=registration_scope,
            )
            return {
                "fused_cloud": None,
                "registrations": registrations,
                "transfer_stats": transfer_stats,
                "qa": qa,
                "report": report,
                "output_path": str(output_path),
                "qa_path": str(qa_path) if qa_path else None,
                "report_path": str(report_path) if report_path else None,
                "metadata_path": str(metadata_path) if metadata_path else None,
                "streaming": True,
            }

        # Full fusion fallback for small/non-LAS datasets: load only the attributes actually required.
        # The loader performs a RAM-budget check before materialising each cloud.
        # performs a RAM-budget check before materialising each cloud.
        if len(reference_paths) != 1:
            raise ValueError(
                "Reference tile collections are supported by the streaming reference_attributes path. "
                "Use a .las/.laz output, transfer_k=1 and a positive transfer_max_distance."
            )
        reference_single = reference_paths[0]
        reference = self.ingestion.load_point_cloud(
            reference_single,
            source=reference_name,
            crs=_point_input_crs(reference_single, reference_crs, self.config.crs),
            attributes=[],
            text_options=reference_text_options,
        )
        self._check_point_cloud_crs(reference, reference_name)
        if aoi_selection is not None:
            reference = self._subset_point_cloud_to_aoi(reference, aoi_selection)

        aligned_sources: dict[str, PointCloudData] = {}
        transfer_attributes: dict[str, list[str]] = {}
        transfer_prefixes: dict[str, str] = {}
        point_variable_metadata: dict[str, dict[str, Any]] = {}
        for key, spec in specs.items():
            attributes, variable_meta, legacy_single = self._resolve_point_attribute_specs(key, spec)
            requested_dims = list(dict.fromkeys(attributes.values()))
            cloud = self.ingestion.load_point_cloud(
                spec["path"],
                source=str(spec.get("source_name") or key),
                crs=_point_input_crs(spec["path"], spec.get("crs"), self.config.crs),
                attributes=requested_dims,
                text_options=spec.get("text_options"),
            )
            self._check_point_cloud_crs(cloud, key)
            if cloud.metadata.get("text_schema") is not None:
                spec["resolved_text_schema"] = cloud.metadata.get("text_schema")
            lut = {x.lower(): x for x in cloud.attributes}
            output_names: list[str] = []
            for output_name, requested in attributes.items():
                actual = requested if requested in cloud.attributes else lut.get(str(requested).lower())
                if actual is None:
                    raise KeyError(f"Configured attribute {requested!r} missing for source {key!r}")
                if output_name != actual:
                    cloud.attributes[output_name] = np.asarray(cloud.attributes[actual])
                output_names.append(output_name)
                point_variable_metadata[output_name] = dict(variable_meta.get(output_name, {}))
            aligned = cloud.transformed(registrations[key].transformation)
            if aoi_selection is not None:
                source_geometry = buffered_geometry(
                    aoi_selection,
                    float(transfer_max_distance or 0.0) if mode == "reference_attributes" else 0.0,
                )
                keep = point_mask(aligned.xyz, source_geometry)
                if np.any(keep):
                    aligned = aligned.subset(keep)
                else:
                    aligned = PointCloudData(
                        xyz=np.empty((0, 3), dtype=np.float64),
                        attributes={name: np.empty((0,), dtype=np.asarray(values).dtype) for name, values in aligned.attributes.items()},
                        crs=aligned.crs, source=aligned.source, metadata=dict(aligned.metadata),
                    )
            aligned_sources[key] = aligned
            transfer_attributes[key] = output_names
            # Preserve the historical single-attribute naming while allowing
            # explicit multi-attribute mappings to define final variable names.
            transfer_prefixes[key] = key if legacy_single else ""

        transfer_stats: dict[str, dict[str, float]] = {}
        if mode == "reference_attributes":
            fused = reference.copy(deep=True)
            for key, cloud in aligned_sources.items():
                fused, stats = self.fusion.transfer_attributes_to_reference(
                    fused,
                    cloud,
                    attributes=transfer_attributes[key],
                    prefix=transfer_prefixes[key],
                    k=transfer_k,
                    max_distance=transfer_max_distance,
                    weighting=transfer_weighting,
                    include_distance=True,
                )
                transfer_stats[key] = stats
        elif mode in {"concatenate", "voxel"}:
            fused = self.fusion.fuse_point_clouds(
                [reference, *aligned_sources.values()], mode=mode, voxel_size=voxel_size
            )
        else:
            raise ValueError("fusion_mode must be 'reference_attributes', 'concatenate' or 'voxel'")

        saved = self.ingestion.save_point_cloud(fused, output_path) if output_path else None
        metadata_path = self._write_point_cloud_metadata(
            output_path=output_path,
            reference_path=reference_paths,
            specs=specs,
            registrations=registrations,
            variable_metadata=point_variable_metadata,
            output_dimensions={},
            fusion_mode=mode,
            aoi=aoi_selection,
            registration_scope=registration_scope,
        ) if output_path else None
        qa = self.validation.registration_report(registrations, transfer_stats)
        qa["aoi"] = (
            {**aoi_selection.to_metadata(), "registration_scope": registration_scope}
            if aoi_selection is not None else {"enabled": False}
        )
        report = self.validation.summary_report(point_cloud=qa)
        qa_path, report_path = self._write_reports(output_path, qa, report)
        return {
            "fused_cloud": fused,
            "registrations": registrations,
            "transfer_stats": transfer_stats,
            "qa": qa,
            "report": report,
            "output_path": str(saved) if saved else None,
            "qa_path": str(qa_path) if qa_path else None,
            "report_path": str(report_path) if report_path else None,
            "metadata_path": str(metadata_path) if metadata_path else None,
        }

    def run_multimodal_point_cloud_fusion(
        self,
        *,
        reference_path: PointCloudReference,
        source_clouds: Mapping[str, PointCloudSource],
        output_path: str | Path,
        **kwargs: Any,
    ) -> dict[str, Any]:
        """Analysis-ready multimodal point-cloud co-registration and fusion.

        This is the explicit multimodal entry point for point clouds carrying
        arbitrary imaging or other numeric attributes. Geometry is registered
        with ICP/GICP and attributes are transferred or aggregated according to
        ``fusion_mode``. Attribute names, wavelengths, units and sensor metadata
        remain user-defined; no channel convention is embedded in the core.
        """
        result = self.run_point_cloud_fusion(
            reference_path=reference_path,
            source_clouds=source_clouds,
            output_path=output_path,
            **kwargs,
        )
        result["fusion_method"] = (
            "geometric point-cloud co-registration and attribute-preserving analysis-ready fusion"
        )
        return result

    # ------------------------------------------------------------------
    # Case 2: raster -> raster fusion, windowed + parallel
    # ------------------------------------------------------------------
    def run_raster_fusion(
        self,
        *,
        raster_sources: Mapping[str, RasterSource],
        output_path: str | Path,
        reference_key: str | None = None,
        target_crs: str | None = None,
        resolution: float | tuple[float, float] | None = None,
        extent: str = "reference",
        resampling: Mapping[str, str] | str = "bilinear",
        compute_indices: bool = True,
        sensor_name: str | None = None,
        index_definitions: Mapping[str, Mapping[str, Any]] | None = None,
        anomaly_keys: Iterable[str] = (),
        anomaly_references: Mapping[str, float] | None = None,
        include_original: bool = True,
        nodata: float = -9999.0,
        fine_transforms: Mapping[str, Any] | None = None,
        fine_transform_grid: str = "full",
        band_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        aoi: Any | None = None,
        aoi_crs: str | None = None,
        aoi_mask: bool = True,
    ) -> dict[str, Any]:
        """Fuse arbitrary rasters with bounded memory and parallel windows.

        No band names are built into the module. Derived indices are computed
        only from explicit definitions or the selected user-configured sensor.
        ``anomaly_keys`` is generic and can be used for any continuous raster;
        when no reference value is supplied the scene mean is estimated in a
        first streaming pass.
        """
        import rasterio

        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        specs, meta, full_profile = self.ingestion.plan_raster_grid(
            raster_sources,
            reference_key=reference_key,
            target_crs=target_crs,
            resolution=resolution,
            extent=extent,
            nodata=nodata,
        )
        aoi_selection = normalize_aoi(aoi, target_crs=full_profile["crs"], aoi_crs=aoi_crs)
        profile, crop_offset = clip_profile_to_aoi(full_profile, aoi_selection)
        aoi_geojson = aoi_selection.geojson if (aoi_selection is not None and aoi_mask) else None

        fine_transform_grid = str(fine_transform_grid).lower().strip()
        if fine_transform_grid not in {"full", "output"}:
            raise ValueError("fine_transform_grid must be 'full' or 'output'")
        fine_transforms_raw = {
            str(k): np.asarray(v, dtype=np.float64)
            for k, v in (fine_transforms or {}).items()
        }
        if fine_transform_grid == "full" and crop_offset != (0, 0):
            fine_transforms_raw = {
                key: transform_for_cropped_grid(
                    matrix, col_offset=crop_offset[0], row_offset=crop_offset[1]
                )
                for key, matrix in fine_transforms_raw.items()
            }
        fine_transforms = {key: matrix.tolist() for key, matrix in fine_transforms_raw.items()}
        unknown_transforms = [k for k in fine_transforms if k not in specs]
        if unknown_transforms:
            raise KeyError(f"Fine registration transforms refer to missing raster sources: {unknown_transforms}")
        band_metadata = {str(k): dict(v) for k, v in (band_metadata or {}).items()}

        definitions = (
            self.config.resolve_index_definitions(sensor_name, index_definitions)
            if compute_indices
            else {}
        )
        if compute_indices and not definitions:
            raise ValueError(
                "compute_indices=True but no index definitions are configured or supplied"
            )

        anomaly_keys = tuple(dict.fromkeys(str(x) for x in anomaly_keys))
        missing_anomaly = [x for x in anomaly_keys if x not in specs]
        if missing_anomaly:
            raise KeyError(f"Anomaly sources are missing: {missing_anomaly}")
        anomaly_refs = {str(k): float(v) for k, v in (anomaly_references or {}).items()}

        original_names = list(specs) if include_original else []
        index_names = list(definitions)
        anomaly_names = [f"{key}_anomaly" for key in anomaly_keys]
        band_names = [*original_names, *index_names, *anomaly_names]
        if not band_names:
            raise ValueError("No raster features selected for output")

        limits = self.config.processing_limits
        tile_size = limits.raster_tile(bands=max(1, len(specs) + len(band_names)))
        windows = list(self.ingestion.raster_windows(profile, tile_size))

        # First pass only for anomaly references that were not explicitly set.
        unresolved = [k for k in anomaly_keys if k not in anomaly_refs]
        if unresolved:
            LOGGER.info("Raster anomaly-reference pass: serial windowed I/O")
            sums = {k: 0.0 for k in unresolved}
            counts = {k: 0 for k in unresolved}
            for window in windows:
                arrays = self.ingestion.read_aligned_raster_window(
                    {k: specs[k] for k in unresolved},
                    profile,
                    window,
                    resampling=resampling,
                    nodata=nodata,
                )
                inside = aoi_mask_for_window(aoi_geojson, profile, window) if aoi_geojson is not None else None
                for key, arr in arrays.items():
                    valid = np.isfinite(arr) & ~np.isclose(arr, nodata)
                    if inside is not None:
                        valid &= inside
                    if np.any(valid):
                        sums[key] += float(np.sum(arr[valid], dtype=np.float64))
                        counts[key] += int(np.count_nonzero(valid))
            for key in unresolved:
                if counts[key] == 0:
                    raise ValueError(f"Cannot derive anomaly reference: source {key!r} has no valid pixels")
                anomaly_refs[key] = sums[key] / counts[key]

        out_profile = dict(profile)
        out_profile.update(
            count=len(band_names), dtype="float32", nodata=float(nodata),
            compress="deflate", tiled=True, BIGTIFF="YES"
        )
        stats = _StreamingStats(band_names, nodata)

        estimated_task = tile_size * tile_size * max(1, len(specs) + len(band_names)) * 4 * 4
        workers = limits.effective_worker_count(len(windows), estimated_task_bytes=estimated_task)
        workers = max(1, workers)
        max_in_flight = limits.in_flight_count(workers)

        worker_task_base = {
            "specs": specs,
            "profile": profile,
            "resampling": resampling,
            "nodata": float(nodata),
            "definitions": definitions,
            "anomaly_keys": anomaly_keys,
            "anomaly_refs": anomaly_refs,
            "include_original": include_original,
            "band_names": band_names,
            "fine_transforms": fine_transforms,
            "aoi_geojson": aoi_geojson,
        }

        with rasterio.open(output_path, "w", **out_profile) as dst:
            dst.update_tags(
                uav_biodiv_fusion="analysis_ready_variable_preserving_stack",
                uav_biodiv_units_preserved="true",
            )
            for i, name in enumerate(band_names, start=1):
                dst.set_band_description(i, name)
                if name in band_metadata:
                    tags = {}
                    for key, value in band_metadata[name].items():
                        if value is None:
                            continue
                        if isinstance(value, (dict, list, tuple)):
                            tags[f"uav_biodiv_{key}"] = json.dumps(value, allow_nan=True)
                        else:
                            tags[f"uav_biodiv_{key}"] = str(value)
                    if tags:
                        dst.update_tags(i, **tags)
            if workers == 1:
                for window in windows:
                    task = dict(worker_task_base)
                    task["window"] = window_to_tuple(window)
                    window_tuple, cube = _process_raster_window_task(task)
                    from rasterio.windows import Window
                    out_window = Window(*window_tuple)
                    dst.write(cube, window=out_window)
                    for bi, name in enumerate(band_names):
                        stats.update(name, cube[bi])
            else:
                LOGGER.info("Raster tile stage: starting %d worker processes for %d tiles", workers, len(windows))
                with process_pool(workers) as executor:
                    iterator = iter(windows)
                    pending: set[Any] = set()

                    def submit_more() -> None:
                        while len(pending) < max_in_flight:
                            try:
                                window = next(iterator)
                            except StopIteration:
                                break
                            task = dict(worker_task_base)
                            task["window"] = window_to_tuple(window)
                            pending.add(executor.submit(_process_raster_window_task, task))

                    submit_more()
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            window_tuple, cube = future.result()
                            from rasterio.windows import Window
                            out_window = Window(*window_tuple)
                            dst.write(cube, window=out_window)
                            for bi, name in enumerate(band_names):
                                stats.update(name, cube[bi])
                        submit_more()

        qa = {
            "grid": {
                "crs": str(profile["crs"]),
                "width": int(profile["width"]),
                "height": int(profile["height"]),
                "transform": tuple(profile["transform"]),
            },
            "sources": {
                name: {
                    "path": str(specs[name][0]),
                    "band": int(specs[name][1]),
                    "source_crs": str(meta[name]["crs"]),
                }
                for name in specs
            },
            "features": stats.result(),
            "band_names": band_names,
            "band_metadata": band_metadata,
            "anomaly_references": anomaly_refs,
            "fine_registration_applied": bool(fine_transforms),
            "fine_transforms": fine_transforms,
            "aoi": (
                {**aoi_selection.to_metadata(), "mask_applied": bool(aoi_mask), "crop_offset_pixels": list(crop_offset)}
                if aoi_selection is not None else {"enabled": False}
            ),
            "processing": {
                "streaming": True,
                "parallel_backend": "process" if workers > 1 else "serial",
                "tile_size": int(tile_size),
                "workers": int(workers),
                "max_in_flight": int(max_in_flight),
                "memory_budget_gb": limits.memory_budget_bytes() / (1024**3),
                "resource_plan": limits.resource_plan(estimated_task_bytes=estimated_task, bands=max(1, len(specs) + len(band_names))),
            },
        }
        report = self.validation.summary_report(raster_fusion=qa)
        qa_path, report_path = self._write_reports(output_path, qa, report)
        return {
            "fused_raster": None,
            "qa": qa,
            "report": report,
            "output_path": str(output_path),
            "qa_path": str(qa_path) if qa_path else None,
            "report_path": str(report_path) if report_path else None,
            "band_names": band_names,
        }

    def run_multimodal_raster_fusion(
        self,
        *,
        raster_sources: Mapping[str, RasterSource],
        source_groups: Mapping[str, Mapping[str, Any]],
        reference_group: str,
        output_path: str | Path,
        reference_key: str | None = None,
        target_crs: str | None = None,
        resolution: float | tuple[float, float] | None = None,
        extent: str = "reference",
        resampling: Mapping[str, str] | str = "bilinear",
        registration_transform: str = "auto",
        registration_preview_max_dimension: int = 2048,
        registration_coarse_preview_dimension: int = 512,
        registration_rms_multiplier: float = 3.0,
        registration_fallback_max_native_pixels: float = 3.0,
        registration_min_score_improvement: float = 0.003,
        registration_require_local_consensus: bool = True,
        max_rotation_deg: float = 0.25,
        max_scale_delta: float = 0.005,
        max_shear_deg: float = 0.10,
        refinement_translation_px: float = 2.0,
        optimizer_maxiter: int = 40,
        raster_metadata: Mapping[str, Mapping[str, Any]] | None = None,
        nodata: float = -9999.0,
        aoi: Any | None = None,
        aoi_crs: str | None = None,
        aoi_mask: bool = True,
        registration_scope: str = "full",
    ) -> dict[str, Any]:
        """Multimodal co-registration and analysis-ready data fusion.

        The method estimates geometric corrections from temporary normalized
        gradient representations. The original variables are then resampled
        onto one common analysis grid and written as separate output bands.
        No cross-sensor arithmetic, normalization or unit conversion is
        applied to the analysis values.

        ``source_groups`` defines which raster variables originate from the
        same source geometry. Every key in ``raster_sources`` must belong to
        exactly one group. ``registration_keys`` can optionally select the
        variables used to estimate a group's transform; the resulting transform
        is applied to every variable in that group.

        With ``registration_transform="auto"`` (default), model selection is
        conservative and hierarchical: identity -> translation -> similarity ->
        affine. The first model that passes the geometric constraints and local
        residual-consensus QA is selected. If no correction is defensible, the
        original georeferencing is retained unchanged.

        ``aoi`` limits the output extent. With the default
        ``registration_scope="full"``, registration is estimated on the full
        available analysis grid and is therefore independent of the AOI. Set
        ``registration_scope="aoi"`` only for an intentionally local alignment;
        in that mode both the AOI crop and polygon mask participate in
        registration.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        if reference_group not in source_groups:
            raise KeyError(f"reference_group {reference_group!r} is not configured")
        if reference_key is None:
            ref_keys = tuple(source_groups[reference_group].get("keys", ()))
            if not ref_keys:
                raise ValueError(f"Reference group {reference_group!r} has no raster keys")
            reference_key = str(ref_keys[0])

        specs, source_meta, full_profile = self.ingestion.plan_raster_grid(
            raster_sources,
            reference_key=reference_key,
            target_crs=target_crs,
            resolution=resolution,
            extent=extent,
            nodata=nodata,
        )
        aoi_selection = normalize_aoi(aoi, target_crs=full_profile["crs"], aoi_crs=aoi_crs)
        output_profile, crop_offset = clip_profile_to_aoi(full_profile, aoi_selection)
        registration_scope = str(registration_scope).lower().strip()
        if registration_scope not in {"full", "aoi"}:
            raise ValueError("registration_scope must be 'full' or 'aoi'")
        if registration_scope == "aoi" and aoi_selection is None:
            raise ValueError("registration_scope='aoi' requires an AOI")
        registration_profile = output_profile if registration_scope == "aoi" else full_profile

        # Source-group settings are product/acquisition specific and therefore
        # take precedence. If geometric accuracy is stored in the configured
        # sensor metadata, inherit it only when the group does not override it.
        effective_source_groups: dict[str, dict[str, Any]] = {
            str(name): dict(group) for name, group in source_groups.items()
        }
        for group in effective_source_groups.values():
            sensor_name = group.get("sensor") or group.get("sensor_name")
            if sensor_name is None or "geometric_accuracy" in group:
                continue
            if str(sensor_name) in self.config.sensors:
                sensor_cfg = self.config.get_sensor(str(sensor_name))
                if isinstance(sensor_cfg.get("geometric_accuracy"), Mapping):
                    group["geometric_accuracy"] = dict(sensor_cfg["geometric_accuracy"])

        LOGGER.info(
            "Multimodal registration: reference group %s; transform=%s; scope=%s; preview<=%d px",
            reference_group,
            registration_transform,
            registration_scope,
            int(registration_preview_max_dimension),
        )
        fine_transforms, registration_results, key_to_group = estimate_multimodal_registrations(
            specs,
            registration_profile,
            effective_source_groups,
            reference_group=reference_group,
            resampling=resampling,
            nodata=nodata,
            transform_type=registration_transform,
            preview_max_dimension=int(registration_preview_max_dimension),
            coarse_preview_dimension=int(registration_coarse_preview_dimension),
            rms_multiplier=float(registration_rms_multiplier),
            fallback_max_native_pixels=float(registration_fallback_max_native_pixels),
            min_score_improvement=float(registration_min_score_improvement),
            require_local_consensus=bool(registration_require_local_consensus),
            max_rotation_deg=float(max_rotation_deg),
            max_scale_delta=float(max_scale_delta),
            max_shear_deg=float(max_shear_deg),
            refinement_translation_px=float(refinement_translation_px),
            optimizer_maxiter=int(optimizer_maxiter),
            aoi_geojson=(aoi_selection.geojson if registration_scope == "aoi" and aoi_selection is not None else None),
        )

        band_metadata = self._build_raster_band_metadata(
            specs=specs,
            source_meta=source_meta,
            profile=output_profile,
            source_groups=effective_source_groups,
            key_to_group=key_to_group,
            reference_group=reference_group,
            resampling=resampling,
            raster_metadata=raster_metadata or {},
        )

        result = self.run_raster_fusion(
            raster_sources=raster_sources,
            output_path=output_path,
            reference_key=reference_key,
            target_crs=target_crs,
            resolution=resolution,
            extent=extent,
            resampling=resampling,
            compute_indices=False,
            include_original=True,
            nodata=nodata,
            fine_transforms=fine_transforms,
            fine_transform_grid="output" if registration_scope == "aoi" else "full",
            band_metadata=band_metadata,
            aoi=aoi_selection,
            aoi_mask=aoi_mask,
        )

        profile = output_profile
        transform = profile["transform"]
        pixel_size_x = float((float(transform.a) ** 2 + float(transform.d) ** 2) ** 0.5)
        pixel_size_y = float((float(transform.b) ** 2 + float(transform.e) ** 2) ** 0.5)
        registration_payload: dict[str, Any] = {
            "method": "multimodal co-registration and analysis-ready data fusion",
            "registration_policy": {
                "default_transform": str(registration_transform),
                "coarse_preview_max_dimension": int(registration_coarse_preview_dimension),
                "fine_preview_max_dimension": int(registration_preview_max_dimension),
                "rms_multiplier": float(registration_rms_multiplier),
                "fallback_max_native_pixels": float(registration_fallback_max_native_pixels),
                "min_score_improvement": float(registration_min_score_improvement),
                "require_local_consensus": bool(registration_require_local_consensus),
                "accuracy_units": "analysis_grid_map_units",
                "failure_safe": "identity transform retained when QA/constraints fail",
                "registration_scope": registration_scope,
                "aoi_influences_registration": bool(registration_scope == "aoi"),
            },
            "aoi": (
                {**aoi_selection.to_metadata(), "mask_applied": bool(aoi_mask), "crop_offset_pixels": list(crop_offset)}
                if aoi_selection is not None else {"enabled": False}
            ),
            "reference_group": str(reference_group),
            "reference_key": str(reference_key),
            "analysis_grid": {
                "crs": str(profile["crs"]),
                "width": int(profile["width"]),
                "height": int(profile["height"]),
                "transform": tuple(profile["transform"]),
                "pixel_size_x": pixel_size_x,
                "pixel_size_y": pixel_size_y,
            },
            "registration_grid": {
                "crs": str(registration_profile["crs"]),
                "width": int(registration_profile["width"]),
                "height": int(registration_profile["height"]),
                "transform": tuple(registration_profile["transform"]),
            },
            "groups": {
                str(reference_group): {
                    "reference": True,
                    "transformation_pixel": np.eye(3, dtype=float).tolist(),
                    "keys": [str(x) for x in effective_source_groups[reference_group].get("keys", ())],
                    "geometric_accuracy": effective_source_groups[reference_group].get("geometric_accuracy"),
                }
            },
        }
        for group_name, registration in registration_results.items():
            LOGGER.info(
                "Registration %s: accepted=%s applied=%s shift=(%.4f, %.4f) map units; reason=%s",
                group_name,
                registration.accepted,
                registration.applied,
                registration.estimated_shift_map_units[0],
                registration.estimated_shift_map_units[1],
                registration.acceptance_reason,
            )
            item = registration.to_dict()
            item["reference"] = False
            item["keys"] = [str(x) for x in effective_source_groups[group_name].get("keys", ())]
            item["geometric_accuracy"] = effective_source_groups[group_name].get("geometric_accuracy")
            item["residual_shift_map_units"] = [
                float(registration.residual_shift_full_px[0] * pixel_size_x),
                float(registration.residual_shift_full_px[1] * pixel_size_y),
            ]
            item["center_displacement_map_units"] = [
                float(registration.center_displacement_full_px[0] * pixel_size_x),
                float(registration.center_displacement_full_px[1] * pixel_size_y),
            ]
            registration_payload["groups"][str(group_name)] = item

        variables = []
        for index, name in enumerate(result["band_names"], start=1):
            item = dict(band_metadata.get(name, {}))
            item["output_band"] = int(index)
            item["name"] = str(name)
            variables.append(item)

        metadata_payload = {
            "schema": "uav_biodiv.analysis_ready.v1",
            "software": {"name": "uav-biodiv", "version": "0.9.1"},
            "fusion_method": "Multimodal co-registration and analysis-ready data fusion",
            "values": {
                "cross_sensor_value_fusion": False,
                "unit_conversion": False,
                "units_preserved": True,
                "note": "Registration representations are used only to estimate geometry; output bands retain their source variable meaning and units. Resampling is recorded per variable.",
            },
            "project": {
                "project_name": self.config.project_name,
                "site_name": self.config.metadata.get("site_name"),
                "campaign_name": self.config.metadata.get("campaign_name"),
                "output_prefix": self.config.output_prefix,
            },
            "aoi": (
                {**aoi_selection.to_metadata(), "mask_applied": bool(aoi_mask), "registration_scope": registration_scope}
                if aoi_selection is not None else {"enabled": False}
            ),
            "output": {
                "path": str(output_path),
                "crs": str(profile["crs"]),
                "width": int(profile["width"]),
                "height": int(profile["height"]),
                "transform": tuple(profile["transform"]),
                "nodata": float(nodata),
            },
            "variables": variables,
            "registration_sidecar": str(sidecar_path(output_path, "registration")),
        }
        metadata_path = write_json(sidecar_path(output_path, "metadata"), metadata_payload)
        registration_path = write_json(sidecar_path(output_path, "registration"), registration_payload)

        result["metadata_path"] = str(metadata_path)
        result["registration_path"] = str(registration_path)
        result["metadata"] = metadata_payload
        result["registration"] = registration_payload
        result["fusion_method"] = metadata_payload["fusion_method"]
        return result

    def _build_raster_band_metadata(
        self,
        *,
        specs: Mapping[str, tuple[Path, int]],
        source_meta: Mapping[str, Mapping[str, Any]],
        profile: Mapping[str, Any],
        source_groups: Mapping[str, Mapping[str, Any]],
        key_to_group: Mapping[str, str],
        reference_group: str,
        resampling: Mapping[str, str] | str,
        raster_metadata: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, dict[str, Any]]:
        """Build explicit variable provenance without interpreting names."""
        transform = profile["transform"]
        output_resolution = [
            float((float(transform.a) ** 2 + float(transform.d) ** 2) ** 0.5),
            float((float(transform.b) ** 2 + float(transform.e) ** 2) ** 0.5),
        ]
        out: dict[str, dict[str, Any]] = {}
        for key, (path, source_band) in specs.items():
            group_name = key_to_group[key]
            group = source_groups[group_name]
            supplied = dict(raster_metadata.get(key, {}))
            sensor_name = supplied.get("sensor_name") or group.get("sensor") or group.get("sensor_name")
            channel_map = group.get("channel_map") or group.get("band_map") or {}
            channel_name = supplied.get("channel_name") or channel_map.get(key) or key

            configured_sensor: dict[str, Any] = {}
            configured_channel: dict[str, Any] = {}
            if sensor_name is not None and str(sensor_name) in self.config.sensors:
                sensor = dict(self.config.get_sensor(str(sensor_name)))
                configured_channel_raw = sensor.get("bands", {}).get(str(channel_name), {})
                configured_channel = (
                    dict(configured_channel_raw)
                    if isinstance(configured_channel_raw, Mapping)
                    else {"value": configured_channel_raw}
                )
                configured_sensor = dict(sensor)
                configured_sensor.pop("bands", None)
                configured_sensor.pop("indices", None)
                configured_sensor.pop("point_attributes", None)

            merged = dict(configured_channel)
            merged.update(supplied)
            spectral = dict(merged.get("spectral") or {})
            for field in ("central_wavelength_nm", "wavelength_min_nm", "wavelength_max_nm", "bandwidth_nm"):
                if field not in spectral and merged.get(field) is not None:
                    spectral[field] = merged.get(field)
            if "central_wavelength_nm" not in spectral and merged.get("wavelength_nm") is not None:
                spectral["central_wavelength_nm"] = merged.get("wavelength_nm")

            acquisition = {}
            if isinstance(configured_sensor.get("acquisition"), Mapping):
                acquisition.update(configured_sensor["acquisition"])
            if isinstance(merged.get("acquisition"), Mapping):
                acquisition.update(merged["acquisition"])
            for field in ("date", "acquisition_date", "start_time", "end_time", "timezone"):
                value = merged.get(field, configured_sensor.get(field))
                if value is not None:
                    acquisition[field] = value

            m = source_meta[key]
            native_transform = m["transform"]
            native_resolution = [
                float((float(native_transform.a) ** 2 + float(native_transform.d) ** 2) ** 0.5),
                float((float(native_transform.b) ** 2 + float(native_transform.e) ** 2) ** 0.5),
            ]
            method = resampling.get(key, "bilinear") if isinstance(resampling, Mapping) else resampling
            out[key] = {
                "description": merged.get("description"),
                "unit": merged.get("unit"),
                "spectral": spectral or None,
                "acquisition": acquisition or None,
                "source": {
                    "sensor_name": str(sensor_name) if sensor_name is not None else None,
                    "sensor_type": configured_sensor.get("sensor_type") or merged.get("sensor_type"),
                    "platform": merged.get("platform", configured_sensor.get("platform")),
                    "file": str(path),
                    "source_band": int(source_band),
                    "group": str(group_name),
                    "channel_name": str(channel_name),
                },
                "native": {
                    "crs": str(m.get("crs")),
                    "resolution": native_resolution,
                    "dtype": str(m.get("dtype")),
                    "nodata": m.get("nodata"),
                },
                "analysis_grid": {
                    "resolution": output_resolution,
                    "resampling": str(method),
                    "fine_coregistration": group_name != reference_group,
                },
                "physical_value_preservation": {
                    "unit_conversion": False,
                    "cross_sensor_value_fusion": False,
                    "unit_preserved": True,
                },
                "user_metadata": {
                    k: v
                    for k, v in merged.items()
                    if k not in {
                        "description", "unit", "spectral", "acquisition", "central_wavelength_nm",
                        "wavelength_min_nm", "wavelength_max_nm", "bandwidth_nm", "wavelength_nm",
                        "date", "acquisition_date", "start_time", "end_time", "timezone",
                        "sensor_name", "channel_name", "sensor_type", "platform", "raster_band",
                        "point_attribute", "role",
                    }
                } or None,
            }
        return out

    # ------------------------------------------------------------------
    # Case 3: point cloud -> raster metrics -> raster fusion
    # ------------------------------------------------------------------
    def run_lidar_raster_fusion(
        self,
        *,
        point_cloud_path: str | Path | Iterable[str | Path],
        raster_sources: Mapping[str, RasterSource],
        output_path: str | Path,
        reference_key: str | None = None,
        target_crs: str | None = None,
        resolution: float | tuple[float, float] | None = None,
        extent: str = "reference",
        resampling: Mapping[str, str] | str = "bilinear",
        point_metrics: Iterable[str] = ("point_density", "z_mean", "z_std", "z_p95"),
        point_sensor_name: str | None = None,
        metric_attributes: Mapping[str, str] | None = None,
        ground_classes: int | Iterable[int] | None = None,
        canopy_threshold: float = 2.0,
        compute_indices: bool = True,
        raster_sensor_name: str | None = None,
        index_definitions: Mapping[str, Mapping[str, Any]] | None = None,
        anomaly_keys: Iterable[str] = (),
        anomaly_references: Mapping[str, float] | None = None,
        include_original_rasters: bool = True,
        keep_point_metrics_raster: bool = True,
        point_cloud_recursive: bool = False,
        nodata: float = -9999.0,
        aoi: Any | None = None,
        aoi_crs: str | None = None,
        aoi_mask: bool = True,
    ) -> dict[str, Any]:
        """Rasterize point-cloud metrics and fuse them with raster sources.

        ``point_cloud_path`` may be one LAS/LAZ file, a directory containing
        LAS/LAZ tiles, a glob, or an explicit iterable of LAS/LAZ files. Tiled
        source clouds are streamed directly and are never merged into an
        intermediate point-cloud file. The caller explicitly selects the
        source(s); the core does not filter inputs by filename. ``aoi`` can
        restrict metric computation and the final fused raster to a study area
        without clipping the input files first.

        The actual point-cloud dimension names are supplied through
        ``metric_attributes`` or configuration. Recognised semantic keys are
        ``height``, ``classification`` and ``intensity`` because those metrics
        require those kinds of values; these are *not* assumed LAS column
        names.
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        _, _, full_target_profile = self.ingestion.plan_raster_grid(
            raster_sources,
            reference_key=reference_key,
            target_crs=target_crs,
            resolution=resolution,
            extent=extent,
            nodata=nodata,
        )
        aoi_selection = normalize_aoi(
            aoi, target_crs=full_target_profile["crs"], aoi_crs=aoi_crs
        ) if aoi is not None else None
        target_profile, _ = clip_profile_to_aoi(full_target_profile, aoi_selection)

        attrs = {str(k): str(v) for k, v in (metric_attributes or {}).items()}
        if point_sensor_name is not None:
            for semantic in ("height", "classification", "intensity"):
                if semantic not in attrs:
                    value = self.config.point_attribute(point_sensor_name, semantic)
                    if value is not None:
                        attrs[semantic] = value
            if ground_classes is None:
                configured = self.config.get_sensor(point_sensor_name).get("ground_classes")
                if configured is not None:
                    ground_classes = configured

        metrics_path = output_path.with_name(f"{output_path.stem}_point_metrics.tif")
        metric_meta = rasterize_lidar_metrics_files(
            point_cloud_path,
            target_profile,
            metrics_path,
            ingestion=self.ingestion,
            processing_limits=self.config.processing_limits,
            metrics=point_metrics,
            height_attribute=attrs.get("height"),
            canopy_threshold=canopy_threshold,
            classification_attribute=attrs.get("classification"),
            ground_classes=ground_classes,
            intensity_attribute=attrs.get("intensity"),
            recursive=point_cloud_recursive,
            nodata=nodata,
            aoi=aoi_selection,
        )

        point_metrics = tuple(dict.fromkeys(str(x) for x in point_metrics))
        combined: dict[str, RasterSource] = dict(raster_sources)
        for band, metric in enumerate(point_metrics, start=1):
            if metric in combined:
                raise ValueError(f"Point metric name {metric!r} collides with a raster source name")
            combined[metric] = (metrics_path, band)

        result = self.run_raster_fusion(
            raster_sources=combined,
            output_path=output_path,
            reference_key=reference_key,
            target_crs=target_crs,
            resolution=resolution,
            extent=extent,
            resampling=resampling,
            compute_indices=compute_indices,
            sensor_name=raster_sensor_name,
            index_definitions=index_definitions,
            anomaly_keys=anomaly_keys,
            anomaly_references=anomaly_references,
            include_original=True,
            nodata=nodata,
            aoi=aoi_selection,
            aoi_mask=aoi_mask,
        )
        result["point_metrics_path"] = str(metrics_path)
        result["point_metrics_processing"] = metric_meta
        if not keep_point_metrics_raster:
            try:
                metrics_path.unlink()
                result["point_metrics_path"] = None
            except OSError:
                pass
        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _subset_point_cloud_to_aoi(
        cloud: PointCloudData,
        aoi: AOISelection | None,
        *,
        max_points: int | None = None,
    ) -> PointCloudData:
        if aoi is None:
            return cloud
        mask = point_mask(cloud.xyz, aoi)
        if not np.any(mask):
            raise ValueError("AOI contains no points from a point-cloud registration sample")
        subset = cloud.subset(mask)
        if subset.n_points < 3:
            raise ValueError("AOI contains fewer than three points; point-cloud registration is not possible")
        if max_points is not None and subset.n_points > int(max_points):
            idx = np.linspace(0, subset.n_points - 1, int(max_points), dtype=np.int64)
            subset = subset.subset(idx)
        subset.metadata = dict(subset.metadata)
        subset.metadata["aoi_filtered"] = True
        subset.metadata["aoi_bounds"] = list(aoi.bounds)
        return subset

    def _sample_reference_point_cloud(
        self,
        paths: Iterable[Path],
        *,
        max_points: int,
        source: str,
        reference_crs: str | None,
        text_options: Mapping[str, Any] | None,
    ) -> PointCloudData:
        """Sample one reference file or a LAS/LAZ tile collection safely."""
        paths = list(paths)
        infos = []
        total = 0
        for path in paths:
            crs = _point_input_crs(path, reference_crs, self.config.crs)
            opts = text_options if path.suffix.lower() in {".txt", ".csv", ".xyz"} else None
            info = self.ingestion.inspect_point_cloud(path, crs=crs, text_options=opts)
            infos.append((path, crs, opts, info))
            total += int(info.get("point_count", 0) or 0)
        if total <= 0:
            raise ValueError("Reference point cloud contains no points")

        xyz_parts: list[np.ndarray] = []
        resolved_crs: str | None = None
        remaining = int(max_points)
        for i, (path, crs, opts, info) in enumerate(infos):
            n = int(info.get("point_count", 0) or 0)
            if i == len(infos) - 1:
                take = max(1, remaining)
            else:
                take = max(1, int(round(max_points * (n / total))))
                take = min(take, remaining - max(0, len(infos) - i - 1))
            sample = self.ingestion.sample_point_cloud(
                path, max_points=min(n, take) if n else take, source=source, crs=crs,
                attributes=[], text_options=opts,
            )
            self._check_point_cloud_crs(sample, f"{source}:{path.name}")
            if resolved_crs is None:
                resolved_crs = sample.crs
            elif sample.crs and resolved_crs:
                try:
                    from pyproj import CRS
                    if CRS.from_user_input(sample.crs) != CRS.from_user_input(resolved_crs):
                        raise ValueError("Reference point-cloud tiles use different CRS definitions")
                except Exception as exc:
                    if str(sample.crs) != str(resolved_crs):
                        raise ValueError("Reference point-cloud tiles use different CRS definitions") from exc
            xyz_parts.append(sample.xyz)
            remaining -= sample.n_points
            if remaining <= 0:
                break
        xyz = np.concatenate(xyz_parts, axis=0)
        if xyz.shape[0] > max_points:
            xyz = xyz[:max_points]
        return PointCloudData(
            xyz=xyz, attributes={}, crs=resolved_crs or reference_crs or self.config.crs, source=source,
            metadata={
                "paths": [str(x[0]) for x in infos],
                "n_original": total,
                "subsampled_to": int(len(xyz)),
                "reference_collection": len(infos) > 1,
            },
        )

    def _resolve_point_attribute_specs(
        self,
        key: str,
        spec: Mapping[str, Any],
    ) -> tuple[dict[str, str], dict[str, dict[str, Any]], bool]:
        """Resolve arbitrary point variables without sensor-specific assumptions."""
        attributes: dict[str, str] = {}
        metadata: dict[str, dict[str, Any]] = {}
        legacy_single = False

        raw_attributes = spec.get("attributes")
        if raw_attributes is not None:
            for output_name, raw in dict(raw_attributes).items():
                if isinstance(raw, Mapping):
                    item = dict(raw)
                    dimension = item.pop("dimension", item.pop("point_attribute", None))
                    if dimension is None:
                        raise ValueError(
                            f"Point variable {output_name!r} in source {key!r} requires a 'dimension'"
                        )
                    attributes[str(output_name)] = str(dimension)
                    metadata[str(output_name)] = item
                else:
                    attributes[str(output_name)] = str(raw)
                    metadata[str(output_name)] = {}
        elif spec.get("bands") is not None:
            sensor_name = spec.get("sensor")
            if sensor_name is None:
                raise ValueError(f"Source {key!r} uses 'bands' but has no configured sensor")
            output_names = dict(spec.get("output_names") or {})
            for channel_name in spec.get("bands") or ():
                channel_name = str(channel_name)
                dimension = self.config.point_attribute(
                    str(sensor_name), "value", band_name=channel_name
                )
                if dimension is None:
                    raise KeyError(
                        f"Configured channel {channel_name!r} for sensor {sensor_name!r} has no point_attribute"
                    )
                output_name = str(output_names.get(channel_name, channel_name))
                attributes[output_name] = str(dimension)
                configured = self.config.sensor_band_metadata(str(sensor_name), channel_name)
                metadata[output_name] = {
                    "sensor_name": configured["sensor_name"],
                    "channel_name": configured["channel_name"],
                    **dict(configured.get("channel_metadata") or {}),
                    "sensor_metadata": configured.get("sensor_metadata"),
                }
        else:
            attribute = spec.get("attribute")
            sensor_name = spec.get("sensor")
            channel_name = spec.get("band")
            if attribute is None and sensor_name is not None:
                attribute = self.config.point_attribute(
                    str(sensor_name),
                    str(spec.get("semantic_attribute", "value")),
                    band_name=str(channel_name) if channel_name is not None else None,
                )
            if attribute is None:
                raise ValueError(
                    f"Source {key!r} requires 'attributes', configured 'bands', or an explicit point attribute"
                )
            output_name = str(spec.get("output_name") or key)
            attributes[output_name] = str(attribute)
            metadata[output_name] = dict(spec.get("attribute_metadata") or {})
            legacy_single = True

        # Merge source-level metadata into every variable without overwriting
        # per-variable values.
        source_metadata = dict(spec.get("metadata") or {})
        for output_name in attributes:
            merged = dict(source_metadata)
            merged.update(metadata.get(output_name, {}))
            merged.setdefault("source_name", str(key))
            merged.setdefault("source_file", str(spec.get("path")))
            if spec.get("sensor") is not None:
                merged.setdefault("sensor_name", str(spec.get("sensor")))
            metadata[output_name] = merged
        return attributes, metadata, legacy_single

    def _write_point_cloud_metadata(
        self,
        *,
        output_path: str | Path | None,
        reference_path: PointCloudReference,
        specs: Mapping[str, Mapping[str, Any]],
        registrations: Mapping[str, RegistrationResult],
        variable_metadata: Mapping[str, Mapping[str, Any]],
        output_dimensions: Mapping[str, Any],
        fusion_mode: str,
        aoi: AOISelection | None = None,
        registration_scope: str = "full",
    ) -> Path | None:
        if output_path is None:
            return None
        variables = []
        for name, raw in variable_metadata.items():
            item = dict(raw)
            item["name"] = str(name)
            item.setdefault("unit", None)
            spectral = dict(item.get("spectral") or {})
            for field in ("central_wavelength_nm", "wavelength_min_nm", "wavelength_max_nm", "bandwidth_nm"):
                if field not in spectral and item.get(field) is not None:
                    spectral[field] = item.get(field)
            if "central_wavelength_nm" not in spectral and item.get("wavelength_nm") is not None:
                spectral["central_wavelength_nm"] = item.get("wavelength_nm")
            item["spectral"] = spectral or None
            item["physical_value_preservation"] = {
                "unit_conversion": False,
                "cross_sensor_value_fusion": False,
                "unit_preserved": True,
            }
            variables.append(item)
        payload = {
            "schema": "uav_biodiv.point_analysis_ready.v1",
            "software": {"name": "uav-biodiv", "version": "0.9.1"},
            "fusion_method": "geometric point-cloud co-registration and attribute-preserving fusion",
            "fusion_mode": str(fusion_mode),
            "aoi": (
                {**aoi.to_metadata(), "registration_scope": str(registration_scope)}
                if aoi is not None else {"enabled": False}
            ),
            "project": {
                "project_name": self.config.project_name,
                "site_name": self.config.metadata.get("site_name"),
                "campaign_name": self.config.metadata.get("campaign_name"),
                "output_prefix": self.config.output_prefix,
            },
            "reference": {
                "paths": [str(p) for p in _resolve_reference_point_paths(reference_path)],
            },
            "sources": {
                str(name): {
                    "path": str(spec.get("path")),
                    "format": Path(spec.get("path")).suffix.lower().lstrip("."),
                    "crs": spec.get("crs") or (self.config.crs if Path(spec.get("path")).suffix.lower() in {".txt", ".csv", ".xyz"} else None),
                    "text_options": spec.get("text_options"),
                    "resolved_text_schema": spec.get("resolved_text_schema"),
                    "registration": registrations[name].to_dict() if name in registrations else None,
                }
                for name, spec in specs.items()
            },
            "variables": variables,
            "output_dimensions": dict(output_dimensions),
            "values": {
                "cross_sensor_value_fusion": False,
                "unit_conversion": False,
                "units_preserved": True,
            },
        }
        return write_json(sidecar_path(output_path, "metadata"), payload)

    @staticmethod
    def _normalize_point_source_spec(key: str, raw: PointCloudSource) -> dict[str, Any]:
        if isinstance(raw, (str, Path)):
            return {"path": raw, "output_name": key}
        spec = dict(raw)
        if "path" not in spec:
            raise ValueError(f"Point-cloud source {key!r} must contain a 'path'")
        spec.setdefault("output_name", key)
        # ``text_options`` is intentionally source-local: TXT/CSV has no embedded
        # CRS or field schema. The mapping is passed unchanged to ingestion.
        if spec.get("text_options") is not None:
            spec["text_options"] = dict(spec["text_options"])
        return spec

    def _check_point_cloud_crs(self, cloud: PointCloudData, name: str) -> None:
        if not self.config.crs or not cloud.crs:
            return
        try:
            from pyproj import CRS
            same = CRS.from_user_input(self.config.crs) == CRS.from_user_input(cloud.crs)
        except Exception:
            same = str(self.config.crs) == str(cloud.crs)
        if not same:
            raise ValueError(
                f"Point cloud {name!r} CRS {cloud.crs!r} differs from configured CRS "
                f"{self.config.crs!r}. Reproject input clouds before point-cloud fusion."
            )

    @staticmethod
    def _write_reports(
        output_path: str | Path | None,
        qa: Mapping[str, Any],
        report_text: str,
    ) -> tuple[Path | None, Path | None]:
        if output_path is None:
            return None, None
        output_path = Path(output_path)
        qa_path = output_path.with_suffix(output_path.suffix + ".qa.json")
        report_path = output_path.with_suffix(output_path.suffix + ".report.txt")
        qa_path.write_text(json.dumps(dict(qa), indent=2, allow_nan=True), encoding="utf-8")
        report_path.write_text(report_text, encoding="utf-8")
        return qa_path, report_path
