from __future__ import annotations

from typing import Any, Mapping
import json

import numpy as np
from scipy.spatial import cKDTree
from sklearn.metrics import confusion_matrix as sk_confusion_matrix, accuracy_score

from .ingestion import PointCloudData
from .fusion import RegistrationResult


class ValidationReporting:
    """Accuracy assessment, uncertainty proxies and reporting (D1.1 §8.1.7)."""

    @staticmethod
    def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, Any]:
        y_true = np.asarray(y_true)
        y_pred = np.asarray(y_pred)
        labels = np.unique(np.concatenate([y_true, y_pred]))
        return {
            "labels": labels.tolist(),
            "matrix": sk_confusion_matrix(y_true, y_pred, labels=labels).tolist(),
            "overall_accuracy": float(accuracy_score(y_true, y_pred)),
        }

    @staticmethod
    def point_cloud_distances(
        reference: PointCloudData,
        aligned: PointCloudData,
        *,
        max_points: int = 500_000,
        random_state: int = 42,
    ) -> dict[str, float]:
        rng = np.random.default_rng(random_state)
        ref_xyz = reference.xyz
        mov_xyz = aligned.xyz
        if len(ref_xyz) > max_points:
            ref_xyz = ref_xyz[rng.choice(len(ref_xyz), max_points, replace=False)]
        if len(mov_xyz) > max_points:
            mov_xyz = mov_xyz[rng.choice(len(mov_xyz), max_points, replace=False)]
        d, _ = cKDTree(ref_xyz).query(mov_xyz, k=1, workers=-1)
        return {
            "rmse": float(np.sqrt(np.mean(d ** 2))),
            "mean": float(np.mean(d)),
            "median": float(np.median(d)),
            "p95": float(np.percentile(d, 95)),
            "max": float(np.max(d)),
        }

    @staticmethod
    def registration_report(
        registrations: Mapping[str, RegistrationResult],
        transfer_stats: Mapping[str, Mapping[str, float]] | None = None,
    ) -> dict[str, Any]:
        return {
            "registration": {name: res.to_dict() for name, res in registrations.items()},
            "attribute_transfer": dict(transfer_stats or {}),
        }

    @staticmethod
    def raster_alignment_report(
        rasters: Mapping[str, Any],
    ) -> dict[str, Any]:
        """Summarise whether in-memory rasters share one analysis grid."""
        if not rasters:
            return {"aligned": True, "rasters": {}}
        items = {}
        ref_shape = None
        ref_crs = None
        ref_transform = None
        aligned = True
        for name, raster in rasters.items():
            data = np.asarray(raster.data)
            shape = tuple(map(int, data.shape[-2:]))
            crs = str(raster.profile.get("crs"))
            transform = raster.profile.get("transform")
            if ref_shape is None:
                ref_shape, ref_crs, ref_transform = shape, crs, transform
            else:
                aligned &= shape == ref_shape and crs == ref_crs and transform == ref_transform
            items[name] = {
                "shape": list(shape),
                "crs": crs,
                "transform": list(transform) if transform is not None else None,
            }
        return {"aligned": bool(aligned), "rasters": items}

    @staticmethod
    def raster_feature_statistics(
        features: Mapping[str, np.ndarray],
        *,
        nodata: float | None = -9999.0,
    ) -> dict[str, dict[str, float | int]]:
        """Basic QA statistics for fused raster features."""
        out: dict[str, dict[str, float | int]] = {}
        for name, values in features.items():
            arr = np.asarray(values, dtype=np.float64)
            valid = np.isfinite(arr)
            if nodata is not None:
                valid &= ~np.isclose(arr, nodata)
            v = arr[valid]
            if v.size == 0:
                out[name] = {
                    "valid_pixels": 0,
                    "valid_fraction": 0.0,
                    "min": float("nan"),
                    "max": float("nan"),
                    "mean": float("nan"),
                    "std": float("nan"),
                }
            else:
                out[name] = {
                    "valid_pixels": int(v.size),
                    "valid_fraction": float(v.size / arr.size),
                    "min": float(np.min(v)),
                    "max": float(np.max(v)),
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v)),
                }
        return out

    @staticmethod
    def summary_report(
        *,
        ebv_metrics: Mapping[str, Any] | None = None,
        canopy_metrics: Mapping[str, Any] | None = None,
        classification: Mapping[str, Any] | None = None,
        point_cloud: Mapping[str, Any] | None = None,
        raster_fusion: Mapping[str, Any] | None = None,
    ) -> str:
        blocks: list[str] = []
        if ebv_metrics is not None:
            blocks.append("[EBV] Essential Biodiversity Variables – Ecosystem Extent (m²)\n" + json.dumps(dict(ebv_metrics), indent=2))
        if canopy_metrics is not None:
            blocks.append("[EBV] Essential Biodiversity Variables – Canopy Structure Metrics\n" + json.dumps(dict(canopy_metrics), indent=2))
        if classification is not None:
            blocks.append("[VALIDATION] Classification\n" + json.dumps(dict(classification), indent=2))
        if point_cloud is not None:
            blocks.append("[VALIDATION] Point-cloud registration and fusion\n" + json.dumps(dict(point_cloud), indent=2))
        if raster_fusion is not None:
            blocks.append("[VALIDATION] Raster harmonisation and fusion\n" + json.dumps(dict(raster_fusion), indent=2))
        return "\n\n".join(blocks)
