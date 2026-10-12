from __future__ import annotations

from typing import Mapping
import numpy as np


class EBVECVDerivation:
    """EBV and ECV indicator derivation (D1.1 §8.1.5)."""

    @staticmethod
    def derive_ecosystem_extent(
        classified: np.ndarray,
        class_mapping: Mapping[int, str],
        pixel_area: float,
    ) -> dict[str, float]:
        arr = np.asarray(classified)
        return {
            name: float(np.count_nonzero(arr == code) * pixel_area)
            for code, name in class_mapping.items()
        }

    @staticmethod
    def derive_canopy_structure_metrics(
        chm: np.ndarray,
        mask: np.ndarray | None = None,
        nodata: float | None = None,
    ) -> dict[str, float]:
        values = np.asarray(chm, dtype=float)
        valid = np.isfinite(values)
        if nodata is not None:
            valid &= ~np.isclose(values, nodata)
        if mask is not None:
            valid &= np.asarray(mask, dtype=bool)
        data = values[valid]
        if data.size == 0:
            return {"mean_height": float("nan"), "std_height": float("nan"), "p95_height": float("nan")}
        return {
            "mean_height": float(np.mean(data)),
            "std_height": float(np.std(data)),
            "p95_height": float(np.percentile(data, 95)),
        }

    @staticmethod
    def derive_lst_anomalies(
        lst: np.ndarray,
        reference: float | None = None,
        nodata: float | None = None,
    ) -> np.ndarray:
        arr = np.asarray(lst, dtype=np.float32).copy()
        if nodata is not None:
            arr[np.isclose(arr, nodata)] = np.nan
        ref = float(np.nanmean(arr)) if reference is None else float(reference)
        return arr - ref

    @staticmethod
    def forest_health_from_indices(
        ndvi: np.ndarray,
        ndre: np.ndarray,
        *,
        ndvi_threshold: float = 0.5,
        ndre_threshold: float = 0.2,
    ) -> np.ndarray:
        return ((np.asarray(ndvi) >= ndvi_threshold) & (np.asarray(ndre) >= ndre_threshold)).astype(np.uint8)
