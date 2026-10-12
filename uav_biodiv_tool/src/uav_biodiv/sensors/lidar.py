from __future__ import annotations

from pathlib import Path
from typing import Iterable, Mapping, Any
import glob
import logging
import time

import numpy as np

from ..ingestion import PointCloudData, RasterData
from ..execution import process_pool

LOGGER = logging.getLogger(__name__)


def compute_chm_from_dsm_dtm(
    dsm_path: str | Path,
    dtm_path: str | Path,
    output_path: str | Path,
    *,
    tile_size: int = 1024,
) -> Path:
    """Compute CHM = DSM - DTM using bounded raster windows."""
    import rasterio
    from rasterio.windows import Window

    dsm_path = Path(dsm_path)
    dtm_path = Path(dtm_path)
    output_path = Path(output_path)

    with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
        if (
            dsm_src.width != dtm_src.width
            or dsm_src.height != dtm_src.height
            or dsm_src.transform != dtm_src.transform
            or dsm_src.crs != dtm_src.crs
        ):
            raise ValueError("DSM and DTM must already be aligned")
        profile = dsm_src.profile.copy()
        profile.update(
            dtype="float32", count=1, nodata=-9999.0, compress="deflate",
            tiled=True, BIGTIFF="YES"
        )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with rasterio.open(output_path, "w", **profile) as dst:
            dst.set_band_description(1, "CHM")
            tile = max(32, int(tile_size))
            for row_off in range(0, dsm_src.height, tile):
                h = min(tile, dsm_src.height - row_off)
                for col_off in range(0, dsm_src.width, tile):
                    w = min(tile, dsm_src.width - col_off)
                    window = Window(col_off, row_off, w, h)
                    dsm = dsm_src.read(1, window=window, masked=True).astype(np.float32)
                    dtm = dtm_src.read(1, window=window, masked=True).astype(np.float32)
                    chm = dsm - dtm
                    dst.write(chm.filled(-9999.0).astype(np.float32), 1, window=window)
    return output_path


def _xy_in_target_crs(
    cloud: PointCloudData,
    target_crs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    x = cloud.xyz[:, 0].astype(np.float64, copy=False)
    y = cloud.xyz[:, 1].astype(np.float64, copy=False)
    if cloud.crs is None or target_crs is None:
        return x, y

    from pyproj import CRS, Transformer

    src = CRS.from_user_input(cloud.crs)
    dst = CRS.from_user_input(target_crs)
    if src == dst:
        return x, y
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    xt, yt = transformer.transform(x, y)
    return np.asarray(xt, dtype=np.float64), np.asarray(yt, dtype=np.float64)


def _point_cells(
    cloud: PointCloudData,
    profile: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return valid point indices, rows, cols and flattened cell ids."""
    transform = profile.get("transform")
    width = int(profile["width"])
    height = int(profile["height"])
    if transform is None:
        raise ValueError("Raster profile requires an affine transform")

    x, y = _xy_in_target_crs(cloud, profile.get("crs"))
    inv = ~transform
    cols_f = inv.a * x + inv.b * y + inv.c
    rows_f = inv.d * x + inv.e * y + inv.f
    cols = np.floor(cols_f).astype(np.int64)
    rows = np.floor(rows_f).astype(np.int64)
    valid = (
        np.isfinite(x)
        & np.isfinite(y)
        & (rows >= 0)
        & (rows < height)
        & (cols >= 0)
        & (cols < width)
    )
    idx = np.flatnonzero(valid)
    rows_v = rows[valid]
    cols_v = cols[valid]
    cells = rows_v * width + cols_v
    return idx, rows_v, cols_v, cells


def _aggregate_basic(
    cells: np.ndarray,
    values: np.ndarray,
    n_cells: int,
    *,
    percentiles: Iterable[int] = (5, 50, 95),
) -> dict[str, np.ndarray]:
    """Aggregate one numeric point attribute by raster cell."""
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    cells = np.asarray(cells, dtype=np.int64)[finite]
    values = values[finite]

    count = np.bincount(cells, minlength=n_cells).astype(np.int64)
    sum_v = np.bincount(cells, weights=values, minlength=n_cells)
    sumsq_v = np.bincount(cells, weights=values * values, minlength=n_cells)

    mean = np.full(n_cells, np.nan, dtype=np.float64)
    std = np.full(n_cells, np.nan, dtype=np.float64)
    occupied = count > 0
    mean[occupied] = sum_v[occupied] / count[occupied]
    var = np.zeros(n_cells, dtype=np.float64)
    var[occupied] = np.maximum(sumsq_v[occupied] / count[occupied] - mean[occupied] ** 2, 0.0)
    std[occupied] = np.sqrt(var[occupied])

    min_v = np.full(n_cells, np.inf, dtype=np.float64)
    max_v = np.full(n_cells, -np.inf, dtype=np.float64)
    np.minimum.at(min_v, cells, values)
    np.maximum.at(max_v, cells, values)
    min_v[~occupied] = np.nan
    max_v[~occupied] = np.nan

    out: dict[str, np.ndarray] = {
        "count": count,
        "min": min_v,
        "max": max_v,
        "mean": mean,
        "std": std,
    }

    pct_list = sorted({int(p) for p in percentiles})
    if pct_list and values.size:
        order = np.argsort(cells, kind="stable")
        c_sorted = cells[order]
        v_sorted = values[order]
        unique, starts, counts = np.unique(c_sorted, return_index=True, return_counts=True)
        pct_arrays = {p: np.full(n_cells, np.nan, dtype=np.float64) for p in pct_list}
        for cell, start, n in zip(unique, starts, counts, strict=True):
            vals = v_sorted[start : start + n]
            qs = np.percentile(vals, pct_list)
            for p, q in zip(pct_list, qs, strict=True):
                pct_arrays[p][cell] = q
        for p, arr in pct_arrays.items():
            out[f"p{p:02d}"] = arr
    return out


def rasterize_lidar_metrics(
    cloud: PointCloudData,
    reference_profile: Mapping[str, Any],
    *,
    metrics: Iterable[str] = (
        "point_count",
        "point_density",
        "z_min",
        "z_max",
        "z_mean",
        "z_std",
        "z_p05",
        "z_p50",
        "z_p95",
    ),
    height_attribute: str | None = None,
    canopy_threshold: float = 2.0,
    classification_attribute: str | None = None,
    ground_classes: int | Iterable[int] | None = None,
    intensity_attribute: str | None = None,
    nodata: float = -9999.0,
) -> dict[str, np.ndarray]:
    """Rasterize LiDAR structural metrics onto an existing raster grid.

    This function implements BioClima fusion case 3: the LiDAR point cloud is
    converted to structural raster metrics first; multispectral/thermal fusion
    happens only afterwards in raster space.

    Supported metric names
    ----------------------
    point_count, point_density,
    z_min, z_max, z_mean, z_std, z_p05, z_p50, z_p95,
    intensity_mean, intensity_std,
    height_mean, height_std, height_p05, height_p50, height_p95,
    canopy_cover, vertical_entropy,
    dtm_ground, dsm_p95, chm_p95.

    ``height_*``, ``canopy_cover`` and ``vertical_entropy`` require a normalized
    user-configured normalized-height attribute. They are not
    computed from absolute Z silently.
    """
    metrics = tuple(dict.fromkeys(str(m) for m in metrics))
    width = int(reference_profile["width"])
    height = int(reference_profile["height"])
    n_cells = width * height
    transform = reference_profile["transform"]
    pixel_area = abs(float(transform.a * transform.e - transform.b * transform.d))
    if pixel_area <= 0:
        raise ValueError("Reference grid has invalid pixel area")
    if "point_density" in metrics and reference_profile.get("crs") is not None:
        from pyproj import CRS
        if not CRS.from_user_input(reference_profile["crs"]).is_projected:
            raise ValueError("point_density requires a projected raster CRS so pixel area is in linear units")

    idx, _, _, cells = _point_cells(cloud, reference_profile)
    z = cloud.xyz[idx, 2]
    z_stats = _aggregate_basic(cells, z, n_cells, percentiles=(5, 50, 95))
    point_count = np.bincount(cells, minlength=n_cells).astype(np.float64)

    result: dict[str, np.ndarray] = {}

    def raster(flat: np.ndarray, *, empty_as_nodata: bool = True) -> np.ndarray:
        arr = np.asarray(flat, dtype=np.float64).reshape(height, width)
        if empty_as_nodata:
            arr = arr.copy()
            arr[~np.isfinite(arr)] = float(nodata)
        return arr.astype(np.float32)

    basic_map = {
        "point_count": point_count,
        "point_density": point_count / pixel_area,
        "z_min": z_stats["min"],
        "z_max": z_stats["max"],
        "z_mean": z_stats["mean"],
        "z_std": z_stats["std"],
        "z_p05": z_stats["p05"],
        "z_p50": z_stats["p50"],
        "z_p95": z_stats["p95"],
    }
    for name, values in basic_map.items():
        if name in metrics:
            result[name] = raster(values, empty_as_nodata=name not in {"point_count", "point_density"})

    if any(m.startswith("intensity_") for m in metrics):
        if intensity_attribute is None:
            raise ValueError(
                "Intensity metrics were requested but no intensity_attribute was configured. "
                "Set it explicitly or map semantic role 'intensity' in UAVProjectConfig."
            )
        if intensity_attribute not in cloud.attributes:
            raise KeyError(f"LiDAR attribute {intensity_attribute!r} is required for intensity metrics")
        intensity = np.asarray(cloud.attributes[intensity_attribute])[idx]
        st = _aggregate_basic(cells, intensity, n_cells, percentiles=())
        if "intensity_mean" in metrics:
            result["intensity_mean"] = raster(st["mean"])
        if "intensity_std" in metrics:
            result["intensity_std"] = raster(st["std"])

    height_stats = None
    height_values = None
    if any(m.startswith("height_") or m in {"canopy_cover", "vertical_entropy"} for m in metrics):
        if height_attribute is None:
            raise ValueError(
                "Normalized-height metrics require height_attribute; absolute Z is not used as canopy height implicitly"
            )
        if height_attribute not in cloud.attributes:
            raise KeyError(f"Point cloud does not contain height attribute {height_attribute!r}")
        height_values = np.asarray(cloud.attributes[height_attribute], dtype=np.float64)[idx]
        height_stats = _aggregate_basic(cells, height_values, n_cells, percentiles=(5, 50, 95))
        hmap = {
            "height_mean": height_stats["mean"],
            "height_std": height_stats["std"],
            "height_p05": height_stats["p05"],
            "height_p50": height_stats["p50"],
            "height_p95": height_stats["p95"],
        }
        for name, values in hmap.items():
            if name in metrics:
                result[name] = raster(values)

    if "canopy_cover" in metrics:
        finite_h = np.isfinite(height_values)
        total = np.bincount(cells[finite_h], minlength=n_cells).astype(np.float64)
        above = np.bincount(
            cells[finite_h & (height_values > float(canopy_threshold))], minlength=n_cells
        ).astype(np.float64)
        cover = np.full(n_cells, np.nan, dtype=np.float64)
        valid = total > 0
        cover[valid] = above[valid] / total[valid]
        result["canopy_cover"] = raster(cover)

    if "vertical_entropy" in metrics:
        finite_h = np.isfinite(height_values) & (height_values >= 0)
        c = cells[finite_h]
        h = height_values[finite_h]
        entropy = np.full(n_cells, np.nan, dtype=np.float64)
        if h.size:
            # Eight relative vertical bins per occupied cell; entropy normalized to [0, 1].
            order = np.argsort(c, kind="stable")
            c_s = c[order]
            h_s = h[order]
            unique, starts, counts = np.unique(c_s, return_index=True, return_counts=True)
            for cell, start, n in zip(unique, starts, counts, strict=True):
                vals = h_s[start : start + n]
                vmax = float(np.max(vals))
                if vmax <= 0 or n < 2:
                    entropy[cell] = 0.0
                    continue
                hist, _ = np.histogram(vals, bins=8, range=(0.0, vmax))
                p = hist[hist > 0].astype(np.float64)
                p /= p.sum()
                entropy[cell] = -np.sum(p * np.log(p)) / np.log(8.0)
        result["vertical_entropy"] = raster(entropy)

    if any(m in {"dtm_ground", "dsm_p95", "chm_p95"} for m in metrics):
        dsm = z_stats["p95"]
        dtm = np.full(n_cells, np.nan, dtype=np.float64)
        if classification_attribute is not None and classification_attribute in cloud.attributes:
            if ground_classes is None:
                raise ValueError(
                    "Ground-class metrics were requested but ground_classes is not configured."
                )
            cls = np.asarray(cloud.attributes[classification_attribute])[idx]
            if isinstance(ground_classes, (int, np.integer)):
                classes = np.array([int(ground_classes)])
            else:
                classes = np.asarray(list(ground_classes))
            ground = np.isin(cls, classes)
            if np.any(ground):
                gst = _aggregate_basic(cells[ground], z[ground], n_cells, percentiles=(5,))
                dtm = gst["p05"]
        elif "dtm_ground" in metrics or "chm_p95" in metrics:
            if classification_attribute is None:
                raise ValueError(
                    "Ground-class metrics were requested but no classification_attribute was configured. "
                    "Set it explicitly or map semantic role 'classification' in UAVProjectConfig."
                )
            raise KeyError(
                f"Ground metric requested but configured classification attribute "
                f"{classification_attribute!r} is missing"
            )
        if "dtm_ground" in metrics:
            result["dtm_ground"] = raster(dtm)
        if "dsm_p95" in metrics:
            result["dsm_p95"] = raster(dsm)
        if "chm_p95" in metrics:
            chm = dsm - dtm
            chm[chm < 0] = 0.0
            result["chm_p95"] = raster(chm)

    return result


def lidar_metrics_to_raster_data(
    metrics: Mapping[str, np.ndarray],
    reference_profile: Mapping[str, Any],
    *,
    nodata: float = -9999.0,
    source: str = "lidar_raster_metrics",
) -> RasterData:
    """Pack rasterized LiDAR metrics into one multiband RasterData object."""
    if not metrics:
        raise ValueError("No LiDAR metrics supplied")
    names = list(metrics)
    arrays = [np.asarray(metrics[name], dtype=np.float32) for name in names]
    shape = arrays[0].shape
    if any(a.shape != shape for a in arrays):
        raise ValueError("All LiDAR metric rasters must have identical shapes")
    profile = dict(reference_profile)
    profile.update(count=len(names), dtype="float32", nodata=float(nodata), compress="deflate")
    return RasterData(np.stack(arrays, axis=0), profile, names, source)

# -----------------------------------------------------------------------------
# Streaming / memory-bounded LiDAR rasterization
# -----------------------------------------------------------------------------

def _requested_percentiles(metrics: Iterable[str], prefix: str) -> tuple[int, ...]:
    out: list[int] = []
    token = f"{prefix}_p"
    for name in metrics:
        if str(name).startswith(token):
            tail = str(name)[len(token):]
            try:
                p = int(tail)
            except ValueError:
                continue
            if 0 <= p <= 100:
                out.append(p)
    return tuple(sorted(set(out)))


def _tile_metric_cube(
    records: np.ndarray,
    *,
    tile_height: int,
    tile_width: int,
    metrics: tuple[str, ...],
    pixel_area: float,
    nodata: float,
    canopy_threshold: float,
    ground_classes: tuple[int, ...] | None,
    has_height: bool,
    has_intensity: bool,
    has_classification: bool,
) -> np.ndarray:
    """Compute exact per-cell metrics for one disk-partitioned tile."""
    n_cells = int(tile_height) * int(tile_width)
    cells = records["row"].astype(np.int64) * int(tile_width) + records["col"].astype(np.int64)
    z = records["z"].astype(np.float64, copy=False)

    z_pct = _requested_percentiles(metrics, "z")
    z_stats = _aggregate_basic(cells, z, n_cells, percentiles=z_pct)
    point_count = np.bincount(cells, minlength=n_cells).astype(np.float64)

    out: dict[str, np.ndarray] = {}
    basic = {
        "point_count": point_count,
        "point_density": point_count / pixel_area,
        "z_min": z_stats["min"],
        "z_max": z_stats["max"],
        "z_mean": z_stats["mean"],
        "z_std": z_stats["std"],
    }
    for p in z_pct:
        basic[f"z_p{p:02d}"] = z_stats[f"p{p:02d}"]
    for name, values in basic.items():
        if name in metrics:
            out[name] = values

    if any(m.startswith("intensity_") for m in metrics):
        if not has_intensity:
            raise ValueError("Intensity metrics requested without a configured point attribute")
        st = _aggregate_basic(cells, records["intensity"], n_cells, percentiles=())
        if "intensity_mean" in metrics:
            out["intensity_mean"] = st["mean"]
        if "intensity_std" in metrics:
            out["intensity_std"] = st["std"]

    height_values = None
    if any(m.startswith("height_") or m in {"canopy_cover", "vertical_entropy"} for m in metrics):
        if not has_height:
            raise ValueError("Normalized-height metrics requested without a configured point attribute")
        height_values = records["height"].astype(np.float64, copy=False)
        h_pct = _requested_percentiles(metrics, "height")
        st = _aggregate_basic(cells, height_values, n_cells, percentiles=h_pct)
        hmap = {"height_mean": st["mean"], "height_std": st["std"]}
        for p in h_pct:
            hmap[f"height_p{p:02d}"] = st[f"p{p:02d}"]
        for name, values in hmap.items():
            if name in metrics:
                out[name] = values

    if "canopy_cover" in metrics:
        finite = np.isfinite(height_values)
        total = np.bincount(cells[finite], minlength=n_cells).astype(np.float64)
        above = np.bincount(
            cells[finite & (height_values > float(canopy_threshold))], minlength=n_cells
        ).astype(np.float64)
        cover = np.full(n_cells, np.nan, dtype=np.float64)
        valid = total > 0
        cover[valid] = above[valid] / total[valid]
        out["canopy_cover"] = cover

    if "vertical_entropy" in metrics:
        finite = np.isfinite(height_values) & (height_values >= 0)
        c = cells[finite]
        h = height_values[finite]
        entropy = np.full(n_cells, np.nan, dtype=np.float64)
        if h.size:
            order = np.argsort(c, kind="stable")
            c_s = c[order]
            h_s = h[order]
            unique, starts, counts = np.unique(c_s, return_index=True, return_counts=True)
            for cell, start, n in zip(unique, starts, counts, strict=True):
                vals = h_s[start : start + n]
                vmax = float(np.max(vals))
                if vmax <= 0 or n < 2:
                    entropy[cell] = 0.0
                    continue
                hist, _ = np.histogram(vals, bins=8, range=(0.0, vmax))
                p = hist[hist > 0].astype(np.float64)
                p /= p.sum()
                entropy[cell] = -np.sum(p * np.log(p)) / np.log(8.0)
        out["vertical_entropy"] = entropy

    if any(m in {"dtm_ground", "dsm_p95", "chm_p95"} for m in metrics):
        # These names are metric semantics, not LAS field names. The actual
        # classification dimension is supplied by configuration/caller.
        dsm = _aggregate_basic(cells, z, n_cells, percentiles=(95,))["p95"]
        dtm = np.full(n_cells, np.nan, dtype=np.float64)
        if "dtm_ground" in metrics or "chm_p95" in metrics:
            if not has_classification or not ground_classes:
                raise ValueError(
                    "Ground-based metrics require a configured classification attribute and ground_classes"
                )
            cls = records["class_value"]
            ground = np.isin(cls, np.asarray(ground_classes))
            if np.any(ground):
                dtm = _aggregate_basic(cells[ground], z[ground], n_cells, percentiles=(5,))["p05"]
        if "dtm_ground" in metrics:
            out["dtm_ground"] = dtm
        if "dsm_p95" in metrics:
            out["dsm_p95"] = dsm
        if "chm_p95" in metrics:
            chm = dsm - dtm
            chm[chm < 0] = 0.0
            out["chm_p95"] = chm

    cube = np.full((len(metrics), tile_height, tile_width), float(nodata), dtype=np.float32)
    for band, name in enumerate(metrics):
        values = out.get(name)
        if values is None:
            raise KeyError(f"Unsupported LiDAR raster metric {name!r}")
        arr = np.asarray(values, dtype=np.float64).reshape(tile_height, tile_width)
        if name in {"point_count", "point_density"}:
            arr = np.where(np.isfinite(arr), arr, 0.0)
        else:
            arr = np.where(np.isfinite(arr), arr, float(nodata))
        cube[band] = arr.astype(np.float32)
    return cube



def _compute_lidar_metric_tile_task(task: Mapping[str, Any]):
    """CPU-heavy tile aggregation executed in a worker process."""
    tile_id = int(task["tile_id"])
    n_tile_cols = int(task["n_tile_cols"])
    tile_size = int(task["tile_size"])
    raster_height = int(task["raster_height"])
    raster_width = int(task["raster_width"])
    tr = tile_id // n_tile_cols
    tc = tile_id % n_tile_cols
    row_off = tr * tile_size
    col_off = tc * tile_size
    h = min(tile_size, raster_height - row_off)
    w = min(tile_size, raster_width - col_off)
    record_dtype = np.dtype(task["record_dtype_descr"])
    records = np.fromfile(task["path"], dtype=record_dtype)
    cube = _tile_metric_cube(
        records,
        tile_height=h,
        tile_width=w,
        metrics=tuple(task["metrics"]),
        pixel_area=float(task["pixel_area"]),
        nodata=float(task["nodata"]),
        canopy_threshold=float(task["canopy_threshold"]),
        ground_classes=(tuple(task["ground_classes"]) if task.get("ground_classes") else None),
        has_height=bool(task["has_height"]),
        has_intensity=bool(task["has_intensity"]),
        has_classification=bool(task["has_classification"]),
    )
    return tile_id, row_off, col_off, cube


def resolve_lidar_sources(
    lidar_sources: str | Path | Iterable[str | Path],
    *,
    recursive: bool = False,
) -> list[Path]:
    """Resolve one LAS/LAZ file, a directory, a glob, or an iterable of them.

    Directories are expanded to ``.las`` and ``.laz`` files.  No files are
    merged: every source remains an independent input stream and all points are
    aggregated only after being mapped to the common raster grid.  This avoids
    creating an intermediate merged LAS/LAZ and therefore avoids artefacts
    introduced by an external merge step.

    The caller explicitly selects the input source(s). No filename-based
    inclusion or exclusion rules are applied by the processing core.
    """

    items: list[str | Path]
    if isinstance(lidar_sources, (str, Path)):
        items = [lidar_sources]
    else:
        items = list(lidar_sources)

    found: list[Path] = []
    for item in items:
        text = str(item)
        if any(ch in text for ch in "*?["):
            for match in glob.glob(text, recursive=recursive):
                path = Path(match)
                if path.is_file() and path.suffix.lower() in {".las", ".laz"}:
                    found.append(path)
            continue

        path = Path(item)
        if path.is_dir():
            iterator = path.rglob("*") if recursive else path.glob("*")
            for candidate in iterator:
                if candidate.is_file() and candidate.suffix.lower() in {".las", ".laz"}:
                    found.append(candidate)
        elif path.is_file():
            # Explicit single-file inputs retain support for every point-cloud
            # format understood by DataIngestionHarmonization (for example
            # NPZ in tests/workflows). Directory/glob discovery is deliberately
            # limited to LAS/LAZ because this function targets tiled LiDAR sets.
            found.append(path)
        else:
            raise FileNotFoundError(f"LiDAR source does not exist: {path}")

    # Stable de-duplication while preserving deterministic source order.
    unique: list[Path] = []
    seen: set[str] = set()
    for path in sorted(found, key=lambda x: str(x).lower()):
        key = str(path.resolve()).lower()
        if key not in seen:
            seen.add(key)
            unique.append(path)

    if not unique:
        raise ValueError("No LAS/LAZ source files were resolved")
    return unique


def rasterize_lidar_metrics_files(
    lidar_paths: str | Path | Iterable[str | Path],
    reference_profile: Mapping[str, Any],
    output_path: str | Path,
    *,
    ingestion: Any,
    processing_limits: Any,
    metrics: Iterable[str] = ("point_count", "point_density", "z_mean", "z_std", "z_p95"),
    height_attribute: str | None = None,
    canopy_threshold: float = 2.0,
    classification_attribute: str | None = None,
    ground_classes: int | Iterable[int] | None = None,
    intensity_attribute: str | None = None,
    recursive: bool = False,
    nodata: float = -9999.0,
    aoi: Any | None = None,
) -> dict[str, Any]:
    """Stream one or many LAS/LAZ tiles directly to raster metrics.

    The input may be a single file, a directory, a glob expression, or an
    iterable of LAS/LAZ files. Source files are *not* merged. Each file is
    streamed independently and its points are appended to the same temporary
    spatial raster tiles; metric aggregation is then performed jointly per
    raster tile. Consequently, ``z_mean``, ``z_std``, percentiles and counts
    are computed from all source points that fall in each raster cell, including
    cells crossed by source-file boundaries.

    This method intentionally does not silently de-duplicate overlapping source
    tiles. If the original LAS/LAZ tiles themselves contain padding or duplicate
    points, those points remain part of the data and should be handled using
    source-specific overlap metadata or a separate explicit cleaning step.
    """
    import shutil
    import tempfile
    from concurrent.futures import wait, FIRST_COMPLETED
    import rasterio
    from rasterio.windows import Window
    from ..aoi import point_mask, aoi_mask_for_window

    source_paths = resolve_lidar_sources(
        lidar_paths, recursive=recursive
    )
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics = tuple(dict.fromkeys(str(x) for x in metrics))
    if not metrics:
        raise ValueError("At least one LiDAR metric is required")

    need_height = any(m.startswith("height_") or m in {"canopy_cover", "vertical_entropy"} for m in metrics)
    need_intensity = any(m.startswith("intensity_") for m in metrics)
    need_class = any(m in {"dtm_ground", "chm_p95"} for m in metrics)
    if need_height and not height_attribute:
        raise ValueError("Normalized-height metrics require height_attribute")
    if need_intensity and not intensity_attribute:
        raise ValueError("Intensity metrics require intensity_attribute")
    if need_class and not classification_attribute:
        raise ValueError("Ground-based metrics require classification_attribute")
    if need_class and ground_classes is None:
        raise ValueError("Ground-based metrics require ground_classes")

    attributes = [x for x in (height_attribute, intensity_attribute, classification_attribute) if x]
    source_info: list[dict[str, Any]] = []
    point_count = 0
    for source_path in source_paths:
        info = ingestion.inspect_point_cloud(source_path)
        count = int(info.get("point_count", 0) or 0)
        point_count += count
        source_info.append({
            "path": str(source_path),
            "point_count": count,
            "crs": info.get("crs"),
            "point_format": info.get("point_format"),
        })

    fields: list[tuple[str, Any]] = [("row", np.uint16), ("col", np.uint16), ("z", np.float64)]
    if need_height:
        fields.append(("height", np.float32))
    if need_intensity:
        fields.append(("intensity", np.float32))
    if need_class:
        fields.append(("class_value", np.int32))
    record_dtype = np.dtype(fields, align=False)

    width = int(reference_profile["width"])
    height = int(reference_profile["height"])
    transform = reference_profile["transform"]
    pixel_area = abs(float(transform.a * transform.e - transform.b * transform.d))
    if pixel_area <= 0:
        raise ValueError("Reference grid has invalid pixel area")
    if "point_density" in metrics and reference_profile.get("crs") is not None:
        from pyproj import CRS
        if not CRS.from_user_input(reference_profile["crs"]).is_projected:
            raise ValueError("point_density requires a projected output CRS")

    limits = processing_limits
    budget = int(limits.memory_budget_bytes())
    base_tile = int(limits.raster_tile(bands=max(1, len(metrics)), work_factor=8.0))

    # Use LAS point density and record size to reduce tile size before spooling.
    n_cells = max(1, width * height)
    avg_density = max(1.0, point_count / n_cells) if point_count else 1.0
    target_tile_bytes = max(16 * 1024**2, int(budget * 0.08))
    density_safety = 8.0
    est_safe = int((target_tile_bytes / (avg_density * record_dtype.itemsize * density_safety)) ** 0.5)
    tile_size = max(32, min(base_tile, max(32, (est_safe // 32) * 32)))
    tile_size = min(tile_size, 65504)  # uint16 local indices

    temp_parent = Path(limits.temp_dir) if limits.temp_dir else output_path.parent
    temp_parent.mkdir(parents=True, exist_ok=True)
    estimated_spool = max(1, point_count) * record_dtype.itemsize
    free_disk = shutil.disk_usage(temp_parent).free
    if estimated_spool * 1.15 > free_disk:
        raise OSError(
            f"Insufficient temporary disk space: about {estimated_spool / (1024**3):.1f} GiB is required "
            f"but only {free_disk / (1024**3):.1f} GiB is free in {temp_parent}"
        )

    classes: tuple[int, ...] | None
    if ground_classes is None:
        classes = None
    elif isinstance(ground_classes, (int, np.integer)):
        classes = (int(ground_classes),)
    else:
        classes = tuple(int(x) for x in ground_classes)

    partition_started = time.perf_counter()
    LOGGER.info("LiDAR partition stage: serial streaming I/O from %d source file(s); no worker pool is active", len(source_paths))
    with tempfile.TemporaryDirectory(prefix="uav_biodiv_lidar_", dir=temp_parent) as tmp:
        tmp_dir = Path(tmp)
        n_tile_cols = int(np.ceil(width / tile_size))
        tile_files: dict[int, Path] = {}
        spooled_points = 0

        source_points_in_grid: dict[str, int] = {}
        for source_index, source_path in enumerate(source_paths, start=1):
            source_spooled = 0
            LOGGER.info(
                "LiDAR source %d/%d: streaming %s",
                source_index, len(source_paths), source_path.name,
            )
            for chunk in ingestion.iter_point_cloud_chunks(
                source_path,
                crs=None,
                attributes=attributes,
            ):
                if aoi is not None:
                    keep = point_mask(chunk.xyz, aoi)
                    if not np.any(keep):
                        continue
                    chunk = chunk.subset(keep)
                idx, rows, cols, _ = _point_cells(chunk, reference_profile)
                if idx.size == 0:
                    continue
                tile_r = rows // tile_size
                tile_c = cols // tile_size
                tile_ids = tile_r * n_tile_cols + tile_c
                order = np.argsort(tile_ids, kind="stable")
                ids_sorted = tile_ids[order]
                unique, starts, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
                for tile_id, start, n in zip(unique, starts, counts, strict=True):
                    sel = order[start : start + n]
                    pidx = idx[sel]
                    tr = int(tile_id) // n_tile_cols
                    tc = int(tile_id) % n_tile_cols
                    rec = np.empty(int(n), dtype=record_dtype)
                    rec["row"] = (rows[sel] - tr * tile_size).astype(np.uint16)
                    rec["col"] = (cols[sel] - tc * tile_size).astype(np.uint16)
                    rec["z"] = chunk.xyz[pidx, 2]
                    if need_height:
                        rec["height"] = np.asarray(chunk.attributes[height_attribute])[pidx].astype(np.float32)
                    if need_intensity:
                        rec["intensity"] = np.asarray(chunk.attributes[intensity_attribute])[pidx].astype(np.float32)
                    if need_class:
                        rec["class_value"] = np.asarray(chunk.attributes[classification_attribute])[pidx].astype(np.int32)
                    path = tile_files.setdefault(int(tile_id), tmp_dir / f"tile_{int(tile_id):08d}.bin")
                    with path.open("ab") as fh:
                        rec.tofile(fh)
                    spooled_points += int(n)
                    source_spooled += int(n)
            source_points_in_grid[str(source_path)] = source_spooled

        if not tile_files:
            raise ValueError("No LiDAR points overlap the target raster grid")

        partition_seconds = time.perf_counter() - partition_started
        LOGGER.info("LiDAR partition stage finished in %.1f s; %d spatial tiles ready", partition_seconds, len(tile_files))

        max_file = max(path.stat().st_size for path in tile_files.values())
        # Reading + sorting + working arrays is typically < 5x spool bytes.
        task_bytes = max(64 * 1024**2, max_file * 5)
        workers = limits.effective_worker_count(len(tile_files), estimated_task_bytes=task_bytes)
        workers = max(1, workers)
        per_worker_budget = max(64 * 1024**2, budget // workers)
        if task_bytes > per_worker_budget:
            # Refuse before the OS starts paging. The user can lower tile size
            # in ProcessingLimits if the scene contains extreme local density.
            raise MemoryError(
                f"The densest LiDAR tile may require ~{task_bytes/(1024**3):.2f} GiB in one worker, "
                f"above the per-worker budget {per_worker_budget/(1024**3):.2f} GiB. "
                "Lower processing.raster_tile_size and rerun."
            )

        out_profile = dict(reference_profile)
        out_profile.update(
            count=len(metrics), dtype="float32", nodata=float(nodata), compress="deflate",
            tiled=True, BIGTIFF="YES"
        )

        task_base = {
            "n_tile_cols": n_tile_cols,
            "tile_size": tile_size,
            "raster_height": height,
            "raster_width": width,
            "record_dtype_descr": record_dtype.descr,
            "metrics": metrics,
            "pixel_area": pixel_area,
            "nodata": float(nodata),
            "canopy_threshold": float(canopy_threshold),
            "ground_classes": classes,
            "has_height": need_height,
            "has_intensity": need_intensity,
            "has_classification": need_class,
        }

        compute_started = time.perf_counter()
        if workers > 1:
            LOGGER.info("LiDAR metric stage: starting %d worker processes for %d tiles", workers, len(tile_files))
        else:
            LOGGER.info("LiDAR metric stage: serial computation for %d tiles", len(tile_files))
        with rasterio.open(output_path, "w", **out_profile) as dst:
            for i, name in enumerate(metrics, start=1):
                dst.set_band_description(i, name)
            # Initialise by windows so cells with no point observations have
            # explicit semantics: count/density = 0, all others = nodata.
            for row_off in range(0, height, tile_size):
                h0 = min(tile_size, height - row_off)
                for col_off in range(0, width, tile_size):
                    w0 = min(tile_size, width - col_off)
                    fill = np.full((len(metrics), h0, w0), float(nodata), dtype=np.float32)
                    out_window = Window(col_off, row_off, w0, h0)
                    inside = aoi_mask_for_window(aoi.geojson, reference_profile, out_window) if aoi is not None else None
                    for bi, metric in enumerate(metrics):
                        if metric in {"point_count", "point_density"}:
                            if inside is None:
                                fill[bi].fill(0.0)
                            else:
                                fill[bi][inside] = 0.0
                    dst.write(fill, window=out_window)

            def make_task(tid: int, tpath: Path) -> dict[str, Any]:
                task = dict(task_base)
                task["tile_id"] = int(tid)
                task["path"] = str(tpath)
                return task

            if workers == 1:
                for tid, tpath in tile_files.items():
                    _, row_off, col_off, cube = _compute_lidar_metric_tile_task(make_task(tid, tpath))
                    window = Window(col_off, row_off, cube.shape[2], cube.shape[1])
                    if aoi is not None:
                        inside = aoi_mask_for_window(aoi.geojson, reference_profile, window)
                        cube[:, ~inside] = float(nodata)
                    dst.write(cube, window=window)
            else:
                with process_pool(workers) as executor:
                    iterator = iter(tile_files.items())
                    pending: set[Any] = set()
                    max_in_flight = limits.in_flight_count(workers)

                    def submit_more() -> None:
                        while len(pending) < max_in_flight:
                            try:
                                tid, tpath = next(iterator)
                            except StopIteration:
                                break
                            pending.add(executor.submit(_compute_lidar_metric_tile_task, make_task(tid, tpath)))

                    submit_more()
                    while pending:
                        done, pending = wait(pending, return_when=FIRST_COMPLETED)
                        for future in done:
                            _, row_off, col_off, cube = future.result()
                            window = Window(col_off, row_off, cube.shape[2], cube.shape[1])
                            if aoi is not None:
                                inside = aoi_mask_for_window(aoi.geojson, reference_profile, window)
                                cube[:, ~inside] = float(nodata)
                            dst.write(cube, window=window)
                        submit_more()

        compute_seconds = time.perf_counter() - compute_started
        return {
            "output_path": str(output_path),
            "metrics": list(metrics),
            "input_mode": "single_file" if len(source_paths) == 1 else "tile_collection",
            "source_file_count": int(len(source_paths)),
            "source_files": [str(p) for p in source_paths],
            "source_info": source_info,
            "source_points_in_grid": source_points_in_grid,
            "aoi": ({**aoi.to_metadata(), "mask_applied": True} if aoi is not None else {"enabled": False}),
            "point_count_header": point_count,
            "points_in_grid": int(spooled_points),
            "tile_size": int(tile_size),
            "tiles": int(len(tile_files)),
            "parallel_backend": "process" if workers > 1 else "serial",
            "partition_stage": "serial_streaming_io",
            "worker_spawn_stage": "parallel_metric_compute",
            "partition_seconds": float(partition_seconds),
            "parallel_compute_seconds": float(compute_seconds),
            "workers": int(workers),
            "memory_budget_gb": budget / (1024**3),
            "resource_plan": limits.resource_plan(estimated_task_bytes=task_bytes, bands=max(1, len(metrics))),
            "max_tile_spool_mb": max_file / (1024**2),
            "temporary_spool_gb": sum(p.stat().st_size for p in tile_files.values()) / (1024**3),
        }

def rasterize_lidar_metrics_file(
    lidar_path: str | Path,
    reference_profile: Mapping[str, Any],
    output_path: str | Path,
    *,
    ingestion: Any,
    processing_limits: Any,
    metrics: Iterable[str] = ("point_count", "point_density", "z_mean", "z_std", "z_p95"),
    height_attribute: str | None = None,
    canopy_threshold: float = 2.0,
    classification_attribute: str | None = None,
    ground_classes: int | Iterable[int] | None = None,
    intensity_attribute: str | None = None,
    nodata: float = -9999.0,
) -> dict[str, Any]:
    """Backward-compatible single-file wrapper.

    New code that may consume tiled source clouds should use
    :func:`rasterize_lidar_metrics_files`.
    """
    return rasterize_lidar_metrics_files(
        lidar_path,
        reference_profile,
        output_path,
        ingestion=ingestion,
        processing_limits=processing_limits,
        metrics=metrics,
        height_attribute=height_attribute,
        canopy_threshold=canopy_threshold,
        classification_attribute=classification_attribute,
        ground_classes=ground_classes,
        intensity_attribute=intensity_attribute,
        nodata=nodata,
    )

