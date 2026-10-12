from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import numpy as np
from scipy import ndimage
from scipy.optimize import minimize


@dataclass(slots=True)
class RasterRegistrationResult:
    """Geometric registration result for one raster source group.

    The 3x3 homogeneous matrix maps moving pixels to reference pixels in the
    final analysis grid using ``[column, row, 1]`` coordinates. Registration
    is conservative by default: an estimated correction is applied only when
    it stays inside the configured geometric-accuracy envelope and passes the
    multimodal QA checks. Otherwise the identity transform is retained.
    """

    group: str
    reference_group: str
    method: str
    transform_type: str
    selected_model: str
    model_selection: dict[str, Any]
    transformation: np.ndarray
    preview_shape: tuple[int, int]
    preview_levels: tuple[tuple[int, int], ...]
    reference_keys: tuple[str, ...]
    moving_keys: tuple[str, ...]
    phase_translation_preview_px: tuple[float, float]
    mutual_information_before: float
    mutual_information_after: float
    gradient_correlation_before: float
    gradient_correlation_after: float
    score_before: float
    score_after: float
    score_improvement: float
    residual_shift_full_px: tuple[float, float]
    center_displacement_full_px: tuple[float, float]
    accepted: bool
    applied: bool
    acceptance_reason: str
    constraint_source: str
    relative_rms_map_units: tuple[float | None, float | None]
    max_allowed_shift_map_units: tuple[float, float]
    estimated_shift_map_units: tuple[float, float]
    maximum_displacement_map_units: tuple[float, float]
    local_validation: dict[str, Any]
    converged: bool
    optimizer_message: str

    def to_dict(self) -> dict[str, Any]:
        return {
            "group": self.group,
            "reference_group": self.reference_group,
            "method": self.method,
            "transform_type": self.transform_type,
            "selected_model": self.selected_model,
            "model_selection": dict(self.model_selection),
            "transformation_pixel": np.asarray(self.transformation, dtype=float).tolist(),
            "preview_shape": [int(self.preview_shape[0]), int(self.preview_shape[1])],
            "preview_levels": [[int(h), int(w)] for h, w in self.preview_levels],
            "reference_keys": list(self.reference_keys),
            "moving_keys": list(self.moving_keys),
            "phase_translation_preview_px": [float(x) for x in self.phase_translation_preview_px],
            "mutual_information_before": float(self.mutual_information_before),
            "mutual_information_after": float(self.mutual_information_after),
            "gradient_correlation_before": float(self.gradient_correlation_before),
            "gradient_correlation_after": float(self.gradient_correlation_after),
            "score_before": float(self.score_before),
            "score_after": float(self.score_after),
            "score_improvement": float(self.score_improvement),
            "residual_shift_full_px": [float(x) for x in self.residual_shift_full_px],
            "center_displacement_full_px": [float(x) for x in self.center_displacement_full_px],
            "accepted": bool(self.accepted),
            "applied": bool(self.applied),
            "acceptance_reason": str(self.acceptance_reason),
            "constraint_source": str(self.constraint_source),
            "relative_rms_map_units": [
                None if x is None else float(x) for x in self.relative_rms_map_units
            ],
            "max_allowed_shift_map_units": [float(x) for x in self.max_allowed_shift_map_units],
            "estimated_shift_map_units": [float(x) for x in self.estimated_shift_map_units],
            "maximum_displacement_map_units": [
                float(x) for x in self.maximum_displacement_map_units
            ],
            "local_validation": dict(self.local_validation),
            "converged": bool(self.converged),
            "optimizer_message": str(self.optimizer_message),
        }


def _resampling_enum(name: str):
    from rasterio.enums import Resampling

    value = str(name).lower()
    if not hasattr(Resampling, value):
        raise ValueError(
            f"Unknown resampling method {name!r}; use a rasterio resampling name such as "
            "nearest, bilinear, cubic or average"
        )
    return getattr(Resampling, value)


def _source_resampling(resampling: Mapping[str, str] | str, key: str) -> str:
    return str(resampling.get(key, "bilinear") if isinstance(resampling, Mapping) else resampling)


def _robust_normalize(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    arr = np.asarray(values, dtype=np.float32)
    mask = np.asarray(valid, dtype=bool) & np.isfinite(arr)
    out = np.zeros(arr.shape, dtype=np.float32)
    if np.count_nonzero(mask) < 32:
        return out
    sample = arr[mask]
    lo, hi = np.percentile(sample, [2.0, 98.0])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(sample))
        hi = float(np.max(sample))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return out
    out[mask] = np.clip((arr[mask] - lo) / (hi - lo), 0.0, 1.0)
    return out


def _gradient_representation(values: np.ndarray, valid: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(valid, dtype=bool)
    # Avoid artificial gradients at nodata boundaries.
    safe_valid = ndimage.binary_erosion(valid, iterations=1, border_value=0)
    normalized = _robust_normalize(values, valid)
    smoothed = ndimage.gaussian_filter(normalized, sigma=1.0, mode="nearest")
    gx = ndimage.sobel(smoothed, axis=1, mode="nearest")
    gy = ndimage.sobel(smoothed, axis=0, mode="nearest")
    grad = np.hypot(gx, gy).astype(np.float32)
    grad = _robust_normalize(grad, safe_valid)
    grad[~safe_valid] = 0.0
    return grad, safe_valid


def _alpha_band_index(src: Any) -> int | None:
    """Return the 1-based alpha-band index when the source exposes one.

    Some orthomosaics carry both a dataset-level NoData value and an alpha
    band. Rasterio warns when a masked read lets the NoData mask shadow the
    alpha band. For geometric co-registration we explicitly prefer alpha as
    the source validity geometry when it is available.
    """
    from rasterio.enums import ColorInterp

    for index, color_interp in enumerate(src.colorinterp, start=1):
        if color_interp == ColorInterp.alpha:
            return int(index)
    return None


def _target_vrt_options(
    src: Any,
    profile: Mapping[str, Any],
    *,
    method: str,
    nodata: float,
) -> tuple[dict[str, Any], int | None]:
    """Build WarpedVRT options and explicitly activate a source alpha band.

    The destination NoData value remains an output fill value. If a source
    alpha band exists, GDAL uses it to define valid source geometry during
    warping/resampling, while the original pixel values remain untouched.
    """
    alpha_band = _alpha_band_index(src)
    options: dict[str, Any] = {
        "crs": profile["crs"],
        "transform": profile["transform"],
        "width": int(profile["width"]),
        "height": int(profile["height"]),
        "resampling": _resampling_enum(method),
        "nodata": float(nodata),
    }
    if alpha_band is not None:
        options["src_alpha"] = int(alpha_band)
    return options, alpha_band


def _read_vrt_with_validity(
    vrt: Any,
    band: int,
    *,
    alpha_band: int | None,
    nodata: float,
    **read_kwargs: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a VRT band and return both data and an explicit validity mask.

    For RGBA/alpha-masked products, masked reads are intentionally avoided:
    Rasterio otherwise emits ``NodataShadowWarning`` when dataset NoData and
    alpha coexist. Because ``src_alpha`` is configured on the WarpedVRT, GDAL
    writes alpha-invalid destination pixels as the requested destination NoData
    value. For sources without alpha, the normal Rasterio masked-read path is
    retained so dataset/internal masks continue to be honored.
    """
    if alpha_band is not None:
        data = np.asarray(
            vrt.read(int(band), masked=False, **read_kwargs),
            dtype=np.float32,
        )
        valid = np.isfinite(data)
        if np.isnan(float(nodata)):
            valid &= ~np.isnan(data)
        else:
            valid &= ~np.isclose(data, float(nodata))
        return data, valid

    arr = vrt.read(int(band), masked=True, **read_kwargs)
    data = np.asarray(arr.filled(float(nodata)), dtype=np.float32)
    valid = ~np.ma.getmaskarray(arr)
    valid &= np.isfinite(data)
    if np.isnan(float(nodata)):
        valid &= ~np.isnan(data)
    else:
        valid &= ~np.isclose(data, float(nodata))
    return data, valid


def _preview_shape(profile: Mapping[str, Any], max_dimension: int) -> tuple[int, int]:
    height = int(profile["height"])
    width = int(profile["width"])
    if max(height, width) <= max_dimension:
        return height, width
    scale = max(height, width) / float(max_dimension)
    return max(64, int(round(height / scale))), max(64, int(round(width / scale)))


def build_registration_preview(
    specs: Mapping[str, tuple[Path, int]],
    profile: Mapping[str, Any],
    keys: Sequence[str],
    *,
    resampling: Mapping[str, str] | str = "bilinear",
    nodata: float = -9999.0,
    max_dimension: int = 512,
    aoi_geojson: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a normalized multimodal registration representation.

    The representation is the mean of per-channel gradient magnitudes. The
    original source values and units are untouched and are never replaced by
    this representation in the analysis-ready output.
    """
    import rasterio
    from rasterio.vrt import WarpedVRT

    if not keys:
        raise ValueError("At least one registration key is required")
    missing = [key for key in keys if key not in specs]
    if missing:
        raise KeyError(f"Registration keys are missing from raster sources: {missing}")

    ph, pw = _preview_shape(profile, int(max_dimension))
    gradients: list[np.ndarray] = []
    masks: list[np.ndarray] = []
    for key in keys:
        path, band = specs[key]
        method = _source_resampling(resampling, key)
        with rasterio.open(path) as src:
            vrt_opts, alpha_band = _target_vrt_options(
                src, profile, method=method, nodata=float(nodata)
            )
            with WarpedVRT(src, **vrt_opts) as vrt:
                data, valid = _read_vrt_with_validity(
                    vrt,
                    int(band),
                    alpha_band=alpha_band,
                    nodata=float(nodata),
                    out_shape=(ph, pw),
                    out_dtype="float32",
                    resampling=_resampling_enum(method),
                )
        grad, grad_valid = _gradient_representation(data, valid)
        gradients.append(grad)
        masks.append(grad_valid)

    stack = np.stack(gradients, axis=0)
    mask_stack = np.stack(masks, axis=0)
    count = np.sum(mask_stack, axis=0)
    representation = np.divide(
        np.sum(np.where(mask_stack, stack, 0.0), axis=0),
        count,
        out=np.zeros((ph, pw), dtype=np.float32),
        where=count > 0,
    ).astype(np.float32)
    valid = count > 0
    if aoi_geojson is not None:
        from affine import Affine
        from rasterio.features import geometry_mask

        preview_transform = profile["transform"] * Affine.scale(
            float(profile["width"]) / float(pw),
            float(profile["height"]) / float(ph),
        )
        inside = geometry_mask(
            [dict(aoi_geojson)],
            out_shape=(ph, pw),
            transform=preview_transform,
            invert=True,
            all_touched=False,
        )
        valid &= inside
    representation[~valid] = 0.0
    return representation, valid


def _parabolic_offset(v_minus: float, v0: float, v_plus: float) -> float:
    denominator = v_minus - 2.0 * v0 + v_plus
    if abs(denominator) < 1e-15:
        return 0.0
    return float(np.clip(0.5 * (v_minus - v_plus) / denominator, -1.0, 1.0))


def phase_translation(
    fixed: np.ndarray,
    moving: np.ndarray,
    *,
    fixed_mask: np.ndarray | None = None,
    moving_mask: np.ndarray | None = None,
    max_shift_x: float | None = None,
    max_shift_y: float | None = None,
) -> tuple[float, float]:
    """Estimate the translation that maps ``moving`` onto ``fixed``.

    Returns ``(dx, dy)`` in pixel coordinates, where positive ``dx`` moves the
    moving image to larger columns and positive ``dy`` to larger rows. When
    ``max_shift_x``/``max_shift_y`` are supplied, the correlation peak search
    is restricted to that physically plausible displacement envelope instead
    of finding an arbitrary global FFT peak.
    """
    f = np.asarray(fixed, dtype=np.float64)
    m = np.asarray(moving, dtype=np.float64)
    if f.shape != m.shape:
        raise ValueError("fixed and moving registration images must have the same shape")
    fm = np.ones(f.shape, dtype=bool) if fixed_mask is None else np.asarray(fixed_mask, dtype=bool)
    mm = np.ones(m.shape, dtype=bool) if moving_mask is None else np.asarray(moving_mask, dtype=bool)

    f = np.where(fm & np.isfinite(f), f, 0.0)
    m = np.where(mm & np.isfinite(m), m, 0.0)
    if np.any(fm):
        f = np.where(fm, f - np.mean(f[fm]), 0.0)
    if np.any(mm):
        m = np.where(mm, m - np.mean(m[mm]), 0.0)

    wy = np.hanning(f.shape[0]) if f.shape[0] > 2 else np.ones(f.shape[0])
    wx = np.hanning(f.shape[1]) if f.shape[1] > 2 else np.ones(f.shape[1])
    window = wy[:, None] * wx[None, :]
    f *= window
    m *= window

    ff = np.fft.fft2(f)
    mmf = np.fft.fft2(m)
    cross = ff * np.conj(mmf)
    magnitude = np.abs(cross)
    cross = np.divide(cross, magnitude, out=np.zeros_like(cross), where=magnitude > 1e-15)
    corr = np.real(np.fft.ifft2(cross))

    row_indices = np.arange(corr.shape[0], dtype=np.float64)
    row_shifts = np.where(
        row_indices > corr.shape[0] // 2, row_indices - corr.shape[0], row_indices
    )
    col_indices = np.arange(corr.shape[1], dtype=np.float64)
    col_shifts = np.where(
        col_indices > corr.shape[1] // 2, col_indices - corr.shape[1], col_indices
    )
    allowed = np.ones(corr.shape, dtype=bool)
    if max_shift_y is not None and np.isfinite(float(max_shift_y)):
        allowed &= np.abs(row_shifts[:, None]) <= max(0.0, float(max_shift_y)) + 1e-9
    if max_shift_x is not None and np.isfinite(float(max_shift_x)):
        allowed &= np.abs(col_shifts[None, :]) <= max(0.0, float(max_shift_x)) + 1e-9
    if not np.any(allowed):
        return 0.0, 0.0

    restricted = np.where(allowed, corr, -np.inf)
    row, col = np.unravel_index(np.argmax(restricted), restricted.shape)
    row_shift = float(row_shifts[int(row)])
    col_shift = float(col_shifts[int(col)])

    # Subpixel parabolic peak refinement. If a neighbour lies outside the
    # configured search envelope, no fractional correction is inferred on that
    # side; this prevents refinement from escaping the physical constraint.
    r0 = int(row)
    c0 = int(col)
    r_minus = (r0 - 1) % corr.shape[0]
    r_plus = (r0 + 1) % corr.shape[0]
    c_minus = (c0 - 1) % corr.shape[1]
    c_plus = (c0 + 1) % corr.shape[1]
    dy = 0.0
    dx = 0.0
    if allowed[r_minus, c0] and allowed[r_plus, c0]:
        dy = _parabolic_offset(
            float(corr[r_minus, c0]), float(corr[r0, c0]), float(corr[r_plus, c0])
        )
    if allowed[r0, c_minus] and allowed[r0, c_plus]:
        dx = _parabolic_offset(
            float(corr[r0, c_minus]), float(corr[r0, c0]), float(corr[r0, c_plus])
        )

    estimate_x = col_shift + dx
    estimate_y = row_shift + dy
    if max_shift_x is not None and np.isfinite(float(max_shift_x)):
        estimate_x = float(np.clip(estimate_x, -abs(float(max_shift_x)), abs(float(max_shift_x))))
    if max_shift_y is not None and np.isfinite(float(max_shift_y)):
        estimate_y = float(np.clip(estimate_y, -abs(float(max_shift_y)), abs(float(max_shift_y))))
    return float(estimate_x), float(estimate_y)


def _translation_matrix(dx: float, dy: float) -> np.ndarray:
    matrix = np.eye(3, dtype=np.float64)
    matrix[0, 2] = float(dx)
    matrix[1, 2] = float(dy)
    return matrix


def _affine_matrix(
    *,
    dx: float,
    dy: float,
    rotation_deg: float,
    scale_x: float,
    scale_y: float,
    shear_deg: float,
    center_col: float,
    center_row: float,
) -> np.ndarray:
    theta = math.radians(float(rotation_deg))
    shear = math.radians(float(shear_deg))
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    scale = np.array([[float(scale_x), 0.0, 0.0], [0.0, float(scale_y), 0.0], [0.0, 0.0, 1.0]])
    shear_matrix = np.array([[1.0, math.tan(shear), 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    to_origin = _translation_matrix(-center_col, -center_row)
    back = _translation_matrix(center_col, center_row)
    translate = _translation_matrix(dx, dy)
    return translate @ back @ rotation @ shear_matrix @ scale @ to_origin


def warp_registration_image(
    image: np.ndarray,
    matrix_moving_to_fixed: np.ndarray,
    *,
    output_shape: tuple[int, int],
    order: int = 1,
    cval: float = 0.0,
) -> np.ndarray:
    """Warp an image using a moving->fixed homogeneous pixel matrix."""
    inv = np.linalg.inv(np.asarray(matrix_moving_to_fixed, dtype=np.float64))
    rows, cols = np.indices(output_shape, dtype=np.float64)
    moving_col = inv[0, 0] * cols + inv[0, 1] * rows + inv[0, 2]
    moving_row = inv[1, 0] * cols + inv[1, 1] * rows + inv[1, 2]
    return ndimage.map_coordinates(
        np.asarray(image, dtype=np.float32),
        [moving_row, moving_col],
        order=int(order),
        mode="constant",
        cval=float(cval),
        prefilter=bool(order > 1),
    ).astype(np.float32)


def _warped_mask(mask: np.ndarray, matrix: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    return warp_registration_image(mask.astype(np.float32), matrix, output_shape=shape, order=0, cval=0.0) > 0.5


def normalized_mutual_information(
    fixed: np.ndarray,
    moving_aligned: np.ndarray,
    valid: np.ndarray,
    *,
    bins: int = 48,
) -> float:
    valid = np.asarray(valid, dtype=bool)
    valid &= np.isfinite(fixed) & np.isfinite(moving_aligned)
    if np.count_nonzero(valid) < 256:
        return float("-inf")
    x = np.clip(np.asarray(fixed, dtype=np.float64)[valid], 0.0, 1.0)
    y = np.clip(np.asarray(moving_aligned, dtype=np.float64)[valid], 0.0, 1.0)
    hist, _, _ = np.histogram2d(x, y, bins=int(bins), range=((0.0, 1.0), (0.0, 1.0)))
    total = float(np.sum(hist))
    if total <= 0:
        return float("-inf")
    pxy = hist / total
    px = np.sum(pxy, axis=1)
    py = np.sum(pxy, axis=0)
    nz = pxy > 0
    denom = px[:, None] * py[None, :]
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / np.maximum(denom[nz], 1e-15))))
    hx = -float(np.sum(px[px > 0] * np.log(px[px > 0])))
    hy = -float(np.sum(py[py > 0] * np.log(py[py > 0])))
    if hx <= 0 or hy <= 0:
        return 0.0
    return float(mi / math.sqrt(hx * hy))


def gradient_correlation(fixed: np.ndarray, moving_aligned: np.ndarray, valid: np.ndarray) -> float:
    valid = np.asarray(valid, dtype=bool)
    valid &= np.isfinite(fixed) & np.isfinite(moving_aligned)
    if np.count_nonzero(valid) < 32:
        return float("nan")
    x = np.asarray(fixed, dtype=np.float64)[valid]
    y = np.asarray(moving_aligned, dtype=np.float64)[valid]
    x -= np.mean(x)
    y -= np.mean(y)
    den = float(np.sqrt(np.sum(x * x) * np.sum(y * y)))
    return float(np.sum(x * y) / den) if den > 0 else float("nan")


def _evaluate_alignment(
    fixed: np.ndarray,
    fixed_mask: np.ndarray,
    moving: np.ndarray,
    moving_mask: np.ndarray,
    matrix: np.ndarray,
) -> tuple[float, float, np.ndarray, np.ndarray]:
    aligned = warp_registration_image(moving, matrix, output_shape=fixed.shape, order=1, cval=0.0)
    aligned_mask = _warped_mask(moving_mask, matrix, fixed.shape)
    valid = fixed_mask & aligned_mask
    return (
        normalized_mutual_information(fixed, aligned, valid),
        gradient_correlation(fixed, aligned, valid),
        aligned,
        valid,
    )



def _alignment_score(mi: float, corr: float) -> float:
    """Combined multimodal score used only for geometric QA/selection."""
    if not np.isfinite(mi):
        return float("-inf")
    corr_term = float(corr) if np.isfinite(corr) else 0.0
    return float(mi + 0.25 * max(-1.0, min(1.0, corr_term)))


def _matrix_displacement_bounds_px(matrix: np.ndarray, shape: tuple[int, int]) -> tuple[float, float]:
    """Maximum absolute X/Y displacement caused by a matrix over an image."""
    h, w = int(shape[0]), int(shape[1])
    pts = np.array(
        [
            [0.0, 0.0, 1.0],
            [max(0, w - 1), 0.0, 1.0],
            [0.0, max(0, h - 1), 1.0],
            [max(0, w - 1), max(0, h - 1), 1.0],
            [(max(0, w - 1)) / 2.0, (max(0, h - 1)) / 2.0, 1.0],
        ],
        dtype=np.float64,
    ).T
    moved = np.asarray(matrix, dtype=np.float64) @ pts
    delta = moved[:2, :] - pts[:2, :]
    return float(np.max(np.abs(delta[0]))), float(np.max(np.abs(delta[1])))


def _local_translation_consensus(
    fixed: np.ndarray,
    fixed_mask: np.ndarray,
    moving_aligned: np.ndarray,
    moving_mask_aligned: np.ndarray,
    *,
    grid_size: int = 3,
    max_residual_px: float = 2.0,
    residual_tolerance_px: float = 0.75,
    dispersion_tolerance_px: float = 1.0,
    min_windows: int = 4,
) -> dict[str, Any]:
    """Validate a global correction using independent local residual shifts.

    This is deliberately a QA step, not a local-warp model. A correction is
    accepted only when several spatial windows agree that little residual
    translation remains after the global transform.
    """
    h, w = fixed.shape
    grid = max(2, int(grid_size))
    estimates: list[tuple[float, float]] = []
    for gy in range(grid):
        r0 = int(round(gy * h / grid))
        r1 = int(round((gy + 1) * h / grid))
        for gx in range(grid):
            c0 = int(round(gx * w / grid))
            c1 = int(round((gx + 1) * w / grid))
            if r1 - r0 < 24 or c1 - c0 < 24:
                continue
            f = fixed[r0:r1, c0:c1]
            m = moving_aligned[r0:r1, c0:c1]
            fm = fixed_mask[r0:r1, c0:c1]
            mm = moving_mask_aligned[r0:r1, c0:c1]
            valid = fm & mm & np.isfinite(f) & np.isfinite(m)
            min_valid = max(128, int(0.12 * valid.size))
            if np.count_nonzero(valid) < min_valid:
                continue
            # Feature-poor windows (uniform field/sky/water/etc.) do not
            # provide a defensible registration constraint.
            if float(np.std(f[valid])) < 1e-4 or float(np.std(m[valid])) < 1e-4:
                continue
            dx, dy = phase_translation(
                f,
                m,
                fixed_mask=fm,
                moving_mask=mm,
                max_shift_x=float(max_residual_px),
                max_shift_y=float(max_residual_px),
            )
            estimates.append((float(dx), float(dy)))

    if len(estimates) < int(min_windows):
        return {
            "accepted": False,
            "reason": "insufficient informative local windows",
            "windows_used": int(len(estimates)),
            "min_windows": int(min_windows),
            "median_residual_px": [None, None],
            "mad_residual_px": [None, None],
            "inlier_fraction": 0.0,
        }

    arr = np.asarray(estimates, dtype=np.float64)
    med = np.median(arr, axis=0)
    mad = 1.4826 * np.median(np.abs(arr - med[None, :]), axis=0)
    dist = np.sqrt(np.sum((arr - med[None, :]) ** 2, axis=1))
    inlier_radius = max(1.0, 2.5 * float(np.hypot(mad[0], mad[1])))
    inlier_fraction = float(np.mean(dist <= inlier_radius))
    residual_norm = float(np.hypot(med[0], med[1]))
    accepted = (
        residual_norm <= float(residual_tolerance_px)
        and float(np.max(mad)) <= float(dispersion_tolerance_px)
        and inlier_fraction >= 0.60
    )
    reason = "local residual consensus passed" if accepted else "local residual consensus failed"
    return {
        "accepted": bool(accepted),
        "reason": reason,
        "windows_used": int(len(estimates)),
        "min_windows": int(min_windows),
        "median_residual_px": [float(med[0]), float(med[1])],
        "mad_residual_px": [float(mad[0]), float(mad[1])],
        "inlier_fraction": float(inlier_fraction),
        "residual_tolerance_px": float(residual_tolerance_px),
        "dispersion_tolerance_px": float(dispersion_tolerance_px),
    }


def _group_accuracy(group: Mapping[str, Any]) -> tuple[float | None, float | None]:
    """Read RMS X/Y from a source-group configuration in analysis map units.

    Accepted forms are either top-level ``rms``/``rms_x``/``rms_y`` or a
    nested ``geometric_accuracy`` mapping containing the same keys. A scalar
    ``rms`` is used for both axes. No unit conversion is guessed: values are
    interpreted in the map units of the analysis grid.
    """
    nested = group.get("geometric_accuracy")
    accuracy = dict(nested) if isinstance(nested, Mapping) else {}
    scalar = accuracy.get("rms", group.get("rms"))
    x = accuracy.get("rms_x", group.get("rms_x", scalar))
    y = accuracy.get("rms_y", group.get("rms_y", scalar))

    def clean(value: Any) -> float | None:
        if value is None:
            return None
        value = float(value)
        if not np.isfinite(value) or value < 0:
            raise ValueError("RMS values must be finite and >= 0")
        return value

    return clean(x), clean(y)


def _source_native_resolution(
    specs: Mapping[str, tuple[Path, int]],
    keys: Sequence[str],
    target_crs: Any,
) -> tuple[float | None, float | None]:
    """Return largest native pixel size when source/target CRS units match."""
    import rasterio

    rx: list[float] = []
    ry: list[float] = []
    for key in keys:
        path, _band = specs[str(key)]
        with rasterio.open(path) as src:
            if src.crs is None or target_crs is None or src.crs != target_crs:
                continue
            transform = src.transform
            x = float((float(transform.a) ** 2 + float(transform.d) ** 2) ** 0.5)
            y = float((float(transform.b) ** 2 + float(transform.e) ** 2) ** 0.5)
            if x > 0 and np.isfinite(x):
                rx.append(x)
            if y > 0 and np.isfinite(y):
                ry.append(y)
    return (max(rx) if rx else None, max(ry) if ry else None)


def _registration_shift_limits(
    *,
    reference_group_cfg: Mapping[str, Any],
    moving_group_cfg: Mapping[str, Any],
    reference_native_resolution: tuple[float | None, float | None],
    moving_native_resolution: tuple[float | None, float | None],
    target_pixel_size: tuple[float, float],
    rms_multiplier: float,
    fallback_max_native_pixels: float,
) -> tuple[tuple[float, float], tuple[float | None, float | None], str]:
    """Return conservative X/Y displacement limits in analysis map units."""
    ref_rms = _group_accuracy(reference_group_cfg)
    mov_rms = _group_accuracy(moving_group_cfg)

    direct = moving_group_cfg.get("max_shift_map_units")
    direct_x = moving_group_cfg.get("max_shift_x")
    direct_y = moving_group_cfg.get("max_shift_y")
    if isinstance(direct, Sequence) and not isinstance(direct, (str, bytes)) and len(direct) >= 2:
        direct_x = direct[0]
        direct_y = direct[1]
    elif direct is not None and not isinstance(direct, Sequence):
        direct_x = direct_y = direct
    if direct_x is not None or direct_y is not None:
        if direct_x is None:
            direct_x = direct_y
        if direct_y is None:
            direct_y = direct_x
        dx = float(direct_x)
        dy = float(direct_y)
        if dx < 0 or dy < 0 or not np.isfinite(dx) or not np.isfinite(dy):
            raise ValueError("max_shift_x/max_shift_y must be finite and >= 0")
        rel = (
            None if ref_rms[0] is None or mov_rms[0] is None else float(math.hypot(ref_rms[0], mov_rms[0])),
            None if ref_rms[1] is None or mov_rms[1] is None else float(math.hypot(ref_rms[1], mov_rms[1])),
        )
        return (dx, dy), rel, "explicit_max_shift"

    if ref_rms[0] is not None and ref_rms[1] is not None and mov_rms[0] is not None and mov_rms[1] is not None:
        rel_x = float(math.hypot(ref_rms[0], mov_rms[0]))
        rel_y = float(math.hypot(ref_rms[1], mov_rms[1]))
        return (
            max(0.0, float(rms_multiplier) * rel_x),
            max(0.0, float(rms_multiplier) * rel_y),
        ), (rel_x, rel_y), "combined_source_rms"

    # If RMS is unavailable, do not permit an unlimited image-registration
    # search. Bound it by a few pixels of the coarser native source geometry.
    tx, ty = target_pixel_size
    rnx, rny = reference_native_resolution
    mnx, mny = moving_native_resolution
    basis_x = max([v for v in (tx, rnx, mnx) if v is not None and np.isfinite(v) and v > 0])
    basis_y = max([v for v in (ty, rny, mny) if v is not None and np.isfinite(v) and v > 0])
    return (
        float(fallback_max_native_pixels) * float(basis_x),
        float(fallback_max_native_pixels) * float(basis_y),
    ), (None, None), "fallback_native_resolution"

def estimate_group_registration(
    fixed: np.ndarray,
    fixed_mask: np.ndarray,
    moving: np.ndarray,
    moving_mask: np.ndarray,
    *,
    transform_type: str = "translation",
    max_translation_x_px: float | None = None,
    max_translation_y_px: float | None = None,
    max_rotation_deg: float = 0.25,
    max_scale_delta: float = 0.005,
    max_shear_deg: float = 0.10,
    refinement_translation_px: float = 2.0,
    optimizer_maxiter: int = 40,
    min_score_improvement: float = 0.003,
    require_local_consensus: bool = True,
    local_grid_size: int = 3,
    local_max_residual_px: float = 2.0,
    local_residual_tolerance_px: float = 0.75,
    local_dispersion_tolerance_px: float = 1.0,
    local_min_windows: int = 4,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Estimate one constrained multimodal transform on a common grid.

    ``translation`` estimates only X/Y residual displacement. ``similarity``
    adds a small rotation and one uniform scale parameter. ``affine`` adds
    independent X/Y scale and shear. Every model is constrained by the same
    maximum spatial-displacement envelope and is rejected unless it improves
    the multimodal score and passes local residual-consensus QA.
    """
    transform_type = str(transform_type).lower().strip()
    if transform_type not in {"translation", "similarity", "affine"}:
        raise ValueError("transform_type must be 'translation', 'similarity' or 'affine'")

    max_x = None if max_translation_x_px is None else max(0.0, float(max_translation_x_px))
    max_y = None if max_translation_y_px is None else max(0.0, float(max_translation_y_px))
    identity = np.eye(3, dtype=np.float64)
    mi_before, corr_before, _, _ = _evaluate_alignment(
        fixed, fixed_mask, moving, moving_mask, identity
    )
    score_before = _alignment_score(mi_before, corr_before)

    phase_dx, phase_dy = phase_translation(
        fixed,
        moving,
        fixed_mask=fixed_mask,
        moving_mask=moving_mask,
        max_shift_x=max_x,
        max_shift_y=max_y,
    )
    translation = _translation_matrix(phase_dx, phase_dy)
    chosen = translation
    converged = True
    message = "constrained phase cross-correlation translation"

    if transform_type in {"similarity", "affine"}:
        center_col = (fixed.shape[1] - 1) / 2.0
        center_row = (fixed.shape[0] - 1) / 2.0
        rotation_values = np.linspace(-abs(float(max_rotation_deg)), abs(float(max_rotation_deg)), 5)
        scale_delta = abs(float(max_scale_delta))
        scale_values = (
            np.array([1.0 - scale_delta, 1.0, 1.0 + scale_delta])
            if scale_delta > 0
            else np.array([1.0])
        )
        best_matrix = translation
        mi_t, corr_t, _, _ = _evaluate_alignment(
            fixed, fixed_mask, moving, moving_mask, translation
        )
        best_score = _alignment_score(mi_t, corr_t)

        if transform_type == "similarity":
            best_params = np.array([phase_dx, phase_dy, 0.0, 1.0], dtype=np.float64)
        else:
            best_params = np.array(
                [phase_dx, phase_dy, 0.0, 1.0, 1.0, 0.0], dtype=np.float64
            )

        # Conservative coarse search: only rotation + uniform scale are used
        # to initialise either model. Full affine anisotropy/shear is allowed
        # only during the bounded refinement below.
        for angle in rotation_values:
            for isotropic_scale in scale_values:
                base = _affine_matrix(
                    dx=0.0,
                    dy=0.0,
                    rotation_deg=float(angle),
                    scale_x=float(isotropic_scale),
                    scale_y=float(isotropic_scale),
                    shear_deg=0.0,
                    center_col=center_col,
                    center_row=center_row,
                )
                prewarped = warp_registration_image(
                    moving, base, output_shape=fixed.shape, order=1, cval=0.0
                )
                prewarped_mask = _warped_mask(moving_mask, base, fixed.shape)
                dx, dy = phase_translation(
                    fixed,
                    prewarped,
                    fixed_mask=fixed_mask,
                    moving_mask=prewarped_mask,
                    max_shift_x=max_x,
                    max_shift_y=max_y,
                )
                candidate = _translation_matrix(dx, dy) @ base
                disp_x, disp_y = _matrix_displacement_bounds_px(candidate, fixed.shape)
                if max_x is not None and disp_x > max_x + 1e-6:
                    continue
                if max_y is not None and disp_y > max_y + 1e-6:
                    continue
                mi, corr, _, _ = _evaluate_alignment(
                    fixed, fixed_mask, moving, moving_mask, candidate
                )
                candidate_score = _alignment_score(mi, corr)
                if candidate_score > best_score:
                    best_score = candidate_score
                    best_matrix = candidate
                    if transform_type == "similarity":
                        best_params = np.array(
                            [dx, dy, float(angle), float(isotropic_scale)],
                            dtype=np.float64,
                        )
                    else:
                        best_params = np.array(
                            [
                                dx,
                                dy,
                                float(angle),
                                float(isotropic_scale),
                                float(isotropic_scale),
                                0.0,
                            ],
                            dtype=np.float64,
                        )

        t_refine = max(0.25, float(refinement_translation_px))
        bx0 = -np.inf if max_x is None else -max_x
        bx1 = np.inf if max_x is None else max_x
        by0 = -np.inf if max_y is None else -max_y
        by1 = np.inf if max_y is None else max_y

        if transform_type == "similarity":
            bounds = [
                (max(bx0, best_params[0] - t_refine), min(bx1, best_params[0] + t_refine)),
                (max(by0, best_params[1] - t_refine), min(by1, best_params[1] + t_refine)),
                (-abs(float(max_rotation_deg)), abs(float(max_rotation_deg))),
                (1.0 - scale_delta, 1.0 + scale_delta),
            ]

            def build_matrix(params: np.ndarray) -> np.ndarray:
                return _affine_matrix(
                    dx=float(params[0]),
                    dy=float(params[1]),
                    rotation_deg=float(params[2]),
                    scale_x=float(params[3]),
                    scale_y=float(params[3]),
                    shear_deg=0.0,
                    center_col=center_col,
                    center_row=center_row,
                )
        else:
            bounds = [
                (max(bx0, best_params[0] - t_refine), min(bx1, best_params[0] + t_refine)),
                (max(by0, best_params[1] - t_refine), min(by1, best_params[1] + t_refine)),
                (-abs(float(max_rotation_deg)), abs(float(max_rotation_deg))),
                (1.0 - scale_delta, 1.0 + scale_delta),
                (1.0 - scale_delta, 1.0 + scale_delta),
                (-abs(float(max_shear_deg)), abs(float(max_shear_deg))),
            ]

            def build_matrix(params: np.ndarray) -> np.ndarray:
                return _affine_matrix(
                    dx=float(params[0]),
                    dy=float(params[1]),
                    rotation_deg=float(params[2]),
                    scale_x=float(params[3]),
                    scale_y=float(params[4]),
                    shear_deg=float(params[5]),
                    center_col=center_col,
                    center_row=center_row,
                )

        def objective(params: np.ndarray) -> float:
            matrix = build_matrix(params)
            disp_x, disp_y = _matrix_displacement_bounds_px(matrix, fixed.shape)
            if max_x is not None and disp_x > max_x:
                return 1e6 + 1e3 * (disp_x - max_x)
            if max_y is not None and disp_y > max_y:
                return 1e6 + 1e3 * (disp_y - max_y)
            mi, corr, _, _ = _evaluate_alignment(
                fixed, fixed_mask, moving, moving_mask, matrix
            )
            value = _alignment_score(mi, corr)
            return 1e6 if not np.isfinite(value) else -value

        result = minimize(
            objective,
            best_params,
            method="Powell",
            bounds=bounds,
            options={"maxiter": int(optimizer_maxiter), "xtol": 1e-3, "ftol": 1e-5},
        )
        candidate = build_matrix(result.x)
        disp_x, disp_y = _matrix_displacement_bounds_px(candidate, fixed.shape)
        within = (max_x is None or disp_x <= max_x + 1e-6) and (
            max_y is None or disp_y <= max_y + 1e-6
        )
        mi_candidate, corr_candidate, _, _ = _evaluate_alignment(
            fixed, fixed_mask, moving, moving_mask, candidate
        )
        if within and _alignment_score(mi_candidate, corr_candidate) >= best_score - 1e-8:
            chosen = candidate
            converged = bool(result.success)
            message = str(result.message)
        else:
            chosen = best_matrix
            converged = True
            message = (
                f"{transform_type} optimizer result rejected; "
                "retained constrained coarse solution"
            )

    mi_after, corr_after, aligned, valid_after = _evaluate_alignment(
        fixed, fixed_mask, moving, moving_mask, chosen
    )
    score_after = _alignment_score(mi_after, corr_after)
    improvement = float(score_after - score_before)
    disp_x, disp_y = _matrix_displacement_bounds_px(chosen, fixed.shape)
    at_boundary = (
        (max_x is not None and max_x > 0 and disp_x >= 0.98 * max_x)
        or (max_y is not None and max_y > 0 and disp_y >= 0.98 * max_y)
    )
    local = _local_translation_consensus(
        fixed,
        fixed_mask,
        aligned,
        valid_after,
        grid_size=int(local_grid_size),
        max_residual_px=float(local_max_residual_px),
        residual_tolerance_px=float(local_residual_tolerance_px),
        dispersion_tolerance_px=float(local_dispersion_tolerance_px),
        min_windows=int(local_min_windows),
    )
    residual_dx, residual_dy = phase_translation(
        fixed,
        aligned,
        fixed_mask=fixed_mask,
        moving_mask=valid_after,
        max_shift_x=float(local_max_residual_px),
        max_shift_y=float(local_max_residual_px),
    )

    accepted = True
    reason = "registration correction accepted"
    if improvement < float(min_score_improvement):
        accepted = False
        reason = (
            f"score improvement {improvement:.6g} is below minimum "
            f"{float(min_score_improvement):.6g}"
        )
    elif at_boundary:
        accepted = False
        reason = "estimated correction reached the geometric-accuracy search boundary"
    elif require_local_consensus and not bool(local.get("accepted", False)):
        accepted = False
        reason = str(local.get("reason") or "local residual consensus failed")

    if not accepted:
        chosen = identity
        mi_after, corr_after, aligned, valid_after = _evaluate_alignment(
            fixed, fixed_mask, moving, moving_mask, identity
        )
        score_after = _alignment_score(mi_after, corr_after)
        improvement = float(score_after - score_before)
        residual_dx, residual_dy = phase_translation(
            fixed,
            aligned,
            fixed_mask=fixed_mask,
            moving_mask=valid_after,
            max_shift_x=float(local_max_residual_px),
            max_shift_y=float(local_max_residual_px),
        )
        converged = False
        message = f"identity retained: {reason}"

    return chosen, {
        "phase_translation_preview_px": (float(phase_dx), float(phase_dy)),
        "mutual_information_before": float(mi_before),
        "mutual_information_after": float(mi_after),
        "gradient_correlation_before": float(corr_before),
        "gradient_correlation_after": float(corr_after),
        "score_before": float(score_before),
        "score_after": float(score_after),
        "score_improvement": float(improvement),
        "residual_shift_preview_px": (float(residual_dx), float(residual_dy)),
        "accepted": bool(accepted),
        "acceptance_reason": str(reason),
        "local_validation": local,
        "converged": bool(converged),
        "optimizer_message": str(message),
    }


def _identity_registration_qa(
    fixed: np.ndarray,
    fixed_mask: np.ndarray,
    moving: np.ndarray,
    moving_mask: np.ndarray,
    *,
    local_grid_size: int,
    local_max_residual_px: float,
    local_residual_tolerance_px: float,
    local_dispersion_tolerance_px: float,
    local_min_windows: int,
) -> dict[str, Any]:
    """Evaluate whether existing georeferencing is already sufficient."""
    identity = np.eye(3, dtype=np.float64)
    mi, corr, aligned, valid = _evaluate_alignment(
        fixed, fixed_mask, moving, moving_mask, identity
    )
    score = _alignment_score(mi, corr)
    local = _local_translation_consensus(
        fixed,
        fixed_mask,
        aligned,
        valid,
        grid_size=int(local_grid_size),
        max_residual_px=float(local_max_residual_px),
        residual_tolerance_px=float(local_residual_tolerance_px),
        dispersion_tolerance_px=float(local_dispersion_tolerance_px),
        min_windows=int(local_min_windows),
    )
    residual_dx, residual_dy = phase_translation(
        fixed,
        aligned,
        fixed_mask=fixed_mask,
        moving_mask=valid,
        max_shift_x=float(local_max_residual_px),
        max_shift_y=float(local_max_residual_px),
    )
    accepted = bool(local.get("accepted", False))
    return {
        "phase_translation_preview_px": (float(residual_dx), float(residual_dy)),
        "mutual_information_before": float(mi),
        "mutual_information_after": float(mi),
        "gradient_correlation_before": float(corr),
        "gradient_correlation_after": float(corr),
        "score_before": float(score),
        "score_after": float(score),
        "score_improvement": 0.0,
        "residual_shift_preview_px": (float(residual_dx), float(residual_dy)),
        "accepted": accepted,
        "acceptance_reason": (
            "existing georeferencing passes local residual-consensus QA"
            if accepted
            else str(local.get("reason") or "existing georeferencing requires evaluation")
        ),
        "local_validation": local,
        "converged": True,
        "optimizer_message": "identity model; no fine transform applied",
    }


def _multiscale_translation_candidate(
    *,
    specs: Mapping[str, tuple[Path, int]],
    profile: Mapping[str, Any],
    moving_keys: Sequence[str],
    fixed_cache: Mapping[int, tuple[np.ndarray, np.ndarray]],
    level_dimensions: Sequence[int],
    resampling: Mapping[str, str] | str,
    nodata: float,
    full_h: int,
    full_w: int,
    max_full_x: float,
    max_full_y: float,
    group: Mapping[str, Any],
    min_score_improvement: float,
    require_local_consensus: bool,
    local_grid_size: int,
    local_max_residual_px: float,
    local_residual_tolerance_px: float,
    local_dispersion_tolerance_px: float,
    local_min_windows: int,
    aoi_geojson: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, dict[str, Any], list[tuple[int, int]], tuple[float, float], tuple[float, float]]:
    """Conservative coarse-to-fine residual translation on the full grid."""
    current_dx_full = 0.0
    current_dy_full = 0.0
    first_phase = (0.0, 0.0)
    preview_shapes: list[tuple[int, int]] = []
    finest_payload: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None

    for level_index, dim in enumerate(level_dimensions):
        fixed, fixed_mask = fixed_cache[int(dim)]
        moving, moving_mask = build_registration_preview(
            specs,
            profile,
            moving_keys,
            resampling=resampling,
            nodata=nodata,
            max_dimension=int(dim),
            aoi_geojson=aoi_geojson,
        )
        ph, pw = fixed.shape
        preview_shapes.append((ph, pw))
        sx = pw / float(full_w)
        sy = ph / float(full_h)
        current_level = _translation_matrix(current_dx_full * sx, current_dy_full * sy)
        prealigned = warp_registration_image(
            moving, current_level, output_shape=fixed.shape, order=1, cval=0.0
        )
        prealigned_mask = _warped_mask(moving_mask, current_level, fixed.shape)
        remaining_x_level = max(0.0, max_full_x - abs(current_dx_full)) * sx
        remaining_y_level = max(0.0, max_full_y - abs(current_dy_full)) * sy
        residual_dx, residual_dy = phase_translation(
            fixed,
            prealigned,
            fixed_mask=fixed_mask,
            moving_mask=prealigned_mask,
            max_shift_x=remaining_x_level,
            max_shift_y=remaining_y_level,
        )
        if level_index == 0:
            first_phase = (float(residual_dx), float(residual_dy))
        candidate_dx_full = float(
            np.clip(
                current_dx_full + residual_dx / max(sx, 1e-15),
                -max_full_x,
                max_full_x,
            )
        )
        candidate_dy_full = float(
            np.clip(
                current_dy_full + residual_dy / max(sy, 1e-15),
                -max_full_y,
                max_full_y,
            )
        )
        candidate_level = _translation_matrix(candidate_dx_full * sx, candidate_dy_full * sy)
        mi_current, corr_current, _, _ = _evaluate_alignment(
            fixed, fixed_mask, moving, moving_mask, current_level
        )
        mi_candidate, corr_candidate, _, _ = _evaluate_alignment(
            fixed, fixed_mask, moving, moving_mask, candidate_level
        )
        if _alignment_score(mi_candidate, corr_candidate) >= _alignment_score(
            mi_current, corr_current
        ):
            current_dx_full = candidate_dx_full
            current_dy_full = candidate_dy_full
        finest_payload = (fixed, fixed_mask, moving, moving_mask)

    assert finest_payload is not None
    fixed, fixed_mask, moving, moving_mask = finest_payload
    ph, pw = fixed.shape
    sx = pw / float(full_w)
    sy = ph / float(full_h)
    matrix_preview = _translation_matrix(current_dx_full * sx, current_dy_full * sy)
    identity_preview = np.eye(3, dtype=np.float64)
    mi_before, corr_before, _, _ = _evaluate_alignment(
        fixed, fixed_mask, moving, moving_mask, identity_preview
    )
    mi_after_candidate, corr_after_candidate, aligned_candidate, valid_candidate = _evaluate_alignment(
        fixed, fixed_mask, moving, moving_mask, matrix_preview
    )
    score_before = _alignment_score(mi_before, corr_before)
    candidate_score = _alignment_score(mi_after_candidate, corr_after_candidate)
    improvement = float(candidate_score - score_before)
    at_boundary = (
        (max_full_x > 0 and abs(current_dx_full) >= 0.98 * max_full_x)
        or (max_full_y > 0 and abs(current_dy_full) >= 0.98 * max_full_y)
    )
    local = _local_translation_consensus(
        fixed,
        fixed_mask,
        aligned_candidate,
        valid_candidate,
        grid_size=int(group.get("local_grid_size", local_grid_size)),
        max_residual_px=float(group.get("local_max_residual_px", local_max_residual_px)),
        residual_tolerance_px=float(
            group.get("local_residual_tolerance_px", local_residual_tolerance_px)
        ),
        dispersion_tolerance_px=float(
            group.get("local_dispersion_tolerance_px", local_dispersion_tolerance_px)
        ),
        min_windows=int(group.get("local_min_windows", local_min_windows)),
    )
    accepted = True
    reason = "registration correction accepted"
    min_improvement = float(group.get("min_score_improvement", min_score_improvement))
    if improvement < min_improvement:
        accepted = False
        reason = f"score improvement {improvement:.6g} is below minimum {min_improvement:.6g}"
    elif at_boundary:
        accepted = False
        reason = "estimated correction reached the geometric-accuracy search boundary"
    elif bool(group.get("require_local_consensus", require_local_consensus)) and not bool(
        local.get("accepted", False)
    ):
        accepted = False
        reason = str(local.get("reason") or "local residual consensus failed")

    if accepted:
        matrix_full = _translation_matrix(current_dx_full, current_dy_full)
        mi_after = mi_after_candidate
        corr_after = corr_after_candidate
        aligned_final = aligned_candidate
        valid_final = valid_candidate
        score_after = candidate_score
        converged = True
        optimizer_message = "conservative coarse-to-fine residual translation accepted"
    else:
        matrix_full = np.eye(3, dtype=np.float64)
        mi_after = mi_before
        corr_after = corr_before
        aligned_final = moving
        valid_final = fixed_mask & moving_mask
        score_after = score_before
        improvement = 0.0
        converged = False
        optimizer_message = f"identity retained: {reason}"

    residual_dx_preview, residual_dy_preview = phase_translation(
        fixed,
        aligned_final,
        fixed_mask=fixed_mask,
        moving_mask=valid_final,
        max_shift_x=float(group.get("local_max_residual_px", local_max_residual_px)),
        max_shift_y=float(group.get("local_max_residual_px", local_max_residual_px)),
    )
    residual_full = (
        float(residual_dx_preview / max(sx, 1e-15)),
        float(residual_dy_preview / max(sy, 1e-15)),
    )
    qa = {
        "mutual_information_before": mi_before,
        "mutual_information_after": mi_after,
        "gradient_correlation_before": corr_before,
        "gradient_correlation_after": corr_after,
        "score_before": score_before,
        "score_after": score_after,
        "score_improvement": improvement,
        "accepted": accepted,
        "acceptance_reason": reason,
        "local_validation": local,
        "converged": converged,
        "optimizer_message": optimizer_message,
        "phase_translation_preview_px": first_phase,
        "residual_shift_preview_px": (float(residual_dx_preview), float(residual_dy_preview)),
    }
    return matrix_full, qa, preview_shapes, residual_full, first_phase


def _candidate_summary(model: str, qa: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "model": str(model),
        "accepted": bool(qa.get("accepted", False)),
        "score_before": float(qa.get("score_before", float("nan"))),
        "score_after": float(qa.get("score_after", float("nan"))),
        "score_improvement": float(qa.get("score_improvement", 0.0)),
        "residual_shift_preview_px": [
            float(x) for x in qa.get("residual_shift_preview_px", (0.0, 0.0))
        ],
        "local_validation": dict(qa.get("local_validation", {})),
        "reason": str(qa.get("acceptance_reason", "")),
    }


def estimate_multimodal_registrations(
    specs: Mapping[str, tuple[Path, int]],
    profile: Mapping[str, Any],
    source_groups: Mapping[str, Mapping[str, Any]],
    *,
    reference_group: str,
    resampling: Mapping[str, str] | str = "bilinear",
    nodata: float = -9999.0,
    transform_type: str = "auto",
    preview_max_dimension: int = 2048,
    coarse_preview_dimension: int = 512,
    rms_multiplier: float = 3.0,
    fallback_max_native_pixels: float = 3.0,
    min_score_improvement: float = 0.003,
    require_local_consensus: bool = True,
    local_grid_size: int = 3,
    local_max_residual_px: float = 2.0,
    local_residual_tolerance_px: float = 0.75,
    local_dispersion_tolerance_px: float = 1.0,
    local_min_windows: int = 4,
    max_rotation_deg: float = 0.25,
    max_scale_delta: float = 0.005,
    max_shear_deg: float = 0.10,
    refinement_translation_px: float = 2.0,
    optimizer_maxiter: int = 40,
    aoi_geojson: Mapping[str, Any] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, RasterRegistrationResult], dict[str, str]]:
    """Estimate conservative fine co-registration transforms per source group.

    ``auto`` is the recommended v0.7 policy. It uses a fail-safe hierarchy:
    existing georeferencing (identity) -> translation -> similarity -> affine.
    A more complex model is attempted only when the simpler model does not
    pass the residual-consensus QA. All corrections remain constrained by the
    RMS/native-resolution displacement envelope. If no model is defensible,
    the existing georeferencing is retained unchanged.
    """
    requested_global = str(transform_type).lower().strip()
    if requested_global not in {"auto", "translation", "similarity", "affine"}:
        raise ValueError(
            "transform_type must be 'auto', 'translation', 'similarity' or 'affine'"
        )
    if reference_group not in source_groups:
        raise KeyError(f"reference_group {reference_group!r} is not configured")

    key_to_group: dict[str, str] = {}
    for group_name, group in source_groups.items():
        keys = tuple(str(k) for k in group.get("keys", ()))
        if not keys:
            raise ValueError(f"Source group {group_name!r} has no 'keys'")
        for key in keys:
            if key not in specs:
                raise KeyError(f"Raster key {key!r} in group {group_name!r} is not a source")
            if key in key_to_group:
                raise ValueError(f"Raster key {key!r} belongs to more than one source group")
            key_to_group[key] = str(group_name)
    missing = [key for key in specs if key not in key_to_group]
    if missing:
        raise ValueError(f"Every raster source must belong to a source group; unassigned: {missing}")

    ref_group_cfg = source_groups[reference_group]
    ref_keys = tuple(
        str(k) for k in ref_group_cfg.get("registration_keys") or ref_group_cfg.get("keys", ())
    )
    full_h = int(profile["height"])
    full_w = int(profile["width"])
    transform = profile["transform"]
    pixel_size_x = float((float(transform.a) ** 2 + float(transform.d) ** 2) ** 0.5)
    pixel_size_y = float((float(transform.b) ** 2 + float(transform.e) ** 2) ** 0.5)
    if pixel_size_x <= 0 or pixel_size_y <= 0:
        raise ValueError("Analysis-grid pixel size must be positive")

    max_dim = max(64, int(preview_max_dimension))
    coarse_dim = max(64, min(int(coarse_preview_dimension), max_dim))
    level_dimensions = sorted(set([coarse_dim, max_dim]))
    fixed_cache: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    for dim in level_dimensions:
        fixed_cache[dim] = build_registration_preview(
            specs,
            profile,
            ref_keys,
            resampling=resampling,
            nodata=nodata,
            max_dimension=dim,
            aoi_geojson=aoi_geojson,
        )

    target_crs = profile.get("crs")
    ref_native = _source_native_resolution(specs, ref_keys, target_crs)
    key_transforms = {
        str(key): np.eye(3, dtype=np.float64) for key in ref_group_cfg.get("keys", ())
    }
    results: dict[str, RasterRegistrationResult] = {}

    for group_name, group in source_groups.items():
        if group_name == reference_group:
            continue
        moving_keys = tuple(
            str(k) for k in group.get("registration_keys") or group.get("keys", ())
        )
        moving_native = _source_native_resolution(specs, moving_keys, target_crs)
        limits_map, relative_rms, constraint_source = _registration_shift_limits(
            reference_group_cfg=ref_group_cfg,
            moving_group_cfg=group,
            reference_native_resolution=ref_native,
            moving_native_resolution=moving_native,
            target_pixel_size=(pixel_size_x, pixel_size_y),
            rms_multiplier=float(group.get("rms_multiplier", rms_multiplier)),
            fallback_max_native_pixels=float(
                group.get("fallback_max_native_pixels", fallback_max_native_pixels)
            ),
        )
        max_full_x = float(limits_map[0] / pixel_size_x)
        max_full_y = float(limits_map[1] / pixel_size_y)
        requested_model = str(group.get("transform_type", requested_global)).lower().strip()
        if requested_model not in {"auto", "translation", "similarity", "affine"}:
            raise ValueError(
                "group transform_type must be 'auto', 'translation', 'similarity' or 'affine'"
            )

        preview_shapes: list[tuple[int, int]] = []
        tested: list[dict[str, Any]] = []
        selected_model = "identity"
        selection_reason = "existing georeferencing retained"
        matrix_full = np.eye(3, dtype=np.float64)
        residual_full = (0.0, 0.0)
        phase_translation_preview = (0.0, 0.0)

        # The finest preview is used for model QA and for similarity/affine.
        fine_dim = int(level_dimensions[-1])
        fixed_fine, fixed_mask_fine = fixed_cache[fine_dim]
        moving_fine, moving_mask_fine = build_registration_preview(
            specs,
            profile,
            moving_keys,
            resampling=resampling,
            nodata=nodata,
            max_dimension=fine_dim,
            aoi_geojson=aoi_geojson,
        )
        ph, pw = fixed_fine.shape
        fine_shape = (ph, pw)
        sx = pw / float(full_w)
        sy = ph / float(full_h)

        identity_qa = _identity_registration_qa(
            fixed_fine,
            fixed_mask_fine,
            moving_fine,
            moving_mask_fine,
            local_grid_size=int(group.get("local_grid_size", local_grid_size)),
            local_max_residual_px=float(group.get("local_max_residual_px", local_max_residual_px)),
            local_residual_tolerance_px=float(
                group.get("local_residual_tolerance_px", local_residual_tolerance_px)
            ),
            local_dispersion_tolerance_px=float(
                group.get("local_dispersion_tolerance_px", local_dispersion_tolerance_px)
            ),
            local_min_windows=int(group.get("local_min_windows", local_min_windows)),
        )

        if requested_model == "auto":
            tested.append(_candidate_summary("identity", identity_qa))
            if bool(identity_qa["accepted"]):
                qa = identity_qa
                selected_model = "identity"
                selection_reason = (
                    "existing georeferencing passes local residual-consensus QA; "
                    "no fine transform is justified"
                )
                preview_shapes.append(fine_shape)
                residual_full = (
                    float(identity_qa["residual_shift_preview_px"][0] / max(sx, 1e-15)),
                    float(identity_qa["residual_shift_preview_px"][1] / max(sy, 1e-15)),
                )
                phase_translation_preview = identity_qa["phase_translation_preview_px"]
            else:
                # 1) Translation: cheapest correction and default practical model.
                t_matrix, t_qa, t_shapes, t_residual, t_phase = _multiscale_translation_candidate(
                    specs=specs,
                    profile=profile,
                    moving_keys=moving_keys,
                    fixed_cache=fixed_cache,
                    level_dimensions=level_dimensions,
                    resampling=resampling,
                    nodata=nodata,
                    full_h=full_h,
                    full_w=full_w,
                    max_full_x=max_full_x,
                    max_full_y=max_full_y,
                    group=group,
                    min_score_improvement=min_score_improvement,
                    require_local_consensus=require_local_consensus,
                    local_grid_size=local_grid_size,
                    local_max_residual_px=local_max_residual_px,
                    local_residual_tolerance_px=local_residual_tolerance_px,
                    local_dispersion_tolerance_px=local_dispersion_tolerance_px,
                    local_min_windows=local_min_windows,
                    aoi_geojson=aoi_geojson,
                )
                tested.append(_candidate_summary("translation", t_qa))
                if bool(t_qa["accepted"]):
                    matrix_full, qa = t_matrix, t_qa
                    preview_shapes = list(t_shapes)
                    residual_full = t_residual
                    phase_translation_preview = t_phase
                    selected_model = "translation"
                    selection_reason = (
                        "translation is the simplest model that passes all registration QA"
                    )
                else:
                    # 2) Similarity: only if translation is insufficient.
                    s_preview, s_qa = estimate_group_registration(
                        fixed_fine,
                        fixed_mask_fine,
                        moving_fine,
                        moving_mask_fine,
                        transform_type="similarity",
                        max_translation_x_px=max_full_x * sx,
                        max_translation_y_px=max_full_y * sy,
                        max_rotation_deg=float(group.get("max_rotation_deg", max_rotation_deg)),
                        max_scale_delta=float(group.get("max_scale_delta", max_scale_delta)),
                        max_shear_deg=0.0,
                        refinement_translation_px=float(
                            group.get("refinement_translation_px", refinement_translation_px)
                        ),
                        optimizer_maxiter=int(group.get("optimizer_maxiter", optimizer_maxiter)),
                        min_score_improvement=float(
                            group.get("min_score_improvement", min_score_improvement)
                        ),
                        require_local_consensus=bool(
                            group.get("require_local_consensus", require_local_consensus)
                        ),
                        local_grid_size=int(group.get("local_grid_size", local_grid_size)),
                        local_max_residual_px=float(
                            group.get("local_max_residual_px", local_max_residual_px)
                        ),
                        local_residual_tolerance_px=float(
                            group.get("local_residual_tolerance_px", local_residual_tolerance_px)
                        ),
                        local_dispersion_tolerance_px=float(
                            group.get("local_dispersion_tolerance_px", local_dispersion_tolerance_px)
                        ),
                        local_min_windows=int(group.get("local_min_windows", local_min_windows)),
                    )
                    tested.append(_candidate_summary("similarity", s_qa))
                    if bool(s_qa["accepted"]):
                        scale = np.diag([full_w / float(pw), full_h / float(ph), 1.0])
                        matrix_full = scale @ s_preview @ np.linalg.inv(scale)
                        qa = s_qa
                        preview_shapes.append(fine_shape)
                        residual_full = (
                            float(s_qa["residual_shift_preview_px"][0] / max(sx, 1e-15)),
                            float(s_qa["residual_shift_preview_px"][1] / max(sy, 1e-15)),
                        )
                        phase_translation_preview = s_qa["phase_translation_preview_px"]
                        selected_model = "similarity"
                        selection_reason = (
                            "translation did not pass QA; similarity is the simplest model "
                            "that passes all registration QA"
                        )
                    else:
                        # 3) Affine: last resort; only after similarity fails.
                        a_preview, a_qa = estimate_group_registration(
                            fixed_fine,
                            fixed_mask_fine,
                            moving_fine,
                            moving_mask_fine,
                            transform_type="affine",
                            max_translation_x_px=max_full_x * sx,
                            max_translation_y_px=max_full_y * sy,
                            max_rotation_deg=float(group.get("max_rotation_deg", max_rotation_deg)),
                            max_scale_delta=float(group.get("max_scale_delta", max_scale_delta)),
                            max_shear_deg=float(group.get("max_shear_deg", max_shear_deg)),
                            refinement_translation_px=float(
                                group.get("refinement_translation_px", refinement_translation_px)
                            ),
                            optimizer_maxiter=int(group.get("optimizer_maxiter", optimizer_maxiter)),
                            min_score_improvement=float(
                                group.get("min_score_improvement", min_score_improvement)
                            ),
                            require_local_consensus=bool(
                                group.get("require_local_consensus", require_local_consensus)
                            ),
                            local_grid_size=int(group.get("local_grid_size", local_grid_size)),
                            local_max_residual_px=float(
                                group.get("local_max_residual_px", local_max_residual_px)
                            ),
                            local_residual_tolerance_px=float(
                                group.get("local_residual_tolerance_px", local_residual_tolerance_px)
                            ),
                            local_dispersion_tolerance_px=float(
                                group.get("local_dispersion_tolerance_px", local_dispersion_tolerance_px)
                            ),
                            local_min_windows=int(group.get("local_min_windows", local_min_windows)),
                        )
                        tested.append(_candidate_summary("affine", a_qa))
                        if bool(a_qa["accepted"]):
                            scale = np.diag([full_w / float(pw), full_h / float(ph), 1.0])
                            matrix_full = scale @ a_preview @ np.linalg.inv(scale)
                            qa = a_qa
                            preview_shapes.append(fine_shape)
                            residual_full = (
                                float(a_qa["residual_shift_preview_px"][0] / max(sx, 1e-15)),
                                float(a_qa["residual_shift_preview_px"][1] / max(sy, 1e-15)),
                            )
                            phase_translation_preview = a_qa["phase_translation_preview_px"]
                            selected_model = "affine"
                            selection_reason = (
                                "identity, translation and similarity did not pass QA; "
                                "affine passes all constraints and residual-consensus QA"
                            )
                        else:
                            qa = identity_qa
                            preview_shapes.append(fine_shape)
                            selected_model = "identity"
                            selection_reason = (
                                "no candidate correction passed all geometric constraints and QA; "
                                "existing georeferencing retained"
                            )
                            residual_full = (
                                float(identity_qa["residual_shift_preview_px"][0] / max(sx, 1e-15)),
                                float(identity_qa["residual_shift_preview_px"][1] / max(sy, 1e-15)),
                            )
                            phase_translation_preview = identity_qa["phase_translation_preview_px"]
        elif requested_model == "translation":
            matrix_full, qa, preview_shapes, residual_full, phase_translation_preview = (
                _multiscale_translation_candidate(
                    specs=specs,
                    profile=profile,
                    moving_keys=moving_keys,
                    fixed_cache=fixed_cache,
                    level_dimensions=level_dimensions,
                    resampling=resampling,
                    nodata=nodata,
                    full_h=full_h,
                    full_w=full_w,
                    max_full_x=max_full_x,
                    max_full_y=max_full_y,
                    group=group,
                    min_score_improvement=min_score_improvement,
                    require_local_consensus=require_local_consensus,
                    local_grid_size=local_grid_size,
                    local_max_residual_px=local_max_residual_px,
                    local_residual_tolerance_px=local_residual_tolerance_px,
                    local_dispersion_tolerance_px=local_dispersion_tolerance_px,
                    local_min_windows=local_min_windows,
                )
            )
            tested.append(_candidate_summary("translation", qa))
            selected_model = "translation" if bool(qa["accepted"]) else "identity"
            selection_reason = (
                "explicit translation accepted"
                if bool(qa["accepted"])
                else "explicit translation rejected; existing georeferencing retained"
            )
        else:
            # Explicit similarity or affine remains available for expert use.
            matrix_preview, qa = estimate_group_registration(
                fixed_fine,
                fixed_mask_fine,
                moving_fine,
                moving_mask_fine,
                transform_type=requested_model,
                max_translation_x_px=max_full_x * sx,
                max_translation_y_px=max_full_y * sy,
                max_rotation_deg=float(group.get("max_rotation_deg", max_rotation_deg)),
                max_scale_delta=float(group.get("max_scale_delta", max_scale_delta)),
                max_shear_deg=(
                    0.0
                    if requested_model == "similarity"
                    else float(group.get("max_shear_deg", max_shear_deg))
                ),
                refinement_translation_px=float(
                    group.get("refinement_translation_px", refinement_translation_px)
                ),
                optimizer_maxiter=int(group.get("optimizer_maxiter", optimizer_maxiter)),
                min_score_improvement=float(
                    group.get("min_score_improvement", min_score_improvement)
                ),
                require_local_consensus=bool(
                    group.get("require_local_consensus", require_local_consensus)
                ),
                local_grid_size=int(group.get("local_grid_size", local_grid_size)),
                local_max_residual_px=float(
                    group.get("local_max_residual_px", local_max_residual_px)
                ),
                local_residual_tolerance_px=float(
                    group.get("local_residual_tolerance_px", local_residual_tolerance_px)
                ),
                local_dispersion_tolerance_px=float(
                    group.get("local_dispersion_tolerance_px", local_dispersion_tolerance_px)
                ),
                local_min_windows=int(group.get("local_min_windows", local_min_windows)),
            )
            scale = np.diag([full_w / float(pw), full_h / float(ph), 1.0])
            matrix_full = scale @ matrix_preview @ np.linalg.inv(scale)
            residual_full = (
                float(qa["residual_shift_preview_px"][0] / max(sx, 1e-15)),
                float(qa["residual_shift_preview_px"][1] / max(sy, 1e-15)),
            )
            phase_translation_preview = qa["phase_translation_preview_px"]
            preview_shapes.append(fine_shape)
            tested.append(_candidate_summary(requested_model, qa))
            selected_model = requested_model if bool(qa["accepted"]) else "identity"
            selection_reason = (
                f"explicit {requested_model} accepted"
                if bool(qa["accepted"])
                else f"explicit {requested_model} rejected; existing georeferencing retained"
            )

        center = np.array([[(full_w - 1) / 2.0], [(full_h - 1) / 2.0], [1.0]])
        moved_center = matrix_full @ center
        center_disp = (
            float(moved_center[0, 0] - center[0, 0]),
            float(moved_center[1, 0] - center[1, 0]),
        )
        estimated_map = (
            float(center_disp[0] * pixel_size_x),
            float(center_disp[1] * pixel_size_y),
        )
        max_disp_px = _matrix_displacement_bounds_px(matrix_full, (full_h, full_w))
        max_disp_map = (
            float(max_disp_px[0] * pixel_size_x),
            float(max_disp_px[1] * pixel_size_y),
        )
        model_selection = {
            "requested": requested_model,
            "policy": "simplest_adequate_model",
            "hierarchy": ["identity", "translation", "similarity", "affine"],
            "tested": tested,
            "selected": selected_model,
            "reason": selection_reason,
        }
        acceptance_reason = (
            selection_reason if requested_model == "auto" else str(qa["acceptance_reason"])
        )
        accepted = bool(selected_model != "identity") if requested_model == "auto" else bool(qa["accepted"])
        if requested_model == "auto" and selected_model == "identity":
            # Identity is a valid automatic outcome, but no correction was applied.
            accepted = bool(identity_qa.get("accepted", False))

        result = RasterRegistrationResult(
            group=str(group_name),
            reference_group=str(reference_group),
            method="conservative_multimodal_coregistration_v070_auto",
            transform_type=requested_model,
            selected_model=selected_model,
            model_selection=model_selection,
            transformation=matrix_full,
            preview_shape=preview_shapes[-1] if preview_shapes else fine_shape,
            preview_levels=tuple(preview_shapes or [fine_shape]),
            reference_keys=ref_keys,
            moving_keys=moving_keys,
            phase_translation_preview_px=(
                float(phase_translation_preview[0]),
                float(phase_translation_preview[1]),
            ),
            mutual_information_before=float(qa["mutual_information_before"]),
            mutual_information_after=float(qa["mutual_information_after"]),
            gradient_correlation_before=float(qa["gradient_correlation_before"]),
            gradient_correlation_after=float(qa["gradient_correlation_after"]),
            score_before=float(qa["score_before"]),
            score_after=float(qa["score_after"]),
            score_improvement=float(qa["score_improvement"]),
            residual_shift_full_px=residual_full,
            center_displacement_full_px=center_disp,
            accepted=accepted,
            applied=bool(selected_model != "identity" and not np.allclose(matrix_full, np.eye(3), atol=1e-12)),
            acceptance_reason=acceptance_reason,
            constraint_source=str(constraint_source),
            relative_rms_map_units=relative_rms,
            max_allowed_shift_map_units=(float(limits_map[0]), float(limits_map[1])),
            estimated_shift_map_units=estimated_map,
            maximum_displacement_map_units=max_disp_map,
            local_validation=dict(qa["local_validation"]),
            converged=bool(qa["converged"]),
            optimizer_message=str(qa["optimizer_message"]),
        )
        results[str(group_name)] = result
        for key in group.get("keys", ()):
            key_transforms[str(key)] = matrix_full

    return key_transforms, results, key_to_group

def _sampling_order(method: str) -> int:
    value = str(method).lower()
    if value in {"nearest", "mode"}:
        return 0
    if value in {"cubic", "cubic_spline", "lanczos"}:
        return 3
    return 1


def read_registered_raster_window(
    specs: Mapping[str, tuple[Path, int]],
    profile: Mapping[str, Any],
    window: Any,
    *,
    transformations: Mapping[str, Any] | None = None,
    resampling: Mapping[str, str] | str = "bilinear",
    nodata: float = -9999.0,
) -> dict[str, np.ndarray]:
    """Read a target-grid window and apply optional fine pixel transforms.

    Source pixels are first georeferenced onto the common grid through GDAL's
    WarpedVRT. Fine multimodal transforms are then sampled in target-grid pixel
    coordinates. The operation is windowed and therefore bounded by tile size.
    """
    import rasterio
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import Window

    transforms = transformations or {}
    out: dict[str, np.ndarray] = {}
    col_off = int(window.col_off)
    row_off = int(window.row_off)
    width = int(window.width)
    height = int(window.height)

    for key, (path, band) in specs.items():
        method = _source_resampling(resampling, key)
        matrix = np.asarray(transforms.get(key, np.eye(3)), dtype=np.float64)
        identity = np.allclose(matrix, np.eye(3), atol=1e-12)
        with rasterio.open(path) as src:
            vrt_opts, alpha_band = _target_vrt_options(
                src, profile, method=method, nodata=float(nodata)
            )
            with WarpedVRT(src, **vrt_opts) as vrt:
                if identity:
                    data, _valid = _read_vrt_with_validity(
                        vrt,
                        int(band),
                        alpha_band=alpha_band,
                        nodata=float(nodata),
                        window=window,
                        out_dtype="float32",
                        boundless=False,
                    )
                    out[str(key)] = data
                    continue

                inv = np.linalg.inv(matrix)
                corners = np.array(
                    [
                        [col_off, row_off, 1.0],
                        [col_off + width - 1, row_off, 1.0],
                        [col_off, row_off + height - 1, 1.0],
                        [col_off + width - 1, row_off + height - 1, 1.0],
                    ],
                    dtype=np.float64,
                ).T
                moving_corners = inv @ corners
                margin = 4
                c0 = max(0, int(math.floor(np.min(moving_corners[0]))) - margin)
                r0 = max(0, int(math.floor(np.min(moving_corners[1]))) - margin)
                c1 = min(int(profile["width"]), int(math.ceil(np.max(moving_corners[0]))) + margin + 1)
                r1 = min(int(profile["height"]), int(math.ceil(np.max(moving_corners[1]))) + margin + 1)
                if c1 <= c0 or r1 <= r0:
                    out[str(key)] = np.full((height, width), float(nodata), dtype=np.float32)
                    continue

                src_window = Window(c0, r0, c1 - c0, r1 - r0)
                data, valid_src = _read_vrt_with_validity(
                    vrt,
                    int(band),
                    alpha_band=alpha_band,
                    nodata=float(nodata),
                    window=src_window,
                    out_dtype="float32",
                    boundless=False,
                )

                rows, cols = np.indices((height, width), dtype=np.float64)
                fixed_col = cols + col_off
                fixed_row = rows + row_off
                moving_col = inv[0, 0] * fixed_col + inv[0, 1] * fixed_row + inv[0, 2] - c0
                moving_row = inv[1, 0] * fixed_col + inv[1, 1] * fixed_row + inv[1, 2] - r0
                order = _sampling_order(method)
                sampled = ndimage.map_coordinates(
                    data,
                    [moving_row, moving_col],
                    order=order,
                    mode="constant",
                    cval=float(nodata),
                    prefilter=bool(order > 1),
                ).astype(np.float32)
                valid = ndimage.map_coordinates(
                    valid_src.astype(np.float32),
                    [moving_row, moving_col],
                    order=0,
                    mode="constant",
                    cval=0.0,
                    prefilter=False,
                ) > 0.5
                sampled[~valid] = float(nodata)
                out[str(key)] = sampled
    return out
