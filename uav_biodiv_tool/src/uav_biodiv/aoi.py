from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
import math

import numpy as np


@dataclass(frozen=True)
class AOISelection:
    """Normalized area of interest in one target CRS.

    The geometry is stored as a Shapely geometry. Public pipeline methods accept
    common user representations (bbox, vector path, GeoJSON or Shapely geometry)
    and normalize them through :func:`normalize_aoi`.
    """

    geometry: Any
    crs: str
    source: str
    input_crs: str | None = None

    @property
    def bounds(self) -> tuple[float, float, float, float]:
        return tuple(float(x) for x in self.geometry.bounds)

    @property
    def geojson(self) -> Mapping[str, Any]:
        from shapely.geometry import mapping

        return mapping(self.geometry)

    def to_metadata(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "source": self.source,
            "input_crs": self.input_crs,
            "analysis_crs": self.crs,
            "bounds": list(self.bounds),
            "geometry_type": str(self.geometry.geom_type),
        }


def _crs_string(value: Any) -> str:
    from pyproj import CRS

    return CRS.from_user_input(value).to_string()


def _transform_geometry(geometry: Any, src_crs: Any, dst_crs: Any) -> Any:
    from pyproj import CRS, Transformer
    from shapely.ops import transform

    src = CRS.from_user_input(src_crs)
    dst = CRS.from_user_input(dst_crs)
    if src == dst:
        return geometry
    transformer = Transformer.from_crs(src, dst, always_xy=True)
    return transform(transformer.transform, geometry)


def normalize_aoi(
    aoi: Any,
    *,
    target_crs: Any,
    aoi_crs: Any | None = None,
) -> AOISelection | None:
    """Normalize an AOI to ``target_crs``.

    Supported inputs
    ----------------
    - ``None``;
    - bbox ``(minx, miny, maxx, maxy)``;
    - path to a vector dataset supported by GeoPandas;
    - GeoJSON geometry/Feature/FeatureCollection mapping;
    - Shapely geometry;
    - GeoPandas GeoSeries/GeoDataFrame.

    A bbox/GeoJSON/Shapely geometry without embedded CRS is interpreted in
    ``aoi_crs`` when supplied, otherwise in ``target_crs``. A vector dataset
    with no CRS must be accompanied by ``aoi_crs``.
    """
    if aoi is None:
        return None
    if target_crs is None:
        raise ValueError("AOI processing requires a known target CRS")
    target = _crs_string(target_crs)

    if isinstance(aoi, AOISelection):
        geometry = _transform_geometry(aoi.geometry, aoi.crs, target)
        return AOISelection(
            geometry=geometry,
            crs=target,
            source=aoi.source,
            input_crs=aoi.input_crs or aoi.crs,
        )

    from shapely.geometry import box, shape
    from shapely.ops import unary_union

    geometry = None
    source = "geometry"
    source_crs = aoi_crs

    # Four-number bbox.
    if (
        isinstance(aoi, Sequence)
        and not isinstance(aoi, (str, bytes, Path, Mapping))
        and len(aoi) == 4
        and all(np.isscalar(v) for v in aoi)
    ):
        minx, miny, maxx, maxy = map(float, aoi)
        if not all(np.isfinite([minx, miny, maxx, maxy])) or minx >= maxx or miny >= maxy:
            raise ValueError("AOI bbox must be finite and ordered as (minx, miny, maxx, maxy)")
        geometry = box(minx, miny, maxx, maxy)
        source = "bbox"

    elif isinstance(aoi, (str, Path)):
        import geopandas as gpd

        path = Path(aoi)
        if not path.exists():
            raise FileNotFoundError(f"AOI vector dataset does not exist: {path}")
        gdf = gpd.read_file(path)
        if gdf.empty:
            raise ValueError(f"AOI vector dataset contains no features: {path}")
        valid = [geom for geom in gdf.geometry if geom is not None and not geom.is_empty]
        if not valid:
            raise ValueError(f"AOI vector dataset contains no valid geometries: {path}")
        geometry = unary_union(valid)
        source = str(path)
        if gdf.crs is not None:
            source_crs = gdf.crs
        elif source_crs is None:
            raise ValueError("AOI vector dataset has no CRS; supply aoi_crs explicitly")

    elif isinstance(aoi, Mapping):
        raw = dict(aoi)
        kind = str(raw.get("type", ""))
        if kind == "Feature":
            geometry = shape(raw["geometry"])
        elif kind == "FeatureCollection":
            geometries = [shape(item["geometry"]) for item in raw.get("features", ())]
            geometries = [g for g in geometries if g is not None and not g.is_empty]
            if not geometries:
                raise ValueError("AOI FeatureCollection contains no valid geometries")
            geometry = unary_union(geometries)
        else:
            geometry = shape(raw)
        source = "geojson"

    elif hasattr(aoi, "geometry") and hasattr(aoi, "crs"):
        # GeoDataFrame / GeoSeries-like object.
        geoms = list(aoi.geometry if hasattr(aoi, "geometry") else aoi)
        geoms = [g for g in geoms if g is not None and not g.is_empty]
        if not geoms:
            raise ValueError("AOI object contains no valid geometries")
        geometry = unary_union(geoms)
        source = "geopandas_object"
        if getattr(aoi, "crs", None) is not None:
            source_crs = aoi.crs

    elif hasattr(aoi, "geom_type") and hasattr(aoi, "bounds"):
        geometry = aoi
        source = "shapely_geometry"

    else:
        raise TypeError(
            "AOI must be None, a 4-value bbox, vector path, GeoJSON mapping, "
            "Shapely geometry or GeoPandas object"
        )

    if geometry is None or geometry.is_empty:
        raise ValueError("AOI geometry is empty")
    if not geometry.is_valid:
        # make_valid is preferred when available; buffer(0) is a compatible fallback.
        try:
            from shapely import make_valid

            geometry = make_valid(geometry)
        except Exception:
            geometry = geometry.buffer(0)
    if geometry.is_empty:
        raise ValueError("AOI geometry is empty after geometry validation")

    source_crs = target if source_crs is None else _crs_string(source_crs)
    geometry = _transform_geometry(geometry, source_crs, target)
    return AOISelection(
        geometry=geometry,
        crs=target,
        source=source,
        input_crs=str(source_crs),
    )


def clip_profile_to_aoi(
    profile: Mapping[str, Any],
    aoi: AOISelection | None,
) -> tuple[dict[str, Any], tuple[int, int]]:
    """Crop a raster profile to the AOI bounding box while preserving grid alignment.

    Returns ``(profile, (col_offset, row_offset))``. The offset is useful for
    converting full-grid pixel transforms to the cropped output grid.
    """
    out = dict(profile)
    if aoi is None:
        return out, (0, 0)

    import rasterio
    from rasterio.windows import Window, from_bounds, transform as window_transform

    if rasterio.crs.CRS.from_user_input(profile["crs"]) != rasterio.crs.CRS.from_user_input(aoi.crs):
        raise ValueError("AOI CRS does not match the raster analysis-grid CRS")

    raw = from_bounds(*aoi.bounds, transform=profile["transform"])
    col0 = max(0, int(math.floor(raw.col_off)))
    row0 = max(0, int(math.floor(raw.row_off)))
    col1 = min(int(profile["width"]), int(math.ceil(raw.col_off + raw.width)))
    row1 = min(int(profile["height"]), int(math.ceil(raw.row_off + raw.height)))
    if col1 <= col0 or row1 <= row0:
        raise ValueError("AOI does not overlap the raster analysis grid")

    window = Window(col0, row0, col1 - col0, row1 - row0)
    out["width"] = int(window.width)
    out["height"] = int(window.height)
    out["transform"] = window_transform(window, profile["transform"])
    return out, (col0, row0)


def aoi_mask_for_window(
    aoi_geojson: Mapping[str, Any] | None,
    profile: Mapping[str, Any],
    window: Any,
) -> np.ndarray | None:
    """Return a boolean inside-AOI mask for one raster window."""
    if aoi_geojson is None:
        return None
    from rasterio.features import geometry_mask
    from rasterio.windows import transform as window_transform

    return geometry_mask(
        [dict(aoi_geojson)],
        out_shape=(int(window.height), int(window.width)),
        transform=window_transform(window, profile["transform"]),
        invert=True,
        all_touched=False,
    )


def point_mask(xyz: np.ndarray, aoi: AOISelection | Any | None) -> np.ndarray:
    """Return mask selecting XYZ points inside an AOI geometry.

    ``aoi`` may be an :class:`AOISelection` or a Shapely geometry already in
    the point-cloud CRS. A fast bbox prefilter is always applied first.
    """
    pts = np.asarray(xyz, dtype=np.float64)
    if aoi is None:
        return np.ones(len(pts), dtype=bool)
    geometry = aoi.geometry if isinstance(aoi, AOISelection) else aoi
    minx, miny, maxx, maxy = map(float, geometry.bounds)
    bbox = (
        np.isfinite(pts[:, 0])
        & np.isfinite(pts[:, 1])
        & (pts[:, 0] >= minx)
        & (pts[:, 0] <= maxx)
        & (pts[:, 1] >= miny)
        & (pts[:, 1] <= maxy)
    )
    idx = np.flatnonzero(bbox)
    if idx.size == 0:
        return bbox
    try:
        from shapely import intersects_xy

        inside = np.asarray(intersects_xy(geometry, pts[idx, 0], pts[idx, 1]), dtype=bool)
    except Exception:
        try:
            from shapely import contains_xy

            inside = np.asarray(contains_xy(geometry, pts[idx, 0], pts[idx, 1]), dtype=bool)
        except Exception:
            from shapely.geometry import Point
            from shapely.prepared import prep

            prepared = prep(geometry)
            inside = np.array([prepared.intersects(Point(x, y)) for x, y in pts[idx, :2]], dtype=bool)
    out = np.zeros(len(pts), dtype=bool)
    out[idx] = inside
    return out


def buffered_geometry(aoi: AOISelection | None, distance: float) -> Any | None:
    if aoi is None:
        return None
    d = max(0.0, float(distance))
    return aoi.geometry if d == 0 else aoi.geometry.buffer(d)


def transform_for_cropped_grid(
    matrix: np.ndarray,
    *,
    col_offset: int,
    row_offset: int,
) -> np.ndarray:
    """Express a full-grid pixel transform in cropped-grid pixel coordinates."""
    m = np.asarray(matrix, dtype=np.float64)
    if m.shape != (3, 3):
        raise ValueError("Raster fine-registration matrix must be 3x3")
    if col_offset == 0 and row_offset == 0:
        return m.copy()
    to_full = np.array(
        [[1.0, 0.0, float(col_offset)], [0.0, 1.0, float(row_offset)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    to_crop = np.array(
        [[1.0, 0.0, -float(col_offset)], [0.0, 1.0, -float(row_offset)], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return to_crop @ m @ to_full
