from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Iterator, Mapping
import logging
import math
import warnings

import numpy as np

from .config import ProcessingLimits

LOGGER = logging.getLogger(__name__)


@dataclass
class PointCloudData:
    """In-memory point cloud with arbitrary per-point attributes.

    Parameters
    ----------
    xyz
        ``(n, 3)`` coordinates in map units.
    attributes
        Per-point one-dimensional arrays. Every array must have length ``n``.
    crs
        CRS identifier/WKT when known.
    source
        Human-readable sensor/source identifier.
    metadata
        Arbitrary acquisition and processing metadata.
    """

    xyz: np.ndarray
    attributes: dict[str, np.ndarray] = field(default_factory=dict)
    crs: str | None = None
    source: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        xyz = np.asarray(self.xyz, dtype=np.float64)
        if xyz.ndim != 2 or xyz.shape[1] != 3:
            raise ValueError("xyz must have shape (n_points, 3)")
        if not np.all(np.isfinite(xyz)):
            raise ValueError("xyz contains non-finite coordinates")
        self.xyz = xyz
        n = xyz.shape[0]

        clean: dict[str, np.ndarray] = {}
        for name, values in self.attributes.items():
            arr = np.asarray(values)
            if arr.ndim != 1 or arr.shape[0] != n:
                raise ValueError(
                    f"Attribute {name!r} must be 1-D with {n} values; got {arr.shape}"
                )
            clean[str(name)] = arr
        self.attributes = clean

    @property
    def n_points(self) -> int:
        return int(self.xyz.shape[0])

    def copy(self, deep: bool = True) -> "PointCloudData":
        xyz = self.xyz.copy() if deep else self.xyz
        attrs = {
            k: (v.copy() if deep else v)
            for k, v in self.attributes.items()
        }
        return PointCloudData(
            xyz=xyz,
            attributes=attrs,
            crs=self.crs,
            source=self.source,
            metadata=dict(self.metadata),
        )

    def transformed(self, matrix: np.ndarray) -> "PointCloudData":
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError("Transformation matrix must have shape (4, 4)")
        xyz_h = np.c_[self.xyz, np.ones(self.n_points)]
        out = (matrix @ xyz_h.T).T[:, :3]
        result = self.copy(deep=True)
        result.xyz = out
        return result

    def subset(self, indices: np.ndarray) -> "PointCloudData":
        idx = np.asarray(indices)
        return PointCloudData(
            xyz=self.xyz[idx],
            attributes={k: v[idx] for k, v in self.attributes.items()},
            crs=self.crs,
            source=self.source,
            metadata=dict(self.metadata),
        )


@dataclass
class RasterData:
    """In-memory raster container using band-first arrays.

    The object is intentionally lightweight: rasterio remains the authority for
    CRS/transform metadata, while the package can pass aligned rasters between
    ingestion, fusion and validation without reopening files.
    """

    data: np.ndarray
    profile: dict[str, Any]
    band_names: list[str] = field(default_factory=list)
    source: str | None = None

    def __post_init__(self) -> None:
        arr = np.asarray(self.data)
        if arr.ndim == 2:
            arr = arr[None, ...]
        if arr.ndim != 3:
            raise ValueError("Raster data must be 2-D or band-first 3-D")
        self.data = arr
        if not self.band_names:
            self.band_names = [f"band_{i+1}" for i in range(arr.shape[0])]
        if len(self.band_names) != arr.shape[0]:
            raise ValueError("band_names length must equal raster band count")

    @property
    def shape(self) -> tuple[int, int]:
        return int(self.data.shape[1]), int(self.data.shape[2])

    @property
    def crs(self):
        return self.profile.get("crs")

    @property
    def transform(self):
        return self.profile.get("transform")

    @property
    def nodata(self):
        return self.profile.get("nodata")


class DataIngestionHarmonization:
    """Data ingestion and harmonisation utilities (D1.1 §8.1.3)."""

    def __init__(
        self,
        target_crs: str | None = None,
        processing_limits: ProcessingLimits | Mapping[str, Any] | None = None,
    ) -> None:
        self.target_crs = target_crs
        if isinstance(processing_limits, ProcessingLimits):
            self.processing_limits = processing_limits
        else:
            self.processing_limits = ProcessingLimits.from_mapping(processing_limits)

    # ------------------------------------------------------------------
    # Raster/vector ingestion and harmonisation (D1.1 §8.1.3)
    # ------------------------------------------------------------------
    def load_raster(self, path: str | Path) -> tuple[np.ndarray, dict[str, Any]]:
        """Load all raster bands and return ``(band_first_array, profile)``."""
        raster = self.load_raster_data(path)
        return raster.data, raster.profile

    def load_raster_data(
        self,
        path: str | Path,
        *,
        bands: Iterable[int] | None = None,
        band_names: Iterable[str] | None = None,
        masked: bool = False,
    ) -> RasterData:
        import rasterio

        path = Path(path)
        with rasterio.open(path) as src:
            indexes = list(bands) if bands is not None else list(range(1, src.count + 1))
            itemsize = max(np.dtype(src.dtypes[i - 1]).itemsize for i in indexes)
            estimated = int(src.width) * int(src.height) * len(indexes) * itemsize
            # rasterio/GDAL plus conversion/masks can require several copies.
            estimated_working = estimated * (4 if masked else 3)
            if estimated_working > self.processing_limits.memory_budget_bytes():
                raise MemoryError(
                    f"Full raster load would require approximately {estimated_working / (1024**3):.2f} GiB, "
                    f"above the current package budget of "
                    f"{self.processing_limits.memory_budget_bytes() / (1024**3):.2f} GiB. "
                    "Use the streaming/windowed pipeline instead of load_raster_data()."
                )
            data = src.read(indexes, masked=masked)
            profile = src.profile.copy()
            profile.update(count=len(indexes))
            descriptions = [src.descriptions[i - 1] for i in indexes]

        if band_names is None:
            names = [d if d else f"band_{idx}" for d, idx in zip(descriptions, indexes, strict=True)]
        else:
            names = list(band_names)
        return RasterData(data=data, profile=profile, band_names=names, source=str(path))

    def load_raster_band(
        self,
        source: str | Path | tuple[str | Path, int],
        *,
        name: str | None = None,
    ) -> RasterData:
        """Load one analysis band.

        ``source`` can be a path (band 1) or ``(path, band_index)``. This makes
        the same API work for separate single-band products and multiband stacks.
        """
        if isinstance(source, tuple):
            path, band = source
        else:
            path, band = source, 1
        return self.load_raster_data(path, bands=[int(band)], band_names=[name or Path(path).stem])

    def load_vector(self, path: str | Path):
        import geopandas as gpd
        return gpd.read_file(path)

    @staticmethod
    def _resampling(name: str):
        from rasterio.enums import Resampling
        try:
            return getattr(Resampling, str(name).lower())
        except AttributeError as exc:
            valid = [x for x in dir(Resampling) if not x.startswith("_")]
            raise ValueError(f"Unknown resampling method {name!r}; valid examples: nearest, bilinear, cubic, average") from exc

    def plan_raster_grid(
        self,
        sources: Mapping[str, str | Path | tuple[str | Path, int]],
        *,
        reference_key: str | None = None,
        target_crs: str | None = None,
        resolution: float | tuple[float, float] | None = None,
        extent: str = "reference",
        nodata: float = -9999.0,
    ) -> tuple[dict[str, tuple[Path, int]], dict[str, dict[str, Any]], dict[str, Any]]:
        """Inspect raster metadata and derive a common output grid without loading pixels."""
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.warp import transform_bounds, calculate_default_transform

        if not sources:
            raise ValueError("At least one raster source is required")
        if reference_key is None:
            reference_key = next(iter(sources))
        if reference_key not in sources:
            raise KeyError(f"reference_key {reference_key!r} is not present in sources")

        meta: dict[str, dict[str, Any]] = {}
        specs: dict[str, tuple[Path, int]] = {}
        for key, spec in sources.items():
            if isinstance(spec, tuple):
                path, band = Path(spec[0]), int(spec[1])
            else:
                path, band = Path(spec), 1
            with rasterio.open(path) as src:
                if band < 1 or band > src.count:
                    raise ValueError(f"Band {band} out of range for {path} ({src.count} bands)")
                meta[key] = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "bounds": src.bounds,
                    "dtype": src.dtypes[band - 1],
                    "nodata": src.nodata,
                    "path": str(path),
                    "band": band,
                }
            specs[key] = (path, band)

        ref = meta[reference_key]
        if ref["crs"] is None:
            raise ValueError("Reference raster has no CRS")
        dst_crs = rasterio.crs.CRS.from_user_input(target_crs or self.target_crs or ref["crs"])

        if extent == "reference" and resolution is None and dst_crs == ref["crs"]:
            dst_transform = ref["transform"]
            dst_width = int(ref["width"])
            dst_height = int(ref["height"])
        else:
            bounds_by_key: dict[str, tuple[float, float, float, float]] = {}
            for key, m in meta.items():
                if m["crs"] is None:
                    raise ValueError(f"Raster {key!r} has no CRS")
                bounds_by_key[key] = transform_bounds(
                    m["crs"], dst_crs, *m["bounds"], densify_pts=21
                )

            if extent == "reference":
                left, bottom, right, top = bounds_by_key[reference_key]
            elif extent == "intersection":
                left = max(b[0] for b in bounds_by_key.values())
                bottom = max(b[1] for b in bounds_by_key.values())
                right = min(b[2] for b in bounds_by_key.values())
                top = min(b[3] for b in bounds_by_key.values())
                if left >= right or bottom >= top:
                    raise ValueError("Raster sources do not have a common intersection")
            elif extent == "union":
                left = min(b[0] for b in bounds_by_key.values())
                bottom = min(b[1] for b in bounds_by_key.values())
                right = max(b[2] for b in bounds_by_key.values())
                top = max(b[3] for b in bounds_by_key.values())
            else:
                raise ValueError("extent must be 'reference', 'intersection' or 'union'")

            if resolution is None:
                t, _, _ = calculate_default_transform(
                    ref["crs"], dst_crs, ref["width"], ref["height"], *ref["bounds"]
                )
                xres, yres = abs(t.a), abs(t.e)
            elif np.isscalar(resolution):
                xres = yres = float(resolution)
            else:
                xres, yres = map(float, resolution)
            if xres <= 0 or yres <= 0:
                raise ValueError("resolution must be positive")
            dst_width = max(1, int(np.ceil((right - left) / xres)))
            dst_height = max(1, int(np.ceil((top - bottom) / yres)))
            dst_transform = from_origin(left, top, xres, yres)

        profile = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": "float32",
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": float(nodata),
            "compress": "deflate",
            "tiled": True,
            "BIGTIFF": "YES",
        }
        return specs, meta, profile

    @staticmethod
    def raster_windows(profile: Mapping[str, Any], tile_size: int) -> Iterator[Any]:
        """Yield bounded rasterio windows covering a profile."""
        from rasterio.windows import Window

        height = int(profile["height"])
        width = int(profile["width"])
        tile = max(32, int(tile_size))
        for row_off in range(0, height, tile):
            h = min(tile, height - row_off)
            for col_off in range(0, width, tile):
                w = min(tile, width - col_off)
                yield Window(col_off, row_off, w, h)

    def read_aligned_raster_window(
        self,
        specs: Mapping[str, tuple[Path, int]],
        profile: Mapping[str, Any],
        window: Any,
        *,
        resampling: Mapping[str, str] | str = "bilinear",
        nodata: float = -9999.0,
    ) -> dict[str, np.ndarray]:
        """Read/reproject one window from arbitrary raster sources.

        Each source is opened through ``WarpedVRT`` on the target grid. Only
        the requested window is materialised, so memory use scales with tile
        size rather than full raster dimensions.
        """
        import rasterio
        from rasterio.vrt import WarpedVRT

        out: dict[str, np.ndarray] = {}
        for key, (path, band) in specs.items():
            method = resampling.get(key, "bilinear") if isinstance(resampling, Mapping) else resampling
            with rasterio.open(path) as src:
                vrt_opts = {
                    "crs": profile["crs"],
                    "transform": profile["transform"],
                    "width": int(profile["width"]),
                    "height": int(profile["height"]),
                    "resampling": self._resampling(method),
                    "nodata": float(nodata),
                }
                with WarpedVRT(src, **vrt_opts) as vrt:
                    arr = vrt.read(
                        int(band),
                        window=window,
                        out_dtype="float32",
                        boundless=False,
                    )
            out[str(key)] = np.asarray(arr, dtype=np.float32)
        return out

    def harmonise_raster_sources(
        self,
        sources: Mapping[str, str | Path | tuple[str | Path, int]],
        *,
        reference_key: str | None = None,
        target_crs: str | None = None,
        resolution: float | tuple[float, float] | None = None,
        extent: str = "reference",
        resampling: Mapping[str, str] | str = "bilinear",
        nodata: float = -9999.0,
    ) -> tuple[dict[str, RasterData], dict[str, Any]]:
        """Reproject/resample named raster bands to one common grid.

        Parameters
        ----------
        extent
            ``reference`` uses the reference raster footprint exactly;
            ``intersection`` uses the common overlap; ``union`` uses all data.
        resampling
            One method for all sources, or a mapping keyed by source name.

        Returns
        -------
        aligned, target_profile
            Every output contains one band on exactly the same CRS, transform,
            width and height.
        """
        import rasterio
        from rasterio.transform import from_origin
        from rasterio.warp import reproject, transform_bounds, calculate_default_transform

        if not sources:
            raise ValueError("At least one raster source is required")
        if reference_key is None:
            reference_key = next(iter(sources))
        if reference_key not in sources:
            raise KeyError(f"reference_key {reference_key!r} is not present in sources")

        # Read only metadata first.
        meta: dict[str, dict[str, Any]] = {}
        specs: dict[str, tuple[Path, int]] = {}
        for key, spec in sources.items():
            if isinstance(spec, tuple):
                path, band = Path(spec[0]), int(spec[1])
            else:
                path, band = Path(spec), 1
            with rasterio.open(path) as src:
                if band < 1 or band > src.count:
                    raise ValueError(f"Band {band} out of range for {path} ({src.count} bands)")
                meta[key] = {
                    "crs": src.crs,
                    "transform": src.transform,
                    "width": src.width,
                    "height": src.height,
                    "bounds": src.bounds,
                    "dtype": src.dtypes[band - 1],
                    "nodata": src.nodata,
                }
            specs[key] = (path, band)

        ref = meta[reference_key]
        dst_crs = rasterio.crs.CRS.from_user_input(target_crs or self.target_crs or ref["crs"])
        if ref["crs"] is None:
            raise ValueError("Reference raster has no CRS")

        # Default: exact reference grid when CRS/resolution do not force a new grid.
        if extent == "reference" and resolution is None and dst_crs == ref["crs"]:
            dst_transform = ref["transform"]
            dst_width = int(ref["width"])
            dst_height = int(ref["height"])
        else:
            bounds_by_key = {}
            for key, m in meta.items():
                if m["crs"] is None:
                    raise ValueError(f"Raster {key!r} has no CRS")
                bounds_by_key[key] = transform_bounds(m["crs"], dst_crs, *m["bounds"], densify_pts=21)

            if extent == "reference":
                left, bottom, right, top = bounds_by_key[reference_key]
            elif extent == "intersection":
                left = max(b[0] for b in bounds_by_key.values())
                bottom = max(b[1] for b in bounds_by_key.values())
                right = min(b[2] for b in bounds_by_key.values())
                top = min(b[3] for b in bounds_by_key.values())
                if left >= right or bottom >= top:
                    raise ValueError("Raster sources do not have a common intersection")
            elif extent == "union":
                left = min(b[0] for b in bounds_by_key.values())
                bottom = min(b[1] for b in bounds_by_key.values())
                right = max(b[2] for b in bounds_by_key.values())
                top = max(b[3] for b in bounds_by_key.values())
            else:
                raise ValueError("extent must be 'reference', 'intersection' or 'union'")

            if resolution is None:
                # Approximate reference resolution in the target CRS.
                t, w, h = calculate_default_transform(
                    ref["crs"], dst_crs, ref["width"], ref["height"], *ref["bounds"]
                )
                xres, yres = abs(t.a), abs(t.e)
            elif np.isscalar(resolution):
                xres = yres = float(resolution)
            else:
                xres, yres = map(float, resolution)
            if xres <= 0 or yres <= 0:
                raise ValueError("resolution must be positive")

            dst_width = max(1, int(np.ceil((right - left) / xres)))
            dst_height = max(1, int(np.ceil((top - bottom) / yres)))
            dst_transform = from_origin(left, top, xres, yres)

        target_profile = {
            "driver": "GTiff",
            "height": dst_height,
            "width": dst_width,
            "count": 1,
            "dtype": "float32",
            "crs": dst_crs,
            "transform": dst_transform,
            "nodata": float(nodata),
            "compress": "deflate",
        }

        estimated_full = int(dst_width) * int(dst_height) * max(1, len(specs)) * 4 * 3
        if estimated_full > self.processing_limits.memory_budget_bytes():
            raise MemoryError(
                f"Full raster harmonisation would require approximately "
                f"{estimated_full / (1024**3):.2f} GiB of working memory, above the current "
                f"package budget of {self.processing_limits.memory_budget_bytes() / (1024**3):.2f} GiB. "
                "Use UAVMonitoringPipeline.run_raster_fusion(), which processes bounded windows."
            )

        aligned: dict[str, RasterData] = {}
        for key, (path, band) in specs.items():
            method = resampling.get(key, "bilinear") if isinstance(resampling, Mapping) else resampling
            with rasterio.open(path) as src:
                src_data = src.read(band)
                dst = np.full((dst_height, dst_width), float(nodata), dtype=np.float32)
                reproject(
                    source=src_data,
                    destination=dst,
                    src_transform=src.transform,
                    src_crs=src.crs,
                    src_nodata=src.nodata,
                    dst_transform=dst_transform,
                    dst_crs=dst_crs,
                    dst_nodata=float(nodata),
                    resampling=self._resampling(method),
                )
            aligned[key] = RasterData(
                data=dst[None, ...],
                profile=dict(target_profile),
                band_names=[key],
                source=str(path),
            )
        return aligned, target_profile

    def harmonise_rasters(
        self,
        paths: Iterable[str | Path],
        *,
        output_dir: str | Path | None = None,
        reference_index: int = 0,
        target_crs: str | None = None,
        resolution: float | tuple[float, float] | None = None,
        extent: str = "reference",
        resampling: str = "bilinear",
        nodata: float = -9999.0,
    ) -> list[Path]:
        """Harmonise complete single-band rasters and optionally write them.

        This keeps the D1.1 method name while delegating the actual work to the
        named-source implementation used by the pipelines.
        """
        items = [Path(p) for p in paths]
        if not items:
            return []
        sources = {f"raster_{i}": p for i, p in enumerate(items)}
        ref_key = f"raster_{reference_index}"
        aligned, _ = self.harmonise_raster_sources(
            sources, reference_key=ref_key, target_crs=target_crs, resolution=resolution,
            extent=extent, resampling=resampling, nodata=nodata
        )
        if output_dir is None:
            # No silent temporary files: return original paths when no destination is requested.
            return items
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        out_paths = []
        for key, raster in aligned.items():
            out = output_dir / f"{Path(raster.source).stem}_harmonised.tif"
            self.save_raster(raster, out)
            out_paths.append(out)
        return out_paths

    @staticmethod
    def save_raster(raster: RasterData, path: str | Path, *, band_names: Iterable[str] | None = None) -> Path:
        import rasterio
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        profile = dict(raster.profile)
        profile.update(count=raster.data.shape[0], dtype=str(raster.data.dtype), BIGTIFF="YES")
        with rasterio.open(path, "w", **profile) as dst:
            dst.write(raster.data)
            names = list(band_names or raster.band_names)
            for i, name in enumerate(names, start=1):
                dst.set_band_description(i, str(name))
        return path

    # ------------------------------------------------------------------
    # Memory-aware point-cloud ingestion
    # ------------------------------------------------------------------
    def inspect_point_cloud(
        self,
        path: str | Path,
        *,
        crs: str | None = None,
        text_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Inspect point-cloud metadata without materialising the complete cloud.

        LAS/LAZ metadata is read from the header. Text point clouds (TXT/CSV/XYZ)
        are scanned in bounded chunks so that point count, bounds and available
        numeric dimensions can be reported without loading the whole file. Text
        files do not carry a CRS; pass ``crs`` (or configure the project CRS).
        """
        path = Path(path)
        suffix = path.suffix.lower()
        if suffix in {".las", ".laz"}:
            try:
                import laspy
            except ImportError as exc:
                raise ImportError("LAS/LAZ support requires laspy") from exc

            with laspy.open(path) as reader:
                header = reader.header
                try:
                    parsed = header.parse_crs()
                    parsed_crs = parsed.to_string() if parsed is not None else None
                except Exception:
                    parsed_crs = None
                return {
                    "path": str(path),
                    "format": suffix.lstrip("."),
                    "point_count": int(header.point_count),
                    "point_format": int(header.point_format.id),
                    "point_record_bytes": int(header.point_format.size),
                    "dimensions": [str(x) for x in header.point_format.dimension_names],
                    "crs": crs or parsed_crs,
                    "mins": [float(x) for x in header.mins],
                    "maxs": [float(x) for x in header.maxs],
                    "streaming": True,
                }

        if suffix in {".txt", ".csv", ".xyz"}:
            schema = self._text_schema(path, text_options)
            count = 0
            mins = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
            maxs = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
            # Geometry-only scan; this stays memory-bounded.
            for chunk in self._iter_text_point_cloud_chunks(
                path,
                source=path.stem,
                crs=crs,
                attributes=[],
                chunk_points=min(500_000, self.processing_limits.point_chunk_points(bytes_per_point=32)),
                text_options=text_options,
                schema=schema,
            ):
                if chunk.n_points == 0:
                    continue
                count += chunk.n_points
                mins = np.minimum(mins, np.min(chunk.xyz, axis=0))
                maxs = np.maximum(maxs, np.max(chunk.xyz, axis=0))
            if count == 0:
                mins[:] = np.nan
                maxs[:] = np.nan
            return {
                "path": str(path),
                "format": suffix.lstrip("."),
                "point_count": int(count),
                "dimensions": ["X", "Y", "Z", *schema["attribute_names"]],
                "crs": crs,
                "mins": [float(x) for x in mins],
                "maxs": [float(x) for x in maxs],
                "streaming": True,
                "text_schema": schema["public"],
            }

        if suffix == ".ply":
            return {"path": str(path), "format": "ply", "crs": crs, "streaming": False}
        if suffix == ".npz":
            with np.load(path, allow_pickle=False) as data:
                if "xyz" not in data:
                    raise ValueError("NPZ point cloud must contain an 'xyz' array")
                xyz = np.asarray(data["xyz"], dtype=np.float64)
                attrs = [k[5:] for k in data.files if k.startswith("attr_")]
            return {
                "path": str(path), "format": "npz", "crs": crs, "streaming": False,
                "point_count": int(len(xyz)), "dimensions": ["X", "Y", "Z", *attrs],
                "mins": [float(x) for x in np.min(xyz, axis=0)] if len(xyz) else [float("nan")]*3,
                "maxs": [float(x) for x in np.max(xyz, axis=0)] if len(xyz) else [float("nan")]*3,
            }
        return {"path": str(path), "format": suffix.lstrip("."), "crs": crs, "streaming": False}

    @staticmethod
    def _split_text_line(line: str, delimiter: str | None) -> list[str]:
        if delimiter is None:
            return line.strip().split()
        return [token.strip() for token in line.strip().split(delimiter)]

    @staticmethod
    def _looks_numeric_tokens(tokens: Iterable[str]) -> bool:
        vals = list(tokens)
        if not vals:
            return False
        try:
            for token in vals:
                float(token)
            return True
        except (TypeError, ValueError):
            return False

    def _text_schema(
        self,
        path: Path,
        text_options: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Resolve a generic numeric text point-cloud schema.

        Supported options:
        ``delimiter``: ``"auto"`` (default), whitespace/``None``, comma, semicolon or tab;
        ``has_header``: ``"auto"`` (default), ``True`` or ``False``;
        ``columns``: mapping of logical names to zero-based indices or header names;
        ``comment``: comment prefix (default ``#``);
        ``encoding``: text encoding (default UTF-8).

        Coordinate logical names are ``x``, ``y`` and ``z``. All other mapped
        columns become arbitrary point attributes. No sensor/band convention is
        inferred.
        """
        opts = dict(text_options or {})
        delimiter_opt = opts.get("delimiter", "auto")
        comment = str(opts.get("comment", "#"))
        encoding = str(opts.get("encoding", "utf-8"))
        has_header_opt = opts.get("has_header", "auto")
        columns_opt = dict(opts.get("columns") or {})

        first_line = None
        first_line_number = 0
        with path.open("r", encoding=encoding, errors="replace") as fh:
            for line_number, line in enumerate(fh, start=1):
                stripped = line.strip()
                if not stripped or (comment and stripped.startswith(comment)):
                    continue
                first_line = stripped
                first_line_number = line_number
                break
        if first_line is None:
            raise ValueError(f"Text point cloud is empty: {path}")

        if delimiter_opt in {"auto", "AUTO"}:
            if "," in first_line:
                delimiter = ","
            elif ";" in first_line:
                delimiter = ";"
            elif "\t" in first_line:
                delimiter = "\t"
            else:
                delimiter = None
        elif delimiter_opt in {None, "", "whitespace", "space"}:
            delimiter = None
        elif delimiter_opt == "tab":
            delimiter = "\t"
        else:
            delimiter = str(delimiter_opt)

        tokens = self._split_text_line(first_line, delimiter)
        if has_header_opt == "auto":
            has_header = not self._looks_numeric_tokens(tokens)
        else:
            has_header = bool(has_header_opt)

        header_names = tokens if has_header else []
        n_columns_hint = len(tokens)
        lower_header = {name.lower(): i for i, name in enumerate(header_names)}

        def resolve_column(value: Any, logical_name: str) -> int:
            if isinstance(value, (int, np.integer)):
                idx = int(value)
            else:
                text = str(value)
                if text.isdigit():
                    idx = int(text)
                elif has_header and text.lower() in lower_header:
                    idx = int(lower_header[text.lower()])
                else:
                    raise KeyError(
                        f"Text column {value!r} for {logical_name!r} was not found. "
                        f"Header columns: {header_names or 'none'}"
                    )
            if idx < 0:
                raise ValueError("Text column indexes must be zero-based non-negative integers")
            return idx

        coord_cols: dict[str, int] = {}
        for coord, default_idx in (("x", 0), ("y", 1), ("z", 2)):
            if coord in columns_opt:
                coord_cols[coord] = resolve_column(columns_opt[coord], coord)
            elif has_header and coord in lower_header:
                coord_cols[coord] = int(lower_header[coord])
            else:
                coord_cols[coord] = default_idx

        if max(coord_cols.values()) >= n_columns_hint:
            raise ValueError(f"Text point cloud requires X/Y/Z columns: {path}")

        attribute_cols: dict[str, int] = {}
        # Explicit mappings have priority and define stable semantic names.
        for name, value in columns_opt.items():
            lname = str(name)
            if lname.lower() in {"x", "y", "z"}:
                continue
            attribute_cols[lname] = resolve_column(value, lname)

        if not attribute_cols:
            coord_idx = set(coord_cols.values())
            if has_header:
                for idx, name in enumerate(header_names):
                    if idx not in coord_idx:
                        attribute_cols[str(name)] = idx
            else:
                # Preserve historical generic names for unlabelled numeric columns.
                for idx in range(n_columns_hint):
                    if idx not in coord_idx:
                        attribute_cols[f"value_{idx - 2}"] = idx

        public_delimiter = "whitespace" if delimiter is None else ("tab" if delimiter == "\t" else delimiter)
        return {
            "delimiter": delimiter,
            "comment": comment,
            "encoding": encoding,
            "has_header": bool(has_header),
            "skiprows": int(first_line_number if has_header else first_line_number - 1),
            "coordinate_columns": coord_cols,
            "attribute_columns": attribute_cols,
            "attribute_names": list(attribute_cols),
            "public": {
                "delimiter": public_delimiter,
                "has_header": bool(has_header),
                "columns": {**coord_cols, **attribute_cols},
            },
        }

    def _iter_text_point_cloud_chunks(
        self,
        path: Path,
        *,
        source: str | None,
        crs: str | None,
        attributes: Iterable[str] | None,
        chunk_points: int | None,
        text_options: Mapping[str, Any] | None,
        schema: Mapping[str, Any] | None = None,
    ) -> Iterator[PointCloudData]:
        schema = dict(schema or self._text_schema(path, text_options))
        attr_cols_all = dict(schema["attribute_columns"])
        if attributes is None:
            requested = list(attr_cols_all)
        else:
            requested = list(dict.fromkeys(str(x) for x in attributes))
        missing = [name for name in requested if name not in attr_cols_all]
        if missing:
            raise KeyError(
                f"Requested text point attributes are missing: {missing}. "
                f"Available attributes: {list(attr_cols_all)}"
            )

        coord_cols = dict(schema["coordinate_columns"])
        selected_columns = [coord_cols["x"], coord_cols["y"], coord_cols["z"]]
        selected_columns.extend(attr_cols_all[name] for name in requested)
        # np.loadtxt returns columns in usecols order; preserve duplicates only once.
        unique_cols = list(dict.fromkeys(int(x) for x in selected_columns))
        position = {col: i for i, col in enumerate(unique_cols)}

        chunk = int(chunk_points or self.processing_limits.point_chunk_points(bytes_per_point=max(32, 24 + 8 * len(requested))))
        # Very large max_rows values cause avoidable parser/string memory spikes.
        chunk = max(1_000, min(chunk, 500_000))
        offset = 0
        with path.open("r", encoding=str(schema["encoding"]), errors="replace") as fh:
            for _ in range(int(schema["skiprows"])):
                next(fh, None)
            while True:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings(
                            "ignore",
                            message="loadtxt: input contained no data.*",
                            category=UserWarning,
                        )
                        arr = np.loadtxt(
                            fh,
                            dtype=np.float64,
                            delimiter=schema["delimiter"],
                            comments=schema["comment"],
                            usecols=unique_cols,
                            max_rows=chunk,
                            ndmin=2,
                        )
                except ValueError as exc:
                    raise ValueError(
                        f"Could not parse numeric text point cloud {path}. "
                        "Provide text_options with delimiter/has_header/columns when auto-detection is insufficient."
                    ) from exc
                if arr.size == 0 or arr.shape[0] == 0:
                    break
                xyz = np.column_stack([
                    arr[:, position[coord_cols["x"]]],
                    arr[:, position[coord_cols["y"]]],
                    arr[:, position[coord_cols["z"]]],
                ])
                finite = np.all(np.isfinite(xyz), axis=1)
                if not np.any(finite):
                    offset += int(arr.shape[0])
                    continue
                xyz = xyz[finite]
                attrs = {
                    name: arr[:, position[attr_cols_all[name]]][finite]
                    for name in requested
                }
                n = int(len(xyz))
                yield PointCloudData(
                    xyz=xyz,
                    attributes=attrs,
                    crs=crs,
                    source=source or path.stem,
                    metadata={
                        "path": str(path),
                        "format": path.suffix.lower().lstrip("."),
                        "chunk_offset": offset,
                        "chunk_size": n,
                        "text_schema": schema["public"],
                    },
                )
                offset += int(arr.shape[0])

    @staticmethod
    def _resolve_las_dimensions(header: Any, attributes: Iterable[str] | None) -> list[str]:
        available = [str(name) for name in header.point_format.dimension_names]
        exact = {name: name for name in available}
        lower = {name.lower(): name for name in available}
        if attributes is None:
            return [name for name in available if name.lower() not in {"x", "y", "z"}]
        requested = [str(x) for x in attributes]
        resolved: list[str] = []
        missing: list[str] = []
        for name in dict.fromkeys(requested):
            actual = exact.get(name) or lower.get(name.lower())
            if actual is None:
                missing.append(name)
            elif actual.lower() not in {"x", "y", "z"}:
                resolved.append(actual)
        if missing:
            raise KeyError(
                f"Requested LAS/LAZ dimensions are missing: {missing}. Available dimensions: {available}"
            )
        return resolved

    def iter_point_cloud_chunks(
        self,
        path: str | Path,
        *,
        source: str | None = None,
        crs: str | None = None,
        attributes: Iterable[str] | None = None,
        chunk_points: int | None = None,
        text_options: Mapping[str, Any] | None = None,
    ) -> Iterator[PointCloudData]:
        """Stream a point cloud as bounded :class:`PointCloudData` chunks.

        LAS/LAZ and numeric TXT/CSV/XYZ are streamed. Only requested attributes
        are materialised. Text columns are defined with ``text_options``; this
        lets photogrammetric point clouds keep physical variables such as
        temperature without first converting the file to LAS.
        """
        path = Path(path)
        if path.suffix.lower() in {".txt", ".csv", ".xyz"}:
            yield from self._iter_text_point_cloud_chunks(
                path, source=source, crs=crs, attributes=attributes,
                chunk_points=chunk_points, text_options=text_options,
            )
            return
        if path.suffix.lower() not in {".las", ".laz"}:
            # PLY/NPZ currently use the guarded in-memory loader.
            yield self.load_point_cloud(
                path, source=source, crs=crs, attributes=attributes,
                text_options=text_options,
            )
            return

        try:
            import laspy
        except ImportError as exc:
            raise ImportError(
                "LAS/LAZ support requires laspy (and lazrs for LAZ): pip install 'laspy[lazrs]'"
            ) from exc

        with laspy.open(path) as reader:
            header = reader.header
            dims = self._resolve_las_dimensions(header, attributes)
            parsed_crs = crs
            if parsed_crs is None:
                try:
                    parsed = header.parse_crs()
                    parsed_crs = parsed.to_string() if parsed is not None else None
                except Exception:
                    parsed_crs = None

            # XYZ float64 plus selected dimensions and temporary conversion
            # arrays. 2x point-record size is a conservative estimate.
            bytes_per_point = max(32, 24 + 2 * int(header.point_format.size))
            chunk = int(chunk_points or self.processing_limits.point_chunk_points(
                bytes_per_point=bytes_per_point
            ))
            offset = 0
            for points in reader.chunk_iterator(chunk):
                xyz = np.column_stack((points.x, points.y, points.z)).astype(np.float64, copy=False)
                attrs: dict[str, np.ndarray] = {}
                for name in dims:
                    try:
                        attrs[name] = np.asarray(points[name])
                    except Exception:
                        if hasattr(points, name):
                            attrs[name] = np.asarray(getattr(points, name))
                n = int(len(xyz))
                yield PointCloudData(
                    xyz=xyz,
                    attributes=attrs,
                    crs=parsed_crs,
                    source=source or path.stem,
                    metadata={
                        "path": str(path),
                        "format": path.suffix.lower().lstrip("."),
                        "chunk_offset": offset,
                        "chunk_size": n,
                        "n_original": int(header.point_count),
                    },
                )
                offset += n

    def sample_point_cloud(
        self,
        path: str | Path,
        *,
        max_points: int,
        source: str | None = None,
        crs: str | None = None,
        attributes: Iterable[str] | None = None,
        text_options: Mapping[str, Any] | None = None,
    ) -> PointCloudData:
        """Create a deterministic spatial-order sample without full-file loading."""
        info = self.inspect_point_cloud(path, crs=crs, text_options=text_options)
        total = int(info.get("point_count", 0) or 0)
        streamable = path_suffix(path) in {"las", "laz", "txt", "csv", "xyz"}
        if total <= max_points or not streamable:
            return self.load_point_cloud(
                path, source=source, crs=crs, attributes=attributes, max_points=max_points,
                text_options=text_options,
            )
        stride = max(1, int(math.ceil(total / int(max_points))))
        xyz_parts: list[np.ndarray] = []
        attr_parts: dict[str, list[np.ndarray]] = {}
        selected = 0
        global_offset = 0
        for chunk in self.iter_point_cloud_chunks(
            path, source=source, crs=crs, attributes=attributes, text_options=text_options
        ):
            n = chunk.n_points
            local = np.arange(n, dtype=np.int64)
            keep = ((local + global_offset) % stride) == 0
            if np.any(keep):
                xyz_parts.append(chunk.xyz[keep])
                for name, values in chunk.attributes.items():
                    attr_parts.setdefault(name, []).append(np.asarray(values)[keep])
                selected += int(np.count_nonzero(keep))
            global_offset += n
            if selected >= max_points:
                break
        xyz = np.concatenate(xyz_parts, axis=0)[:max_points]
        attrs = {
            name: np.concatenate(parts, axis=0)[:max_points]
            for name, parts in attr_parts.items()
        }
        return PointCloudData(
            xyz=xyz,
            attributes=attrs,
            crs=crs or info.get("crs"),
            source=source or Path(path).stem,
            metadata={
                "path": str(path),
                "n_original": total,
                "subsampled_to": int(len(xyz)),
                "streaming_sample": True,
            },
        )

    # ------------------------------------------------------------------
    # Point-cloud ingestion added without changing package structure
    # ------------------------------------------------------------------
    def load_point_cloud(
        self,
        path: str | Path,
        *,
        source: str | None = None,
        crs: str | None = None,
        attributes: Iterable[str] | None = None,
        max_points: int | None = None,
        random_state: int = 42,
        text_options: Mapping[str, Any] | None = None,
    ) -> PointCloudData:
        """Load LAS/LAZ, NPZ, CSV/TXT or PLY point-cloud data.

        LAS/LAZ is the recommended interchange format for BioClima. ``laspy``
        is imported lazily so the core numerical tests do not depend on it.
        """
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix in {".las", ".laz"}:
            cloud = self._load_las_laz(path, source=source, crs=crs, attributes=attributes)
        elif suffix == ".npz":
            cloud = self._load_npz(path, source=source, crs=crs)
        elif suffix in {".csv", ".txt", ".xyz"}:
            cloud = self._load_text(
                path, source=source, crs=crs, attributes=attributes, text_options=text_options
            )
        elif suffix == ".ply":
            cloud = self._load_ply(path, source=source, crs=crs)
        else:
            raise ValueError(f"Unsupported point-cloud format: {suffix}")

        if max_points is not None and cloud.n_points > max_points:
            rng = np.random.default_rng(random_state)
            idx = np.sort(rng.choice(cloud.n_points, size=max_points, replace=False))
            cloud = cloud.subset(idx)
            cloud.metadata["subsampled_from"] = int(cloud.metadata.get("n_original", cloud.n_points))
            cloud.metadata["subsampled_to"] = int(max_points)

        return cloud

    def _load_las_laz(
        self,
        path: Path,
        *,
        source: str | None,
        crs: str | None,
        attributes: Iterable[str] | None,
    ) -> PointCloudData:
        try:
            import laspy
        except ImportError as exc:
            raise ImportError(
                "LAS/LAZ support requires laspy (and lazrs for LAZ): "
                "pip install 'laspy[lazrs]'"
            ) from exc

        with laspy.open(path) as reader:
            header = reader.header
            dims = self._resolve_las_dimensions(header, attributes)
            estimated = int(header.point_count) * max(32, 24 + 2 * int(header.point_format.size))
            if estimated > self.processing_limits.memory_budget_bytes():
                raise MemoryError(
                    f"Full LAS/LAZ load is estimated at {estimated / (1024**3):.2f} GiB, above the "
                    f"current package RAM budget of "
                    f"{self.processing_limits.memory_budget_bytes() / (1024**3):.2f} GiB. "
                    "Use iter_point_cloud_chunks(), sample_point_cloud(), or a streaming pipeline."
                )
        las = laspy.read(path)
        xyz = np.column_stack((las.x, las.y, las.z)).astype(np.float64, copy=False)

        # Keep LAS/LAZ ingestion sensor-agnostic. By default every available
        # non-coordinate point dimension is loaded exactly as it is named in
        # the file, including Extra Bytes dimensions. Users may restrict the
        # list explicitly with ``attributes`` when memory is a concern.
        available_names = [str(name) for name in las.point_format.dimension_names]
        attrs: dict[str, np.ndarray] = {}
        for actual_name in dims:
            try:
                attrs[actual_name] = np.asarray(las[actual_name])
            except Exception:
                if hasattr(las, actual_name):
                    attrs[actual_name] = np.asarray(getattr(las, actual_name))

        parsed_crs = crs
        if parsed_crs is None:
            try:
                parsed = las.header.parse_crs()
                parsed_crs = parsed.to_string() if parsed is not None else None
            except Exception:
                parsed_crs = None

        return PointCloudData(
            xyz=xyz,
            attributes=attrs,
            crs=parsed_crs,
            source=source or path.stem,
            metadata={"path": str(path), "format": suffix_name(path), "n_original": int(len(xyz))},
        )

    @staticmethod
    def _load_npz(path: Path, *, source: str | None, crs: str | None) -> PointCloudData:
        with np.load(path, allow_pickle=False) as data:
            if "xyz" not in data:
                raise ValueError("NPZ point cloud must contain an 'xyz' array")
            xyz = np.asarray(data["xyz"], dtype=np.float64)
            attrs = {
                k[5:]: np.asarray(data[k])
                for k in data.files
                if k.startswith("attr_")
            }
        return PointCloudData(
            xyz=xyz,
            attributes=attrs,
            crs=crs,
            source=source or path.stem,
            metadata={"path": str(path), "format": "npz", "n_original": int(len(xyz))},
        )

    def _load_text(
        self,
        path: Path,
        *,
        source: str | None,
        crs: str | None,
        attributes: Iterable[str] | None,
        text_options: Mapping[str, Any] | None,
    ) -> PointCloudData:
        """Load a text point cloud using the same schema as the streaming reader."""
        info = self.inspect_point_cloud(path, crs=crs, text_options=text_options)
        requested = list(attributes) if attributes is not None else [
            d for d in info.get("dimensions", []) if str(d).lower() not in {"x", "y", "z"}
        ]
        n = int(info.get("point_count", 0) or 0)
        estimated = n * max(32, 24 + 8 * len(requested)) * 3
        if estimated > self.processing_limits.memory_budget_bytes():
            raise MemoryError(
                f"Full text point-cloud load is estimated at {estimated / (1024**3):.2f} GiB, "
                f"above the current package RAM budget of "
                f"{self.processing_limits.memory_budget_bytes() / (1024**3):.2f} GiB. "
                "Use iter_point_cloud_chunks(), sample_point_cloud(), or the streaming fusion pipeline."
            )
        xyz_parts: list[np.ndarray] = []
        attr_parts: dict[str, list[np.ndarray]] = {str(name): [] for name in requested}
        for chunk in self.iter_point_cloud_chunks(
            path, source=source, crs=crs, attributes=requested, text_options=text_options
        ):
            xyz_parts.append(chunk.xyz)
            for name in requested:
                attr_parts[str(name)].append(np.asarray(chunk.attributes[str(name)]))
        if not xyz_parts:
            return PointCloudData(
                np.empty((0, 3), dtype=np.float64), {}, crs, source or path.stem,
                {"path": str(path), "format": "text", "text_schema": info.get("text_schema")},
            )
        xyz = np.concatenate(xyz_parts, axis=0)
        attrs = {
            name: np.concatenate(parts, axis=0)
            for name, parts in attr_parts.items() if parts
        }
        return PointCloudData(
            xyz, attrs, crs, source or path.stem,
            {
                "path": str(path), "format": path.suffix.lower().lstrip("."),
                "n_original": int(len(xyz)), "text_schema": info.get("text_schema"),
            },
        )

    @staticmethod
    def _load_ply(path: Path, *, source: str | None, crs: str | None) -> PointCloudData:
        try:
            from plyfile import PlyData
        except ImportError as exc:
            raise ImportError("PLY support requires plyfile: pip install plyfile") from exc

        ply = PlyData.read(path)
        vertex = ply["vertex"].data
        names = list(vertex.dtype.names or [])
        for req in ("x", "y", "z"):
            if req not in names:
                raise ValueError(f"PLY is missing vertex property {req!r}")
        xyz = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float64)
        attrs = {
            name: np.asarray(vertex[name])
            for name in names
            if name not in {"x", "y", "z"} and np.issubdtype(vertex[name].dtype, np.number)
        }
        return PointCloudData(xyz, attrs, crs, source or path.stem, {"path": str(path), "format": "ply"})

    def save_point_cloud(self, cloud: PointCloudData, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        suffix = path.suffix.lower()
        if suffix == ".npz":
            payload = {"xyz": cloud.xyz}
            payload.update({f"attr_{k}": v for k, v in cloud.attributes.items()})
            np.savez_compressed(path, **payload)
            return path
        if suffix in {".las", ".laz"}:
            return self._save_las_laz(cloud, path)
        if suffix == ".csv":
            names = ["x", "y", "z", *cloud.attributes.keys()]
            arrays = [cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2], *cloud.attributes.values()]
            matrix = np.column_stack(arrays)
            np.savetxt(path, matrix, delimiter=",", header=",".join(names), comments="")
            return path
        raise ValueError("Output point cloud must be .las, .laz, .npz or .csv")

    @staticmethod
    def _save_las_laz(cloud: PointCloudData, path: Path) -> Path:
        try:
            import laspy
        except ImportError as exc:
            raise ImportError(
                "LAS/LAZ output requires laspy (and lazrs for LAZ): "
                "pip install 'laspy[lazrs]'"
            ) from exc

        header = laspy.LasHeader(point_format=3, version="1.2")
        header.scales = np.array([0.001, 0.001, 0.001])
        header.offsets = np.floor(np.min(cloud.xyz, axis=0))
        if cloud.crs:
            try:
                from pyproj import CRS
                header.add_crs(CRS.from_user_input(cloud.crs))
            except Exception:
                LOGGER.warning("Could not write CRS %r to LAS header", cloud.crs)

        las = laspy.LasData(header)
        las.x, las.y, las.z = cloud.xyz[:, 0], cloud.xyz[:, 1], cloud.xyz[:, 2]

        standard = {str(name) for name in las.point_format.dimension_names}
        for name, values in cloud.attributes.items():
            arr = np.asarray(values)
            safe_name = _safe_las_name(name)
            if safe_name in standard:
                try:
                    setattr(las, safe_name, arr)
                    continue
                except Exception:
                    pass

            # Extra dimensions use compact numeric types and NaN for missing floats.
            if np.issubdtype(arr.dtype, np.bool_):
                arr_out = arr.astype(np.uint8)
            elif np.issubdtype(arr.dtype, np.integer):
                arr_out = arr.astype(np.int32, copy=False)
            else:
                arr_out = arr.astype(np.float32, copy=False)
            las.add_extra_dim(laspy.ExtraBytesParams(name=safe_name, type=arr_out.dtype))
            las[safe_name] = arr_out

        las.write(path)
        return path


def _safe_las_name(name: str) -> str:
    # LAS extra-byte names are limited; keep them deterministic and readable.
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    return safe[:32] or "value"


def suffix_name(path: Path) -> str:
    return path.suffix.lower().lstrip(".")


def path_suffix(path: str | Path) -> str:
    return Path(path).suffix.lower().lstrip(".")
