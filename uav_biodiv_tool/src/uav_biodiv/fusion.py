from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping, Sequence
import logging

import numpy as np
from scipy.spatial import cKDTree

from .ingestion import PointCloudData
from .execution import process_pool

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class RegistrationResult:
    method: str
    transformation: np.ndarray
    rmse: float
    median_distance: float
    p95_distance: float
    overlap_fraction: float
    correspondences: int
    iterations: int
    converged: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "transformation": self.transformation.tolist(),
            "rmse": float(self.rmse),
            "median_distance": float(self.median_distance),
            "p95_distance": float(self.p95_distance),
            "overlap_fraction": float(self.overlap_fraction),
            "correspondences": int(self.correspondences),
            "iterations": int(self.iterations),
            "converged": bool(self.converged),
        }


class MultiSensorFusion:
    """Multi-sensor fusion (D1.1 §8.1.4).

    Point-cloud and raster workflows are kept as separate representations.
    Case 3 converts LiDAR to raster metrics first and only then performs
    raster-domain feature fusion.
    """

    # ------------------------------------------------------------------
    # Raster fusion (D1.1 §8.1.4)
    # ------------------------------------------------------------------
    @staticmethod
    def stack_rasters(*arrays: np.ndarray) -> np.ndarray:
        """Stack already aligned rasters into a band-first feature cube."""
        if not arrays:
            raise ValueError("At least one raster array is required")
        layers = []
        shape = None
        for arr in arrays:
            a = np.asarray(arr)
            if a.ndim == 2:
                a = a[None, ...]
            if a.ndim != 3:
                raise ValueError("Each raster must be 2-D or band-first 3-D")
            if shape is None:
                shape = a.shape[1:]
            elif a.shape[1:] != shape:
                raise ValueError("Rasters must already be aligned before stacking")
            layers.append(a)
        return np.concatenate(layers, axis=0)

    @staticmethod
    def stack_named_rasters(
        rasters: Mapping[str, np.ndarray],
        *,
        nodata: float | None = -9999.0,
        dtype: str | np.dtype = np.float32,
    ) -> tuple[np.ndarray, list[str]]:
        """Stack named 2-D feature rasters and return ``(cube, band_names)``."""
        if not rasters:
            raise ValueError("At least one named raster is required")
        names = list(rasters)
        arrays = []
        shape = None
        for name in names:
            a = np.asarray(rasters[name], dtype=dtype)
            if a.ndim != 2:
                raise ValueError(f"Raster feature {name!r} must be 2-D")
            if shape is None:
                shape = a.shape
            elif a.shape != shape:
                raise ValueError("All named rasters must have identical shapes")
            if nodata is not None:
                a = a.copy()
                a[~np.isfinite(a)] = float(nodata)
            arrays.append(a)
        return np.stack(arrays, axis=0), names

    @staticmethod
    def compute_indices(
        features: Mapping[str, np.ndarray],
        definitions: Mapping[str, Mapping[str, Any]],
        *,
        nodata: float | None = None,
    ) -> dict[str, np.ndarray]:
        """Compute user-configured raster indices from arbitrary band names.

        Supported methods are intentionally simple and transparent:
        ``normalized_difference``, ``ratio``, ``difference``, ``sum`` and
        ``product``. Each definition uses concrete input names ``a`` and ``b``.
        Role-to-band resolution is handled by :class:`UAVProjectConfig`.
        """
        def clean(a: np.ndarray) -> np.ndarray:
            out = np.asarray(a, dtype=np.float32).copy()
            if nodata is not None:
                out[np.isclose(out, nodata)] = np.nan
            return out

        def get(name: str) -> np.ndarray:
            if name not in features:
                raise KeyError(
                    f"Index input {name!r} is missing. Available raster features: {sorted(features)}"
                )
            return clean(features[name])

        out: dict[str, np.ndarray] = {}
        for index_name, raw in definitions.items():
            spec = dict(raw)
            method = str(spec.get("method", "normalized_difference")).lower()
            if "a" not in spec or "b" not in spec:
                raise ValueError(f"Index {index_name!r} requires 'a' and 'b' input names")
            a = get(str(spec["a"]))
            b = get(str(spec["b"]))

            if method in {"normalized_difference", "nd"}:
                den = a + b
                value = np.divide(
                    a - b,
                    den,
                    out=np.full_like(a, np.nan),
                    where=np.isfinite(den) & (np.abs(den) > 1e-12),
                )
            elif method == "ratio":
                value = np.divide(
                    a, b, out=np.full_like(a, np.nan),
                    where=np.isfinite(b) & (np.abs(b) > 1e-12),
                )
            elif method == "difference":
                value = a - b
            elif method == "sum":
                value = a + b
            elif method == "product":
                value = a * b
            else:
                raise ValueError(
                    f"Unsupported index method {method!r} for {index_name!r}; "
                    "use normalized_difference, ratio, difference, sum or product"
                )
            out[str(index_name)] = value.astype(np.float32, copy=False)
        return out

    @staticmethod
    def compute_thermal_anomaly(
        thermal: np.ndarray,
        *,
        reference: float | None = None,
        nodata: float | None = None,
    ) -> np.ndarray:
        arr = np.asarray(thermal, dtype=np.float32).copy()
        if nodata is not None:
            arr[np.isclose(arr, nodata)] = np.nan
        ref = float(np.nanmean(arr)) if reference is None else float(reference)
        return arr - ref

    @staticmethod
    def fuse_lidar_and_spectral(
        lidar_metrics: Mapping[str, np.ndarray | float],
        spectral_indices: Mapping[str, np.ndarray | float],
    ) -> dict[str, np.ndarray | float]:
        """Feature-level fusion of *rasterized* LiDAR metrics and raster features.

        This method intentionally does not sample rasters onto point clouds. In
        case 3, LiDAR is first rasterized to the common grid and only then
        merged with multispectral/thermal raster features.
        """
        out = dict(lidar_metrics)
        overlap = set(out).intersection(spectral_indices)
        if overlap:
            raise ValueError(f"Duplicate feature names: {sorted(overlap)}")
        out.update(spectral_indices)
        return out

    @staticmethod
    def pca_feature_fusion(
        stack: np.ndarray,
        *,
        n_components: int | float = 0.95,
        nodata: float = -9999.0,
        standardize: bool = True,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        """Optional feature-level PCA for an aligned raster stack.

        Missing pixels are excluded from fitting and remain nodata in the
        output. ``n_components`` can be an integer or explained-variance ratio.
        """
        from sklearn.decomposition import PCA
        from sklearn.preprocessing import StandardScaler

        cube = np.asarray(stack, dtype=np.float32)
        if cube.ndim != 3:
            raise ValueError("stack must be band-first with shape (bands, rows, cols)")
        bands, rows, cols = cube.shape
        x = cube.reshape(bands, -1).T.astype(np.float64)
        valid = np.all(np.isfinite(x), axis=1) & np.all(~np.isclose(x, nodata), axis=1)
        if np.count_nonzero(valid) < 2:
            raise ValueError("Not enough valid pixels for PCA")
        x_valid = x[valid]
        scaler = StandardScaler() if standardize else None
        x_fit = scaler.fit_transform(x_valid) if scaler is not None else x_valid
        pca = PCA(n_components=n_components)
        scores = pca.fit_transform(x_fit)
        out = np.full((scores.shape[1], rows * cols), nodata, dtype=np.float32)
        out[:, valid] = scores.T.astype(np.float32)
        meta = {
            "explained_variance_ratio": pca.explained_variance_ratio_.astype(float).tolist(),
            "components": pca.components_.astype(float).tolist(),
            "standardized": bool(standardize),
            "n_valid_pixels": int(np.count_nonzero(valid)),
        }
        return out.reshape(scores.shape[1], rows, cols), meta

    # ------------------------------------------------------------------
    # Point-cloud registration
    # ------------------------------------------------------------------
    def register_point_cloud(
        self,
        reference: PointCloudData,
        moving: PointCloudData,
        *,
        method: str = "icp",
        voxel_size: float = 0.25,
        max_correspondence_distance: float | None = 1.0,
        max_registration_points: int = 250_000,
        max_iterations: int = 60,
        tolerance: float = 1e-5,
        trim_fraction: float = 0.90,
        initial_transform: np.ndarray | None = None,
        random_state: int = 42,
    ) -> tuple[PointCloudData, RegistrationResult]:
        """Rigidly register ``moving`` onto ``reference``.

        ``icp`` is a pure NumPy/SciPy implementation. ``gicp`` uses Open3D
        when installed. Both operate on downsampled points for registration,
        then apply the estimated transformation to the complete moving cloud.
        """
        method = method.lower()
        if method not in {"icp", "gicp"}:
            raise ValueError("method must be 'icp' or 'gicp'")
        if not (0 < trim_fraction <= 1):
            raise ValueError("trim_fraction must be in (0, 1]")

        ref_pts = self._registration_sample(
            reference.xyz, voxel_size=voxel_size, max_points=max_registration_points,
            random_state=random_state,
        )
        mov_pts = self._registration_sample(
            moving.xyz, voxel_size=voxel_size, max_points=max_registration_points,
            random_state=random_state + 1,
        )

        if len(ref_pts) < 3 or len(mov_pts) < 3:
            raise ValueError("At least three points are required in each cloud")

        if method == "icp":
            result = self._icp_numpy(
                ref_pts,
                mov_pts,
                max_correspondence_distance=max_correspondence_distance,
                max_iterations=max_iterations,
                tolerance=tolerance,
                trim_fraction=trim_fraction,
                initial_transform=initial_transform,
            )
        else:
            result = self._gicp_open3d(
                ref_pts,
                mov_pts,
                max_correspondence_distance=max_correspondence_distance,
                voxel_size=voxel_size,
                max_iterations=max_iterations,
                initial_transform=initial_transform,
            )

        aligned = moving.transformed(result.transformation)
        aligned.metadata = dict(aligned.metadata)
        aligned.metadata["registration"] = result.to_dict()
        return aligned, result

    @staticmethod
    def _registration_sample(
        xyz: np.ndarray,
        *,
        voxel_size: float,
        max_points: int,
        random_state: int,
    ) -> np.ndarray:
        pts = np.asarray(xyz, dtype=np.float64)
        if voxel_size > 0:
            origin = np.min(pts, axis=0)
            vox = np.floor((pts - origin) / voxel_size).astype(np.int64)
            _, first = np.unique(vox, axis=0, return_index=True)
            pts = pts[np.sort(first)]
        if len(pts) > max_points:
            rng = np.random.default_rng(random_state)
            idx = rng.choice(len(pts), size=max_points, replace=False)
            pts = pts[idx]
        return pts

    @staticmethod
    def _icp_numpy(
        reference: np.ndarray,
        moving: np.ndarray,
        *,
        max_correspondence_distance: float | None,
        max_iterations: int,
        tolerance: float,
        trim_fraction: float,
        initial_transform: np.ndarray | None,
    ) -> RegistrationResult:
        # Use a common local origin to avoid numerical instability with projected coordinates.
        origin = np.mean(reference, axis=0)
        ref = reference - origin
        mov = moving - origin

        to_local = np.eye(4)
        to_local[:3, 3] = -origin
        to_world = np.eye(4)
        to_world[:3, 3] = origin

        if initial_transform is None:
            transform = np.eye(4)
        else:
            init = np.asarray(initial_transform, dtype=np.float64)
            if init.shape != (4, 4):
                raise ValueError("initial_transform must have shape (4, 4)")
            transform = to_local @ init @ to_world

        tree = cKDTree(ref)
        previous_rmse = np.inf
        converged = False
        last_dist = np.array([], dtype=np.float64)
        last_keep = np.array([], dtype=bool)

        for iteration in range(1, max_iterations + 1):
            transformed = _apply_transform(mov, transform)
            distances, nn_idx = tree.query(transformed, k=1, workers=-1)
            keep = np.isfinite(distances)
            if max_correspondence_distance is not None:
                keep &= distances <= max_correspondence_distance
            if np.count_nonzero(keep) < 3:
                raise RuntimeError(
                    "ICP found fewer than three correspondences. Increase "
                    "max_correspondence_distance or provide a better initial transform."
                )

            valid_idx = np.flatnonzero(keep)
            if trim_fraction < 1.0 and len(valid_idx) > 3:
                n_keep = max(3, int(np.floor(len(valid_idx) * trim_fraction)))
                ranked = valid_idx[np.argsort(distances[valid_idx])[:n_keep]]
                keep = np.zeros_like(keep)
                keep[ranked] = True

            src = transformed[keep]
            dst = ref[nn_idx[keep]]
            delta = _best_fit_rigid_transform(src, dst)
            transform = delta @ transform

            # Evaluate after applying delta.
            transformed2 = _apply_transform(mov, transform)
            distances2, _ = tree.query(transformed2, k=1, workers=-1)
            keep2 = np.isfinite(distances2)
            if max_correspondence_distance is not None:
                keep2 &= distances2 <= max_correspondence_distance
            if np.count_nonzero(keep2) < 3:
                raise RuntimeError("ICP lost valid correspondences during iteration")
            vals = distances2[keep2]
            if trim_fraction < 1.0 and len(vals) > 3:
                n_keep = max(3, int(np.floor(len(vals) * trim_fraction)))
                vals = np.partition(vals, n_keep - 1)[:n_keep]
            rmse = float(np.sqrt(np.mean(vals ** 2)))

            delta_translation = float(np.linalg.norm(delta[:3, 3]))
            delta_rotation = float(np.linalg.norm(delta[:3, :3] - np.eye(3), ord="fro"))
            if abs(previous_rmse - rmse) < tolerance and max(delta_translation, delta_rotation) < np.sqrt(tolerance):
                converged = True
                previous_rmse = rmse
                last_dist = distances2
                last_keep = keep2
                break

            previous_rmse = rmse
            last_dist = distances2
            last_keep = keep2
        else:
            iteration = max_iterations

        # Convert local transform back to world/map coordinates.
        world_transform = to_world @ transform @ to_local

        if last_dist.size == 0:
            transformed = _apply_transform(mov, transform)
            last_dist, _ = tree.query(transformed, k=1, workers=-1)
            last_keep = np.isfinite(last_dist)
            if max_correspondence_distance is not None:
                last_keep &= last_dist <= max_correspondence_distance

        kept = last_dist[last_keep]
        return RegistrationResult(
            method="icp",
            transformation=world_transform,
            rmse=float(np.sqrt(np.mean(kept ** 2))),
            median_distance=float(np.median(kept)),
            p95_distance=float(np.percentile(kept, 95)),
            overlap_fraction=float(np.count_nonzero(last_keep) / len(last_keep)),
            correspondences=int(np.count_nonzero(last_keep)),
            iterations=int(iteration),
            converged=bool(converged),
        )

    @staticmethod
    def _gicp_open3d(
        reference: np.ndarray,
        moving: np.ndarray,
        *,
        max_correspondence_distance: float | None,
        voxel_size: float,
        max_iterations: int,
        initial_transform: np.ndarray | None,
    ) -> RegistrationResult:
        try:
            import open3d as o3d
        except ImportError as exc:
            raise ImportError(
                "Generalized ICP requires Open3D. Reinstall the package with: python -m pip install -e ."
            ) from exc

        origin = np.mean(reference, axis=0)
        ref = reference - origin
        mov = moving - origin
        threshold = max_correspondence_distance
        if threshold is None:
            threshold = max(3.0 * voxel_size, 1.0)

        ref_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(ref))
        mov_pc = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(mov))
        init = np.eye(4)
        if initial_transform is not None:
            init_world = np.asarray(initial_transform, dtype=np.float64)
            to_local = np.eye(4); to_local[:3, 3] = -origin
            to_world = np.eye(4); to_world[:3, 3] = origin
            init = to_local @ init_world @ to_world

        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=max_iterations)
        reg = o3d.pipelines.registration.registration_generalized_icp(
            mov_pc,
            ref_pc,
            threshold,
            init,
            o3d.pipelines.registration.TransformationEstimationForGeneralizedICP(),
            criteria,
        )

        to_local = np.eye(4); to_local[:3, 3] = -origin
        to_world = np.eye(4); to_world[:3, 3] = origin
        world_transform = to_world @ np.asarray(reg.transformation) @ to_local

        aligned_local = _apply_transform(mov, np.asarray(reg.transformation))
        distances, _ = cKDTree(ref).query(aligned_local, k=1, workers=-1)
        keep = np.isfinite(distances) & (distances <= threshold)
        kept = distances[keep]
        return RegistrationResult(
            method="gicp",
            transformation=world_transform,
            rmse=float(np.sqrt(np.mean(kept ** 2))) if len(kept) else float("nan"),
            median_distance=float(np.median(kept)) if len(kept) else float("nan"),
            p95_distance=float(np.percentile(kept, 95)) if len(kept) else float("nan"),
            overlap_fraction=float(np.mean(keep)),
            correspondences=int(np.count_nonzero(keep)),
            iterations=int(max_iterations),
            converged=bool(reg.fitness > 0),
        )

    # ------------------------------------------------------------------
    # Point-domain fusion
    # ------------------------------------------------------------------
    def transfer_attributes_to_reference(
        self,
        reference: PointCloudData,
        source: PointCloudData,
        *,
        attributes: Sequence[str] | None = None,
        prefix: str | None = None,
        k: int = 1,
        max_distance: float | None = 0.50,
        weighting: str = "idw",
        include_distance: bool = True,
    ) -> tuple[PointCloudData, dict[str, float]]:
        """Transfer attributes from a registered source cloud to reference points.

        This keeps the final geometry on the selected reference cloud and attaches
        arbitrary numeric source attributes entirely within point-cloud space.
        """
        if k < 1:
            raise ValueError("k must be >= 1")
        if weighting not in {"nearest", "mean", "idw"}:
            raise ValueError("weighting must be 'nearest', 'mean' or 'idw'")

        attrs = list(attributes) if attributes is not None else list(source.attributes)
        missing = [a for a in attrs if a not in source.attributes]
        if missing:
            raise KeyError(f"Source attributes not found: {missing}")

        if source.n_points == 0:
            out = reference.copy(deep=True)
            tag = f"{prefix}_" if prefix else ""
            for attr in attrs:
                out.attributes[f"{tag}{attr}"] = np.full(reference.n_points, np.nan, dtype=np.float64)
            if include_distance:
                out.attributes[f"{tag}nn_distance"] = np.full(reference.n_points, np.nan, dtype=np.float32)
                out.attributes[f"{tag}matched"] = np.zeros(reference.n_points, dtype=np.uint8)
            return out, {
                "matched_fraction": 0.0,
                "median_nn_distance": float("nan"),
                "p95_nn_distance": float("nan"),
                "max_distance": float(max_distance) if max_distance is not None else float("nan"),
            }

        tree = cKDTree(source.xyz)
        distances, indices = tree.query(reference.xyz, k=k, workers=-1)
        if k == 1:
            distances = distances[:, None]
            indices = indices[:, None]

        nearest = distances[:, 0]
        matched = np.isfinite(nearest)
        if max_distance is not None:
            matched &= nearest <= max_distance

        out = reference.copy(deep=True)
        tag = f"{prefix}_" if prefix else ""

        for attr in attrs:
            values = np.asarray(source.attributes[attr])
            if not np.issubdtype(values.dtype, np.number):
                raise TypeError(f"Attribute {attr!r} is not numeric and cannot be fused")
            gathered = values[indices]
            result = np.full(reference.n_points, np.nan, dtype=np.float64)

            if k == 1 or weighting == "nearest":
                result[matched] = gathered[matched, 0].astype(np.float64)
            elif weighting == "mean":
                valid = np.isfinite(gathered)
                tmp = np.divide(
                    np.nansum(gathered.astype(float), axis=1),
                    np.sum(valid, axis=1),
                    out=np.full(reference.n_points, np.nan),
                    where=np.sum(valid, axis=1) > 0,
                )
                result[matched] = tmp[matched]
            else:  # inverse-distance weighting
                d = distances.astype(np.float64)
                finite = np.isfinite(d)
                if max_distance is not None:
                    finite &= d <= max_distance
                exact = finite & (d <= 1e-12)
                weights = np.zeros_like(d, dtype=np.float64)
                weights[finite & ~exact] = 1.0 / np.maximum(d[finite & ~exact], 1e-12)
                any_exact = np.any(exact, axis=1)
                weights[any_exact] = exact[any_exact].astype(float)
                numerator = np.sum(weights * gathered.astype(float), axis=1)
                denominator = np.sum(weights, axis=1)
                tmp = np.divide(
                    numerator,
                    denominator,
                    out=np.full(reference.n_points, np.nan),
                    where=denominator > 0,
                )
                result[matched] = tmp[matched]

            out.attributes[f"{tag}{attr}"] = result

        if include_distance:
            out.attributes[f"{tag}nn_distance"] = nearest.astype(np.float32)
            out.attributes[f"{tag}matched"] = matched.astype(np.uint8)

        stats = {
            "matched_fraction": float(np.mean(matched)),
            "median_nn_distance": float(np.median(nearest[matched])) if np.any(matched) else float("nan"),
            "p95_nn_distance": float(np.percentile(nearest[matched], 95)) if np.any(matched) else float("nan"),
            "max_distance": float(max_distance) if max_distance is not None else float("nan"),
        }
        return out, stats

    def fuse_point_clouds(
        self,
        clouds: Sequence[PointCloudData],
        *,
        mode: str = "concatenate",
        voxel_size: float = 0.25,
        origin: np.ndarray | None = None,
    ) -> PointCloudData:
        """Fuse registered point clouds by concatenation or voxel aggregation."""
        if not clouds:
            raise ValueError("At least one point cloud is required")
        mode = mode.lower()
        if mode == "concatenate":
            return self._concatenate_clouds(clouds)
        if mode == "voxel":
            return self._voxel_fusion(clouds, voxel_size=voxel_size, origin=origin)
        raise ValueError("mode must be 'concatenate' or 'voxel'")

    @staticmethod
    def _concatenate_clouds(clouds: Sequence[PointCloudData]) -> PointCloudData:
        names = sorted(set().union(*(c.attributes.keys() for c in clouds)))
        xyz = np.vstack([c.xyz for c in clouds])
        attrs: dict[str, np.ndarray] = {}
        source_id = []
        source_map: dict[int, str] = {}

        for name in names:
            pieces = []
            for cloud in clouds:
                if name in cloud.attributes:
                    pieces.append(np.asarray(cloud.attributes[name], dtype=np.float64))
                else:
                    pieces.append(np.full(cloud.n_points, np.nan, dtype=np.float64))
            attrs[name] = np.concatenate(pieces)

        for i, cloud in enumerate(clouds):
            source_map[i] = cloud.source or f"source_{i}"
            source_id.append(np.full(cloud.n_points, i, dtype=np.uint16))
        attrs["source_id"] = np.concatenate(source_id)

        return PointCloudData(
            xyz=xyz,
            attributes=attrs,
            crs=clouds[0].crs,
            source="fused_concatenated",
            metadata={"fusion_mode": "concatenate", "source_map": source_map},
        )

    @staticmethod
    def _voxel_fusion(
        clouds: Sequence[PointCloudData],
        *,
        voxel_size: float,
        origin: np.ndarray | None,
    ) -> PointCloudData:
        if voxel_size <= 0:
            raise ValueError("voxel_size must be > 0")
        all_xyz = np.vstack([c.xyz for c in clouds])
        if origin is None:
            origin = np.min(all_xyz, axis=0)
        origin = np.asarray(origin, dtype=np.float64)
        vox = np.floor((all_xyz - origin) / voxel_size).astype(np.int64)
        unique, inv, counts = np.unique(vox, axis=0, return_inverse=True, return_counts=True)
        n_vox = len(unique)

        xyz_sum = np.zeros((n_vox, 3), dtype=np.float64)
        np.add.at(xyz_sum, inv, all_xyz)
        xyz = xyz_sum / counts[:, None]

        attrs: dict[str, np.ndarray] = {"point_count": counts.astype(np.int32)}
        attr_names = sorted(set().union(*(c.attributes.keys() for c in clouds)))
        offset = 0
        for name in attr_names:
            sums = np.zeros(n_vox, dtype=np.float64)
            nums = np.zeros(n_vox, dtype=np.int64)
            for cloud in clouds:
                n = cloud.n_points
                if name in cloud.attributes and np.issubdtype(np.asarray(cloud.attributes[name]).dtype, np.number):
                    vals = np.asarray(cloud.attributes[name], dtype=np.float64)
                    ok = np.isfinite(vals)
                    ids = inv[offset:offset+n][ok]
                    np.add.at(sums, ids, vals[ok])
                    np.add.at(nums, ids, 1)
                offset += n
            attrs[name] = np.divide(
                sums,
                nums,
                out=np.full(n_vox, np.nan, dtype=np.float64),
                where=nums > 0,
            )
            offset = 0

        offset = 0
        source_map: dict[int, str] = {}
        for i, cloud in enumerate(clouds):
            n = cloud.n_points
            source_map[i] = cloud.source or f"source_{i}"
            occupancy = np.zeros(n_vox, dtype=np.int32)
            np.add.at(occupancy, inv[offset:offset+n], 1)
            attrs[f"source_{i}_count"] = occupancy
            offset += n

        return PointCloudData(
            xyz=xyz,
            attributes=attrs,
            crs=clouds[0].crs,
            source="fused_voxel",
            metadata={
                "fusion_mode": "voxel",
                "voxel_size": float(voxel_size),
                "voxel_origin": origin.tolist(),
                "source_map": source_map,
            },
        )


def _apply_transform(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    return (matrix[:3, :3] @ points.T).T + matrix[:3, 3]


def _best_fit_rigid_transform(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Kabsch/SVD rigid transform mapping source points to target points."""
    src_centroid = np.mean(source, axis=0)
    dst_centroid = np.mean(target, axis=0)
    src0 = source - src_centroid
    dst0 = target - dst_centroid
    h = src0.T @ dst0
    u, _, vt = np.linalg.svd(h)
    r = vt.T @ u.T
    if np.linalg.det(r) < 0:
        vt[-1, :] *= -1
        r = vt.T @ u.T
    t = dst_centroid - r @ src_centroid
    out = np.eye(4)
    out[:3, :3] = r
    out[:3, 3] = t
    return out


def _safe_extra_name(name: str) -> str:
    safe = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
    return (safe or "value")[:32]



def _query_point_tile_task(task: Mapping[str, Any]):
    """Nearest-neighbour multi-attribute transfer for one reference XY tile."""
    from pathlib import Path

    tid = int(task["tile_id"])
    nx = int(task["nx"])
    ny = int(task["ny"])
    tr = tid // nx
    tc = tid % nx
    xyz_q = np.asarray(task["xyz_q"], dtype=np.float64)
    max_distance = float(task["max_distance"])
    result: dict[str, dict[str, Any]] = {}
    for source_name, source_info in task["source_info"].items():
        src_dir = Path(source_info["dir"])
        source_dtype = np.dtype(source_info["dtype_descr"])
        pieces: list[np.ndarray] = []
        for rr in range(max(0, tr - 1), min(ny, tr + 2)):
            for cc in range(max(0, tc - 1), min(nx, tc + 2)):
                path = src_dir / f"{rr * nx + cc:08d}.bin"
                if path.exists() and path.stat().st_size:
                    pieces.append(np.fromfile(path, dtype=source_dtype))
        if not pieces:
            continue
        rec = pieces[0] if len(pieces) == 1 else np.concatenate(pieces)
        xyz_s = np.column_stack((rec["x"], rec["y"], rec["z"]))
        distances, nearest = cKDTree(xyz_s).query(xyz_q, k=1, workers=1)
        matched = np.isfinite(distances) & (distances <= max_distance)
        dst = np.full(len(xyz_q), np.nan, dtype=np.float32)
        if np.any(matched):
            dst[matched] = distances[matched].astype(np.float32)
        values: dict[str, np.ndarray] = {}
        for output_name, field_name in source_info["field_map"].items():
            vals = np.full(len(xyz_q), np.nan, dtype=np.float32)
            if np.any(matched):
                vals[matched] = np.asarray(rec[field_name][nearest[matched]], dtype=np.float32)
            values[str(output_name)] = vals
        result[source_name] = {
            "values": values,
            "distance": dst,
            "matched": matched.astype(np.uint8),
        }
    return np.asarray(task["ref_idx"], dtype=np.int64), result


def stream_reference_attribute_fusion(
    reference_paths: str | Path | Iterable[str | Path],
    sources: Mapping[str, Mapping[str, Any]],
    output_path: str,
    *,
    ingestion: Any,
    processing_limits: Any,
    max_distance: float,
    reference_crs: str | None = None,
    reference_text_options: Mapping[str, Any] | None = None,
    aoi: Any | None = None,
) -> dict[str, Any]:
    """Out-of-core nearest-neighbour transfer of arbitrary point attributes.

    The reference geometry may be one LAS/LAZ file, a collection of LAS/LAZ
    tiles, or one numeric TXT/CSV/XYZ file. Sources may independently be
    LAS/LAZ or numeric text point clouds. Text schemas are supplied by each
    source's ``text_options`` mapping. The output is LAS/LAZ so arbitrary
    physical source variables can be written as Extra Bytes dimensions.

    No spectral, thermal or sensor-specific column names are inferred.
    """
    from collections import defaultdict
    from concurrent.futures import wait, FIRST_COMPLETED
    from copy import deepcopy
    from pathlib import Path
    import shutil
    import tempfile
    from .aoi import point_mask, buffered_geometry

    try:
        import laspy
    except ImportError as exc:
        raise ImportError("Streaming point-cloud fusion requires laspy") from exc

    if isinstance(reference_paths, (str, Path)):
        reference_files = [Path(reference_paths)]
    else:
        reference_files = [Path(x) for x in reference_paths]
    if not reference_files:
        raise ValueError("At least one reference point-cloud file is required")
    for path in reference_files:
        if not path.is_file():
            raise FileNotFoundError(f"Reference point cloud does not exist: {path}")
        if path.suffix.lower() not in {".las", ".laz", ".txt", ".csv", ".xyz"}:
            raise ValueError(
                f"Streaming reference format {path.suffix!r} is not supported; "
                "use LAS/LAZ or numeric TXT/CSV/XYZ"
            )

    output_path = Path(output_path)
    if output_path.suffix.lower() not in {".las", ".laz"}:
        raise ValueError("Streaming point-cloud fusion output must be .las or .laz")
    if not sources:
        raise ValueError("At least one source is required")
    if max_distance is None or float(max_distance) <= 0:
        raise ValueError("Out-of-core nearest transfer requires a positive max_distance")

    normalized_sources: dict[str, dict[str, Any]] = {}
    for source_name, raw in sources.items():
        spec = dict(raw)
        attrs = spec.get("attributes")
        if attrs is None:
            if spec.get("attribute") is None:
                raise ValueError(f"Source {source_name!r} has no attributes configured")
            attrs = {str(spec.get("output_name") or source_name): str(spec["attribute"])}
        attrs = {str(k): str(v) for k, v in dict(attrs).items()}
        if not attrs:
            raise ValueError(f"Source {source_name!r} has an empty attribute mapping")
        path = Path(spec["path"])
        if path.suffix.lower() not in {".las", ".laz", ".txt", ".csv", ".xyz"}:
            raise ValueError(
                f"Streaming source {source_name!r} has unsupported format {path.suffix!r}"
            )
        spec["attributes"] = attrs
        normalized_sources[str(source_name)] = spec
    sources = normalized_sources

    # Aggregate reference bounds/count without merging the original files.
    ref_infos: list[dict[str, Any]] = []
    total_ref = 0
    ref_min = np.array([np.inf, np.inf, np.inf], dtype=np.float64)
    ref_max = np.array([-np.inf, -np.inf, -np.inf], dtype=np.float64)
    resolved_reference_crs: str | None = None
    for ref_path in reference_files:
        opts = reference_text_options if ref_path.suffix.lower() in {".txt", ".csv", ".xyz"} else None
        explicit_crs = reference_crs if ref_path.suffix.lower() in {".txt", ".csv", ".xyz"} else None
        info = ingestion.inspect_point_cloud(ref_path, crs=explicit_crs, text_options=opts)
        ref_infos.append(info)
        n = int(info.get("point_count", 0) or 0)
        total_ref += n
        mins = np.asarray(info.get("mins", [np.nan, np.nan, np.nan]), dtype=np.float64)
        maxs = np.asarray(info.get("maxs", [np.nan, np.nan, np.nan]), dtype=np.float64)
        if np.all(np.isfinite(mins)):
            ref_min = np.minimum(ref_min, mins)
        if np.all(np.isfinite(maxs)):
            ref_max = np.maximum(ref_max, maxs)
        info_crs = info.get("crs")
        if info_crs and resolved_reference_crs is None:
            resolved_reference_crs = str(info_crs)
        elif info_crs and resolved_reference_crs:
            try:
                from pyproj import CRS
                if CRS.from_user_input(info_crs) != CRS.from_user_input(resolved_reference_crs):
                    raise ValueError("Reference point-cloud files use different CRS definitions")
            except Exception as exc:
                if str(info_crs) != str(resolved_reference_crs):
                    raise ValueError("Reference point-cloud files use different CRS definitions") from exc
    if total_ref <= 0 or not np.all(np.isfinite(ref_min[:2])) or not np.all(np.isfinite(ref_max[:2])):
        raise ValueError("Reference point cloud has no valid points/bounds")

    if aoi is not None:
        from shapely.geometry import box

        overlap = box(float(ref_min[0]), float(ref_min[1]), float(ref_max[0]), float(ref_max[1])).intersection(aoi.geometry)
        if overlap.is_empty:
            raise ValueError("AOI does not overlap the reference point cloud")
        minx, miny, maxx, maxy = map(float, overlap.bounds)
        ref_min[0], ref_min[1] = minx, miny
        ref_max[0], ref_max[1] = maxx, maxy

    margin = float(max_distance) * 1.5
    origin_x = float(ref_min[0] - margin)
    origin_y = float(ref_min[1] - margin)
    extent_x = float(ref_max[0] - ref_min[0] + 2 * margin)
    extent_y = float(ref_max[1] - ref_min[1] + 2 * margin)

    limits = processing_limits
    budget = int(limits.memory_budget_bytes())
    workers = limits.worker_count(estimated_task_bytes=256 * 1024**2)

    densities: list[float] = []
    estimated_spool = 0
    source_infos: dict[str, dict[str, Any]] = {}
    for source_name, spec in sources.items():
        info = ingestion.inspect_point_cloud(
            spec["path"], crs=spec.get("crs"), text_options=spec.get("text_options")
        )
        source_infos[source_name] = info
        n = int(info.get("point_count", 0) or 0)
        estimated_spool += n * (8 * 3 + 4 * len(spec["attributes"]))
        mins = np.asarray(info.get("mins", [0, 0, 0]), dtype=float)
        maxs = np.asarray(info.get("maxs", [1, 1, 1]), dtype=float)
        area = max(1e-6, float((maxs[0] - mins[0]) * (maxs[1] - mins[1])))
        if n:
            densities.append(n / area)
    density = max(densities) if densities else 1.0
    max_attribute_count = max(len(spec["attributes"]) for spec in sources.values())
    record_bytes = 8 * 3 + 4 * max_attribute_count
    per_worker = max(128 * 1024**2, budget // max(1, workers))
    target_points = max(50_000, int((per_worker * 0.10) / max(1, record_bytes)))
    auto_side = float(np.sqrt(target_points / max(density, 1e-9)))
    min_side = max(float(max_distance) * 4.0, 1.0)
    configured = getattr(limits, "point_tile_size", None)
    tile_side = float(configured) if configured is not None else min(50.0, max(min_side, auto_side))
    tile_side = max(min_side, tile_side)

    nx = max(1, int(np.ceil(extent_x / tile_side)))
    ny = max(1, int(np.ceil(extent_y / tile_side)))

    tmp_parent = Path(limits.temp_dir) if limits.temp_dir else output_path.parent
    tmp_parent.mkdir(parents=True, exist_ok=True)
    free = shutil.disk_usage(tmp_parent).free
    if estimated_spool * 1.15 > free:
        raise OSError(
            f"Insufficient temporary disk space for point fusion: about "
            f"{estimated_spool/(1024**3):.1f} GiB required, {free/(1024**3):.1f} GiB free"
        )

    # Preserve the original reference LAS dimensions when all reference tiles
    # use an identical LAS point format. Otherwise emit a generic LAS 1.4 point
    # geometry with the transferred variables as Extra Bytes.
    preserve_reference_records = all(p.suffix.lower() in {".las", ".laz"} for p in reference_files)
    out_header = None
    if preserve_reference_records:
        headers = []
        for ref_path in reference_files:
            with laspy.open(ref_path) as rr:
                headers.append(deepcopy(rr.header))
        first = headers[0]
        signature = (
            int(first.point_format.id),
            tuple(str(x) for x in first.point_format.dimension_names),
            str(first.version),
        )
        for h in headers[1:]:
            sig = (int(h.point_format.id), tuple(str(x) for x in h.point_format.dimension_names), str(h.version))
            if sig != signature:
                preserve_reference_records = False
                break
        if preserve_reference_records:
            out_header = first

    if out_header is None:
        out_header = laspy.LasHeader(point_format=6, version="1.4")
        try:
            from pyproj import CRS
            crs_obj = CRS.from_user_input(resolved_reference_crs or reference_crs) if (resolved_reference_crs or reference_crs) else None
            if crs_obj is not None:
                out_header.add_crs(crs_obj)
                out_header.scales = np.array([1e-8, 1e-8, 0.001]) if crs_obj.is_geographic else np.array([0.001, 0.001, 0.001])
            else:
                out_header.scales = np.array([0.001, 0.001, 0.001])
        except Exception:
            out_header.scales = np.array([0.001, 0.001, 0.001])
        out_header.offsets = np.floor(ref_min)

    with tempfile.TemporaryDirectory(prefix="uav_biodiv_pointfusion_", dir=tmp_parent) as tmp:
        root = Path(tmp)
        source_dirs: dict[str, Path] = {}
        source_spooled: dict[str, int] = {}
        source_dtypes: dict[str, np.dtype] = {}
        source_field_maps: dict[str, dict[str, str]] = {}

        # Partition each source once, no matter whether it is LAS/LAZ or text.
        for source_name, spec in sources.items():
            src_dir = root / _safe_extra_name(source_name)
            src_dir.mkdir(parents=True, exist_ok=True)
            source_dirs[source_name] = src_dir
            attributes = dict(spec["attributes"])
            requested_dimensions = list(dict.fromkeys(attributes.values()))
            field_map = {output_name: f"v{i}" for i, output_name in enumerate(attributes)}
            dtype_fields: list[tuple[str, Any]] = [("x", np.float64), ("y", np.float64), ("z", np.float64)]
            dtype_fields.extend((field, np.float32) for field in field_map.values())
            source_dtype = np.dtype(dtype_fields)
            source_dtypes[source_name] = source_dtype
            source_field_maps[source_name] = field_map

            transform = np.asarray(spec["transformation"], dtype=np.float64)
            if transform.shape != (4, 4):
                raise ValueError(f"Source {source_name!r} transformation must be 4x4")
            source_aoi = buffered_geometry(aoi, float(max_distance)) if aoi is not None else None
            count = 0
            for chunk in ingestion.iter_point_cloud_chunks(
                spec["path"],
                attributes=requested_dimensions,
                crs=spec.get("crs"),
                text_options=spec.get("text_options"),
            ):
                lut = {k.lower(): k for k in chunk.attributes}
                resolved: dict[str, str] = {}
                for output_name, requested in attributes.items():
                    actual = requested if requested in chunk.attributes else lut.get(requested.lower())
                    if actual is None:
                        raise KeyError(
                            f"Source {source_name!r} does not contain configured attribute {requested!r}"
                        )
                    resolved[output_name] = actual

                xyz = _apply_transform(chunk.xyz, transform)
                if source_aoi is not None:
                    aoi_inside = point_mask(xyz, source_aoi)
                else:
                    aoi_inside = np.ones(len(xyz), dtype=bool)
                inside = (
                    aoi_inside
                    & (xyz[:, 0] >= origin_x)
                    & (xyz[:, 0] < origin_x + nx * tile_side)
                    & (xyz[:, 1] >= origin_y)
                    & (xyz[:, 1] < origin_y + ny * tile_side)
                    & np.all(np.isfinite(xyz), axis=1)
                )
                if not np.any(inside):
                    continue
                xyz = xyz[inside]
                values = {
                    output_name: np.asarray(chunk.attributes[actual], dtype=np.float32)[inside]
                    for output_name, actual in resolved.items()
                }
                tx = np.floor((xyz[:, 0] - origin_x) / tile_side).astype(np.int64)
                ty = np.floor((xyz[:, 1] - origin_y) / tile_side).astype(np.int64)
                tile_id = ty * nx + tx
                order = np.argsort(tile_id, kind="stable")
                ids = tile_id[order]
                unique, starts, counts = np.unique(ids, return_index=True, return_counts=True)
                for tid, start_i, n in zip(unique, starts, counts, strict=True):
                    sel = order[start_i : start_i + n]
                    rec = np.empty(int(n), dtype=source_dtype)
                    rec["x"] = xyz[sel, 0]
                    rec["y"] = xyz[sel, 1]
                    rec["z"] = xyz[sel, 2]
                    for output_name, field_name in field_map.items():
                        rec[field_name] = values[output_name][sel]
                    with (src_dir / f"{int(tid):08d}.bin").open("ab") as fh:
                        rec.tofile(fh)
                    count += int(n)
            source_spooled[source_name] = count

        existing = {str(x).lower() for x in out_header.point_format.dimension_names}
        output_dims: dict[str, dict[str, Any]] = {}

        def unique_name(candidate: str) -> str:
            candidate = _safe_extra_name(candidate)
            root_name = candidate[:28]
            name = candidate
            i = 1
            while name.lower() in existing:
                name = _safe_extra_name(f"{root_name}_{i}")
                i += 1
            existing.add(name.lower())
            return name

        for source_name, spec in sources.items():
            attr_dims: dict[str, str] = {}
            for output_name in spec["attributes"]:
                value_name = unique_name(output_name)
                out_header.add_extra_dim(laspy.ExtraBytesParams(name=value_name, type=np.float32))
                attr_dims[output_name] = value_name
            base = _safe_extra_name(str(spec.get("quality_prefix") or source_name))
            distance_name = unique_name(f"{base}_dist")
            matched_name = unique_name(f"{base}_ok")
            out_header.add_extra_dim(laspy.ExtraBytesParams(name=distance_name, type=np.float32))
            out_header.add_extra_dim(laspy.ExtraBytesParams(name=matched_name, type=np.uint8))
            output_dims[source_name] = {
                "attributes": attr_dims,
                "distance": distance_name,
                "matched": matched_name,
            }

        output_path.parent.mkdir(parents=True, exist_ok=True)
        matched_counts = defaultdict(int)
        distance_values: dict[str, list[np.ndarray]] = defaultdict(list)
        source_info = {
            source_name: {
                "dir": str(source_dirs[source_name]),
                "dtype_descr": source_dtypes[source_name].descr,
                "field_map": source_field_maps[source_name],
            }
            for source_name in sources
        }

        reference_written = 0

        def process_reference_chunk(ref_xyz: np.ndarray, ref_points: Any | None, writer: Any) -> None:
            nonlocal reference_written
            if aoi is not None:
                keep = point_mask(ref_xyz, aoi)
                if not np.any(keep):
                    return
                ref_xyz = ref_xyz[keep]
                if ref_points is not None:
                    ref_points = ref_points[keep]
            n_ref = len(ref_xyz)
            reference_written += int(n_ref)
            tx = np.floor((ref_xyz[:, 0] - origin_x) / tile_side).astype(np.int64)
            ty = np.floor((ref_xyz[:, 1] - origin_y) / tile_side).astype(np.int64)
            valid_tile = (tx >= 0) & (tx < nx) & (ty >= 0) & (ty < ny)
            tile_id = ty * nx + tx

            values_out = {
                source_name: {
                    output_name: np.full(n_ref, np.nan, dtype=np.float32)
                    for output_name in spec["attributes"]
                }
                for source_name, spec in sources.items()
            }
            dist_out = {name: np.full(n_ref, np.nan, dtype=np.float32) for name in sources}
            ok_out = {name: np.zeros(n_ref, dtype=np.uint8) for name in sources}

            valid_indices = np.flatnonzero(valid_tile)
            if valid_indices.size:
                ids = tile_id[valid_indices]
                order = np.argsort(ids, kind="stable")
                ids_sorted = ids[order]
                unique, starts, counts = np.unique(ids_sorted, return_index=True, return_counts=True)
                task_bytes = max(
                    64 * 1024**2,
                    int(len(valid_indices) / max(1, len(unique))) * (96 + 16 * max_attribute_count),
                )
                q_workers = limits.effective_worker_count(len(unique), estimated_task_bytes=task_bytes)

                def make_task(tid: int, ref_idx: np.ndarray) -> dict[str, Any]:
                    return {
                        "tile_id": int(tid),
                        "ref_idx": np.asarray(ref_idx, dtype=np.int64),
                        "xyz_q": ref_xyz[ref_idx].copy(),
                        "nx": nx,
                        "ny": ny,
                        "source_info": source_info,
                        "max_distance": float(max_distance),
                    }

                groups = []
                for tid, start_i, count_i in zip(unique, starts, counts, strict=True):
                    local = order[start_i : start_i + count_i]
                    ref_idx = valid_indices[local]
                    groups.append((int(tid), ref_idx))

                def consume(ref_idx_out: np.ndarray, query_result: Mapping[str, Any]) -> None:
                    for source_name, payload in query_result.items():
                        for output_name, vals in payload["values"].items():
                            values_out[source_name][output_name][ref_idx_out] = vals
                        dist_out[source_name][ref_idx_out] = payload["distance"]
                        ok_out[source_name][ref_idx_out] = payload["matched"]

                if q_workers == 1:
                    for tid, ref_idx in groups:
                        consume(*_query_point_tile_task(make_task(tid, ref_idx)))
                else:
                    with process_pool(q_workers) as executor:
                        pending = set()
                        iterator = iter(groups)
                        max_in_flight = limits.in_flight_count(q_workers)

                        def submit_more():
                            while len(pending) < max_in_flight:
                                try:
                                    tid, ref_idx = next(iterator)
                                except StopIteration:
                                    break
                                pending.add(executor.submit(_query_point_tile_task, make_task(tid, ref_idx)))

                        submit_more()
                        while pending:
                            done, pending = wait(pending, return_when=FIRST_COMPLETED)
                            for fut in done:
                                consume(*fut.result())
                            submit_more()

            out_points = laspy.ScaleAwarePointRecord.zeros(n_ref, header=out_header)
            if ref_points is not None and preserve_reference_records:
                for field in ref_points.array.dtype.names or ():
                    if field in (out_points.array.dtype.names or ()):
                        out_points.array[field] = ref_points.array[field]
            else:
                out_points.x = ref_xyz[:, 0]
                out_points.y = ref_xyz[:, 1]
                out_points.z = ref_xyz[:, 2]

            for source_name, dims in output_dims.items():
                for output_name, dimension_name in dims["attributes"].items():
                    out_points[dimension_name] = values_out[source_name][output_name]
                out_points[dims["distance"]] = dist_out[source_name]
                out_points[dims["matched"]] = ok_out[source_name]
                matched_counts[source_name] += int(np.count_nonzero(ok_out[source_name]))
                finite_d = dist_out[source_name][np.isfinite(dist_out[source_name])]
                if finite_d.size:
                    if finite_d.size > 50_000:
                        step = max(1, finite_d.size // 50_000)
                        finite_d = finite_d[::step][:50_000]
                    distance_values[source_name].append(finite_d.copy())
            writer.write_points(out_points)

        chunk_points = limits.point_chunk_points(bytes_per_point=max(64, int(out_header.point_format.size) * 2))
        with laspy.open(output_path, mode="w", header=out_header) as writer:
            for ref_path in reference_files:
                if ref_path.suffix.lower() in {".las", ".laz"}:
                    with laspy.open(ref_path) as ref_reader:
                        for ref_points in ref_reader.chunk_iterator(chunk_points):
                            ref_xyz = np.column_stack((ref_points.x, ref_points.y, ref_points.z)).astype(np.float64, copy=False)
                            process_reference_chunk(ref_xyz, ref_points, writer)
                else:
                    for ref_chunk in ingestion.iter_point_cloud_chunks(
                        ref_path,
                        attributes=[],
                        crs=reference_crs or resolved_reference_crs,
                        text_options=reference_text_options,
                        chunk_points=chunk_points,
                    ):
                        process_reference_chunk(ref_chunk.xyz, None, writer)

        stats: dict[str, dict[str, float]] = {}
        for source_name in sources:
            parts = distance_values.get(source_name, [])
            distances = np.concatenate(parts) if parts else np.array([], dtype=np.float32)
            stats[source_name] = {
                "matched_fraction": (matched_counts[source_name] / reference_written) if reference_written else 0.0,
                "median_nn_distance": float(np.median(distances)) if distances.size else float("nan"),
                "p95_nn_distance": float(np.percentile(distances, 95)) if distances.size else float("nan"),
                "max_distance": float(max_distance),
            }

        if reference_written <= 0:
            try:
                output_path.unlink()
            except OSError:
                pass
            raise ValueError("AOI contains no reference points")

        return {
            "output_path": str(output_path),
            "reference_points": int(reference_written),
            "reference_points_header_total": int(total_ref),
            "aoi": ({**aoi.to_metadata(), "output_filtered": True} if aoi is not None else {"enabled": False}),
            "reference_files": [str(p) for p in reference_files],
            "reference_file_count": len(reference_files),
            "reference_records_preserved": bool(preserve_reference_records),
            "source_points_spooled": source_spooled,
            "source_formats": {name: info.get("format") for name, info in source_infos.items()},
            "tile_side": tile_side,
            "parallel_backend": "process" if workers > 1 else "serial",
            "partition_stage": "serial_streaming_io",
            "workers": workers,
            "memory_budget_gb": budget / (1024**3),
            "resource_plan": limits.resource_plan(estimated_task_bytes=256 * 1024**2),
            "transfer_stats": stats,
            "output_dimensions": output_dims,
            "attribute_count": {name: len(spec["attributes"]) for name, spec in sources.items()},
        }
