from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
from typing import Any, Mapping


_GIB = 1024 ** 3
_MIB = 1024 ** 2


@dataclass(slots=True)
class ProcessingLimits:
    """Hardware-adaptive resource policy for bounded parallel processing.

    ``mode="auto"`` is the default and is intended for normal users.  It
    derives a safe RAM budget, worker count, point-cloud chunk size and raster
    tile size from the machine that is actually running the workflow.  Every
    auto value can still be capped explicitly.

    ``mode="manual"`` uses the supplied caps directly, while still refusing
    plans that exceed currently available RAM.  The processing core therefore
    does not assume a specific workstation size.
    """

    mode: str = "auto"
    max_ram_gb: float | None = None
    max_ram_fraction: float | None = None
    reserve_ram_gb: float | None = None
    max_workers: int | None = None
    reserve_cpus: int = 1
    auto_worker_cap: int = 16
    point_chunk_size: int | None = None
    raster_tile_size: int | None = None
    point_tile_size: float | None = None
    max_in_flight: int | None = None
    min_parallel_tasks: int = 4
    min_tasks_per_worker: int = 2
    temp_dir: str | None = None

    def __post_init__(self) -> None:
        self.mode = str(self.mode).lower().strip()
        if self.mode not in {"auto", "manual"}:
            raise ValueError("mode must be 'auto' or 'manual'")
        if self.max_ram_gb is not None and self.max_ram_gb <= 0:
            raise ValueError("max_ram_gb must be > 0 or None")
        if self.max_ram_fraction is not None and not 0 < float(self.max_ram_fraction) <= 1:
            raise ValueError("max_ram_fraction must be in (0, 1] or None")
        if self.reserve_ram_gb is not None and float(self.reserve_ram_gb) < 0:
            raise ValueError("reserve_ram_gb must be >= 0 or None")
        if self.max_workers is not None and int(self.max_workers) < 1:
            raise ValueError("max_workers must be >= 1 or None")
        if int(self.reserve_cpus) < 0:
            raise ValueError("reserve_cpus must be >= 0")
        if int(self.auto_worker_cap) < 1:
            raise ValueError("auto_worker_cap must be >= 1")
        if self.point_chunk_size is not None and int(self.point_chunk_size) < 1:
            raise ValueError("point_chunk_size must be >= 1 or None")
        if self.raster_tile_size is not None and int(self.raster_tile_size) < 32:
            raise ValueError("raster_tile_size must be >= 32 or None")
        if self.point_tile_size is not None and float(self.point_tile_size) <= 0:
            raise ValueError("point_tile_size must be > 0 or None")
        if self.max_in_flight is not None and int(self.max_in_flight) < 1:
            raise ValueError("max_in_flight must be >= 1 or None")
        if int(self.min_parallel_tasks) < 2:
            raise ValueError("min_parallel_tasks must be >= 2")
        if int(self.min_tasks_per_worker) < 1:
            raise ValueError("min_tasks_per_worker must be >= 1")
        if self.temp_dir is not None:
            self.temp_dir = str(Path(self.temp_dir))

    @classmethod
    def from_mapping(cls, value: Mapping[str, Any] | None) -> "ProcessingLimits":
        return cls(**dict(value or {}))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @staticmethod
    def system_memory() -> tuple[int, int]:
        """Return ``(total_bytes, available_bytes)`` with portable fallbacks."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            return int(vm.total), int(vm.available)
        except Exception:
            if os.name == "nt":
                try:
                    import ctypes

                    class MEMORYSTATUSEX(ctypes.Structure):
                        _fields_ = [
                            ("dwLength", ctypes.c_ulong),
                            ("dwMemoryLoad", ctypes.c_ulong),
                            ("ullTotalPhys", ctypes.c_ulonglong),
                            ("ullAvailPhys", ctypes.c_ulonglong),
                            ("ullTotalPageFile", ctypes.c_ulonglong),
                            ("ullAvailPageFile", ctypes.c_ulonglong),
                            ("ullTotalVirtual", ctypes.c_ulonglong),
                            ("ullAvailVirtual", ctypes.c_ulonglong),
                            ("sullAvailExtendedVirtual", ctypes.c_ulonglong),
                        ]

                    stat = MEMORYSTATUSEX()
                    stat.dwLength = ctypes.sizeof(stat)
                    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(stat))
                    return int(stat.ullTotalPhys), int(stat.ullAvailPhys)
                except Exception:
                    pass
            try:
                page = os.sysconf("SC_PAGE_SIZE")
                total = page * os.sysconf("SC_PHYS_PAGES")
                avail = page * os.sysconf("SC_AVPHYS_PAGES")
                return int(total), int(avail)
            except Exception:
                return 8 * _GIB, 2 * _GIB

    @staticmethod
    def system_cpu_counts() -> tuple[int, int]:
        """Return ``(physical_or_logical, logical)`` CPU counts."""
        logical = max(1, os.cpu_count() or 1)
        physical = None
        try:
            import psutil
            physical = psutil.cpu_count(logical=False)
        except Exception:
            physical = None
        preferred = int(physical) if physical else logical
        return max(1, preferred), logical

    def _auto_ram_fraction(self) -> float:
        return float(self.max_ram_fraction if self.max_ram_fraction is not None else 0.70)

    def reserve_ram_bytes(self) -> int:
        total, _ = self.system_memory()
        if self.reserve_ram_gb is not None:
            return int(float(self.reserve_ram_gb) * _GIB)
        # Keep a meaningful OS/application reserve on small machines while not
        # withholding excessive RAM on large servers.
        automatic = max(2 * _GIB, int(total * 0.20))
        return min(automatic, 16 * _GIB)

    def memory_budget_bytes(self) -> int:
        """Live package RAM budget after physical and free-memory caps."""
        total, available = self.system_memory()
        fraction_cap = int(total * self._auto_ram_fraction())
        absolute_cap = int(float(self.max_ram_gb) * _GIB) if self.max_ram_gb is not None else fraction_cap
        configured_cap = min(fraction_cap, absolute_cap)
        free_cap = max(256 * _MIB, int(available) - self.reserve_ram_bytes())
        return max(256 * _MIB, min(configured_cap, free_cap))

    def cpu_budget(self) -> int:
        preferred, _ = self.system_cpu_counts()
        auto = max(1, preferred - int(self.reserve_cpus))
        if self.max_workers is None:
            auto = min(auto, int(self.auto_worker_cap))
        else:
            auto = min(auto, int(self.max_workers))
        return max(1, auto)

    def worker_count(self, *, estimated_task_bytes: int = 256 * _MIB) -> int:
        """Choose real parallel workers from both CPU and live RAM."""
        task = max(32 * _MIB, int(estimated_task_bytes))
        by_ram = max(1, int(self.memory_budget_bytes() // task))
        return max(1, min(self.cpu_budget(), by_ram))

    def effective_worker_count(
        self,
        task_count: int,
        *,
        estimated_task_bytes: int = 256 * _MIB,
    ) -> int:
        """Choose workers from CPU, RAM and the amount of ready work.

        Small batches remain serial and larger batches avoid creating more
        processes than can receive a useful amount of work. This reduces idle
        child processes on both small laptops and large workstations.
        """
        tasks = max(0, int(task_count))
        if tasks < int(self.min_parallel_tasks):
            return 1
        base = self.worker_count(estimated_task_bytes=estimated_task_bytes)
        by_work = max(1, (tasks + int(self.min_tasks_per_worker) - 1) // int(self.min_tasks_per_worker))
        return max(1, min(base, by_work, tasks))

    def in_flight_count(self, workers: int) -> int:
        if self.max_in_flight is not None:
            return int(self.max_in_flight)
        # Process pools duplicate task buffers. Keep the queue deliberately
        # shallow to make memory use predictable on every machine.
        return max(1, min(int(workers) * 2, int(workers) + 4))

    def point_chunk_points(self, *, bytes_per_point: int = 64) -> int:
        """Auto-size a bounded LAS/LAZ chunk for the current machine."""
        if self.point_chunk_size is not None:
            return int(self.point_chunk_size)
        budget = self.memory_budget_bytes()
        workers = self.worker_count(estimated_task_bytes=256 * _MIB)
        per_worker = max(64 * _MIB, budget // max(1, workers))
        points = int((per_worker * 0.15) // max(1, int(bytes_per_point)))
        return max(25_000, min(points, 2_000_000))

    def raster_tile(self, *, bands: int, bytes_per_sample: int = 4, work_factor: float = 6.0) -> int:
        """Auto-size a square raster processing tile."""
        workers = self.worker_count(estimated_task_bytes=256 * _MIB)
        per_worker = max(64 * _MIB, self.memory_budget_bytes() // max(1, workers))
        allowed = max(16 * _MIB, int(per_worker * 0.30))
        bytes_per_pixel = max(1.0, int(bands) * int(bytes_per_sample) * float(work_factor))
        safe = int((allowed / bytes_per_pixel) ** 0.5)
        requested = int(self.raster_tile_size) if self.raster_tile_size is not None else 1024
        size = max(32, min(requested, safe))
        return max(32, (size // 32) * 32)

    def resource_plan(self, *, estimated_task_bytes: int = 256 * _MIB, bands: int = 1) -> dict[str, Any]:
        total, available = self.system_memory()
        preferred_cpu, logical_cpu = self.system_cpu_counts()
        workers = self.worker_count(estimated_task_bytes=estimated_task_bytes)
        return {
            "mode": self.mode,
            "total_ram_gb": total / _GIB,
            "available_ram_gb": available / _GIB,
            "reserved_ram_gb": self.reserve_ram_bytes() / _GIB,
            "memory_budget_gb": self.memory_budget_bytes() / _GIB,
            "preferred_cpu_count": preferred_cpu,
            "logical_cpu_count": logical_cpu,
            "workers": workers,
            "max_in_flight": self.in_flight_count(workers),
            "min_parallel_tasks": int(self.min_parallel_tasks),
            "min_tasks_per_worker": int(self.min_tasks_per_worker),
            "point_chunk_size": self.point_chunk_points(),
            "raster_tile_size": self.raster_tile(bands=max(1, int(bands))),
        }


@dataclass(slots=True)
class UAVProjectConfig:
    """Project-level configuration and metadata management.

    The processing core is deliberately sensor-agnostic.  Sensor names, band
    names, semantic roles, raster-band numbers, point-cloud dimensions and
    derived-index definitions are all supplied by the user in configuration.
    No manufacturer, platform or sensor preset is embedded in the package.
    """

    project_name: str
    base_data_dir: str | Path
    crs: str | None = None
    reference_dem: str | Path | None = None
    reference_dtm: str | Path | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    processing: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.base_data_dir = str(Path(self.base_data_dir))
        if self.reference_dem is not None:
            self.reference_dem = str(Path(self.reference_dem))
        if self.reference_dtm is not None:
            self.reference_dtm = str(Path(self.reference_dtm))
        self.metadata.setdefault("sensors", {})
        # Validate once while keeping JSON-friendly dict storage.
        self.processing = ProcessingLimits.from_mapping(self.processing).to_dict()

    @property
    def output_prefix(self) -> str:
        """Stable output prefix derived only from project configuration.

        ``site_name`` and ``campaign_name`` are preferred when present.  The
        processing core therefore never hard-codes a study-site name.
        """
        parts = [
            self.metadata.get("site_name"),
            self.metadata.get("campaign_name"),
        ]
        parts = [str(x).strip() for x in parts if x is not None and str(x).strip()]
        if not parts:
            parts = [str(self.project_name).strip() or "uav_biodiv"]
        raw = "_".join(parts)
        safe = "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in raw)
        while "__" in safe:
            safe = safe.replace("__", "_")
        return safe.strip("_") or "uav_biodiv"

    @property
    def processing_limits(self) -> ProcessingLimits:
        return ProcessingLimits.from_mapping(self.processing)

    def set_processing_limits(self, **kwargs: Any) -> ProcessingLimits:
        current = self.processing_limits.to_dict()
        current.update(kwargs)
        limits = ProcessingLimits(**current)
        self.processing = limits.to_dict()
        return limits

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")
        return path

    @classmethod
    def load_json(cls, path: str | Path) -> "UAVProjectConfig":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**data)

    @property
    def sensors(self) -> dict[str, dict[str, Any]]:
        return self.metadata.setdefault("sensors", {})

    def add_sensor(
        self,
        name: str,
        *,
        sensor_type: str | None = None,
        bands: Mapping[str, Mapping[str, Any] | str | int] | None = None,
        point_attributes: Mapping[str, str] | None = None,
        indices: Mapping[str, Mapping[str, Any]] | None = None,
        overwrite: bool = False,
        **sensor_metadata: Any,
    ) -> dict[str, Any]:
        """Register a fully user-defined sensor.

        ``bands`` accepts dictionaries, a semantic-role string, or a raster
        band number.  All names are arbitrary and are never interpreted by the
        processing core unless an index definition explicitly refers to them.
        """
        if name in self.sensors and not overwrite:
            raise ValueError(f"Sensor {name!r} already exists; use overwrite=True to replace it")

        definition: dict[str, Any] = {}
        if sensor_type is not None:
            definition["sensor_type"] = str(sensor_type)

        if bands is not None:
            norm_bands: dict[str, dict[str, Any]] = {}
            for band_name, spec in bands.items():
                if isinstance(spec, Mapping):
                    norm_bands[str(band_name)] = dict(spec)
                elif isinstance(spec, str):
                    norm_bands[str(band_name)] = {"role": spec}
                elif isinstance(spec, int):
                    norm_bands[str(band_name)] = {"raster_band": int(spec)}
                else:
                    raise TypeError(
                        f"Band definition for {band_name!r} must be mapping, role string, or integer band index"
                    )
            definition["bands"] = norm_bands

        if point_attributes is not None:
            definition["point_attributes"] = {str(k): str(v) for k, v in point_attributes.items()}
        if indices is not None:
            definition["indices"] = {str(k): dict(v) for k, v in indices.items()}

        definition.update(sensor_metadata)
        self.sensors[str(name)] = definition
        return definition

    def get_sensor(self, name: str) -> dict[str, Any]:
        try:
            return self.sensors[name]
        except KeyError as exc:
            raise KeyError(f"Sensor {name!r} is not configured. Available: {sorted(self.sensors)}") from exc

    def get_band(self, sensor_name: str, band_name: str) -> dict[str, Any]:
        sensor = self.get_sensor(sensor_name)
        bands = sensor.get("bands", {})
        if band_name not in bands:
            raise KeyError(
                f"Band {band_name!r} is not configured for sensor {sensor_name!r}. "
                f"Available: {sorted(bands)}"
            )
        spec = bands[band_name]
        return dict(spec) if isinstance(spec, Mapping) else {"value": spec}

    def band_for_role(self, sensor_name: str, role: str) -> str | None:
        sensor = self.get_sensor(sensor_name)
        for band_name, spec in sensor.get("bands", {}).items():
            if isinstance(spec, Mapping) and spec.get("role") == role:
                return str(band_name)
        return None

    def point_attribute(
        self,
        sensor_name: str,
        semantic_name: str,
        *,
        band_name: str | None = None,
        default: str | None = None,
    ) -> str | None:
        """Resolve an actual point-cloud dimension from configuration only."""
        sensor = self.get_sensor(sensor_name)
        if band_name is not None:
            spec = sensor.get("bands", {}).get(band_name, {})
            if isinstance(spec, Mapping) and spec.get("point_attribute"):
                return str(spec["point_attribute"])
        attrs = sensor.get("point_attributes", {})
        if semantic_name in attrs:
            return str(attrs[semantic_name])
        return default

    def sensor_band_metadata(self, sensor_name: str, band_name: str) -> dict[str, Any]:
        """Return merged sensor + channel metadata for provenance sidecars.

        The function does not interpret channel names or wavelengths. Any
        physical metadata is supplied by the user and is copied verbatim.
        """
        sensor = dict(self.get_sensor(sensor_name))
        bands = sensor.pop("bands", {})
        sensor.pop("indices", None)
        sensor.pop("point_attributes", None)
        band = bands.get(band_name, {})
        band_meta = dict(band) if isinstance(band, Mapping) else {"value": band}
        return {
            "sensor_name": str(sensor_name),
            "sensor_metadata": sensor,
            "channel_name": str(band_name),
            "channel_metadata": band_meta,
        }

    def resolve_index_definitions(
        self,
        sensor_name: str | None = None,
        definitions: Mapping[str, Mapping[str, Any]] | None = None,
    ) -> dict[str, dict[str, Any]]:
        """Resolve arbitrary user-defined index inputs to configured bands."""
        if definitions is None:
            if sensor_name is not None:
                definitions = self.get_sensor(sensor_name).get("indices", {})
            else:
                definitions = self.metadata.get("indices", {})

        resolved: dict[str, dict[str, Any]] = {}
        for index_name, raw_spec in (definitions or {}).items():
            spec = dict(raw_spec)
            if "a_role" in spec:
                if sensor_name is None:
                    raise ValueError(f"Index {index_name!r} uses a_role but no sensor_name was supplied")
                role = str(spec.pop("a_role"))
                band = self.band_for_role(sensor_name, role)
                if band is None:
                    raise KeyError(f"Cannot resolve role {role!r} for index {index_name!r}")
                spec["a"] = band
            if "b_role" in spec:
                if sensor_name is None:
                    raise ValueError(f"Index {index_name!r} uses b_role but no sensor_name was supplied")
                role = str(spec.pop("b_role"))
                band = self.band_for_role(sensor_name, role)
                if band is None:
                    raise KeyError(f"Cannot resolve role {role!r} for index {index_name!r}")
                spec["b"] = band
            resolved[str(index_name)] = spec
        return resolved

    def raster_sources_from_sensor(
        self,
        sensor_name: str,
        *,
        raster_path: str | Path | None = None,
        band_paths: Mapping[str, str | Path] | None = None,
    ) -> dict[str, str | Path | tuple[str | Path, int]]:
        """Build named raster sources from user-defined sensor metadata."""
        sensor = self.get_sensor(sensor_name)
        bands = sensor.get("bands", {})
        if not bands:
            raise ValueError(f"Sensor {sensor_name!r} has no configured bands")

        out: dict[str, str | Path | tuple[str | Path, int]] = {}
        if band_paths is not None:
            for band_name, path in band_paths.items():
                if band_name not in bands:
                    raise KeyError(f"Band {band_name!r} is not configured for sensor {sensor_name!r}")
                out[str(band_name)] = path
            return out

        if raster_path is None:
            raise ValueError("Provide either raster_path or band_paths")

        for band_name, spec in bands.items():
            if not isinstance(spec, Mapping) or "raster_band" not in spec:
                raise KeyError(
                    f"Band {band_name!r} has no raster_band configured; use band_paths for separate files"
                )
            out[str(band_name)] = (raster_path, int(spec["raster_band"]))
        return out
