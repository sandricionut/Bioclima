"""Optional sensor presets and sensor-domain helpers.

The processing core remains sensor-agnostic. Presets only supply metadata,
semantic band roles and ready-to-use index definitions.
"""

from .lidar import (
    LIDAR_SENSORS,
    compute_chm_from_dsm_dtm,
    get_lidar_sensor,
    rasterize_lidar_metrics,
    rasterize_lidar_metrics_files,
    register_lidar_sensor,
    resolve_lidar_sources,
)
from .multispectral import (
    MULTISPECTRAL_SENSORS,
    get_multispectral_sensor,
    multispectral_band_map,
    multispectral_index_definitions,
    register_multispectral_sensor,
)

__all__ = [
    "LIDAR_SENSORS",
    "MULTISPECTRAL_SENSORS",
    "compute_chm_from_dsm_dtm",
    "get_lidar_sensor",
    "get_multispectral_sensor",
    "multispectral_band_map",
    "multispectral_index_definitions",
    "rasterize_lidar_metrics",
    "rasterize_lidar_metrics_files",
    "register_lidar_sensor",
    "register_multispectral_sensor",
    "resolve_lidar_sources",
]
