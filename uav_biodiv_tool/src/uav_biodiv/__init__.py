"""BioClima UAV biodiversity/climate multi-sensor processing package."""

from .config import UAVProjectConfig, ProcessingLimits
from .pipeline import UAVMonitoringPipeline
from .ingestion import PointCloudData, RasterData, DataIngestionHarmonization
from .fusion import MultiSensorFusion, RegistrationResult
from .raster_registration import RasterRegistrationResult
from .validation import ValidationReporting
from .ebv_ecv import EBVECVDerivation
from .ai import AIAnalytics
from .aoi import AOISelection
from .sensors import (
    LIDAR_SENSORS,
    MULTISPECTRAL_SENSORS,
    get_lidar_sensor,
    get_multispectral_sensor,
)

__all__ = [
    "UAVProjectConfig",
    "ProcessingLimits",
    "UAVMonitoringPipeline",
    "PointCloudData",
    "RasterData",
    "DataIngestionHarmonization",
    "MultiSensorFusion",
    "RegistrationResult",
    "RasterRegistrationResult",
    "ValidationReporting",
    "EBVECVDerivation",
    "AIAnalytics",
    "AOISelection",
    "LIDAR_SENSORS",
    "MULTISPECTRAL_SENSORS",
    "get_lidar_sensor",
    "get_multispectral_sensor",
]

__version__ = "0.9.3"
