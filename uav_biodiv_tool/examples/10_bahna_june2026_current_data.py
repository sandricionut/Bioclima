"""BioClima campaign-specific test for v0.9.1.

The file paths are user-side inputs. Study-site and campaign names used in
outputs come from UAVProjectConfig metadata; the processing core contains no
site, platform, sensor or channel naming convention.
"""

from pathlib import Path
import logging

from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline
from uav_biodiv.sensors.lidar import resolve_lidar_sources

# Replace these placeholders with local inputs. BAHNA_LAS may be:
#   - one LAS/LAZ file;
#   - a directory containing only the LAS/LAZ tiles to process; or
#   - an explicit list of LAS/LAZ files.
BAHNA_LAS = Path("bahna_las")
BAHNA_RGB = Path("bahna_rgb.tif")
BAHNA_THERMAL = Path("bahna_thermal.tif")
BAHNA_OUTPUT = Path("bahna_output")
# Optional AOI: None, bbox, vector path, GeoJSON or Shapely geometry.
# With registration_scope="full" the AOI does not influence co-registration.
BAHNA_AOI = None

# Optional product-level geometric accuracy from the UAV processing report.
# Values are in the analysis-grid map units (metres for EPSG:32634). Leave as
# None when unavailable; v0.9.1 then uses a conservative native-resolution
# fallback and refuses corrections that hit that limit.
RGB_RMS_X = None
RGB_RMS_Y = None
THERMAL_RMS_X = None
THERMAL_RMS_Y = None


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    cfg = UAVProjectConfig(
        project_name="BioClima",
        base_data_dir=BAHNA_OUTPUT,
        crs=None,
        metadata={
            "site_name": "Bahna",
            "campaign_name": "June2026",
        },
    )
    cfg.set_processing_limits(mode="auto")

    output_dir = BAHNA_OUTPUT / cfg.output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    prefix = cfg.output_prefix

    # Physical/spectral metadata should be filled from the acquisition record
    # or sensor specification. The core never invents wavelengths or units.
    cfg.add_sensor(
        "imaging_source_alpha",
        sensor_type="imaging",
        bands={
            "rgb_red": {"raster_band": 1, "unit": None, "central_wavelength_nm": None},
            "rgb_green": {"raster_band": 2, "unit": None, "central_wavelength_nm": None},
            "rgb_blue": {"raster_band": 3, "unit": None, "central_wavelength_nm": None},
        },
        acquisition={"campaign": "June2026"},
    )
    cfg.add_sensor(
        "imaging_source_beta",
        sensor_type="imaging",
        bands={
            "thermal_value": {
                "raster_band": 1,
                "unit": None,
                "wavelength_min_nm": None,
                "wavelength_max_nm": None,
            }
        },
        acquisition={"campaign": "June2026"},
    )

    pipeline = UAVMonitoringPipeline(cfg)

    print("\nAutomatic resource plan")
    for key, value in cfg.processing_limits.resource_plan().items():
        print(f"  {key}: {value}")

    lidar_sources = resolve_lidar_sources(BAHNA_LAS)
    print("\nLiDAR source tiles")
    print(f"  files: {len(lidar_sources)}")
    total_points = 0
    for path in lidar_sources:
        info = pipeline.ingestion.inspect_point_cloud(path)
        total_points += int(info.get("point_count", 0) or 0)
    print(f"  total header points: {total_points}")
    print(f"  first tile: {lidar_sources[0]}")

    analysis_ready_path = output_dir / f"{prefix}_rgb_thermal_analysis_ready.tif"
    raster_result = pipeline.run_multimodal_raster_fusion(
        raster_sources={
            "rgb_red": (BAHNA_RGB, 1),
            "rgb_green": (BAHNA_RGB, 2),
            "rgb_blue": (BAHNA_RGB, 3),
            "thermal_value": (BAHNA_THERMAL, 1),
        },
        source_groups={
            "rgb_geometry": {
                "keys": ["rgb_red", "rgb_green", "rgb_blue"],
                "registration_keys": ["rgb_red", "rgb_green", "rgb_blue"],
                "sensor": "imaging_source_alpha",
                "geometric_accuracy": {"rms_x": RGB_RMS_X, "rms_y": RGB_RMS_Y},
            },
            "thermal_geometry": {
                "keys": ["thermal_value"],
                "registration_keys": ["thermal_value"],
                "sensor": "imaging_source_beta",
                "geometric_accuracy": {"rms_x": THERMAL_RMS_X, "rms_y": THERMAL_RMS_Y},
            },
        },
        reference_group="rgb_geometry",
        reference_key="rgb_red",
        output_path=analysis_ready_path,
        registration_transform="auto",
        aoi=BAHNA_AOI,
        registration_scope="full",
    )
    print("\nMultimodal raster processing")
    print(raster_result["qa"]["processing"])
    print("Metadata:", raster_result["metadata_path"])
    print("Registration QA:", raster_result["registration_path"])
    thermal_registration = raster_result["registration"]["groups"]["thermal_geometry"]
    print("Registration accepted:", thermal_registration["accepted"])
    print("Registration applied:", thermal_registration["applied"])
    print("Registration reason:", thermal_registration["acceptance_reason"])
    print("Estimated shift (map units):", thermal_registration["estimated_shift_map_units"])
    print("Allowed shift (map units):", thermal_registration["max_allowed_shift_map_units"])

    # LiDAR metrics are added only after the image products have been
    # co-registered. The already registered analysis-ready raster is used as
    # the common target grid.
    point_result = pipeline.run_lidar_raster_fusion(
        point_cloud_path=BAHNA_LAS,
        raster_sources={
            "rgb_red": (analysis_ready_path, 1),
            "rgb_green": (analysis_ready_path, 2),
            "rgb_blue": (analysis_ready_path, 3),
            "thermal_value": (analysis_ready_path, 4),
        },
        reference_key="rgb_red",
        point_metrics=("point_count", "point_density", "z_mean", "z_std", "z_p95"),
        compute_indices=False,
        output_path=output_dir / f"{prefix}_imagery_lidar_analysis_ready.tif",
        aoi=BAHNA_AOI,
    )
    print("\nPoint-raster processing")
    print(point_result["point_metrics_processing"])


if __name__ == "__main__":
    main()
