from pathlib import Path

from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline


def main() -> None:
    cfg = UAVProjectConfig(
        project_name="BioClima",
        base_data_dir=r"D:\BioClima",
        crs="EPSG:32634",
        metadata={"site_name": "site_alpha", "campaign_name": "campaign_one"},
    )
    cfg.set_processing_limits(mode="auto")

    # Physical metadata is user supplied. The generic example intentionally
    # does not invent wavelengths, channel semantics or geometric accuracies.
    cfg.add_sensor(
        "sensor_alpha",
        sensor_type="imaging",
        bands={
            "channel_alpha": {
                "raster_band": 1,
                "unit": None,
                "central_wavelength_nm": None,
            },
            "channel_beta": {
                "raster_band": 2,
                "unit": None,
                "central_wavelength_nm": None,
            },
        },
        acquisition={"date": None, "timezone": None},
    )
    cfg.add_sensor(
        "sensor_beta",
        sensor_type="imaging",
        bands={
            "channel_gamma": {
                "raster_band": 1,
                "unit": None,
                "central_wavelength_nm": None,
            }
        },
        acquisition={"date": None, "timezone": None},
    )

    output_dir = Path(cfg.base_data_dir) / cfg.output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)
    pipeline = UAVMonitoringPipeline(cfg)
    result = pipeline.run_multimodal_raster_fusion(
        raster_sources={
            "channel_alpha": (r"D:\BioClima\source_alpha.tif", 1),
            "channel_beta": (r"D:\BioClima\source_alpha.tif", 2),
            "channel_gamma": (r"D:\BioClima\source_beta.tif", 1),
        },
        source_groups={
            "geometry_alpha": {
                "keys": ["channel_alpha", "channel_beta"],
                "sensor": "sensor_alpha",
                # Optional, in analysis-grid map units:
                # "geometric_accuracy": {"rms_x": 0.10, "rms_y": 0.10},
            },
            "geometry_beta": {
                "keys": ["channel_gamma"],
                "sensor": "sensor_beta",
                # Optional, in analysis-grid map units:
                # "geometric_accuracy": {"rms_x": 0.15, "rms_y": 0.15},
            },
        },
        reference_group="geometry_alpha",
        output_path=output_dir / f"{cfg.output_prefix}_analysis_ready.tif",
        # v0.9.0 default: automatic conservative model selection.
        registration_transform="auto",
    )
    print(result["metadata_path"])
    print(result["registration_path"])


if __name__ == "__main__":
    main()
