from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline


def main() -> None:
    cfg = UAVProjectConfig.load_json("bioclima_config.json")
    pipeline = UAVMonitoringPipeline(cfg)
    sources = cfg.raster_sources_from_sensor(
        "sensor_A",
        band_paths={
            "channel_alpha": r"D:\BioClima\channel_alpha.tif",
            "channel_beta": r"D:\BioClima\channel_beta.tif",
        },
    )
    result = pipeline.run_lidar_raster_fusion(
        point_cloud_path=r"D:\BioClima\cloud.las",
        point_sensor_name="point_sensor_A",
        raster_sources=sources,
        raster_sensor_name="sensor_A",
        reference_key="channel_alpha",
        point_metrics=("point_density", "height_p95", "canopy_cover", "intensity_mean"),
        output_path=r"D:\BioClima\point_raster_fusion.tif",
    )
    print(result["point_metrics_processing"])


if __name__ == "__main__":
    main()
