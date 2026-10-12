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
    result = pipeline.run_raster_fusion(
        raster_sources=sources,
        sensor_name="sensor_A",
        reference_key="channel_alpha",
        output_path=r"D:\BioClima\raster_fusion.tif",
    )
    print(result["qa"]["processing"])


if __name__ == "__main__":
    main()
