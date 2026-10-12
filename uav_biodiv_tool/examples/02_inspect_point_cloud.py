from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline


def main() -> None:
    cfg = UAVProjectConfig.load_json("bioclima_config.json")
    pipeline = UAVMonitoringPipeline(cfg)
    info = pipeline.ingestion.inspect_point_cloud(r"D:\BioClima\cloud.las")
    for key, value in info.items():
        print(f"{key}: {value}")


if __name__ == "__main__":
    main()
