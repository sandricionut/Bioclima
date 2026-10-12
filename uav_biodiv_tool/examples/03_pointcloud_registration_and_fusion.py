"""Generic multimodal point-cloud fusion example.

All point variable names and physical metadata are supplied by the user. The
core package does not contain predefined sensor/channel naming conventions.
"""

from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline


def main() -> None:
    cfg = UAVProjectConfig.load_json("bioclima_config.json")
    pipeline = UAVMonitoringPipeline(cfg)

    result = pipeline.run_multimodal_point_cloud_fusion(
        reference_path=r"D:\BioClima\reference.las",
        source_clouds={
            "source_alpha": {
                "path": r"D:\BioClima\source_alpha.las",
                "attributes": {
                    "signal_alpha": {"dimension": "SignalAlpha", "unit": None},
                    "signal_beta": {"dimension": "SignalBeta", "unit": None},
                },
            },
            "source_beta": {
                "path": r"D:\BioClima\source_beta.las",
                "attributes": {
                    "physical_value": {"dimension": "PhysicalValue", "unit": None},
                },
            },
        },
        output_path=r"D:\BioClima\point_analysis_ready.laz",
        fusion_mode="reference_attributes",
    )
    print(result["fusion_method"])
    print(result["metadata_path"])


if __name__ == "__main__":
    main()
