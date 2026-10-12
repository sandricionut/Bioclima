"""Example: fuse an attribute stored in LAS/LAZ Extra Bytes onto a reference cloud."""

from pathlib import Path

from uav_biodiv import UAVMonitoringPipeline, UAVProjectConfig


REFERENCE_LIDAR = Path(r"D:\data\reference.laz")
SOURCE_LAZ = Path(r"D:\data\source_with_physical_value.laz")
OUTPUT_LAZ = Path(r"D:\data\analysis_ready_points.laz")


def main() -> None:
    cfg = UAVProjectConfig(
        project_name="example_project",
        base_data_dir=OUTPUT_LAZ.parent,
        crs="EPSG:32634",
    )
    cfg.set_processing_limits(mode="auto")
    pipeline = UAVMonitoringPipeline(cfg)

    # Replace "Temperature" with the actual Extra Bytes dimension name shown by:
    # pipeline.ingestion.inspect_point_cloud(SOURCE_LAZ)
    result = pipeline.run_multimodal_point_cloud_fusion(
        reference_path=REFERENCE_LIDAR,
        source_clouds={
            "source_temperature": {
                "path": SOURCE_LAZ,
                "attributes": {
                    "temperature": {
                        "dimension": "Temperature",
                        "unit": "degC",
                    }
                },
            }
        },
        output_path=OUTPUT_LAZ,
        transfer_max_distance=0.50,
        registration_max_points=250_000,
    )

    print("Output:", result["output_path"])
    print("Metadata:", result["metadata_path"])


if __name__ == "__main__":
    main()
