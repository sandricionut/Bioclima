"""Example: original LiDAR LAS/LAZ tiles as reference + one TXT source cloud.

This avoids creating a merged LiDAR reference file. Every original LAS/LAZ
reference tile is streamed directly into one analysis-ready LAZ output.
"""

from pathlib import Path

from uav_biodiv import UAVMonitoringPipeline, UAVProjectConfig


LIDAR_TILES_DIR = Path(r"D:\data\lidar_tiles")
SOURCE_TXT = Path(r"D:\data\source_points.txt")
OUTPUT_LAZ = Path(r"D:\data\analysis_ready_from_tiles.laz")


def main() -> None:
    cfg = UAVProjectConfig(
        project_name="example_project",
        base_data_dir=OUTPUT_LAZ.parent,
        crs="EPSG:32634",
    )
    cfg.set_processing_limits(mode="auto")
    pipeline = UAVMonitoringPipeline(cfg)

    result = pipeline.run_multimodal_point_cloud_fusion(
        reference_path=LIDAR_TILES_DIR,
        source_clouds={
            "source_temperature": {
                "path": SOURCE_TXT,
                "text_options": {
                    "delimiter": "whitespace",
                    "has_header": False,
                    "columns": {"x": 0, "y": 1, "z": 2, "temperature": 3},
                },
                "attributes": {
                    "temperature": {
                        "dimension": "temperature",
                        "unit": "degC",
                    }
                },
            }
        },
        output_path=OUTPUT_LAZ,
        transfer_max_distance=0.50,
        registration_max_points=250_000,
    )

    print("Reference files:", result["qa"]["streaming"]["reference_file_count"])
    print("Reference records preserved:", result["qa"]["streaming"]["reference_records_preserved"])
    print("Output:", result["output_path"])


if __name__ == "__main__":
    main()
