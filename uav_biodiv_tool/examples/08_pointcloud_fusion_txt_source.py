"""Example: fuse a numeric TXT point cloud onto a LAS/LAZ reference geometry.

The text cloud can come directly from photogrammetric processing. The example
assumes four whitespace-separated columns: X Y Z and one physical variable.
Column meanings are declared explicitly; uav_biodiv does not infer sensor or
band conventions.
"""

from pathlib import Path

from uav_biodiv import UAVMonitoringPipeline, UAVProjectConfig


REFERENCE_LIDAR = Path(r"D:\data\reference.laz")
SOURCE_TXT = Path(r"D:\data\source_points.txt")
OUTPUT_LAZ = Path(r"D:\data\analysis_ready_points.laz")


def main() -> None:
    cfg = UAVProjectConfig(
        project_name="example_project",
        base_data_dir=OUTPUT_LAZ.parent,
        crs="EPSG:32634",  # required for text inputs because TXT has no embedded CRS
        metadata={"site_name": "example_site", "campaign_name": "campaign_01"},
    )
    cfg.set_processing_limits(mode="auto")
    pipeline = UAVMonitoringPipeline(cfg)

    result = pipeline.run_multimodal_point_cloud_fusion(
        reference_path=REFERENCE_LIDAR,
        source_clouds={
            "source_temperature": {
                "path": SOURCE_TXT,
                "crs": "EPSG:32634",
                "text_options": {
                    "delimiter": "whitespace",
                    "has_header": False,
                    "columns": {
                        "x": 0,
                        "y": 1,
                        "z": 2,
                        "temperature": 3,
                    },
                },
                "attributes": {
                    "temperature": {
                        "dimension": "temperature",
                        "unit": "degC",
                        "description": "Temperature value exported with the source point cloud",
                    }
                },
                "metadata": {
                    "source_type": "photogrammetric_point_cloud",
                    "acquisition_date": None,
                    "sensor": None,
                },
            }
        },
        output_path=OUTPUT_LAZ,
        fusion_mode="reference_attributes",
        registration_method="icp",
        transfer_k=1,
        transfer_max_distance=0.50,
        registration_max_points=250_000,
    )

    print("Output:", result["output_path"])
    print("Metadata:", result["metadata_path"])
    print("Transfer QA:", result["transfer_stats"])


if __name__ == "__main__":
    main()
