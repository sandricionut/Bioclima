"""Generic AOI examples for uav_biodiv v0.9.3.

The AOI controls the output/fusion extent. By default, registration is still
estimated from the full available overlap so changing the AOI does not change
the geometric correction. Set registration_scope="aoi" only when a local
alignment is scientifically intended.
"""

from pathlib import Path

from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline


AOI = Path("study_area.gpkg")
# AOI can alternatively be a bbox: (minx, miny, maxx, maxy)


def make_pipeline() -> UAVMonitoringPipeline:
    cfg = UAVProjectConfig(
        project_name="BioClima",
        base_data_dir="outputs",
        crs="EPSG:32634",
    )
    cfg.set_processing_limits(mode="auto")
    return UAVMonitoringPipeline(cfg)


def raster_example() -> None:
    pipeline = make_pipeline()
    result = pipeline.run_multimodal_raster_fusion(
        raster_sources={
            "channel_alpha": ("source_alpha.tif", 1),
            "channel_beta": ("source_beta.tif", 1),
        },
        source_groups={
            "geometry_alpha": {"keys": ["channel_alpha"]},
            "geometry_beta": {"keys": ["channel_beta"]},
        },
        reference_group="geometry_alpha",
        output_path="outputs/analysis_ready_aoi.tif",
        aoi=AOI,
        # Not needed when the vector file already has a CRS.
        # aoi_crs="EPSG:32634",
        registration_scope="full",
    )
    print(result["output_path"])
    print(result["registration_path"])


def point_cloud_example() -> None:
    pipeline = make_pipeline()
    result = pipeline.run_multimodal_point_cloud_fusion(
        reference_path="reference_las_tiles",
        source_clouds={
            "source_variable": {
                "path": "source_points.txt",
                "crs": "EPSG:32634",
                "text_options": {
                    "delimiter": "whitespace",
                    "has_header": False,
                    "field_names": ["x", "y", "z", "value"],
                },
                "attributes": {
                    "value": {"dimension": "value", "unit": None},
                },
            }
        },
        output_path="outputs/analysis_ready_aoi.laz",
        aoi=AOI,
        registration_scope="full",
        transfer_max_distance=0.50,
    )
    print(result["output_path"])


if __name__ == "__main__":
    # Uncomment the workflow for which the example paths above exist.
    # raster_example()
    # point_cloud_example()
    print("Configure the example paths, then call raster_example() or point_cloud_example().")
