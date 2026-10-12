"""Process a directory of LAS/LAZ tiles directly, without merging them first."""

from pathlib import Path

from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline

TILE_DIRECTORY = Path(r"D:\data\lidar_tiles")
REFERENCE_RASTER = Path(r"D:\data\analysis_grid.tif")
OUTPUT = Path(r"D:\data\outputs\lidar_raster_analysis_ready.tif")


def main() -> None:
    cfg = UAVProjectConfig(
        project_name="example",
        base_data_dir=OUTPUT.parent,
    )
    cfg.set_processing_limits(mode="auto")
    pipeline = UAVMonitoringPipeline(cfg)

    result = pipeline.run_lidar_raster_fusion(
        point_cloud_path=TILE_DIRECTORY,
        raster_sources={"reference_variable": (REFERENCE_RASTER, 1)},
        reference_key="reference_variable",
        point_metrics=("point_count", "point_density", "z_mean", "z_std", "z_p95"),
        compute_indices=False,
        output_path=OUTPUT,
    )

    info = result["point_metrics_processing"]
    print("Input mode:", info["input_mode"])
    print("Source files:", info["source_file_count"])
    print("Header points:", info["point_count_header"])
    print("Points in output grid:", info["points_in_grid"])


if __name__ == "__main__":
    main()
