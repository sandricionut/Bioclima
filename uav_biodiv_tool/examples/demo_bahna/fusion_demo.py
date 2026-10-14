
from pathlib import Path
from contextlib import contextmanager
from threading import Event, Thread
import multiprocessing

from tqdm import tqdm

from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline


# ============================================================
# SETTINGS - edit only this section for a similar dataset
# ============================================================

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE / "output"

PROJECT = "BioClima"
SITE = "Bahna"
CAMPAIGN = "June2026"

USE_AOI = False
TRANSFER_MAX_DISTANCE = 0.50

THERMAL_FIELDS = [
    "x", "y", "z", "temperature",
    "normal_x", "normal_y", "normal_z",
]

MULTISPECTRAL_FIELDS = [
    "x", "y", "z", "green", "red", "red_edge", "nir",
    "normal_x", "normal_y", "normal_z",
]


# ============================================================
# SMALL HELPERS
# ============================================================

def find(name):
    matches = [p for p in DATA.rglob(name) if "__MACOSX" not in p.parts]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {name!r} under {DATA}, found {len(matches)}"
        )
    return matches[0]


@contextmanager
def progress(label, refresh_interval=0.5):
    """Keep one live tqdm line while a long pipeline operation is running."""
    stop = Event()
    bar = tqdm(
        total=None,
        desc=label,
        dynamic_ncols=True,
        leave=True,
        bar_format="{desc}  {elapsed}",
    )

    def refresher():
        while not stop.wait(refresh_interval):
            bar.refresh()

    thread = Thread(target=refresher, daemon=True)
    thread.start()
    try:
        yield
    finally:
        stop.set()
        thread.join(timeout=0.2)
        bar.set_description_str(f"{label} ✓")
        bar.refresh()
        bar.close()


# ============================================================
# MAIN
# ============================================================

def main():

    # Inputs
    lidar = find("demo_lidar.las")
    thermal_txt = find("demo_thermal.txt")
    multispectral_txt = find("demo_multispectral.txt")

    rgb = find("demo_rgb.tif")
    thermal_raster = find("demo_thermal.tif")
    multispectral_raster = find("demo_multispectral.tif")

    aoi = find("demo_aoi.gpkg") if USE_AOI else None

    # Project
    cfg = UAVProjectConfig(
        project_name=PROJECT,
        base_data_dir=OUTPUT,
        crs=None,
        metadata={
            "site_name": SITE,
            "campaign_name": CAMPAIGN,
        },
    )

    cfg.set_processing_limits(mode="auto")
    pipeline = UAVMonitoringPipeline(cfg)

    output_dir = OUTPUT / cfg.output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\nBioClima fusion demo")
    print(f"  Site     : {SITE}")
    print(f"  Campaign : {CAMPAIGN}")
    print(f"  AOI      : {aoi if aoi else 'full dataset'}")
    print(f"  Output   : {output_dir}")

    # ========================================================
    # 1. POINT-CLOUD FUSION
    # ========================================================

    lidar_crs = pipeline.ingestion.inspect_point_cloud(lidar).get("crs")
    if lidar_crs is None:
        raise ValueError("The LiDAR file must contain a CRS.")

    point_cloud_output = (
        output_dir / f"{cfg.output_prefix}_pointcloud_analysis_ready.laz"
    )

    with progress("[1/3] Point-cloud fusion"):
        pc_result = pipeline.run_multimodal_point_cloud_fusion(
            reference_path=lidar,
            source_clouds={
                "thermal": {
                    "path": thermal_txt,
                    "crs": lidar_crs,
                    "text_options": {
                        "delimiter": "whitespace",
                        "has_header": False,
                        "field_names": THERMAL_FIELDS,
                    },
                    "attributes": {
                        "temperature": {
                            "dimension": "temperature",
                            "unit": "degC",
                        },
                    },
                },
                "multispectral": {
                    "path": multispectral_txt,
                    "crs": lidar_crs,
                    "text_options": {
                        "delimiter": "whitespace",
                        "has_header": False,
                        "field_names": MULTISPECTRAL_FIELDS,
                    },
                    "attributes": {
                        name: {"dimension": name, "unit": None}
                        for name in ("green", "red", "red_edge", "nir")
                    },
                },
            },
            output_path=point_cloud_output,
            fusion_mode="reference_attributes",
            registration_method="icp",
            registration_max_points=250_000,
            transfer_k=1,
            transfer_max_distance=TRANSFER_MAX_DISTANCE,
            aoi=aoi,
            registration_scope="full",
        )

    print("  output:", pc_result["output_path"])

    # ========================================================
    # 2. RASTER FUSION
    # ========================================================

    raster_sources = {
        "rgb_red": (rgb, 1),
        "rgb_green": (rgb, 2),
        "rgb_blue": (rgb, 3),
        "thermal_temperature": (thermal_raster, 1),
        "ms_green": (multispectral_raster, 1),
        "ms_red": (multispectral_raster, 2),
        "ms_red_edge": (multispectral_raster, 3),
        "ms_nir": (multispectral_raster, 4),
    }

    source_groups = {
        "rgb": {
            "keys": ["rgb_red", "rgb_green", "rgb_blue"],
            "registration_keys": ["rgb_red", "rgb_green", "rgb_blue"],
            "sensor": "rgb",
            "geometric_accuracy": {"rms_x": None, "rms_y": None},
        },
        "thermal": {
            "keys": ["thermal_temperature"],
            "registration_keys": ["thermal_temperature"],
            "sensor": "thermal",
            "geometric_accuracy": {"rms_x": None, "rms_y": None},
        },
        "multispectral": {
            "keys": ["ms_green", "ms_red", "ms_red_edge", "ms_nir"],
            "registration_keys": ["ms_green", "ms_red", "ms_red_edge", "ms_nir"],
            "sensor": "multispectral",
            "geometric_accuracy": {"rms_x": None, "rms_y": None},
        },
    }

    analysis_ready = (
        output_dir / f"{cfg.output_prefix}_imagery_analysis_ready.tif"
    )

    with progress("[2/3] Raster fusion and co-registration"):
        raster_result = pipeline.run_multimodal_raster_fusion(
            raster_sources=raster_sources,
            source_groups=source_groups,
            reference_group="rgb",
            reference_key="rgb_red",
            output_path=analysis_ready,
            registration_transform="auto",
            aoi=aoi,
            registration_scope="full",
        )

    print("  output:", analysis_ready)
    print("  registration QA:", raster_result["registration_path"])

    # ========================================================
    # 3. LIDAR METRICS
    # ========================================================

    fused_sources = {
        key: (analysis_ready, band)
        for band, key in enumerate(raster_sources, start=1)
    }

    final_output = (
        output_dir / f"{cfg.output_prefix}_imagery_lidar_analysis_ready.tif"
    )

    with progress("[3/3] LiDAR metrics and final raster"):
        pipeline.run_lidar_raster_fusion(
            point_cloud_path=lidar,
            raster_sources=fused_sources,
            reference_key="rgb_red",
            point_metrics=(
                "point_count",
                "point_density",
                "z_mean",
                "z_std",
                "z_p95",
            ),
            compute_indices=False,
            output_path=final_output,
            aoi=aoi,
        )

    print("  output:", final_output)
    print("\nDone.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
