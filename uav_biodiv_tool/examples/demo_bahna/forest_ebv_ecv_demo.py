"""Bahna forest EBV/ECV demo for uav_biodiv 0.9.3.

Run after::

    python download_demo_data.py

This example starts from the downloaded Bahna LiDAR and Mavic 3M raster and
produces:

- CHM
- NDVI, GNDVI and NDRE
- forest_health.tif
- lst_anomaly.tif
- ecosystem extent metrics
- canopy structure metrics
- Random Forest habitat classification
- confusion matrix / overall accuracy
- summary report
- metrics JSON

The downloaded demo does not contain an independent habitat map. Therefore,
for software demonstration only, this script creates a simple three-class
reference map from CHM and NDVI thresholds before running the high-level
forest workflow. Replace that generated raster with an independently mapped
habitat raster for scientific analysis.
"""

from pathlib import Path
import logging
import multiprocessing

import numpy as np
import rasterio

from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline
from uav_biodiv.sensors.lidar import rasterize_lidar_metrics_file


# ============================================================
# SETTINGS - edit only this section for a similar dataset
# ============================================================

HERE = Path(__file__).resolve().parent
DATA = HERE / "data"
OUTPUT = HERE / "output" / "forest_ebv_ecv"

PROJECT = "BioClima"
SITE = "Bahna"
CAMPAIGN = "June2026"

MULTISPECTRAL_SENSOR = "dji_mavic_3m"
LIDAR_SENSOR = "dji_zenmuse_l2"

# The Bahna demo LAS uses the LAS classification dimension.
# LAS class 2 is ground. Change these explicitly for another dataset.
LIDAR_CLASSIFICATION_FIELD = "classification"
GROUND_CLASSES = (2,)

# Demo-only habitat thresholds.
# 1 = Forest, 2 = Grassland, 3 = Bare / sparse vegetation.
FOREST_MIN_HEIGHT_M = 2.0
FOREST_MIN_NDVI = 0.50
VEGETATED_MIN_NDVI = 0.30

CLASS_MAPPING = {
    1: "Forest",
    2: "Grassland",
    3: "Bare_or_sparse_vegetation",
}


# ============================================================
# SMALL HELPERS
# ============================================================

def find(name: str) -> Path:
    matches = [
        path
        for path in DATA.rglob(name)
        if "__MACOSX" not in path.parts
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"Expected exactly one {name!r} under {DATA}, found {len(matches)}. "
            "Run download_demo_data.py first."
        )
    return matches[0]


def write_band(
    path: Path,
    array: np.ndarray,
    profile: dict,
    *,
    description: str,
    nodata: float | int,
    dtype: str,
) -> Path:
    """Write one analysis-grid GeoTIFF."""
    out = np.asarray(array).copy()
    if np.issubdtype(np.dtype(dtype), np.floating):
        out = out.astype(dtype, copy=False)
        out[~np.isfinite(out)] = nodata
    else:
        out = out.astype(dtype, copy=False)

    dst_profile = dict(profile)
    dst_profile.update(
        driver="GTiff",
        count=1,
        dtype=dtype,
        nodata=nodata,
        compress="deflate",
        tiled=True,
        BIGTIFF="YES",
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(path, "w", **dst_profile) as dst:
        dst.write(out, 1)
        dst.set_band_description(1, description)
    return path


def prepare_dsm_dtm(
    pipeline: UAVMonitoringPipeline,
    lidar: Path,
    multispectral: Path,
    work_dir: Path,
) -> tuple[Path, Path]:
    """Create DTM and DSM on the multispectral grid from classified LiDAR."""
    lidar_info = pipeline.ingestion.inspect_point_cloud(lidar)
    dimensions = set(lidar_info.get("dimensions", []))

    if LIDAR_CLASSIFICATION_FIELD not in dimensions:
        raise ValueError(
            f"LiDAR dimension {LIDAR_CLASSIFICATION_FIELD!r} is not present. "
            f"Available dimensions: {sorted(dimensions)}"
        )

    with rasterio.open(multispectral) as src:
        reference_profile = src.profile.copy()
        reference_profile.update(count=1)

    metrics_path = work_dir / "lidar_surface_metrics.tif"

    print("\n[1/3] Deriving DTM and DSM from classified LiDAR ...", flush=True)
    rasterize_lidar_metrics_file(
        lidar,
        reference_profile,
        metrics_path,
        ingestion=pipeline.ingestion,
        processing_limits=pipeline.config.processing_limits,
        metrics=("dtm_ground", "dsm_p95"),
        classification_attribute=LIDAR_CLASSIFICATION_FIELD,
        ground_classes=GROUND_CLASSES,
    )

    with rasterio.open(metrics_path) as src:
        profile = src.profile.copy()
        dtm = src.read(1).astype(np.float32)
        dsm = src.read(2).astype(np.float32)
        nodata = src.nodata if src.nodata is not None else -9999.0

    dtm_path = write_band(
        work_dir / "demo_dtm.tif",
        dtm,
        profile,
        description="DTM_ground",
        nodata=nodata,
        dtype="float32",
    )
    dsm_path = write_band(
        work_dir / "demo_dsm.tif",
        dsm,
        profile,
        description="DSM_p95",
        nodata=nodata,
        dtype="float32",
    )

    return dsm_path, dtm_path


def prepare_demo_habitat(
    pipeline: UAVMonitoringPipeline,
    multispectral: Path,
    dsm_path: Path,
    dtm_path: Path,
    output_path: Path,
) -> Path:
    """Create a transparent, demo-only habitat reference for the workflow."""
    bands = pipeline.fusion.load_multispectral(
        multispectral,
        sensor=MULTISPECTRAL_SENSOR,
    )
    indices = pipeline.fusion.compute_multispectral_indices(
        bands,
        sensor=MULTISPECTRAL_SENSOR,
        indices=("ndvi", "gndvi", "ndre"),
    )

    with rasterio.open(dsm_path) as dsm_src, rasterio.open(dtm_path) as dtm_src:
        dsm = dsm_src.read(1, masked=True).astype(np.float32)
        dtm = dtm_src.read(1, masked=True).astype(np.float32)
        profile = dsm_src.profile.copy()

    chm = np.ma.filled(dsm - dtm, np.nan)
    chm[chm < 0] = 0.0
    ndvi = np.asarray(indices["ndvi"], dtype=np.float32)

    valid = np.isfinite(chm) & np.isfinite(ndvi)
    habitat = np.zeros(chm.shape, dtype=np.uint8)

    forest = valid & (chm >= FOREST_MIN_HEIGHT_M) & (ndvi >= FOREST_MIN_NDVI)
    grass = valid & ~forest & (ndvi >= VEGETATED_MIN_NDVI)
    bare = valid & ~forest & ~grass

    habitat[forest] = 1
    habitat[grass] = 2
    habitat[bare] = 3

    print("\n[2/3] Creating demo habitat reference ...", flush=True)
    print("  IMPORTANT: this is a threshold-based software-demo reference,", flush=True)
    print("  not an independently validated habitat map.", flush=True)

    return write_band(
        output_path,
        habitat,
        profile,
        description="demo_habitat_reference",
        nodata=0,
        dtype="uint8",
    )


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    lidar = find("demo_lidar.las")
    multispectral = find("demo_multispectral.tif")
    lst = find("demo_thermal.tif")

    OUTPUT.mkdir(parents=True, exist_ok=True)
    work_dir = OUTPUT / "prepared_inputs"
    work_dir.mkdir(parents=True, exist_ok=True)

    cfg = UAVProjectConfig(
        project_name=PROJECT,
        base_data_dir=OUTPUT,
        metadata={
            "site_name": SITE,
            "campaign_name": CAMPAIGN,
        },
    )
    cfg.set_processing_limits(mode="auto")
    pipeline = UAVMonitoringPipeline(cfg)

    dsm_path, dtm_path = prepare_dsm_dtm(
        pipeline,
        lidar,
        multispectral,
        work_dir,
    )

    habitat_path = prepare_demo_habitat(
        pipeline,
        multispectral,
        dsm_path,
        dtm_path,
        work_dir / "demo_habitat_reference.tif",
    )

    print("\n[3/3] Running forest EBV/ECV workflow ...", flush=True)
    result = pipeline.run_forest_dji_m3m_lidar_example(
        dsm_path=dsm_path,
        dtm_path=dtm_path,
        multispectral_path=multispectral,
        habitat_path=habitat_path,
        output_dir=OUTPUT,
        lst_path=lst,
        lst_band=1,
        lst_reference=None,  # None = mean valid LST of the harmonised scene
        sensor=MULTISPECTRAL_SENSOR,
        lidar_sensor=LIDAR_SENSOR,
        class_mapping=CLASS_MAPPING,
        forest_class=1,
        indices=("ndvi", "gndvi", "ndre"),
        train_random_forest=True,
    )

    print("\nOutputs")
    for name, path in result["outputs"].items():
        print(f"  {name:18s}: {path}")

    print("\nEcosystem extent (m²)")
    for name, area in result["ecosystem_extent"].items():
        print(f"  {name:28s}: {area:.2f}")

    print("\nCanopy structure")
    for name, value in result["canopy_metrics"].items():
        print(f"  {name:28s}: {value:.3f}")

    if result.get("lst_anomaly") is not None:
        print("\nLST anomaly")
        for name, value in result["lst_anomaly"].items():
            if value is None:
                print(f"  {name:28s}: None")
            else:
                print(f"  {name:28s}: {value:.3f}")

    classification = result.get("classification")
    if classification is not None:
        print("\nRF habitat classification")
        print("  overall accuracy:", classification["overall_accuracy"])
        print("  confusion matrix:", classification["matrix"])

    print("\nDone.")


if __name__ == "__main__":
    multiprocessing.freeze_support()
    main()
