
"""BioClima Bahna demo for uav_biodiv v0.9.1.

The demo can download Bahna_demo.zip from a direct public HTTP/HTTPS URL,
extract it, validate the declared TXT schemas, and run:

1. point-cloud multimodal fusion:
      LiDAR LAS/LAZ reference geometry
      + thermal TXT attributes
      + multispectral TXT attributes;

2. raster multimodal co-registration and analysis-ready fusion:
      RGB GeoTIFF
      + thermal GeoTIFF
      + multispectral GeoTIFF;

3. LiDAR raster metrics on the analysis-ready raster grid.

Important design rules
----------------------
- The processing core contains no site-, platform-, sensor- or satellite-band
  naming convention.
- TXT files exported without a header MUST be described explicitly through
  field_names.
- Multispectral raster channels are discovered dynamically and are named
  generically as multispectral_1, multispectral_2, ... unless the user supplies
  richer metadata elsewhere.
- Physical units and wavelengths are never invented by this demo.
"""

from __future__ import annotations

from pathlib import Path
import logging
import re
import shutil
import zipfile

import gdown
import rasterio

from uav_biodiv import UAVProjectConfig, UAVMonitoringPipeline
from uav_biodiv.sensors.lidar import resolve_lidar_sources


def assert_v091_api() -> None:
    """Fail early if the example is run against an older/stale package source."""
    import inspect
    from importlib.metadata import version

    installed = version("uav-biodiv")
    signature = inspect.signature(UAVMonitoringPipeline.run_point_cloud_fusion)
    required = {"aoi", "aoi_crs", "registration_scope"}
    missing = required.difference(signature.parameters)

    if installed != "0.9.1" or missing:
        raise RuntimeError(
            "This demo requires the synchronized uav-biodiv 0.9.1 API. "
            f"Detected version={installed!r}; missing parameters={sorted(missing)}. "
            "Reinstall from this repository with: python -m pip install -e ."
        )


# ============================================================================
# DEMO DATA DOWNLOAD
# ============================================================================

# Direct URL to the demo ZIP archive.
#
# Replace this with the final public URL of Bahna_demo.zip.
# No Google Drive-specific handling is required.
data_zip_url = (
    "https://drive.google.com/file/d/"
    "1vlXTva6-0y9K1gdzHrRCW_JBoz_BdokC/view?usp=sharing"
)


DEMO_ROOT = Path("demo_data")
DATA_ZIP = DEMO_ROOT / "Bahna_demo.zip"
DATA_DIR = DEMO_ROOT / "Bahna_demo"


# ============================================================================
# DEMO INPUTS
# ============================================================================

# Demo inputs are resolved recursively after the ZIP is downloaded/extracted.
# This makes the demo independent of whether the archive contains:
#
#   Bahna_demo/<files>
#
# or an extra directory level such as:
#
#   Bahna_demo/Bahna_demo/<files>
#
# __MACOSX metadata directories are ignored.
BAHNA_LAS: Path | None = None
BAHNA_THERMAL: Path | None = None
BAHNA_MULTISPECTRAL: Path | None = None

BAHNA_RGB: Path | None = None
BAHNA_THERMAL_RASTER: Path | None = None
BAHNA_MULTISPECTRAL_RASTER: Path | None = None

# Output
BAHNA_OUTPUT = Path("bahna_output")

# Optional AOI supported by v0.9.1.
# Examples:
#   BAHNA_AOI = None
#   BAHNA_AOI = (minx, miny, maxx, maxy)
#   BAHNA_AOI = Path("demo_aoi.gpkg")
BAHNA_AOI = None


# ============================================================================
# TXT SCHEMA
# ============================================================================

# Thermal TXT exported without a header:
#   X Y Z temperature
THERMAL_FIELD_NAMES = [
    "x",
    "y",
    "z",
    "temperature",
    "normal_x",
    "normal_y",
    "normal_z",
]

# Only attributes listed here are transferred to the LiDAR reference geometry.
# Surface normals describe the source photogrammetric cloud geometry and are kept
# in the TXT schema/metadata, but are not transferred by default.
THERMAL_TRANSFER_FIELDS = [
    "temperature",
]

# Multispectral TXT exported without a header.
#
# IMPORTANT:
# These names MUST match the actual number and order of columns in demo_ms.txt.
# They are semantic user-side names, not conventions imposed by uav_biodiv.
#
# Replace this list if the exported TXT has a different schema.
MULTISPECTRAL_FIELD_NAMES = [
    "x",
    "y",
    "z",
    "green",
    "red",
    "red_edge",
    "nir",
    "normal_x",
    "normal_y",
    "normal_z",
]

MULTISPECTRAL_TRANSFER_FIELDS = [
    "green",
    "red",
    "red_edge",
    "nir",
]

# If you do not yet know the schema, use:
# MULTISPECTRAL_FIELD_NAMES = None
# The script will stop before expensive processing and report the column count.

THERMAL_UNITS = {
    "temperature": "degC",
}

# Units are intentionally left undefined until established from the export
# settings / sensor processing record.
MULTISPECTRAL_UNITS: dict[str, str | None] = {}

TXT_DELIMITER = "whitespace"
TXT_HAS_HEADER = False

# Maximum point-to-point transfer distance in map units for point-cloud fusion.
# This should ultimately be selected from point spacing and registration QA.
TRANSFER_MAX_DISTANCE = 0.50


# ============================================================================
# OPTIONAL GEOMETRIC ACCURACY
# ============================================================================

# Product-level geometric accuracy from UAV processing reports.
# Values are in the map units of the analysis CRS.
#
# Leave as None when unavailable. uav_biodiv then uses its conservative
# fallback and rejects unjustified large corrections.

RGB_RMS_X = None
RGB_RMS_Y = None

THERMAL_RASTER_RMS_X = None
THERMAL_RASTER_RMS_Y = None

MULTISPECTRAL_RASTER_RMS_X = None
MULTISPECTRAL_RASTER_RMS_Y = None


# ============================================================================
# DOWNLOAD / EXTRACTION HELPERS
# ============================================================================

def download_demo_zip(url: str, output_path: Path) -> Path:
    """Download Bahna_demo.zip from a public Google Drive share link."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temporary = output_path.with_suffix(output_path.suffix + ".part")
    temporary.unlink(missing_ok=True)

    print(f"Downloading demo data -> {output_path}")

    result = gdown.download(
        url=url,
        output=str(temporary),
        quiet=False,
        fuzzy=True,
        resume=True,
    )

    if result is None or not temporary.exists():
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Google Drive download failed. Check that Bahna_demo.zip is "
            "shared as 'Anyone with the link' and that data_zip_url points "
            "to the ZIP file."
        )

    if not zipfile.is_zipfile(temporary):
        temporary.unlink(missing_ok=True)
        raise RuntimeError(
            "Downloaded object is not a valid ZIP archive. "
            "Check that data_zip_url points to Bahna_demo.zip."
        )

    temporary.replace(output_path)
    return output_path


def _safe_extract_zip(zip_path: Path, destination: Path) -> None:
    """Extract a ZIP while rejecting paths that escape the destination."""
    destination = destination.resolve()
    destination.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as archive:
        for member in archive.infolist():
            target = (destination / member.filename).resolve()
            if destination != target and destination not in target.parents:
                raise RuntimeError(
                    f"Unsafe path in ZIP archive: {member.filename}"
                )
        archive.extractall(destination)


def find_demo_file(root: Path, filename: str) -> Path:
    """Find exactly one demo file recursively, ignoring macOS metadata."""
    wanted = filename.lower()

    matches = [
        path
        for path in root.rglob("*")
        if path.is_file()
        and "__MACOSX" not in path.parts
        and path.name.lower() == wanted
    ]

    if not matches:
        raise FileNotFoundError(
            f"Cannot find {filename!r} anywhere under {root}."
        )

    if len(matches) > 1:
        listing = "\n".join(f"  - {path}" for path in matches)
        raise RuntimeError(
            f"More than one copy of {filename!r} was found under {root}:\n"
            f"{listing}\n"
            "Keep only one copy of each demo input."
        )

    return matches[0]


def resolve_demo_inputs() -> None:
    """Resolve all six demo inputs recursively after extraction."""
    global BAHNA_LAS
    global BAHNA_THERMAL
    global BAHNA_MULTISPECTRAL
    global BAHNA_RGB
    global BAHNA_THERMAL_RASTER
    global BAHNA_MULTISPECTRAL_RASTER

    BAHNA_LAS = find_demo_file(DEMO_ROOT, "demo_lidar.las")
    BAHNA_THERMAL = find_demo_file(DEMO_ROOT, "demo_thermal.txt")
    BAHNA_MULTISPECTRAL = find_demo_file(DEMO_ROOT, "demo_multispectral.txt")

    BAHNA_RGB = find_demo_file(DEMO_ROOT, "demo_rgb.tif")
    BAHNA_THERMAL_RASTER = find_demo_file(
        DEMO_ROOT,
        "demo_thermal.tif",
    )
    BAHNA_MULTISPECTRAL_RASTER = find_demo_file(
        DEMO_ROOT,
        "demo_multispectral.tif",
    )


def ensure_demo_data() -> None:
    """Ensure that the six demo inputs are available locally.

    The archive structure itself is not assumed. The required files are found
    recursively by exact filename after extraction.
    """
    DEMO_ROOT.mkdir(parents=True, exist_ok=True)

    # First try to use an already extracted demo dataset.
    try:
        resolve_demo_inputs()
        print("Demo data already available; download/extraction skipped.")
        return
    except FileNotFoundError:
        pass

    # Download only when the ZIP is not already present.
    if not DATA_ZIP.exists():
        download_demo_zip(data_zip_url, DATA_ZIP)

    # Extract into DEMO_ROOT. The ZIP may itself contain a Bahna_demo folder.
    print(f"Extracting {DATA_ZIP} -> {DEMO_ROOT}")
    _safe_extract_zip(DATA_ZIP, DEMO_ROOT)

    # Remove macOS archive metadata if present.
    macosx_dirs = [
        p
        for p in DEMO_ROOT.rglob("__MACOSX")
        if p.is_dir()
    ]
    for directory in sorted(
        macosx_dirs,
        key=lambda p: len(p.parts),
        reverse=True,
    ):
        shutil.rmtree(directory, ignore_errors=True)

    # Resolve the actual paths, regardless of directory nesting.
    resolve_demo_inputs()

    print("\nResolved demo inputs")
    print(f"  LiDAR LAS           : {BAHNA_LAS}")
    print(f"  thermal TXT         : {BAHNA_THERMAL}")
    print(f"  multispectral TXT   : {BAHNA_MULTISPECTRAL}")
    print(f"  RGB raster          : {BAHNA_RGB}")
    print(f"  thermal raster      : {BAHNA_THERMAL_RASTER}")
    print(f"  multispectral raster: {BAHNA_MULTISPECTRAL_RASTER}")


# ============================================================================
# TXT SCHEMA HELPERS
# ============================================================================

def _split_text_line(line: str, delimiter: str) -> list[str]:
    stripped = line.strip()

    if delimiter in {"whitespace", ""}:
        return stripped.split()

    return stripped.split(delimiter)


def text_column_count(
    path: Path,
    *,
    delimiter: str,
    comment: str = "#",
) -> int:
    """Return the column count from the first valid TXT data row."""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            stripped = line.strip()

            if not stripped or stripped.startswith(comment):
                continue

            return len(_split_text_line(stripped, delimiter))

    raise ValueError(f"TXT file contains no data rows: {path}")


def validate_field_names(
    path: Path,
    field_names: list[str] | None,
    label: str,
) -> list[str]:
    """Validate explicit TXT field_names before expensive processing."""
    if not path.exists():
        raise FileNotFoundError(f"Missing {label} TXT file: {path}")

    actual = text_column_count(
        path,
        delimiter=TXT_DELIMITER,
    )

    if not field_names:
        raise ValueError(
            f"{label} TXT contains {actual} columns. "
            f"Define {label.upper()}_FIELD_NAMES with exactly one semantic "
            "name per column, in source order."
        )

    if len(field_names) != actual:
        raise ValueError(
            f"{label} TXT schema mismatch: file contains {actual} columns "
            f"but field_names defines {len(field_names)} fields: "
            f"{field_names}"
        )

    if len(set(field_names)) != len(field_names):
        raise ValueError(
            f"{label} field_names contains duplicate names: {field_names}"
        )

    return list(field_names)


def attributes_from_field_names(
    field_names: list[str],
    *,
    transfer_fields: list[str] | None = None,
    units: dict[str, str | None] | None = None,
) -> dict[str, dict[str, object]]:
    """Build the explicit set of TXT attributes transferred during fusion.

    ``field_names`` describes every source column. ``transfer_fields`` selects
    which non-coordinate fields are scientifically transferred to the reference
    geometry. This prevents source-cloud geometry fields such as normals from
    being silently reinterpreted as attributes of the LiDAR reference cloud.
    """
    units = units or {}
    coordinate_names = {"x", "y", "z"}
    available = set(field_names)

    if transfer_fields is None:
        selected = [
            name for name in field_names
            if name.lower() not in coordinate_names
        ]
    else:
        missing = [name for name in transfer_fields if name not in available]
        if missing:
            raise ValueError(
                f"Transfer fields are missing from field_names: {missing}. "
                f"Available fields: {field_names}"
            )
        selected = list(transfer_fields)

    attributes: dict[str, dict[str, object]] = {}
    for name in selected:
        if name.lower() in coordinate_names:
            raise ValueError(
                f"Coordinate field {name!r} cannot be transferred as an attribute."
            )
        attributes[name] = {
            "dimension": name,
            "unit": units.get(name),
        }

    if not attributes:
        raise ValueError(
            "At least one non-coordinate transfer field is required for point-cloud fusion."
        )

    return attributes


# ============================================================================
# RASTER HELPERS
# ============================================================================

def inspect_multispectral_raster(path: Path) -> tuple[list[str], int]:
    """Return generic keys for every multispectral raster band."""
    if not path.exists():
        raise FileNotFoundError(f"Missing multispectral raster: {path}")

    with rasterio.open(path) as src:
        count = int(src.count)

    if count < 1:
        raise ValueError(
            f"Multispectral raster contains no bands: {path}"
        )

    keys = [
        f"multispectral_{band_index}"
        for band_index in range(1, count + 1)
    ]

    return keys, count


def build_raster_configuration():
    """Build generic raster_sources/source_groups without spectral assumptions."""
    multispectral_keys, multispectral_count = inspect_multispectral_raster(
        BAHNA_MULTISPECTRAL_RASTER
    )

    raster_sources: dict[str, tuple[Path, int]] = {
        "rgb_red": (BAHNA_RGB, 1),
        "rgb_green": (BAHNA_RGB, 2),
        "rgb_blue": (BAHNA_RGB, 3),
        "thermal_value": (BAHNA_THERMAL_RASTER, 1),
    }

    for band_index, key in enumerate(multispectral_keys, start=1):
        raster_sources[key] = (
            BAHNA_MULTISPECTRAL_RASTER,
            band_index,
        )

    source_groups = {
        "rgb_geometry": {
            "keys": [
                "rgb_red",
                "rgb_green",
                "rgb_blue",
            ],
            "registration_keys": [
                "rgb_red",
                "rgb_green",
                "rgb_blue",
            ],
            "sensor": "imaging_source_rgb",
            "geometric_accuracy": {
                "rms_x": RGB_RMS_X,
                "rms_y": RGB_RMS_Y,
            },
        },

        "thermal_geometry": {
            "keys": [
                "thermal_value",
            ],
            "registration_keys": [
                "thermal_value",
            ],
            "sensor": "imaging_source_thermal",
            "geometric_accuracy": {
                "rms_x": THERMAL_RASTER_RMS_X,
                "rms_y": THERMAL_RASTER_RMS_Y,
            },
        },

        "multispectral_geometry": {
            "keys": multispectral_keys,
            "registration_keys": multispectral_keys,
            "sensor": "imaging_source_multispectral",
            "geometric_accuracy": {
                "rms_x": MULTISPECTRAL_RASTER_RMS_X,
                "rms_y": MULTISPECTRAL_RASTER_RMS_Y,
            },
        },
    }

    # The output band order follows raster_sources insertion order.
    output_band_keys = list(raster_sources.keys())

    return (
        raster_sources,
        source_groups,
        multispectral_keys,
        multispectral_count,
        output_band_keys,
    )


# ============================================================================
# MAIN
# ============================================================================

def main() -> None:
    assert_v091_api()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    ensure_demo_data()

    if any(
        value is None
        for value in (
            BAHNA_LAS,
            BAHNA_THERMAL,
            BAHNA_MULTISPECTRAL,
            BAHNA_RGB,
            BAHNA_THERMAL_RASTER,
            BAHNA_MULTISPECTRAL_RASTER,
        )
    ):
        raise RuntimeError("Demo inputs were not resolved correctly.")

    cfg = UAVProjectConfig(
        project_name="BioClima",
        base_data_dir=BAHNA_OUTPUT,
        crs=None,
        metadata={
            "site_name": "Bahna",
            "campaign_name": "June2026",
        },
    )

    cfg.set_processing_limits(mode="auto")

    output_dir = BAHNA_OUTPUT / cfg.output_prefix
    output_dir.mkdir(parents=True, exist_ok=True)

    prefix = cfg.output_prefix
    pipeline = UAVMonitoringPipeline(cfg)

    print("\nAutomatic resource plan")
    for key, value in cfg.processing_limits.resource_plan().items():
        print(f"  {key}: {value}")

    # ------------------------------------------------------------------------
    # LiDAR reference discovery / CRS
    # ------------------------------------------------------------------------

    lidar_sources = resolve_lidar_sources(BAHNA_LAS)

    print("\nLiDAR reference")
    print(f"  files: {len(lidar_sources)}")

    total_points = 0
    lidar_crs = None

    for path in lidar_sources:
        info = pipeline.ingestion.inspect_point_cloud(path)
        total_points += int(info.get("point_count", 0) or 0)

        if lidar_crs is None and info.get("crs"):
            lidar_crs = info["crs"]

    if lidar_crs is None:
        raise ValueError(
            "The LiDAR reference has no CRS. Define a CRS explicitly before "
            "using TXT point clouds because TXT does not embed CRS metadata."
        )

    print(f"  total header points: {total_points:,}")
    print(f"  CRS: {lidar_crs}")
    print(f"  first file: {lidar_sources[0]}")

    # ------------------------------------------------------------------------
    # Explicit TXT schemas
    # ------------------------------------------------------------------------

    thermal_fields = validate_field_names(
        BAHNA_THERMAL,
        THERMAL_FIELD_NAMES,
        "thermal",
    )

    multispectral_fields = validate_field_names(
        BAHNA_MULTISPECTRAL,
        MULTISPECTRAL_FIELD_NAMES,
        "multispectral",
    )

    thermal_attributes = attributes_from_field_names(
        thermal_fields,
        transfer_fields=THERMAL_TRANSFER_FIELDS,
        units=THERMAL_UNITS,
    )

    multispectral_attributes = attributes_from_field_names(
        multispectral_fields,
        transfer_fields=MULTISPECTRAL_TRANSFER_FIELDS,
        units=MULTISPECTRAL_UNITS,
    )

    print("\nTXT schemas")
    print("  thermal:", thermal_fields)
    print("  multispectral:", multispectral_fields)

    # ------------------------------------------------------------------------
    # POINT-CLOUD MULTIMODAL FUSION
    #
    # Reference geometry:
    #   LiDAR LAS/LAZ
    #
    # Attribute sources:
    #   thermal TXT
    #   multispectral TXT
    # ------------------------------------------------------------------------

    point_cloud_output = (
        output_dir
        / f"{prefix}_pointcloud_analysis_ready.laz"
    )

    pc_result = pipeline.run_multimodal_point_cloud_fusion(
        reference_path=BAHNA_LAS,

        source_clouds={
            "thermal_source": {
                "path": BAHNA_THERMAL,
                "crs": lidar_crs,

                "text_options": {
                    "delimiter": TXT_DELIMITER,
                    "has_header": TXT_HAS_HEADER,
                    "field_names": thermal_fields,
                },

                "attributes": thermal_attributes,

                "metadata": {
                    "source_type": "photogrammetric_point_cloud",
                    "campaign": "June2026",
                },
            },

            "multispectral_source": {
                "path": BAHNA_MULTISPECTRAL,
                "crs": lidar_crs,

                "text_options": {
                    "delimiter": TXT_DELIMITER,
                    "has_header": TXT_HAS_HEADER,
                    "field_names": multispectral_fields,
                },

                "attributes": multispectral_attributes,

                "metadata": {
                    "source_type": "photogrammetric_point_cloud",
                    "campaign": "June2026",
                },
            },
        },

        output_path=point_cloud_output,
        fusion_mode="reference_attributes",
        registration_method="icp",
        transfer_k=1,
        transfer_max_distance=TRANSFER_MAX_DISTANCE,
        registration_max_points=250_000,
        aoi=BAHNA_AOI,
        registration_scope="full",
    )

    print("\nPoint-cloud multimodal fusion")
    print("  output:", pc_result["output_path"])
    print("  metadata:", pc_result.get("metadata_path"))
    print("  transfer QA:", pc_result.get("transfer_stats"))

    # ------------------------------------------------------------------------
    # RASTER MULTIMODAL FUSION
    #
    # RGB + thermal + all channels in the multispectral GeoTIFF.
    #
    # No physical wavelength names are inferred here.
    # ------------------------------------------------------------------------

    required_rasters = (
        BAHNA_RGB,
        BAHNA_THERMAL_RASTER,
        BAHNA_MULTISPECTRAL_RASTER,
    )

    missing_rasters = [
        path
        for path in required_rasters
        if not path.exists()
    ]

    if missing_rasters:
        print("\nRaster workflow skipped")
        for path in missing_rasters:
            print(f"  missing: {path}")
        return

    (
        raster_sources,
        source_groups,
        multispectral_keys,
        multispectral_count,
        output_band_keys,
    ) = build_raster_configuration()

    print("\nRaster inputs")
    print(f"  RGB: {BAHNA_RGB}")
    print(f"  thermal: {BAHNA_THERMAL_RASTER}")
    print(
        f"  multispectral: {BAHNA_MULTISPECTRAL_RASTER} "
        f"({multispectral_count} bands)"
    )

    # ------------------------------------------------------------------------
    # Sensor metadata
    #
    # The demo stores only metadata that is actually known. Wavelengths,
    # reflectance units, calibration details, etc. should be added from the
    # acquisition/processing record, not guessed here.
    # ------------------------------------------------------------------------

    cfg.add_sensor(
        "imaging_source_rgb",
        sensor_type="imaging",
        bands={
            "rgb_red": {
                "raster_band": 1,
                "unit": None,
                "central_wavelength_nm": None,
            },
            "rgb_green": {
                "raster_band": 2,
                "unit": None,
                "central_wavelength_nm": None,
            },
            "rgb_blue": {
                "raster_band": 3,
                "unit": None,
                "central_wavelength_nm": None,
            },
        },
        acquisition={
            "campaign": "June2026",
        },
    )

    cfg.add_sensor(
        "imaging_source_thermal",
        sensor_type="imaging",
        bands={
            "thermal_value": {
                "raster_band": 1,
                "unit": None,
                "wavelength_min_nm": None,
                "wavelength_max_nm": None,
            },
        },
        acquisition={
            "campaign": "June2026",
        },
    )

    cfg.add_sensor(
        "imaging_source_multispectral",
        sensor_type="imaging",
        bands={
            key: {
                "raster_band": band_index,
                "unit": None,
                "central_wavelength_nm": None,
                "wavelength_min_nm": None,
                "wavelength_max_nm": None,
            }
            for band_index, key in enumerate(
                multispectral_keys,
                start=1,
            )
        },
        acquisition={
            "campaign": "June2026",
        },
    )

    analysis_ready_path = (
        output_dir
        / f"{prefix}_rgb_thermal_multispectral_analysis_ready.tif"
    )

    raster_result = pipeline.run_multimodal_raster_fusion(
        raster_sources=raster_sources,
        source_groups=source_groups,

        reference_group="rgb_geometry",
        reference_key="rgb_red",

        output_path=analysis_ready_path,

        registration_transform="auto",

        aoi=BAHNA_AOI,

        # The AOI clips the output but does not influence registration.
        registration_scope="full",
    )

    print("\nMultimodal raster processing")
    print(raster_result["qa"]["processing"])
    print("  metadata:", raster_result["metadata_path"])
    print("  registration QA:", raster_result["registration_path"])

    registration_groups = raster_result["registration"]["groups"]

    for group_name in (
        "thermal_geometry",
        "multispectral_geometry",
    ):
        registration = registration_groups[group_name]

        print()
        print(f"Registration: {group_name}")
        print("  accepted:", registration["accepted"])
        print("  applied:", registration["applied"])
        print("  reason:", registration["acceptance_reason"])
        print(
            "  estimated shift:",
            registration["estimated_shift_map_units"],
        )
        print(
            "  allowed shift:",
            registration["max_allowed_shift_map_units"],
        )

    # ------------------------------------------------------------------------
    # LiDAR raster metrics
    #
    # Use the already co-registered analysis-ready imagery raster as the
    # common target grid.
    # ------------------------------------------------------------------------

    # run_multimodal_raster_fusion writes bands following raster_sources
    # insertion order. Rebuild the mapping to the fused GeoTIFF here.
    fused_raster_sources = {
        key: (
            analysis_ready_path,
            band_index,
        )
        for band_index, key in enumerate(
            output_band_keys,
            start=1,
        )
    }

    point_raster_output = (
        output_dir
        / f"{prefix}_imagery_lidar_analysis_ready.tif"
    )

    point_raster_result = pipeline.run_lidar_raster_fusion(
        point_cloud_path=BAHNA_LAS,

        raster_sources=fused_raster_sources,

        reference_key="rgb_red",

        point_metrics=(
            "point_count",
            "point_density",
            "z_mean",
            "z_std",
            "z_p95",
        ),

        compute_indices=False,

        output_path=point_raster_output,

        aoi=BAHNA_AOI,
    )

    print("\nPoint-raster processing")
    print(point_raster_result["point_metrics_processing"])
    print("  output:", point_raster_output)


if __name__ == "__main__":
    main()


