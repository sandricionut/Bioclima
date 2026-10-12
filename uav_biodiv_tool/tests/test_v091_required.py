import inspect

import numpy as np
import pytest

from uav_biodiv import DataIngestionHarmonization, UAVMonitoringPipeline


def test_v091_point_cloud_fusion_exposes_aoi_api():
    signature = inspect.signature(UAVMonitoringPipeline.run_point_cloud_fusion)
    for name in ("aoi", "aoi_crs", "registration_scope"):
        assert name in signature.parameters


def test_headerless_txt_requires_field_names(tmp_path):
    path = tmp_path / "points.txt"
    path.write_text("0 0 1 20.5\n1 0 1.1 21.0\n", encoding="utf-8")

    ingestion = DataIngestionHarmonization()
    with pytest.raises(ValueError, match="field_names.*required"):
        ingestion.inspect_point_cloud(
            path,
            crs="EPSG:32634",
            text_options={
                "delimiter": "whitespace",
                "has_header": False,
            },
        )


def test_field_names_define_all_txt_columns_and_can_rename_fields(tmp_path):
    path = tmp_path / "thermal.txt"
    path.write_text(
        "100 200 10 20.5 -0.1 -0.2 0.97\n"
        "101 201 11 21.0 -0.2 -0.1 0.97\n",
        encoding="utf-8",
    )

    options = {
        "delimiter": "whitespace",
        "has_header": False,
        "field_names": [
            "easting",
            "northing",
            "elevation",
            "temperature",
            "normal_x",
            "normal_y",
            "normal_z",
        ],
        "coordinate_fields": {
            "x": "easting",
            "y": "northing",
            "z": "elevation",
        },
    }

    ingestion = DataIngestionHarmonization()
    info = ingestion.inspect_point_cloud(
        path,
        crs="EPSG:32634",
        text_options=options,
    )
    assert info["text_schema"]["field_names"] == options["field_names"]

    chunks = list(
        ingestion.iter_point_cloud_chunks(
            path,
            crs="EPSG:32634",
            attributes=["temperature"],
            chunk_points=1,
            text_options=options,
        )
    )
    temperature = np.concatenate(
        [chunk.attributes["temperature"] for chunk in chunks]
    )
    np.testing.assert_allclose(temperature, [20.5, 21.0])


def test_txt_column_count_must_match_field_names(tmp_path):
    path = tmp_path / "points.txt"
    path.write_text("0 0 1 20.5 -0.1 -0.2 0.97\n", encoding="utf-8")

    ingestion = DataIngestionHarmonization()
    with pytest.raises(ValueError, match="contains 7 columns but field_names defines 4"):
        ingestion.inspect_point_cloud(
            path,
            text_options={
                "delimiter": "whitespace",
                "has_header": False,
                "field_names": ["x", "y", "z", "temperature"],
            },
        )
