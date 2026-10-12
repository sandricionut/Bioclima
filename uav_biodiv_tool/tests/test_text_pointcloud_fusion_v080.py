import numpy as np
import pytest

try:
    import laspy
except ImportError:
    laspy = None

from uav_biodiv import DataIngestionHarmonization, UAVMonitoringPipeline, UAVProjectConfig


def _write_las(path, xyz):
    if laspy is None:
        pytest.skip("laspy is not installed")
    header = laspy.LasHeader(point_format=3, version="1.2")
    header.scales = np.array([0.001, 0.001, 0.001])
    las = laspy.LasData(header)
    las.x = xyz[:, 0]
    las.y = xyz[:, 1]
    las.z = xyz[:, 2]
    las.write(path)


def test_text_reader_streams_xyz_and_arbitrary_attribute(tmp_path):
    path = tmp_path / "thermal.txt"
    path.write_text(
        "0 0 1 20.5\n1 0 1.1 21.0\n0 1 1.2 22.25\n",
        encoding="utf-8",
    )
    ingestion = DataIngestionHarmonization()
    options = {
        "delimiter": "whitespace",
        "has_header": False,
        "columns": {"x": 0, "y": 1, "z": 2, "temperature": 3},
    }
    info = ingestion.inspect_point_cloud(path, crs="EPSG:32634", text_options=options)
    assert info["point_count"] == 3
    assert "temperature" in info["dimensions"]
    chunks = list(
        ingestion.iter_point_cloud_chunks(
            path,
            crs="EPSG:32634",
            attributes=["temperature"],
            chunk_points=2,
            text_options=options,
        )
    )
    xyz = np.concatenate([c.xyz for c in chunks])
    temp = np.concatenate([c.attributes["temperature"] for c in chunks])
    np.testing.assert_allclose(xyz[:, :2], [[0, 0], [1, 0], [0, 1]])
    np.testing.assert_allclose(temp, [20.5, 21.0, 22.25])


@pytest.mark.skipif(laspy is None, reason="laspy is not installed")
def test_mixed_las_reference_txt_source_streaming_fusion(tmp_path):
    rows, cols = np.indices((6, 6))
    xyz = np.column_stack((cols.ravel(), rows.ravel(), 0.05 * rows.ravel())).astype(float)
    ref = tmp_path / "reference.las"
    txt = tmp_path / "thermal.txt"
    out = tmp_path / "fused.laz"
    _write_las(ref, xyz)
    temperature = 15.0 + np.arange(len(xyz), dtype=float) * 0.1
    matrix = np.column_stack((xyz, temperature))
    np.savetxt(txt, matrix, fmt="%.6f")

    cfg = UAVProjectConfig("test", tmp_path)
    cfg.set_processing_limits(mode="manual", max_workers=1, point_chunk_size=20)
    result = UAVMonitoringPipeline(cfg).run_multimodal_point_cloud_fusion(
        reference_path=ref,
        source_clouds={
            "thermal": {
                "path": txt,
                "text_options": {
                    "delimiter": "whitespace",
                    "has_header": False,
                    "columns": {"x": 0, "y": 1, "z": 2, "temperature": 3},
                },
                "attributes": {
                    "temperature": {"dimension": "temperature", "unit": "degC"},
                },
            }
        },
        output_path=out,
        voxel_size=0.0,
        max_correspondence_distance=0.2,
        transfer_max_distance=0.2,
        registration_max_points=1000,
    )
    assert result["streaming"] is True
    fused = laspy.read(out)
    names = set(str(x) for x in fused.point_format.dimension_names)
    assert "temperature" in names
    np.testing.assert_allclose(fused["temperature"], temperature, atol=1e-5)


@pytest.mark.skipif(laspy is None, reason="laspy is not installed")
def test_las_tile_collection_reference(tmp_path):
    xyz_a = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float)
    xyz_b = np.array([[2, 0, 0], [2, 1, 0], [1, 1, 0]], dtype=float)
    ref_dir = tmp_path / "tiles"
    ref_dir.mkdir()
    _write_las(ref_dir / "tile_a.las", xyz_a)
    _write_las(ref_dir / "tile_b.las", xyz_b)
    xyz = np.vstack([xyz_a, xyz_b])
    txt = tmp_path / "thermal.txt"
    temperature = np.arange(len(xyz), dtype=float) + 20
    np.savetxt(txt, np.column_stack((xyz, temperature)), fmt="%.6f")
    out = tmp_path / "fused.las"

    cfg = UAVProjectConfig("test", tmp_path)
    cfg.set_processing_limits(mode="manual", max_workers=1, point_chunk_size=3)
    result = UAVMonitoringPipeline(cfg).run_multimodal_point_cloud_fusion(
        reference_path=ref_dir,
        source_clouds={
            "thermal": {
                "path": txt,
                "text_options": {
                    "has_header": False,
                    "columns": {"x": 0, "y": 1, "z": 2, "temperature": 3},
                },
                "attributes": {"temperature": {"dimension": "temperature", "unit": "degC"}},
            }
        },
        output_path=out,
        voxel_size=0.0,
        max_correspondence_distance=0.25,
        transfer_max_distance=0.25,
        registration_max_points=100,
    )
    assert result["streaming"] is True
    assert result["qa"]["streaming"]["reference_file_count"] == 2
    fused = laspy.read(out)
    assert len(fused.points) == 6
    assert "temperature" in set(str(x) for x in fused.point_format.dimension_names)
