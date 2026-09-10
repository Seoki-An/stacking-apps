import csv
import json
from pathlib import Path

import numpy as np
import pytest

from stacking_common.pointcloud import PointCloudData, RegistrationResult
from stacking_tabletop.config import load_perception_profile
from stacking_tabletop.perception import (
    DepthFilterConfig,
    LoadedCaptureView,
    ManualObjectExtractor,
    ManualObjectSelection,
    ModeledGripperExtractor,
    TabletopCaptureView,
    TabletopRawCapture,
    TriangleMeshData,
    filter_depth_burst,
)
from stacking_tabletop.perception.in_hand import IncrementalInHandAligner, _PrepareProgress
from stacking_tabletop.perception.pose_graph import build_pose_graph_edge_pairs


ROOT = Path(__file__).resolve().parents[1]


def test_depth_burst_uses_distinct_temporal_support() -> None:
    frames = tuple(
        np.array([[1000, 0], [1000 + index, 2000]], dtype=np.uint16)
        for index in range(5)
    )

    depth, keep, stats = filter_depth_burst(
        frames,
        0.001,
        DepthFilterConfig(minimum_temporal_support=5),
    )

    assert np.allclose(depth[keep], [1.0, 1.002, 2.0], atol=2e-7)
    assert keep.tolist() == [[True, False], [True, True]]
    assert stats.kept_pixels == 3


def test_raw_capture_loads_frame_explicit_metadata(tmp_path: Path) -> None:
    root = tmp_path / "capture"
    (root / "raw_pcd").mkdir(parents=True)
    (root / "raw_pcd" / "10.pcd").touch()
    (root / "camera_frame.json").write_text(
        json.dumps(
            {
                "cloud_frame": "camera",
                "depth_mode": "mode",
                "depth_scale": 1.0,
                "depth_offset_m": 0.0,
            }
        )
    )
    (root / "depth_camera_info.json").write_text(
        json.dumps(
            {
                "frame_id": "camera",
                "width": 1,
                "height": 1,
                "k": [1, 0, 0, 0, 1, 0, 0, 0, 1],
                "d": [],
            }
        )
    )
    (root / "organized_depth_manifest.json").write_text(
        json.dumps(
            {
                "values_are_raw": True,
                "depth_unit_m": 0.001,
                "depth_mode": "mode",
                "camera_info": "depth_camera_info.json",
                "views": [{"representative_pcd": "raw_pcd/10.pcd", "depth_frames": []}],
            }
        )
    )
    row = {
        "filename": "10.pcd",
        "cloud_frame": "camera",
        "view_index": "3",
        "gripper_position": "900",
        "transform_status": "exact",
        **{name: "0.0" for name in (
            "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
            "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
        )},
    }
    for prefix in ("base_T_cloud", "base_T_tool"):
        for i in range(4):
            for j in range(4):
                row[f"{prefix}_{i}{j}"] = "1.0" if i == j else "0.0"
    with (root / "frames.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(row))
        writer.writeheader()
        writer.writerow(row)

    capture = TabletopRawCapture.load(root)

    assert capture.views[0].capture_index == 3
    assert np.allclose(capture.views[0].tool_T_camera, np.eye(4))


def test_manual_object_selection_is_reused_per_view() -> None:
    selection = ManualObjectSelection(
        np.array([-1, 1, -1, 1, -1, 1]),
        np.array([[0.5, 0.0, 0.0]]),
        0.1,
        0.9,
    )
    cloud = PointCloudData([[0, 0, 0], [0.5, 0, 0], [2, 0, 0]])

    result = ManualObjectExtractor(selection).extract(cloud)

    assert np.allclose(result.points, [[0, 0, 0]])


def test_modeled_gripper_extractor_removes_surface_points() -> None:
    pytest.importorskip("open3d")
    vertices = np.array(
        [[x, y, z] for x in (-0.1, 0.1) for y in (-0.1, 0.1) for z in (-0.1, 0.1)]
    )
    triangles = np.array(
        [
            [0, 1, 3], [0, 3, 2], [4, 6, 7], [4, 7, 5],
            [0, 4, 5], [0, 5, 1], [2, 3, 7], [2, 7, 6],
            [0, 2, 6], [0, 6, 4], [1, 5, 7], [1, 7, 3],
        ]
    )
    mesh = TriangleMeshData(vertices, triangles)
    extractor = ModeledGripperExtractor(lambda _: (mesh,), 0.01)
    view = _view(0)

    result = extractor.extract(PointCloudData([[0.1, 0, 0], [0.5, 0, 0]]), view)

    assert np.allclose(result.points, [[0.5, 0, 0]])


def test_incremental_gicp_aligns_to_accumulated_reference() -> None:
    pytest.importorskip("open3d")
    rng = np.random.default_rng(55)
    reference = PointCloudData(
        rng.uniform([-0.03, -0.02, -0.01], [0.04, 0.025, 0.035], (1800, 3))
    )
    offset = np.eye(4)
    offset[:3, 3] = [0.004, -0.002, 0.001]
    stats = type("Stats", (), {"kept_pixels": 1800})()
    loaded = (
        LoadedCaptureView(_view(0), reference, stats),
        LoadedCaptureView(_view(1), reference.transformed(offset), stats),
    )
    profile = load_perception_profile(ROOT / "apps/tabletop/config/perception.yml")

    result = IncrementalInHandAligner(profile).align(loaded)

    assert result[1].decision.accepted
    assert np.allclose(result[1].correction, np.linalg.inv(offset), atol=4e-4)


def test_icp_inspection_reprocesses_reference_after_going_back() -> None:
    profile = load_perception_profile(ROOT / "apps/tabletop/config/perception.yml")
    aligner = IncrementalInHandAligner(profile)

    class IdentityBackend:
        def register(self, source, target, initial):
            return RegistrationResult(initial, 1.0, 1.0, 0.0, len(source), ())

    class Inspector:
        def __init__(self):
            self.actions = iter(("accept", "accept", "back", "reject", "accept"))
            self.filenames = []
            self.reference_sizes = []

        def inspect_icp(self, case):
            self.filenames.append(case.metadata.filename)
            self.reference_sizes.append(
                0 if case.reference is None else len(case.reference)
            )
            return next(self.actions)

    aligner.backend = IdentityBackend()
    stats = type("Stats", (), {"kept_pixels": 4})()
    base = np.array(
        [[0.0, 0.0, 0.0], [0.004, 0.0, 0.0], [0.0, 0.004, 0.0], [0.0, 0.0, 0.004]]
    )
    loaded = tuple(
        LoadedCaptureView(_view(index), PointCloudData(base + 0.02 * index), stats)
        for index in range(3)
    )
    inspector = Inspector()
    progress = []

    result = aligner.align(
        loaded,
        inspector=inspector,
        progress=lambda completed, total: progress.append((completed, total)),
    )

    assert inspector.filenames == ["0.pcd", "1.pcd", "2.pcd", "1.pcd", "2.pcd"]
    assert [view.selected for view in result] == [True, False, True]
    assert inspector.reference_sizes[-1] == inspector.reference_sizes[1]
    assert progress == [(1, 3), (2, 3), (3, 3)]


def test_prepare_progress_includes_preceding_phases_without_resetting() -> None:
    updates = []
    progress = _PrepareProgress(
        3,
        lambda completed, total, message: updates.append(
            (completed, total, message)
        ),
    )

    progress.update(0, 2, "load")
    progress.finish(0, "loaded")
    progress.update(1, 1, "align")
    progress.finish(1, "aligned")
    progress.update(2, 2, "preliminary merge")
    progress.finish(2, "preliminary merged")
    progress.finish(3, "extracted")
    progress.update(4, 0, "pose graph")
    progress.finish(4, "pose graph complete")
    progress.update(5, 2, "final merge")
    progress.finish(5, "final merged")
    progress.finish(6, "exported")

    completed = [update[0] for update in updates]
    assert completed == sorted(completed)
    assert updates[4][:2] == (8, 17)
    assert updates[-1][:2] == (17, 17)


def test_pose_graph_topology_does_not_bridge_missing_capture_frames() -> None:
    viewpoints = tuple(np.eye(4) for _ in range(3))
    viewpoints[1][:3, 3] = [0.1, 0.0, 0.0]
    viewpoints[2][:3, 3] = [0.1, 0.0, 0.0]

    edges = build_pose_graph_edge_pairs(
        3,
        capture_indices=(0, 2, 8),
        viewpoints=viewpoints,
    )

    assert (1, 0, False) not in edges
    assert (2, 1, True) in edges


def _view(index: int) -> TabletopCaptureView:
    return TabletopCaptureView(
        filename=f"{index}.pcd",
        capture_index=index,
        joints=np.zeros(6),
        gripper_position=900.0,
        base_T_camera=np.eye(4),
        base_T_tool=np.eye(4),
    )
