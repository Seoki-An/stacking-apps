"""Desktop-only perception, recorded import and durable review acceptance."""

from dataclasses import replace
from pathlib import Path
import json

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.pointcloud.types import PointCloudData
from stacking_common.pointcloud.identification import identify_pose
from stacking_common.pointcloud.registration import (
    RegistrationResult, RegistrationScaleResult, RegistrationQualityConfig,
)
from stacking_excavator.acquisition.capture import CaptureFrame, CropRequest, digest
from stacking_excavator.acquisition.emulation import run_demonstration
from stacking_excavator.acquisition.recorded import import_recording
from stacking_excavator.acquisition.storage import (
    atomic_json, create_capture, file_hash, load_manifest, load_product, prepare_crop,
)
from stacking_excavator.perception.pipeline import DesktopPerception, orientation_seeds
from stacking_excavator.perception.review import ReplayReviewStore
from stacking_excavator.perception.roi import enclosing_crop

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def desktop(tmp_path_factory):
    pytest.importorskip("open3d")
    directory = tmp_path_factory.mktemp("perception") / "capture"
    run_demonstration(ROOT, directory, kinds=("field", "inhand"))
    return directory, DesktopPerception(ROOT)


def prior(kind):
    import open3d as o3d
    mesh = o3d.io.read_triangle_mesh(str(ROOT / "apps/excavator/assets/objects/model_10_mesh.ply"))
    pose = np.eye(4)
    if kind == "inhand":
        pose[:3, 3] = [0, .65, 0] - mesh.get_center()
    else:
        pose[:3, 3] = [3 - mesh.get_center()[0], -mesh.get_center()[1], -mesh.get_min_bound()[2]]
    return pose


@pytest.mark.parametrize("kind", ["field", "inhand"])
def test_cropped_and_raw_perception_recover_same_emulated_pose(desktop, tmp_path, kind):
    directory, pipeline = desktop
    expected = prior(kind)
    proposal, _ = pipeline.identify(directory / kind / "desktop", {10: expected}, expected_revision=1)
    assert proposal["valid"], proposal
    assert proposal["timestamp_sources"] == ["emulated"]
    assert proposal["objects"][0]["candidate_count"] == (64 if kind == "field" else 4)
    actual = np.array(proposal["objects"][0]["pose"])
    # Rays intersect reconstructed PLY; PoseID uses the measured PCD catalog.
    # These surfaces differ by centimetres. The exact-model scene test below
    # separately checks millimetric registration without that model discrepancy.
    np.testing.assert_allclose(actual[:3, 3], expected[:3, 3], atol=.04)
    assert np.rad2deg(Rotation.from_matrix(actual[:3, :3]).magnitude()) < 5.
    crop = CropRequest(**load_manifest(directory / kind / "desktop")["crop"])
    prepare_crop(directory / kind / "raw", tmp_path / "wide", "wide", replace(crop, half_extents=(100.,)*3),
                 maximum_timestamp_skew_ns=0)
    uncropped, _ = pipeline.identify(tmp_path / "wide", {10: expected}, expected_revision=1)
    assert uncropped["valid"], uncropped
    np.testing.assert_allclose(uncropped["objects"][0]["pose"], actual, atol=.02)
    if kind == "inhand":
        assert all(o["valid"] for o in proposal["opening"])
        assert all(abs(o["estimated"] + .45) < .2 for o in proposal["opening"])


def test_perception_rejects_stale_revision_and_wrong_id(desktop):
    directory, pipeline = desktop
    with pytest.raises(ValueError, match="expected scene revision"):
        pipeline.identify(directory / "field/desktop", {10: prior("field")}, expected_revision=True)
    with pytest.raises(ValueError, match="stale scene"):
        pipeline.identify(directory / "field/desktop", {10: prior("field")}, expected_revision=2)
    with pytest.raises(ValueError, match="object IDs"):
        pipeline.identify(directory / "field/desktop", {11: prior("field")}, expected_revision=1)


def test_empty_crop_is_unknown_not_object_absence(desktop, tmp_path):
    directory, pipeline = desktop
    crop = CropRequest(**load_manifest(directory / "field/desktop")["crop"])
    transform = np.eye(4)
    transform[:3, 3] = [100, 100, 100]
    prepare_crop(directory / "field/raw", tmp_path / "empty", "empty",
                 replace(crop, reference_T_box=transform), maximum_timestamp_skew_ns=0)
    proposal, _ = pipeline.identify(tmp_path / "empty", {10: prior("field")}, expected_revision=1)
    assert not proposal["valid"]
    assert proposal["objects"][0]["pose"] is None
    assert proposal["objects"][0]["reasons"] == ["insufficient_evidence"]
    assert proposal["next_action"] == "recrop_or_reinitialize"


def test_scene_points_are_assigned_to_distinct_expected_stones(desktop, tmp_path):
    _, pipeline = desktop
    poses = {10: np.eye(4), 11: np.eye(4)}
    poses[10][:3, 3] = [0., 0., 1.]
    poses[11][:3, 3] = [1.3, 0., 1.5]  # overhang above the first stone
    clouds = [pipeline.models.stone(i)[0].transformed(p).points for i, p in poses.items()]
    points = np.concatenate(clouds)
    calibration = digest({"scene": "emulated"})
    frame = CaptureFrame("f0", "lidar", 1, 1, np.eye(4), np.eye(4), calibration, points, (0.,)*6, -.45)
    crop = CropRequest("scene", "world", np.eye(4), (3.,)*3, (.1,)*3, calibration, 0, (10, 11))
    create_capture(tmp_path / "raw", "scene", "scene", 0, [frame])
    prepare_crop(tmp_path / "raw", tmp_path / "crop", "scene-crop", crop, maximum_timestamp_skew_ns=0)
    proposal, _ = pipeline.identify(tmp_path / "crop", poses, expected_revision=0)
    assert proposal["valid"], proposal
    assert sum(o["target_points"] for o in proposal["objects"]) <= len(points)
    for obj in proposal["objects"]:
        np.testing.assert_allclose(obj["pose"], poses[obj["object_id"]], atol=.003)


def test_legacy_orientation_conventions():
    initial = np.eye(4)
    initial[:3, :3] = Rotation.from_euler("xyz", [.1, .2, .3]).as_matrix()
    field = orientation_seeds("field", initial)
    assert len(field) == 64
    np.testing.assert_array_equal(field[0][:3, :3], np.eye(3))
    inhand = orientation_seeds("inhand", initial)
    np.testing.assert_allclose(inhand[1][:3, :3],
        Rotation.from_euler("x", 90, degrees=True).as_matrix() @ initial[:3, :3])


def test_distinct_equal_registration_scores_are_ambiguous():
    first = np.eye(4)
    second = np.eye(4)
    second[:3, :3] = Rotation.from_euler("z", 90, degrees=True).as_matrix()

    class Backend:
        def register(self, source, target, seed):
            return RegistrationResult(seed, 1., 1., .001, 100,
                (RegistrationScaleResult(.01, 1., .001, 100),))

    cloud = PointCloudData(np.zeros((100, 3)))
    quality = RegistrationQualityConfig(.3, .2, .03, .6, 180.)
    kwargs = dict(minimum_points=100, ambiguity_score_gap=.01,
                  distinct_translation=.05, distinct_rotation_degrees=15.)
    result = identify_pose(cloud, cloud, [first, second], Backend(), quality, **kwargs)
    assert not result.valid and result.reasons == ("ambiguous_pose",)
    result = identify_pose(cloud, cloud, [first, first.copy()], Backend(), quality, **kwargs)
    assert result.valid


def test_opening_empty_and_wrong_measurement_are_invalid(desktop):
    directory, pipeline = desktop
    assert not pipeline.estimate_opening(PointCloudData(np.empty((0, 3))), -.45)["valid"]
    _, frames = load_product(directory / "inhand/desktop")
    f = frames[0]
    cloud = PointCloudData(f.points).transformed(np.linalg.inv(f.world_T_gripper) @ f.world_T_sensor)
    result = pipeline.estimate_opening(cloud, -1.1)
    assert not result["valid"]


@pytest.mark.parametrize("extension", ["npy", "npz", "ply", "pcd"])
def test_recorded_import_preserves_channels_and_unknown_timing(tmp_path, extension):
    o3d = pytest.importorskip("open3d")
    points = np.array([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    intensity = np.array([[1], [7]], dtype=np.uint16)
    source = tmp_path / ("source." + extension)
    if extension == "npy":
        np.save(source, points)
    elif extension == "npz":
        np.savez(source, xyz=points, attr_intensity=intensity)
    else:
        cloud = o3d.t.geometry.PointCloud(o3d.core.Tensor(points))
        cloud.point["intensity"] = o3d.core.Tensor(intensity)
        assert o3d.t.io.write_point_cloud(str(source), cloud)
    frame = CaptureFrame("frame", "lidar", None, None, np.eye(4), np.eye(4),
        digest({"calibration": "recorded"}), points, (0.,)*6, 0., timestamp_source="recorded_unknown")
    sidecar = {"schema_version": 1, "capture_id": "recording", "scene_revision": 0, "kind": "scene",
        "frames": [{**frame.metadata(), "source": source.name, "source_sha256": file_hash(source)}]}
    atomic_json(tmp_path / "sidecar.json", sidecar)
    imported = import_recording(tmp_path / "sidecar.json", tmp_path / "raw")
    assert imported["recording"]["sources"][0]["sha256"] == file_hash(source)
    crop = CropRequest("scene", "world", np.eye(4), (10.,)*3, (0.,)*3, frame.calibration_sha256, 0, (10,))
    prepare_crop(tmp_path / "raw", tmp_path / "crop", "crop", crop, maximum_timestamp_skew_ns=0)
    _, frames = load_product(tmp_path / "crop")
    np.testing.assert_array_equal(frames[0].points, points)
    assert frames[0].timestamp_ns is None
    if extension != "npy":
        np.testing.assert_array_equal(frames[0].attributes["intensity"], intensity)
    source.write_bytes(b"corrupted")
    with pytest.raises(ValueError, match="source hash"):
        import_recording(tmp_path / "sidecar.json", tmp_path / "bad")


def review_proposal(revision=0, object_id=10, valid=True):
    result = {"schema_version": 1, "scene_revision": revision, "kind": "scene", "frame": "world",
        "valid": valid, "opening": [], "objects": [{"object_id": object_id, "pose": np.eye(4).tolist(),
        "valid": valid, "reasons": [] if valid else ["insufficient_evidence"]}]}
    result["sha256"] = digest(result)
    return result


def test_review_is_durable_and_rejects_stale_or_repeated_decisions(tmp_path):
    store = ReplayReviewStore(tmp_path / "review.db")
    first = store.stage(review_proposal())
    stale = store.stage(review_proposal(object_id=11))
    store.decide(first, accept=True, expected_revision=0)
    restarted = ReplayReviewStore(tmp_path / "review.db")
    assert restarted.snapshot()["revision"] == 1
    with pytest.raises(ValueError, match="consumed"):
        restarted.decide(first, accept=True, expected_revision=1)
    with pytest.raises(ValueError, match="stale"):
        restarted.decide(stale, accept=True, expected_revision=1)
    second = restarted.stage(review_proposal(revision=1, object_id=11))
    snapshot = restarted.decide(second, accept=True, expected_revision=1)
    assert set(snapshot["poses"]) == {"10", "11"}  # unseen objects are never deleted


def test_invalid_proposal_can_be_rejected_but_cannot_be_accepted(tmp_path):
    store = ReplayReviewStore(tmp_path / "review.db")
    token = store.stage(review_proposal(valid=False))
    with pytest.raises(ValueError, match="invalid perception"):
        store.decide(token, accept=True, expected_revision=0)
    assert store.decide(token, accept=False, expected_revision=0)["revision"] == 0


def test_scene_roi_keeps_overhang_and_ground_context(desktop):
    _, pipeline = desktop
    pose = np.eye(4)
    pose[:3, 3] = [4., 0., 1.]
    target_corners = np.array([[0., 0., 0.], [1., 1., 2.]])
    crop = enclosing_crop("scene", pipeline.models, {10: pose}, target_corners=target_corners,
        calibration_sha256="a"*64, scene_revision=2, ground_height=-.5)
    lower = crop.reference_T_box[:3, 3] - crop.half_extents
    upper = crop.reference_T_box[:3, 3] + crop.half_extents
    stone = pipeline.models.stone(10)[0].transformed(pose).points
    assert np.all(stone.min(axis=0) >= lower)
    assert np.all(stone.max(axis=0) <= upper)
    assert lower[2] <= -.5 and upper[0] > 4.


def test_only_one_concurrent_review_can_advance_a_revision(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    path = tmp_path / "review.db"
    store = ReplayReviewStore(path)
    tokens = [store.stage(review_proposal(object_id=i)) for i in (10, 11)]
    def decide(token):
        try:
            ReplayReviewStore(path).decide(token, accept=True, expected_revision=0)
            return True
        except ValueError as error:
            assert "stale" in str(error)
            return False
    with ThreadPoolExecutor(2) as pool:
        assert sum(pool.map(decide, tokens)) == 1
    assert store.snapshot()["revision"] == 1


def test_migrated_measured_catalog_has_verified_xyz(desktop):
    _, pipeline = desktop
    for object_id in (1, 10, 17):
        cloud, sha = pipeline.models.stone(object_id)
        entry = pipeline.models.catalog[str(object_id)]
        assert len(cloud) == entry["points"]
        assert sha == entry["sha256"]


def test_recorded_field_retains_ground_and_reproduces_legacy_pose(desktop, tmp_path):
    _, pipeline = desktop
    assert pipeline.profile['field_ground_removal'] is False
    fixture = ROOT / 'tests/fixtures/legacy_planner/field_recovery.npz'
    with np.load(fixture, allow_pickle=False) as arrays:
        points, initial, expected = (arrays[k] for k in ('points', 'initial', 'expected'))
    calibration = digest({'fixture': 'recorded field recovery'})
    frame = CaptureFrame('f0', 'world', None, None, np.eye(4), np.eye(4), calibration,
        points, (0.,)*6, None, timestamp_source='recorded_unknown')
    crop = CropRequest('field', 'world', np.eye(4), (100.,)*3, (0.,)*3, calibration, 0, (17,))
    create_capture(tmp_path / 'raw', 'field', 'field', 0, [frame])
    prepare_crop(tmp_path / 'raw', tmp_path / 'crop', 'field-crop', crop, maximum_timestamp_skew_ns=0)
    proposal, cloud = pipeline.identify(tmp_path / 'crop', {17: initial}, expected_revision=0)
    np.testing.assert_allclose(proposal['objects'][0]['pose'], expected, atol=1e-7)
    assert proposal['objects'][0]['reasons'] == ['low_target_coverage']
    assert np.count_nonzero(cloud.points[:, 2] <= .08) > 1000
    try:
        pipeline.profile['field_ground_removal'] = True
        experimental, cloud = pipeline.identify(tmp_path / 'crop', {17: initial}, expected_revision=0)
        assert np.all(cloud.points[:, 2] > .08)
        assert np.linalg.norm(np.array(experimental['objects'][0]['pose'])[:3, 3] - expected[:3, 3]) > .1
    finally:
        pipeline.profile['field_ground_removal'] = False


@pytest.mark.parametrize('session,index', [(s, i) for s in range(2) for i in (0, 4, 8)])
def test_recorded_inhand_views_have_bounded_gripper_alignment(desktop, session, index):
    from stacking_common.pointcloud.processing import voxel_downsample
    _, pipeline = desktop
    with np.load(ROOT / 'tests/fixtures/legacy_planner/perception_views.npz', allow_pickle=False) as arrays:
        p = arrays[f'inhand_{session}_{index}']
    p = p[(abs(p[:, 0]) < 2) & (p[:, 1] > 0) & (p[:, 1] < 2) & (abs(p[:, 2]) < 2)]
    cloud = voxel_downsample(PointCloudData(p), .003)
    opening = pipeline.estimate_opening(cloud, None)
    assert opening['valid'] and not opening['measured_valid']
    aligned, report = pipeline.align_gripper(cloud, opening['estimated'])
    assert report['valid'], report
    assert report['translation_m'] < .04 and report['rotation_degrees'] < 5
    assert len(aligned) == len(cloud)
    # A distant cloud must not be pulled onto the jaws and accepted.
    distant = PointCloudData(cloud.points + [5., 0., 0.])
    unchanged, rejected = pipeline.align_gripper(distant, opening['estimated'])
    assert not rejected['valid']
    np.testing.assert_array_equal(unchanged.points, distant.points)


@pytest.mark.parametrize('session', [0, 1])
def test_recorded_scene_does_not_invent_an_unobserved_stone(desktop, tmp_path, session):
    _, pipeline = desktop
    with np.load(ROOT / 'tests/fixtures/legacy_planner/perception_views.npz', allow_pickle=False) as arrays:
        points = arrays[f'scene_{session}']
    assert len(points) > 10000
    initial = np.eye(4)
    initial[:3, 3] = [20., 20., 3.]  # Deliberately unobserved expected object.
    calibration = digest({'fixture': session})
    frame = CaptureFrame('f0', 'world', None, None, np.eye(4), np.eye(4), calibration,
        points, (0.,)*6, None, timestamp_source='recorded_unknown')
    crop = CropRequest('scene', 'world', np.eye(4), (100.,)*3, (0.,)*3, calibration, 0, (10,))
    create_capture(tmp_path / 'raw', 'scene', 'scene', 0, [frame])
    prepare_crop(tmp_path / 'raw', tmp_path / 'crop', 'scene-crop', crop, maximum_timestamp_skew_ns=0)
    proposal, _ = pipeline.identify(tmp_path / 'crop', {10: initial}, expected_revision=0)
    assert not proposal['valid']
    assert proposal['objects'][0]['pose'] is None
    assert proposal['objects'][0]['reasons'] == ['insufficient_evidence']
    store = ReplayReviewStore(tmp_path / 'review.db')
    token = store.stage(proposal)
    with pytest.raises(ValueError, match='invalid perception'):
        store.decide(token, accept=True, expected_revision=0)
    assert store.snapshot()['revision'] == 0
