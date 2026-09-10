"""Import explicit legacy log layouts without importing ROS or legacy Python."""

from pathlib import Path
import re
import numpy as np

from stacking_common.profiles.planning import load_planning_profile
from stacking_excavator.planning.trajectory import ExcavatorRobotTrajectoryProjector, pose_matrix
from .capture import CaptureFrame, digest
from .recorded import read_sensor_file
from .storage import create_capture, file_hash


def import_legacy_log(source, destination, *, repository, kind, capture_id, scene_revision):
    """Keep original numeric frames, and label missing timing/opening as unknown.

    Field/scene: *_raw_TOPIC_swingN.npy plus logged LiDAR transforms and joints.
    In-hand: pcd_N.ply plus q_list.npy, already in grip_body geometry frame.
    In-hand logs cannot reconstruct raw sensor rays lost by the legacy writer.
    """
    source, repository = Path(source), Path(repository)
    profile = load_planning_profile(repository / "apps/excavator/config/planning.yml")
    projector = ExcavatorRobotTrajectoryProjector(profile, repository)
    sources = {}

    def load(path):
        sources[path.name] = file_hash(path)
        return np.load(path, allow_pickle=False)

    if kind == "inhand":
        joints = load(source / "q_list.npy")
        paths = sorted(source.glob("pcd_*.ply"), key=lambda p: int(p.stem.split("_")[-1]))
        if len(paths) != len(joints) or [int(p.stem.split("_")[-1]) for p in paths] != list(range(len(joints))):
            raise ValueError("in-hand cloud count differs from recorded joint samples")
    elif kind in {"field", "scene"}:
        paths = sorted(source.glob(f"{kind}_raw_*_swing*.npy"))
    else:
        raise ValueError("legacy import requires field, inhand or scene")
    if not paths:
        raise ValueError("no frames matching the supported legacy log layout")
    pending = []
    for index, path in enumerate(paths):
        if kind == "inhand":
            q = joints[index]
        else:
            match = re.fullmatch(fr"{kind}_raw_(.+)_swing([0-9]+)", path.stem)
            topic, swing = match.groups()
            q = load(source / f"{kind}_scan_q_actual_swing{swing}.npy")
        if np.asarray(q).shape != (6,):
            raise ValueError("legacy joints must be six canonical radian coordinates")
        body = next(p for p in projector.project(np.array(q), 0.)
                    if p.part_id.startswith("gripper/grip_body/"))
        gripper = pose_matrix(body.pose)  # Body geometry is independent of jaw opening.
        if kind == "inhand":
            sensor = gripper.copy()
            sensor_frame = "recorded-stone-grab-geometry"
        else:
            sensor = load(source / f"{kind}_lidar_frame_{topic}_swing{swing}.npy")
            sensor_frame = topic
        sources[path.name] = file_hash(path)
        points, channels = read_sensor_file(path)
        pending.append((path.stem, sensor_frame, sensor, gripper, q, points, channels))
    model_hashes = {str(name): file_hash(repository / name)
                   for name in (profile.robot.urdf, profile.gripper.urdf)}
    calibration = digest({"provenance": "legacy_transform_snapshot", "sources": sources,
                          "models": model_hashes})
    frames = [CaptureFrame(name, sensor_frame, None, None, sensor, gripper, calibration,
                           points, tuple(q), None, attributes=channels, timestamp_source="recorded_unknown")
              for name, sensor_frame, sensor, gripper, q, points, channels in pending]
    return create_capture(destination, capture_id, kind, scene_revision, frames,
        recording={"layout": "legacy-" + kind, "source_hashes": sources, "model_hashes": model_hashes,
                   "original_sensor_frame": kind != "inhand", "timing_verified": False})
