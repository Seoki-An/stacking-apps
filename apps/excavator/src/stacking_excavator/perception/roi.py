"""Generous purpose-specific transfer boxes; fine filtering stays on desktop."""

import numpy as np

from stacking_common.pointcloud.types import PointCloudData, validate_rigid_transform
from stacking_excavator.acquisition.capture import CropRequest, GRIPPER_FRAME


def enclosing_crop(kind, models, poses, *, calibration_sha256, scene_revision,
                   margins=(.5, .5, .5), target_corners=None, gripper_points=None,
                   ground_height=0.):
    """Enclose expected/observed surfaces, not only the ideal target structure.

    Poses are world-relative, except in-hand poses in stone_grab geometry frame.
    Passing updated observed poses includes displacement and placement overhang.
    """
    if not poses:
        raise ValueError("ROI requires explicit object priors")
    points = []
    for object_id, pose in poses.items():
        model, _ = models.stone(object_id)
        points.append(model.transformed(validate_rigid_transform(pose)).points)
        if kind == "field":
            center = np.asarray(pose)[:3, 3]
            points.append(np.array([center - [.8, .8, .5], center + [.8, .8, 1.]]))
    if kind == "scene":
        if target_corners is None:
            raise ValueError("scene ROI requires target structure corners")
        points.append(PointCloudData(target_corners).points)
    if kind == "inhand":
        if gripper_points is None or not len(gripper_points):
            raise ValueError("in-hand ROI requires jaw envelope including opening uncertainty")
        points.append(PointCloudData(gripper_points).points)
    points = np.concatenate(points)
    lower, upper = points.min(axis=0), points.max(axis=0)
    if kind != "inhand":
        if not np.isfinite(ground_height):
            raise ValueError("ROI ground height must be finite")
        lower[2] = min(lower[2], ground_height)
    pose = np.eye(4)
    pose[:3, 3] = (lower + upper) / 2
    return CropRequest(kind, GRIPPER_FRAME if kind == "inhand" else "world", pose,
        tuple(np.maximum((upper-lower)/2, .001)), margins, calibration_sha256,
        scene_revision, tuple(poses))
