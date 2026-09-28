"""Rigid pose errors against a known transform, independent of registration residuals."""
import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
from .types import PointCloudData, validate_rigid_transform


def pose_errors(model_points, estimate, truth):
    """Report translation, rotation, ADD and nearest-surface ADD-S in metric units.

    ADD uses corresponding model samples. ADD-S allows nearest-neighbor matches;
    it can hide rotation ambiguity and does not declare an object symmetric.
    """
    points = PointCloudData(model_points).points
    if not len(points):
        raise ValueError('pose error needs nonempty model samples')
    estimate, truth = [validate_rigid_transform(p) for p in (estimate, truth)]
    predicted = points @ estimate[:3, :3].T + estimate[:3, 3]
    actual = points @ truth[:3, :3].T + truth[:3, 3]
    add = np.linalg.norm(predicted-actual, axis=1)
    adds = cKDTree(actual).query(predicted)[0]
    return dict(translation_mm=float(np.linalg.norm(estimate[:3, 3]-truth[:3, 3])*1000),
                rotation_deg=float(np.rad2deg(Rotation.from_matrix(estimate[:3, :3] @ truth[:3, :3].T).magnitude())),
                add_mean_mm=float(np.mean(add)*1000), add_p95_mm=float(np.percentile(add,95)*1000),
                adds_mean_mm=float(np.mean(adds)*1000), sample_count=len(points))
