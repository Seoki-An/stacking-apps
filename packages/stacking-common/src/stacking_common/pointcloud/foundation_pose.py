"""Optional FoundationPose adapter; returns mesh-frame poses without ICP."""
from pathlib import Path
import sys
import time

import numpy as np


class FoundationPoseEstimator:
    """Share the learned scorer, refiner and CUDA renderer across objects."""

    def __init__(self, repository):
        sys.path.insert(0, str(Path(repository).resolve()))
        import nvdiffrast.torch as dr
        from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor

        self.estimator_type = FoundationPose
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.context = dr.RasterizeCudaContext()

    def estimate(self, mesh_file, rgb, depth, mask, matrix, debug_directory):
        import torch
        import trimesh

        mesh = trimesh.load(mesh_file, process=False)
        if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
            raise ValueError('FoundationPose requires a triangle mesh in metres and catalog coordinates')
        started = time.monotonic()
        estimator = self.estimator_type(
            model_pts=mesh.vertices, model_normals=mesh.vertex_normals, mesh=mesh,
            scorer=self.scorer, refiner=self.refiner, glctx=self.context,
            debug=0, debug_dir=str(debug_directory))
        pose = estimator.register(K=matrix, rgb=rgb, depth=depth, ob_mask=mask, iteration=5)
        torch.cuda.synchronize()
        pose = np.asarray(pose, dtype=float)
        if (pose.shape != (4, 4) or not np.isfinite(pose).all()
                or not np.allclose(pose[3], [0, 0, 0, 1])
                or not np.allclose(pose[:3, :3].T @ pose[:3, :3], np.eye(3), atol=1e-4)
                or not np.isclose(np.linalg.det(pose[:3, :3]), 1, atol=1e-4)):
            raise ValueError('FoundationPose returned an invalid rigid pose')
        return pose, time.monotonic() - started
