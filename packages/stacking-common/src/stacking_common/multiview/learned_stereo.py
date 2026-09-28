"""Metric rectification and optional Fast-FoundationStereo inference for known poses."""
from pathlib import Path
import sys

import numpy as np
from stacking_common.pointcloud import validate_rigid_transform


def select_pairs(poses, minimum_baseline=.02):
    """Connect every camera to its nearest distinct viewpoint; deduplicate edges."""
    poses = np.asarray([validate_rigid_transform(p) for p in poses])
    pairs = set()
    for i, pose in enumerate(poses):
        distances = np.linalg.norm(poses[:, :3, 3] - pose[:3, 3], axis=1)
        candidates = np.flatnonzero(distances >= minimum_baseline)
        if not len(candidates):
            raise ValueError(f'view {i} has no stereo baseline')
        j = min(candidates, key=lambda k: (1 - poses[k, :3, 2] @ pose[:3, 2], distances[k]))
        pairs.add(tuple(sorted((i, int(j)))))
    return sorted(pairs)


def rectify_pair(images, matrices, distortions, poses, max_size=960):
    """Rotate both cameras into one horizontal stereo frame, including vertical pairs.

    Returns rectified RGB, valid sensor masks, shared intrinsics and left world pose.
    Pixel coordinates follow OpenCV's integer pixel-center convention.
    """
    import cv2
    a, b = [validate_rigid_transform(p) for p in poses]
    x = b[:3, 3] - a[:3, 3]
    baseline = float(np.linalg.norm(x))
    if baseline < 1e-6:
        raise ValueError('stereo requires distinct camera centers')
    x /= baseline
    z = a[:3, 2] + b[:3, 2]
    z -= x * (z @ x)
    if np.linalg.norm(z) < 1e-6:
        raise ValueError('degenerate stereo viewing directions')
    z /= np.linalg.norm(z)
    rotation = np.column_stack((x, np.cross(z, x), z))
    height, width = images[0].shape[:2]
    scale = min(1., max_size / max(height, width))
    width, height = round(width * scale), round(height * scale)
    if abs(x @ a[:3, 1]) > abs(x @ a[:3, 0]):
        width, height = height, width
    focal = float(np.mean([k[0, 0] + k[1, 1] for k in matrices]) / 2 * scale)
    matrix = np.array([[focal, 0, (width-1)/2], [0, focal, (height-1)/2], [0, 0, 1.]])
    rectified, masks = [], []
    for rgb, k, d, pose in zip(images, matrices, distortions, (a, b)):
        if rgb.dtype != np.uint8 or rgb.ndim != 3 or rgb.shape[2] != 3:
            raise ValueError('stereo input must be uint8 RGB')
        mx, my = cv2.initUndistortRectifyMap(np.asarray(k), np.asarray(d),
            rotation.T @ pose[:3, :3], matrix, (width, height), cv2.CV_32FC1)
        masks.append((mx >= 0) & (mx < rgb.shape[1]-1) & (my >= 0) & (my < rgb.shape[0]-1))
        rectified.append(cv2.remap(rgb, mx, my, cv2.INTER_LINEAR))
    world_pose = np.eye(4)
    world_pose[:3, :3], world_pose[:3, 3] = rotation, a[:3, 3]
    return rectified, masks, matrix, world_pose, baseline


def disparity_points(disparity, reverse_disparity, masks, matrix, pose, baseline, tolerance=1.):
    """Backproject positive, in-sensor, left/right-consistent disparity to metric XYZ."""
    import cv2
    disparity, reverse_disparity = np.asarray(disparity), np.asarray(reverse_disparity)
    if disparity.shape != reverse_disparity.shape or baseline <= 0:
        raise ValueError('invalid stereo disparity geometry')
    y, x = np.indices(disparity.shape, dtype=np.float32)
    xr = x - disparity
    safe_x = np.where(np.isfinite(xr), xr, -1).astype(np.float32)
    right = cv2.remap(reverse_disparity.astype(np.float32), safe_x, y, cv2.INTER_LINEAR)
    right_valid = cv2.remap(masks[1].astype(np.float32), safe_x, y, cv2.INTER_LINEAR) > .999
    valid = (masks[0] & right_valid & np.isfinite(disparity) & (disparity > 0)
             & (xr >= 0) & (xr <= disparity.shape[1]-1)
             & np.isfinite(right) & (right > 0) & (np.abs(disparity - right) <= tolerance))
    pixels = np.column_stack((x[valid], y[valid], np.ones(np.count_nonzero(valid))))
    depth = matrix[0, 0] * baseline / disparity[valid]
    points = (pixels @ np.linalg.inv(matrix).T) * depth[:, None]
    return points @ pose[:3, :3].T + pose[:3, 3], valid


class FastFoundationStereo:
    """Headless optional adapter. Only load a trusted official serialized checkpoint."""
    def __init__(self, repository, checkpoint):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError('Fast-FoundationStereo requires CUDA')
        repository = Path(repository).resolve()
        if not (repository / 'core/foundation_stereo.py').is_file():
            raise ValueError('Fast-FoundationStereo repository is missing')
        sys.path.insert(0, str(repository))
        import core.foundation_stereo  # Register serialized model module aliases.
        from core.utils.utils import InputPadder
        self.torch, self.padder = torch, InputPadder
        self.model = torch.load(checkpoint, map_location='cpu', weights_only=False).cuda().eval()

    def predict(self, left, right, max_disparity):
        torch = self.torch
        images = [torch.from_numpy(np.ascontiguousarray(im)).cuda().float().permute(2, 0, 1)[None]
                  for im in (left, right)]
        padder = self.padder(images[0].shape, divis_by=32, force_square=False)
        images = padder.pad(*images)
        self.model.args.max_disp = int(max_disparity)
        self.model.args.valid_iters = 8
        with torch.inference_mode(), torch.amp.autocast('cuda', dtype=torch.float16):
            result = self.model.forward(*images, iters=8, test_mode=True, optimize_build_volume='pytorch1')
        return padder.unpad(result.float()).cpu().numpy().reshape(left.shape[:2])
