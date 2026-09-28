"""Known-pose COLMAP export and dense optical-depth backprojection.

COLMAP pixels have centres at half-integers. Camera poses supplied here map
optical-camera coordinates into the metric world frame.
"""
from pathlib import Path
from itertools import product

import numpy as np
from scipy.spatial.transform import Rotation
from stacking_common.pointcloud.types import validate_rigid_transform


def write_known_cameras(directory, matrix, distortion, image_size, poses, names):
    directory = Path(directory)
    k, d = np.asarray(matrix, float), np.asarray(distortion, float).ravel()
    poses = np.asarray(poses, float)
    width, height = image_size
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or d.size not in (4, 5, 8) or not np.isfinite(d).all()
            or width < 1 or height < 1 or poses.shape != (len(names), 4, 4) or not len(names)
            or len(set(names)) != len(names)
            or any(Path(n).name != n or any(c.isspace() for c in n) for n in names)):
        raise ValueError('invalid known-camera export')
    d = np.pad(d, (0, 8-len(d)))
    params = [k[0, 0], k[1, 1], k[0, 2]+.5, k[1, 2]+.5, *d]
    lines = []
    for index, (pose, name) in enumerate(zip(poses, names), 1):
        inverse = np.linalg.inv(validate_rigid_transform(pose))
        quat = Rotation.from_matrix(inverse[:3, :3]).as_quat()[[3, 0, 1, 2]]
        values = ' '.join(format(float(x), '.17g') for x in [*quat, *inverse[:3, 3]])
        lines.append(f'{index} {values} 1 {name}\n\n')
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'cameras.txt').write_text(f'1 FULL_OPENCV {width} {height} '+ ' '.join(format(float(x), '.17g') for x in params)+'\n')
    (directory/'images.txt').write_text(''.join(lines))
    (directory/'points3D.txt').write_text('')


def depth_bounds_from_roi(poses, minimum, maximum):
    low, high = np.asarray(minimum, float), np.asarray(maximum, float)
    if low.shape != (3,) or high.shape != (3,) or not np.isfinite([low, high]).all() or np.any(high <= low):
        raise ValueError('invalid world region')
    corners = np.array(list(product(*zip(low, high))))
    depths = []
    for pose in poses:
        p = np.linalg.inv(validate_rigid_transform(pose))
        depths.extend((corners@p[:3, :3].T+p[:3, 3])[:, 2])
    if not depths or min(depths) <= .02:
        raise ValueError('ROI must be in front of all cameras')
    return float(min(depths)-.02), float(max(depths)+.02)



def depth_bounds_from_boxes(poses, boxes):
    """Bound depth over actual regions without including empty workspace corners."""
    poses = tuple(poses)
    values = np.asarray(boxes, dtype=float)
    if values.ndim != 3 or values.shape[1:] != (2, 3) or not len(values):
        raise ValueError('depth bounds need nonempty 3D boxes')
    bounds = [depth_bounds_from_roi(poses, low, high) for low, high in values]
    return min(value[0] for value in bounds), max(value[1] for value in bounds)

def read_colmap_depth(path):
    """Read COLMAP's width&height&channels& plus float32 Fortran-order payload."""
    with Path(path).open('rb') as stream:
        header = bytearray()
        while header.count(b'&') < 3:
            byte = stream.read(1)
            if not byte or len(header) > 100:
                raise ValueError('invalid COLMAP depth header')
            header.extend(byte)
        width, height, channels = map(int, header[:-1].split(b'&'))
        if min(width, height) < 1 or channels != 1 or width*height > 100_000_000:
            raise ValueError('invalid COLMAP depth dimensions')
        data = np.frombuffer(stream.read(), dtype='<f4')
    if len(data) != width*height:
        raise ValueError('COLMAP depth payload size mismatch')
    return data.reshape((width, height), order='F').T.copy()


def backproject_colmap_depth(depth, matrix, world_T_camera):
    depth, k = np.asarray(depth, float), np.asarray(matrix, float)
    pose = validate_rigid_transform(world_T_camera)
    if depth.ndim != 2 or k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('expected depth image and positive pinhole intrinsics')
    valid = np.isfinite(depth)&(depth > 0)
    y, x = np.nonzero(valid)
    homogeneous = np.column_stack((x+.5, y+.5, np.ones(len(x))))
    camera = (homogeneous@np.linalg.inv(k).T)*depth[valid, None]
    points = camera@pose[:3, :3].T+pose[:3, 3]
    return points, np.ravel_multi_index((y, x), depth.shape)


def roi_samples(boxes, *, steps=3):
    """Sample each world-frame box separately for visibility and overlap checks."""
    boxes = np.asarray(boxes, float)
    if (boxes.ndim != 3 or boxes.shape[1:] != (2, 3) or not len(boxes)
            or not np.isfinite(boxes).all() or np.any(boxes[:, 1] <= boxes[:, 0])):
        raise ValueError('expected nonempty finite ROI boxes with positive extents')
    return np.array([list(product(*(np.linspace(a, b, steps) for a, b in zip(lo, hi))))
                     for lo, hi in boxes])


def projected_roi_crop(matrix, world_T_camera, image_size, boxes, *, padding=16):
    """Return a clipped xyxy crop and shifted COLMAP pinhole matrix, without resizing."""
    k = np.asarray(matrix, float)
    width, height = image_size
    if (k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0
            or min(width, height) <= 0 or padding < 0):
        raise ValueError('invalid camera or crop padding')
    camera_T_world = np.linalg.inv(validate_rigid_transform(world_T_camera))
    corners = roi_samples(boxes, steps=2).reshape(-1, 3)
    camera = corners @ camera_T_world[:3, :3].T + camera_T_world[:3, 3]
    if np.any(camera[:, 2] <= .02):
        raise ValueError('crop ROI must be in front of the camera')
    projected = camera @ k.T
    uv = projected[:, :2] / projected[:, 2, None]
    low = np.maximum(np.floor(uv.min(axis=0) - padding), 0).astype(int)
    high = np.minimum(np.ceil(uv.max(axis=0) + padding), [width, height]).astype(int)
    if np.any(high <= low):
        raise ValueError('crop ROI is outside the image')
    shifted = k.copy()
    shifted[:2, 2] -= low
    return np.r_[low, high], shifted


def write_pinhole_cameras(directory, matrices, image_sizes, poses, names):
    """Export separate undistorted cameras; matrices already use COLMAP pixel centres."""
    matrices = np.asarray(matrices, float)
    if (matrices.shape != (len(names), 3, 3) or not len(names)
            or not np.isfinite(matrices).all() or len(image_sizes) != len(names)
            or len(poses) != len(names) or len(set(names)) != len(names)
            or any(Path(n).name != n or any(c.isspace() for c in n) for n in names)):
        raise ValueError('invalid per-image camera export')
    cameras, images = [], []
    for index, (k, size, pose, name) in enumerate(zip(matrices, image_sizes, poses, names), 1):
        width, height = map(int, size)
        if min(width, height, k[0, 0], k[1, 1]) <= 0:
            raise ValueError('invalid pinhole dimensions or focal length')
        inverse = np.linalg.inv(validate_rigid_transform(pose))
        quat = Rotation.from_matrix(inverse[:3, :3]).as_quat()[[3, 0, 1, 2]]
        params = ' '.join(format(float(x), '.17g') for x in [k[0, 0], k[1, 1], k[0, 2], k[1, 2]])
        cameras.append(f'{index} PINHOLE {width} {height} {params}\n')
        values = ' '.join(format(float(x), '.17g') for x in [*quat, *inverse[:3, 3]])
        images.append(f'{index} {values} {index} {name}\n\n')
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    (directory/'cameras.txt').write_text(''.join(cameras))
    (directory/'images.txt').write_text(''.join(images))
    (directory/'points3D.txt').write_text('')


def select_source_views(matrices, image_sizes, poses, boxes, *, maximum_sources=6):
    """Choose explicit sources using shared ROI visibility, parallax and direction diversity.

    All boxes visible to a reference must have at least two selected sources
    covering half their visible samples. No sparse COLMAP tracks are required.
    """
    points = roi_samples(boxes)
    poses = np.asarray([validate_rigid_transform(p) for p in poses])
    if len(poses) < 3 or len(matrices) != len(poses) or len(image_sizes) != len(poses) or maximum_sources < 2:
        raise ValueError('source selection needs at least three cameras and two sources')
    visible = []
    for matrix, size, pose in zip(matrices, image_sizes, poses):
        k = np.asarray(matrix, float)
        if k.shape != (3, 3) or not np.isfinite(k).all() or min(size) <= 0:
            raise ValueError('invalid source camera')
        inverse = np.linalg.inv(pose)
        camera = points @ inverse[:3, :3].T + inverse[:3, 3]
        projected = camera @ k.T
        uv = projected[..., :2] / np.maximum(projected[..., 2, None], 1e-12)
        visible.append((camera[..., 2] > .02) & np.all((uv >= 0) & (uv < size), axis=-1))
    visible = np.asarray(visible)
    centres = poses[:, :3, 3]
    target = points.mean(axis=(0, 1))
    directions = centres - target
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1e-12)
    graph = []
    for ref in range(len(poses)):
        active = visible[ref].any(axis=1)
        if not active.any():
            raise ValueError(f'view {ref} sees no reconstruction region')
        candidates = {}
        rays = centres[ref] - points
        rays /= np.maximum(np.linalg.norm(rays, axis=-1, keepdims=True), 1e-12)
        for src in range(len(poses)):
            if src == ref:
                continue
            shared = visible[ref] & visible[src]
            overlap = shared.sum(axis=1) / np.maximum(visible[ref].sum(axis=1), 1)
            other = centres[src] - points
            other /= np.maximum(np.linalg.norm(other, axis=-1, keepdims=True), 1e-12)
            angles = np.rad2deg(np.arccos(np.clip(np.sum(rays * other, axis=-1), -1, 1)))
            angle = float(np.median(angles[shared])) if shared.any() else 0.
            if angle < 1. or not np.any(overlap[active] >= .5):
                continue
            score = float(overlap[active].mean() * min(angle / 8., 1.))
            candidates[src] = (score, overlap >= .5)
        chosen = []
        coverage = np.zeros(len(points), int)
        while candidates and len(chosen) < maximum_sources:
            def score(src):
                quality, covers = candidates[src]
                deficit = int(np.count_nonzero(active & (coverage < 2) & covers))
                separation = (min(np.rad2deg(np.arccos(np.clip(directions[src] @ directions[j], -1, 1)))
                                  for j in chosen) if chosen else 8.)
                return (deficit, quality * (.65 + .35 * min(separation / 8., 1.)), -src)
            best = max(candidates, key=score)
            coverage += candidates.pop(best)[1]
            chosen.append(best)
        if np.any(coverage[active] < 2):
            raise ValueError(f'view {ref} lacks two overlapping sources with useful parallax for every ROI')
        graph.append(chosen)
    return graph
