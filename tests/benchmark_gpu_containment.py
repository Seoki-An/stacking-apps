"""Compare CPU/GPU containment cache misses; requires optional CUDA PyTorch.

This is a diagnostic benchmark, not a production backend. Geometry and target
voxels stay on the GPU. Timings include per-call pose preparation, transfers,
and result retrieval; one-time CUDA/upload costs are reported separately.
"""
from __future__ import annotations

import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
from statistics import median
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'packages/stacking-common/src'), str(ROOT / 'apps/excavator/src')]
from stacking_common.domain.geometry import quaternion_matrix
from stacking_common.environment import VoxelOverlapEvaluator
from stacking_excavator import ExcavatorApplication


class GPUContainment:
    def __init__(self, evaluator, objects):
        self.resolution = evaluator.resolution
        self.tolerance = self.resolution * 1e-6
        self.grid = torch.as_tensor(evaluator._grid_points, dtype=torch.float64, device='cuda')
        self.planes = {}
        self.expanded = {}
        self.objects = objects
        for object_id, obj in objects.items():
            planes, vertices = [], []
            for geometry in obj.geometries:
                centers, normals, _ = geometry.face_data()
                offsets = np.einsum('ij,ij->i', centers, normals)
                planes.append((torch.as_tensor(normals.T.copy(), device='cuda'),
                               torch.as_tensor(offsets + self.tolerance, device='cuda')))
                center = geometry.points.mean(axis=0)
                margin = np.einsum('ij,ij->i', normals, centers - center).min()
                if margin <= 0:
                    raise ValueError('benchmark requires strictly interior geometry centroids')
                vertices.append(center + (geometry.points - center) * (1 + 2 * self.tolerance / margin))
            self.planes[object_id] = planes
            self.expanded[object_id] = np.concatenate(vertices)
        torch.cuda.synchronize()

    def score(self, queries, batch_size):
        groups = defaultdict(list)
        for index, (object_id, pose) in enumerate(queries):
            groups[object_id].append((index, pose))
        result = np.empty(len(queries))
        for object_id, group in groups.items():
            for start in range(0, len(group), batch_size):
                chunk = group[start:start + batch_size]
                poses = np.asarray([pose for _, pose in chunk])
                rotations = np.asarray([quaternion_matrix(pose[3:]) for pose in poses])
                world = self.expanded[object_id][None, :, :] @ rotations.transpose(0, 2, 1) + poses[:, None, :3]
                lower = torch.as_tensor(world.min(axis=1) - self.resolution, device='cuda')
                upper = torch.as_tensor(world.max(axis=1) + self.resolution, device='cuda')
                positions = torch.as_tensor(poses[:, :3].copy(), device='cuda')
                rotations = torch.as_tensor(rotations, device='cuda')
                mask = torch.ones((len(chunk), len(self.grid)), dtype=torch.bool, device='cuda')
                for axis in range(3):
                    values = self.grid[None, :, axis]
                    mask &= (values >= lower[:, axis, None]) & (values <= upper[:, axis, None])
                candidate, voxel = mask.nonzero(as_tuple=True)
                local = torch.bmm((self.grid[voxel] - positions[candidate]).unsqueeze(1),
                                  rotations[candidate]).squeeze(1)
                inside = torch.zeros(len(local), dtype=torch.bool, device='cuda')
                for normals, offsets in self.planes[object_id]:
                    inside |= torch.all(local @ normals <= offsets, dim=1)
                counts = torch.zeros(len(chunk), dtype=torch.int64, device='cuda')
                counts.scatter_add_(0, candidate, inside.to(torch.int64))
                # Keep the same final scalar arithmetic as the CPU reference.
                for (index, _), count in zip(chunk, counts.cpu().numpy()):
                    volume = float(count) * self.resolution**3
                    result[index] = np.clip(volume / self.objects[object_id].volume, 0., 1.)
        return result


def measure(function, repeats):
    times = []
    for _ in range(repeats):
        torch.cuda.synchronize()
        started = time.perf_counter()
        result = function()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - started)
    return result, times


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable in this process')
    started = time.perf_counter()
    torch.cuda.init()
    torch.cuda.synchronize()
    cuda_init_seconds = time.perf_counter() - started
    app = ExcavatorApplication.from_repository(ROOT)
    ids = list(app.inventory_profile.object_ids)
    inventory = app.load_inventory(ids, {i: [0., 0., 0., 0., 0., 0., 1.] for i in ids})
    objects = {obj.object_id: obj for obj in inventory}
    evaluator = VoxelOverlapEvaluator(inventory.target, app.environment_profile.overlap_resolution)
    source = ROOT / 'docs/benchmarks/excavator-mcts-profile.json'
    reference = json.loads(source.read_text())['containment_reference']
    base = [(row['object_id'], np.asarray(row['pose'], dtype=float)) for row in reference]
    started = time.perf_counter()
    gpu = GPUContainment(evaluator, objects)
    upload_seconds = time.perf_counter() - started
    rows = []
    for count in (96, len(base), 4 * len(base)):
        queries = []
        for i in range(count):
            object_id, pose = base[i % len(base)]
            pose = pose.copy()
            pose[:3] += (i // len(base)) * np.array([.011, -.007, .003])
            queries.append((object_id, pose))
        assert len({(obj, pose.tobytes()) for obj, pose in queries}) == count
        def cpu_score():
            # Explicit cache misses on both sides; otherwise repeated benchmark
            # calls would compare GPU work against CPU cache lookup alone.
            return np.asarray([np.clip(evaluator._compute_intersection_volume(objects[obj], pose)
                                      / objects[obj].volume, 0., 1.) for obj, pose in queries])
        cpu_score()
        expected, cpu_times = measure(cpu_score, 3)
        with ThreadPoolExecutor(max_workers=12) as pool:
            def one_query(query):
                obj, pose = query
                return np.clip(evaluator._compute_intersection_volume(objects[obj], pose)
                               / objects[obj].volume, 0., 1.)
            def parallel_cpu_score():
                return np.asarray(list(pool.map(one_query, queries)))
            parallel_cpu_score()
            parallel_result, parallel_times = measure(parallel_cpu_score, 3)
            np.testing.assert_array_equal(parallel_result, expected)
        for batch_size in (1, 16, 64):
            warm = gpu.score(queries, batch_size)
            np.testing.assert_array_equal(warm, expected)
            torch.cuda.reset_peak_memory_stats()
            actual, gpu_times = measure(lambda: gpu.score(queries, batch_size), 5)
            np.testing.assert_array_equal(actual, expected)
            row = {'queries': count, 'max_batch_per_object': batch_size,
                   'cpu_seconds': cpu_times, 'gpu_seconds': gpu_times,
                   'parallel_cpu_seconds': parallel_times,
                   'cpu_median_seconds': median(cpu_times), 'gpu_median_seconds': median(gpu_times),
                   'parallel_cpu_median_seconds': median(parallel_times),
                   'speedup': median(cpu_times) / median(gpu_times),
                   'speedup_vs_parallel_cpu': median(parallel_times) / median(gpu_times),
                   'max_absolute_error': float(np.max(np.abs(actual - expected))),
                   'peak_cuda_allocated_bytes': torch.cuda.max_memory_allocated()}
            rows.append(row)
            print(json.dumps(row), flush=True)
    args.output.write_text(json.dumps({
        'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'cuda': torch.version.cuda,
        'dtype': 'float64', 'target_grid_points': len(evaluator._grid_points),
        'cuda_init_seconds': cuda_init_seconds, 'geometry_upload_seconds': upload_seconds,
        'reference_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
        'benchmark_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'results': rows,
    }, indent=2) + '\n')


if __name__ == '__main__':
    main()
