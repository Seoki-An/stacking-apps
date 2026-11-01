"""Convex Voronoi prisms with an exact polyhedral reference arrangement.

The reference certifies geometric tiling, not native DSF/simulation feasibility
or optimality of a discounted multi-term reward. No native solver is imported.
"""

from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np

from stacking_common.domain import ConvexGeometry


@dataclass(frozen=True)
class VoronoiPiece:
    object_id: int
    layer: int
    geometry: ConvexGeometry
    reference_pose: np.ndarray
    inertia: np.ndarray  # Unit-density inertia about the body-frame centroid.


@dataclass(frozen=True)
class VoronoiPuzzle:
    seed: int
    width: float
    length: float
    layer_height: float
    layers: int
    cells_per_layer: int
    pieces: tuple[VoronoiPiece, ...]

    @property
    def target_volume(self):
        return self.width * self.length * self.layer_height * self.layers

    @property
    def reference_order(self):
        return tuple(p.object_id for p in sorted(self.pieces, key=lambda p: (p.layer, p.object_id)))


def clip_polygon(polygon, normal, offset):
    """Clip a CCW convex polygon to normal.dot(x) <= offset."""
    polygon = np.asarray(polygon, dtype=float)
    if not len(polygon):
        return np.empty((0, 2))
    values = polygon @ normal - offset
    result = []
    for i, end in enumerate(polygon):
        start = polygon[i - 1]
        before, after = values[i - 1], values[i]
        if (before <= 0.) != (after <= 0.):
            result.append(start + before / (before - after) * (end - start))
        if after <= 0.:
            result.append(end)
    if not result:
        return np.empty((0, 2))
    result = np.asarray(result)
    keep = np.linalg.norm(result - np.roll(result, 1, axis=0), axis=1) > 1e-10
    return result[keep]


def voronoi_cells(sites, width, length):
    sites = np.asarray(sites, dtype=float)
    if sites.ndim != 2 or sites.shape[1] != 2 or not np.isfinite(sites).all():
        raise ValueError('Voronoi sites must be finite Nx2 values')
    if width <= 0 or length <= 0 or not np.isfinite([width, length]).all():
        raise ValueError('Voronoi bounds must be positive and finite')
    if len(sites) < 2 or len(np.unique(sites, axis=0)) != len(sites):
        raise ValueError('Voronoi sites must be distinct and contain at least two points')
    if np.any(np.abs(sites) >= np.array([width, length]) / 2):
        raise ValueError('Voronoi sites must be strictly inside the target')
    boundary = np.array([[-width, -length], [width, -length],
                         [width, length], [-width, length]]) / 2
    result = []
    for i, site in enumerate(sites):
        cell = boundary.copy()
        for j, other in enumerate(sites):
            if j != i:
                cell = clip_polygon(cell, 2 * (other - site), other @ other - site @ site)
        if len(cell) < 3:
            raise ValueError('Voronoi cell is degenerate')
        result.append(cell)
    return tuple(result)


def _area_centroid(polygon):
    following = np.roll(polygon, -1, axis=0)
    cross = polygon[:, 0] * following[:, 1] - following[:, 0] * polygon[:, 1]
    area = float(cross.sum() / 2)
    if area <= 1e-12:
        raise ValueError('Polygon must have positive CCW area')
    return area, ((polygon + following) * cross[:, None]).sum(axis=0) / (6 * area)


def _prism(polygon, height, yaw):
    area, centroid = _area_centroid(polygon)
    c, s = np.cos(yaw), np.sin(yaw)
    local = (polygon - centroid) @ np.array([[c, s], [-s, c]])
    n = len(local)
    points = np.vstack([np.column_stack([local, np.full(n, -height / 2)]),
                        np.column_stack([local, np.full(n, height / 2)])])
    faces = []
    for i in range(1, n - 1):
        faces.extend([[0, i + 1, i], [n, n + i, n + i + 1]])
    for i in range(n):
        j = (i + 1) % n
        faces.extend([[i, j, n + j], [i, n + j, n + i]])
    following = np.roll(local, -1, axis=0)
    x, y = local.T
    u, v = following.T
    cross = x * v - u * y
    xx = float(((x*x + x*u + u*u) * cross).sum() / 12)
    yy = float(((y*y + y*v + v*v) * cross).sum() / 12)
    xy = float(((2*x*y + x*v + u*y + 2*u*v) * cross).sum() / 24)
    zz = area * height**3 / 12
    inertia = np.array([[height*yy + zz, -height*xy, 0.],
                        [-height*xy, height*xx + zz, 0.],
                        [0., 0., height*(xx + yy)]])
    return ConvexGeometry(points, np.asarray(faces), sharpness=20), centroid, inertia


def make_layered_voronoi(seed, *, cells_per_layer=4, layers=2,
                         width=2.4, length=1.6, layer_height=.6):
    """Generate independent layer partitions; shuffle IDs and local yaw frames.

    Rejection sampling excludes tiny cells using geometry alone, before any
    planner is run. No instance is selected because greedy happens to fail.
    """
    if (not isinstance(cells_per_layer, int) or cells_per_layer < 2
            or not isinstance(layers, int) or layers < 1):
        raise ValueError('Layer/cell counts must be positive integers, with at least two cells')
    if not np.isfinite([width, length, layer_height]).all() or min(width, length, layer_height) <= 0:
        raise ValueError('Puzzle dimensions must be positive and finite')
    rng = np.random.default_rng(seed)
    ids = iter(rng.permutation(np.arange(1, cells_per_layer * layers + 1)))
    pieces = []
    for layer in range(layers):
        for _ in range(1000):
            sites = rng.uniform(-.4, .4, (cells_per_layer, 2)) * [width, length]
            polygons = voronoi_cells(sites, width, length)
            if min(_area_centroid(p)[0] for p in polygons) >= .35 * width * length / cells_per_layer:
                break
        else:
            raise RuntimeError('Could not sample non-tiny Voronoi cells')
        for polygon in polygons:
            yaw = rng.uniform(-np.pi, np.pi)
            geometry, centroid, inertia = _prism(polygon, layer_height, yaw)
            pose = np.array([*centroid, (layer + .5) * layer_height,
                             0., 0., np.sin(-yaw/2), np.cos(-yaw/2)])
            pose.setflags(write=False);inertia.setflags(write=False)
            pieces.append(VoronoiPiece(int(next(ids)), layer, geometry, pose, inertia))
    return VoronoiPuzzle(int(seed), width, length, layer_height, layers,
                         cells_per_layer, tuple(sorted(pieces, key=lambda p: p.object_id)))


def export_puzzle(puzzle, directory):
    """Write native-compatible OBJ/PLY assets, with privileged truth separate."""
    import yaml

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=False)
    assets = directory / 'assets'
    assets.mkdir()
    for piece in puzzle.pieces:
        points, faces = piece.geometry.points, piece.geometry.faces
        obj = [f'# mass: {piece.geometry.volume:.17g}',
               '# inertia: ' + json.dumps(piece.inertia.tolist()), '# dsf_count: 1',
               'o convex.0', '# sharpness: 20', '# mu: 0.5']
        obj.extend('v ' + ' '.join(f'{x:.17g}' for x in point) for point in points)
        obj.extend('f ' + ' '.join(str(int(i) + 1) for i in face) for face in faces)
        (assets / f'model_{piece.object_id}.obj').write_text('\n'.join(obj) + '\n')
        ply = ['ply', 'format ascii 1.0', f'element vertex {len(points)}',
               'property double x', 'property double y', 'property double z',
               f'element face {len(faces)}', 'property list uchar int vertex_indices', 'end_header']
        ply.extend(' '.join(f'{x:.17g}' for x in point) for point in points)
        ply.extend('3 ' + ' '.join(str(int(i)) for i in face) for face in faces)
        (assets / f'model_{piece.object_id}_mesh.ply').write_text('\n'.join(ply) + '\n')
    target = dict(width=puzzle.width, length=puzzle.length,
                  height=puzzle.layers*puzzle.layer_height, origin=[0., 0., 0.],
                  support_overlap=0., taper_degrees=dict(width_positive=0., width_negative=0.,
                                                         length_positive=0., length_negative=0.))
    inventory = dict(units='si', objects=dict(model_directory=str(assets.resolve()),
        ids=[p.object_id for p in puzzle.pieces], density=1000., density_overrides={}), target=target)
    (directory / 'inventory.yml').write_text(yaml.safe_dump(inventory, sort_keys=False))
    (directory / 'case.json').write_text(json.dumps(dict(schema_version=1, units='si',
        seed=puzzle.seed, layers=puzzle.layers, cells_per_layer=puzzle.cells_per_layer,
        target=target, geometric_reference_fill=1., native_reference_status='not_run',
        note='Exact polyhedral tiling; native DSF feasibility and reward optimality are not certified.'), indent=2))
    reference = dict(order=list(puzzle.reference_order),
        poses={str(p.object_id): p.reference_pose.tolist() for p in puzzle.pieces},
        layers={str(p.object_id): p.layer for p in puzzle.pieces},
        privileged=True, use='Reference validation/evaluation only; never a planner proposal source.')
    (directory / 'reference.json').write_text(json.dumps(reference, indent=2))
