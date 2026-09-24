"""Conservative RGB agreement gates and explicitly provisional coverage labels."""
from dataclasses import dataclass, asdict
import numpy as np
from scipy.spatial import cKDTree
from stacking_common.pointcloud.types import PointCloudData


@dataclass(frozen=True)
class ConfidenceGates:
    pair_distance_m: float = .0015
    neighbor_radius_m: float = .003
    minimum_neighbors: int = 6

    def __post_init__(self):
        if (not np.isfinite([self.pair_distance_m, self.neighbor_radius_m]).all()
                or min(self.pair_distance_m, self.neighbor_radius_m) <= 0
                or isinstance(self.minimum_neighbors, bool) or not isinstance(self.minimum_neighbors, int)
                or self.minimum_neighbors < 1):
            raise ValueError('confidence gates require positive distances and neighbor count')


def voxel_observations(points, views, pixels, voxel_size=.001):
    """Voxel centroids with CSR observation provenance, before any selection.

    Observation rows in offsets[i]:offsets[i+1] identify the view and flattened
    image pixel supporting voxel i. Output coordinates are never moved by gates.
    """
    points = PointCloudData(points).points
    views, pixels = np.asarray(views), np.asarray(pixels)
    if (views.shape != (len(points),) or pixels.shape != views.shape
            or views.dtype.kind not in 'iu' or pixels.dtype.kind not in 'iu'
            or np.any(views < 0) or np.any(pixels < 0)
            or not np.isfinite(voxel_size) or voxel_size <= 0):
        raise ValueError('invalid voxel observation provenance')
    keys = np.floor(points/voxel_size).astype(np.int64)
    keys, inverse, counts = np.unique(keys, axis=0, return_inverse=True, return_counts=True)
    sums = np.zeros((len(keys), 3)); np.add.at(sums, inverse, points)
    order = np.argsort(inverse, kind='stable')
    return dict(points=sums/counts[:, None], voxel_keys=keys,
                observation_offsets=np.r_[0, np.cumsum(counts)],
                observation_views=views[order], observation_pixels=pixels[order])


def filter_rgb_agreement(points, pair_clouds, gates=ConfidenceGates()):
    """Require two nonempty disjoint-pair clouds, then local support.

    The caller establishes disjoint image identities. Distances alone are not
    probabilities or guarantees against shared calibration/matching bias.
    """
    points = PointCloudData(points).points
    if len(pair_clouds) != 2:
        raise ValueError('exactly two disjoint image-pair clouds are required')
    pairs = [PointCloudData(p).points for p in pair_clouds]
    if any(not len(p) for p in pairs):
        raise ValueError('pair clouds must be nonempty')
    distances = np.column_stack([cKDTree(p).query(points, workers=2)[0] for p in pairs])
    agreement = np.all(distances <= gates.pair_distance_m, axis=1)
    ids = np.flatnonzero(agreement)
    local = np.zeros(len(points), bool)
    if len(ids):
        distance = cKDTree(points[ids]).query(points[ids], k=gates.minimum_neighbors+1,
                                            distance_upper_bound=gates.neighbor_radius_m, workers=2)[0]
        # First neighbor is the point itself. Inputs are voxel centroids.
        local[ids] = np.isfinite(distance[:, -1])
    return dict(keep=agreement & local, pair_agreement=agreement, local_support=local,
                pair_distances_m=distances, gates=asdict(gates))


def classify_reference_coverage(points, reference, *, overlap_m=.003, footprint_cell_m=.002):
    """1=3D overlap, 2=far but inside padded XY footprint, 3=candidate extension.

    Footprint is a top-view proxy, not visibility or surface area. A one-cell
    padding around observed reference cells reduces sampling-hole artifacts.
    No point outside the footprint is certified to be a real new surface.
    """
    points, reference = PointCloudData(points).points, PointCloudData(reference).points
    if not len(reference) or not np.isfinite([overlap_m, footprint_cell_m]).all() or min(overlap_m, footprint_cell_m) <= 0:
        raise ValueError('nonempty reference and positive coverage scales required')
    distances = cKDTree(reference).query(points, workers=2)[0]
    keys = np.floor(reference[:, :2]/footprint_cell_m).astype(np.int64)
    footprint = {(int(x+dx), int(y+dy)) for x, y in np.unique(keys, axis=0)
                 for dx in (-1, 0, 1) for dy in (-1, 0, 1)}
    point_keys = np.floor(points[:, :2]/footprint_cell_m).astype(np.int64)
    inside = np.array([tuple(k) in footprint for k in point_keys], bool)
    labels = np.where(distances <= overlap_m, 1, np.where(inside, 2, 3)).astype(np.uint8)
    extension_cells = len(np.unique(point_keys[labels == 3], axis=0))
    return dict(labels=labels, reference_distance_m=distances,
                candidate_xy_cells=extension_cells,
                candidate_projected_area_mm2=extension_cells*(footprint_cell_m*1000)**2)
