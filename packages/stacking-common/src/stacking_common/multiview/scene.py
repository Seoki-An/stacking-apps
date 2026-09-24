"""Partition RGB evidence without changing coordinates or accepting a fusion."""
import numpy as np
from stacking_common.pointcloud import PointCloudData


RGB_LAYERS = {
    'rgb_background': 'Retained RGB outside stone regions',
    'rgb_overlap': 'RGB reference overlap',
    'rgb_unresolved': 'RGB unresolved difference',
    'rgb_extensions': 'RGB possible extensions',
    'rgb_rejected': 'Rejected RGB evidence',
}


def rgb_layer_indices(observations):
    """Validate selection/provenance and return disjoint, exhaustive voxel IDs."""
    points = PointCloudData(observations['points']).points
    n = len(points)
    keep = observations['keep']
    stone = observations['stone']
    labels = observations['coverage_class']
    if (keep.shape != (n,) or keep.dtype.kind != 'b' or stone.shape != (n,)
            or stone.dtype.kind not in 'iu' or np.any(stone < -1)
            or labels.shape != (n,) or labels.dtype.kind not in 'iu'
            or not np.isin(labels, [0, 1, 2, 3]).all()):
        raise ValueError('invalid RGB selection labels')
    selected_stones = keep & (stone >= 0)
    if not np.array_equal(labels > 0, selected_stones):
        raise ValueError('coverage classes must label exactly the retained stone samples')
    offsets = observations['observation_offsets']
    views = observations['observation_views']
    pixels = observations['observation_pixels']
    if (offsets.shape != (n+1,) or offsets.dtype.kind not in 'iu'
            or offsets[0] != 0 or np.any(offsets[1:] <= offsets[:-1])
            or views.ndim != 1 or pixels.shape != views.shape
            or offsets[-1] != len(views) or views.dtype.kind not in 'iu'
            or pixels.dtype.kind not in 'iu' or np.any(views < 0) or np.any(pixels < 0)):
        raise ValueError('invalid RGB observation provenance')
    return {
        'rgb_background': np.flatnonzero(keep & (stone < 0)),
        'rgb_overlap': np.flatnonzero(labels == 1),
        'rgb_unresolved': np.flatnonzero(labels == 2),
        'rgb_extensions': np.flatnonzero(labels == 3),
        'rgb_rejected': np.flatnonzero(~keep),
    }
