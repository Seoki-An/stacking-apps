"""On-demand diagnostics for the pre-region production XY scorer."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull

from stacking_common.actions.config import PlanarSamplingConfig
from stacking_common.actions.floor import ActiveFloorPolicy
from stacking_common.actions.orientation import MappingOrientationProvider
from stacking_common.actions.scored import ScoredPlanarPoseSampler
from stacking_common.domain.geometry import transform_points
from .heightmap import HeightMapRenderer


@dataclass(frozen=True)
class SamplingScoreView:
    x: np.ndarray
    y: np.ndarray
    fields: dict[str, np.ndarray]
    valid: np.ndarray
    scene_height: np.ndarray
    scene_xlim: np.ndarray
    scene_ylim: np.ndarray
    initial_xy: np.ndarray
    footprint: tuple[np.ndarray, ...]
    target_bounds: tuple[np.ndarray, np.ndarray]
    weights: dict[str, float]


def initial_pose_score_view(inventory, state, object_id, initial_pose, *, planar_config,
                            settings, floor_settings, height_map_resolution, height_map_margin):
    """Recompute a fixed initial orientation in the pre-search scene.

    The separate sampler cannot affect planning RNG or caches. Height scores are
    normalized over this orientation's eligible cells. This is a diagnostic
    reconstruction with normal floor gates, not a replay of retries or the
    multi-orientation normalization used during search.
    """
    if settings.get('strategy') or not settings.get('score') or not settings.get('floor_fill'):
        raise ValueError('Score inspection requires pre-region sampler settings')
    index = inventory.object_ids.index(object_id)
    if index in state.placed_object_indices:
        raise ValueError('score-map candidate is already in the scene')
    pose = np.asarray(initial_pose, dtype=float)
    renderer = HeightMapRenderer(inventory, height_map_resolution, height_map_margin)
    observation = SimpleNamespace(scene_height_map=renderer.render(state),
        height_map_xlim=renderer.xlim, height_map_ylim=renderer.ylim)
    config = PlanarSamplingConfig(tuple(planar_config['grid_shape']), (0.,),
        planar_config['minimum_xy_separation'], planar_config.get('target_anchor_inset_scale'))
    provider = MappingOrientationProvider({oid: pose[3:] for oid in inventory.object_ids})
    sampler = ScoredPlanarPoseSampler(inventory, config, provider, settings,
        floor_policy=ActiveFloorPolicy(inventory, **floor_settings))
    data = sampler.score_map(state, observation, index)
    fields = {'score': data['scores'].copy()}
    for name in ('height', 'support_likelihood', 'open_area', 'fill_area', 'frontier',
                 'target_boundary', 'excavator_distance', 'placement_height'):
        source = 'height_term' if name == 'height' else name
        fields[name] = np.asarray(data.get(source, np.full(config.grid_shape, np.nan))).copy()
    valid = data['candidate_mask'] & np.isfinite(fields['score'])
    footprint = []
    for component in inventory[index].geometries:
        xy = transform_points(component.points, pose)[:, :2]
        footprint.append(xy[ConvexHull(xy).vertices])
    half = np.array([inventory.target.width, inventory.target.length]) / 2
    target_bounds = (inventory.target.origin[:2] - half, inventory.target.origin[:2] + half)
    weights = {name: float(settings['score'][name]) for name in fields if name in settings['score']}
    return SamplingScoreView(data['x_coords'], data['y_coords'], fields, valid,
        observation.scene_height_map, renderer.xlim, renderer.ylim,
        pose[:2].copy(), tuple(footprint), target_bounds, weights)
