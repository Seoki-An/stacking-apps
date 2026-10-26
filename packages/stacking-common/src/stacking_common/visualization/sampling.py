"""On-demand diagnostics for the pre-region production XY scorer."""

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
from scipy.spatial import ConvexHull

from stacking_common.actions.config import PlanarSamplingConfig
from stacking_common.actions.floor import ActiveFloorPolicy, floor_policy_settings
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
                            settings, height_map_resolution, height_map_margin, floor_settings=None):
    """Recompute a fixed initial orientation in the pre-search scene.

    The separate sampler cannot affect planning RNG or caches. Height scores are
    normalized over this orientation's eligible cells. This is a diagnostic
    reconstruction with normal floor gates, not a replay of retries or the
    multi-orientation normalization used during search.
    """
    from stacking_common.domain import CompositeTargetGeometry
    if isinstance(inventory.target, CompositeTargetGeometry):
        raise ValueError("Composite target rendering is supported; per-region score-map inspection is not yet available")
    if settings.get('strategy') or not settings.get('score') or not (settings.get('floor') or settings.get('floor_fill')):
        raise ValueError('Score inspection requires pre-region sampler settings')
    removed_scores = {'open_area', 'frontier'}
    if any(float(settings['score'].get(key, 0.)) != 0. for key in removed_scores):
        raise ValueError('This run used removed open-area or frontier scores; its score maps cannot be reconstructed')
    settings = {**settings, 'score': {key: value for key, value in settings['score'].items()
                                    if key not in removed_scores}}
    recorded_floor = settings.get('floor', settings.get('floor_fill'))
    if recorded_floor.get('u_shape_filter', 'floor' not in settings):
        raise ValueError('This run enabled the removed U-shape filter; its score maps cannot be reconstructed')
    recorded_minimum = recorded_floor.get('minimum_contact_cells',
        recorded_floor.get('u_shape_min_frontier_contact_cells', 1))
    delayed = 0 < len(state.placed_object_indices) < int(recorded_floor.get('connected_after', 1))
    if float(recorded_minimum) != 1. or delayed:
        raise ValueError('This run used a different connectivity rule; its score maps cannot be reconstructed')
    if 'floor' not in settings:
        # Historical runs kept proposal settings and authoritative layer settings
        # separately. Translate at the artifact boundary, preserving their values.
        legacy = settings['floor_fill']
        if floor_settings is None:
            raise ValueError('Historical score inspection requires recorded floor settings')
        floor = {'coverage_mode': 'bottom_slice', **floor_settings,
            'footprint_scale': legacy.get('u_shape_footprint_scale', .9)}
        settings = {key: value for key, value in settings.items() if key != 'floor_fill'}
        settings['floor'] = floor
    recorded_max_layers = settings['floor'].get('max_layers')
    # Discard historical keys only when their rule matches current sampling.
    floor = {key: value for key, value in settings['floor'].items()
             if key not in {'connected_after', 'minimum_contact_cells', 'max_layers'}}
    settings = {**settings, 'floor': floor}
    index = inventory.object_ids.index(object_id)
    if index in state.placed_object_indices:
        raise ValueError('score-map candidate is already in the scene')
    pose = np.asarray(initial_pose, dtype=float)
    renderer = HeightMapRenderer(inventory, height_map_resolution, height_map_margin)
    observation = SimpleNamespace(scene_height_map=renderer.render(state),
        height_map_xlim=renderer.xlim, height_map_ylim=renderer.ylim)
    config = PlanarSamplingConfig(tuple(planar_config['grid_shape']), (0.,),
        target_anchor_inset_scale=planar_config.get('target_anchor_inset_scale'))
    provider = MappingOrientationProvider({oid: pose[3:] for oid in inventory.object_ids})
    policy = ActiveFloorPolicy(inventory, **floor_policy_settings(settings['floor']))
    if (recorded_max_layers is not None
            and len(policy.context_details(state)['completed_layers']) >= int(recorded_max_layers)):
        raise ValueError('This run reached its historical layer limit; its score maps cannot be reconstructed')
    sampler = ScoredPlanarPoseSampler(inventory, config, provider, settings, floor_policy=policy)
    data = sampler.score_map(state, observation, index)
    fields = {'score': data['scores'].copy()}
    for name in ('height', 'support_likelihood', 'fill_area',
                 'target_boundary', 'planar_bias', 'placement_height'):
        source = 'height_term' if name == 'height' else name
        fields[name] = np.asarray(data.get(source, np.full(config.grid_shape, np.nan))).copy()
    valid = data['candidate_mask'] & np.isfinite(fields['score'])
    footprint = []
    for component in inventory[index].geometries:
        xy = transform_points(component.points, pose)[:, :2]
        footprint.append(xy[ConvexHull(xy).vertices])
    half = np.array([inventory.target.width, inventory.target.length]) / 2
    target_bounds = (inventory.target.origin[:2] - half, inventory.target.origin[:2] + half)
    weights = dict(data.get('weights', {}))
    return SamplingScoreView(data['x_coords'], data['y_coords'], fields, valid,
        observation.scene_height_map, renderer.xlim, renderer.ylim,
        pose[:2].copy(), tuple(footprint), target_bounds, weights)
