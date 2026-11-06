"""Metric XY connectivity for consecutive placements."""

import numpy as np
from scipy.spatial import ConvexHull

from stacking_common.domain.geometry import transform_points
from .types import PlacementFilterDecision


def projected_footprint(points):
    xy = np.asarray(points, dtype=float)[:, :2]
    return xy[ConvexHull(xy).vertices]


def footprint_distances(previous, candidate, translations):
    """Exact distance between convex XY polygons at candidate translations.

    Separating axes detect overlap. Outside, the gap is the distance from the
    translation to the Minkowski difference of the two polygons. Its boundary
    has at most m+n edges, avoiding a vertex/edge tensor of size N*m*n.
    """
    previous, candidate = np.asarray(previous), np.asarray(candidate)
    translations = np.asarray(translations).reshape(-1, 2)
    if not len(translations):
        return np.empty(0)
    edges = np.concatenate((np.roll(previous, -1, axis=0) - previous,
                            np.roll(candidate, -1, axis=0) - candidate))
    axes = np.column_stack((-edges[:, 1], edges[:, 0]))
    pa, pb = previous @ axes.T, candidate @ axes.T
    offsets = translations @ axes.T
    separated = np.any((pb.min(0) + offsets > pa.max(0) + 1e-12)
                       | (pb.max(0) + offsets < pa.min(0) - 1e-12), axis=1)
    distances = np.zeros(len(translations))
    if not np.any(separated):
        return distances
    differences = (previous[:, None] - candidate[None, :]).reshape(-1, 2)
    boundary = differences[ConvexHull(differences).vertices]
    edge = np.roll(boundary, -1, axis=0) - boundary
    delta = translations[separated, None] - boundary
    fraction = np.clip(np.sum(delta * edge, axis=-1)
                       / np.sum(edge * edge, axis=-1), 0., 1.)
    delta = delta - fraction[..., None] * edge
    distances[separated] = np.sqrt(np.min(np.sum(delta * delta, axis=-1), axis=1))
    return distances


def footprints_within_gap(previous, candidate, translations, max_gap):
    """Admit nearby polygons, pruning only by a conservative AABB lower bound.

    Bounding boxes can reject distant anchors but never establish connectivity.
    Survivors use the same exact distance and cutoff as the diagnostic path.
    """
    previous = np.asarray(previous, dtype=float)
    candidate = np.asarray(candidate, dtype=float)
    translations = np.asarray(translations, dtype=float).reshape(-1, 2)
    accepted = np.zeros(len(translations), dtype=bool)
    if not len(translations):
        return accepted
    lower = candidate.min(axis=0) + translations
    upper = candidate.max(axis=0) + translations
    separation = np.maximum(np.maximum(previous.min(axis=0) - upper,
                                       lower - previous.max(axis=0)), 0.)
    # Roundoff at the broad-phase boundary must go to the exact calculation.
    scale = max(1., float(np.abs(previous).max()), float(np.abs(lower).max()),
                float(np.abs(upper).max()))
    slack = 1e-12 + 32. * np.finfo(float).eps * scale
    near = np.hypot(separation[:, 0], separation[:, 1]) <= max_gap + slack
    accepted[near] = footprint_distances(previous, candidate, translations[near]) <= max_gap
    return accepted


def bounded_distance_penalty(previous, candidate, translations, gap, length):
    """Dimensionless soft cost: zero within gap, approaching one at infinity."""
    if not np.isfinite(length) or length <= 0.:
        raise ValueError("distance penalty length must be finite and positive")
    excess = np.maximum(footprint_distances(previous, candidate, translations) - gap, 0.)
    return excess / (length + excess)


class PreviousActionConnectivity:
    """Check actual projected footprints before and after physical evaluation."""

    def __init__(self, inventory, max_gap):
        self.inventory = inventory
        self.max_gap = float(max_gap)
        if not np.isfinite(self.max_gap) or self.max_gap < 0:
            raise ValueError('connectivity gap must be finite and non-negative')

    def check(self, parent, object_index, pose, *, object_poses=None):
        if not parent.placed_object_indices:
            return PlacementFilterDecision.accept()
        previous = parent.placed_object_indices[-1]
        previous_id = parent.object_ids[previous]
        poses = parent.object_poses if object_poses is None else object_poses
        previous_points, _ = self.inventory[previous].mesh_arrays()
        points, _ = self.inventory[object_index].mesh_arrays()
        a = projected_footprint(transform_points(previous_points, poses[previous_id]))
        b = projected_footprint(transform_points(points, pose))
        gap = float(footprint_distances(a, b, [[0., 0.]])[0])
        diagnostics = dict(previous_action_object_id=previous_id,
                           previous_action_xy_gap=gap,
                           previous_action_xy_gap_limit=self.max_gap)
        if gap > self.max_gap + 1e-9:
            return PlacementFilterDecision.reject('previous_action_disconnected', **diagnostics)
        return PlacementFilterDecision.accept(**diagnostics)

    def evaluate(self, inventory, state, object_index, initial, solved):
        return self.check(state, object_index, solved.pose)


class RootConnectivityPreference(PreviousActionConnectivity):
    """Record root connectivity for allocation without rejecting a root jump."""

    def evaluate(self, inventory, state, object_index, initial, solved):
        decision = self.check(state, object_index, solved.pose)
        return PlacementFilterDecision.accept(**decision.diagnostics,
                                              root_connected=decision.accepted)
