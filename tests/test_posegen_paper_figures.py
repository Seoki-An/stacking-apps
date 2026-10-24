"""Scientific checks for the standalone posegen paper-figure experiment."""

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

pytest.importorskip("stacking_core")

from stacking_common.actions.core_solver import CorePlacementPoseSolver
from stacking_common.actions.types import InitializedPlacement

from posegen_paper_figures import force_incidences
from posegen_paper_rendering import make_force_arrow
from posegen_paper_trace import trace_pose_optimization
from posegen_structure_toy import CUBE, StructureToy


def test_trace_matches_normal_solve_and_true_prefix_poses():
    toy = StructureToy()
    profile = replace(toy.solvers["structure"].profile, schedule={
        **toy.solvers["structure"].profile.schedule,
        "times": (.5, 1.), "max_iters_warmup": 8, "max_iters_final": 8,
    })
    initial = InitializedPlacement(toy.initial_pose(.35), 0.)
    solver = CorePlacementPoseSolver(toy.inventory, profile)
    reference = solver.solve(toy.inventory, toy.state, 2, initial)
    trace = trace_pose_optimization(toy.inventory, profile, toy.state, 2, initial, max_samples=20)
    np.testing.assert_allclose(trace.result.pose, reference.pose, atol=1e-12)
    np.testing.assert_allclose(trace.samples[0].pose, initial.pose)
    np.testing.assert_allclose(trace.samples[-1].pose, reference.pose)
    assert any(s.stage == 1 and 0 < s.iteration < 8 for s in trace.samples)
    # Independently stop the original native solver at a sampled iteration.
    sample = next(s for s in trace.samples if s.stage == 1 and 0 < s.iteration < 8)
    prefix_profile = replace(profile, schedule={**profile.schedule, "times": (.5,),
                                                "max_iters_final": sample.iteration})
    prefix = CorePlacementPoseSolver(toy.inventory, prefix_profile).solve(
        toy.inventory, toy.state, 2, initial)
    np.testing.assert_allclose(sample.pose, prefix.pose, atol=1e-12)
    # Whole-scene forces, with exactly the physical scaling used by search.
    for entity, forces in reference.diagnostics["posegen_contact_forces"].items():
        np.testing.assert_allclose(trace.result.diagnostics["posegen_contact_forces"][entity], forces)
    assert len(force_incidences(toy.inventory, trace.result.diagnostics)) > 1
    assert CUBE in {i for i, _, _ in force_incidences(toy.inventory, trace.result.diagnostics)}


def test_force_entities_map_to_inventory_ids_and_keep_both_contact_sides():
    inventory = [SimpleNamespace(object_id=56), SimpleNamespace(object_id=8)]
    points = {1: [[0., 0., .5]], 2: [[0., 0., .5]]}
    forces = {1: [[0., 0., -100.]], 2: [[0., 0., 100.]]}
    rows = force_incidences(inventory, dict(posegen_contact_points=points, posegen_contact_forces=forces))
    assert [i for i, _, _ in rows] == [56, 8]
    np.testing.assert_allclose(sum(f for _, _, f in rows), np.zeros(3))


def test_force_export_rejects_mismatched_contact_counts():
    with pytest.raises(ValueError, match="counts differ"):
        force_incidences([SimpleNamespace(object_id=56)], dict(
            posegen_contact_points={1: []}, posegen_contact_forces={1: [[0., 0., 100.]]}))


def test_force_arrows_have_equal_widths_and_balanced_monotonic_lengths():
    pytest.importorskip("pyvista")
    arrows = [make_force_arrow(np.zeros(3), np.array([force, 0., 0.]), 10000.)
              for force in (1., 100., 10000.)]
    sizes = np.array([np.ptp(arrow.points, axis=0) for arrow in arrows])
    assert np.all(np.diff(sizes[:, 0]) > 0)
    assert sizes[-1, 0] / sizes[0, 0] < 4
    np.testing.assert_allclose(sizes[:, 1:], np.tile(sizes[0, 1:], (3, 1)), atol=1e-7)
    # The shaft rings also retain their radius, independently of head width.
    shaft_radii = [np.linalg.norm(arrow.points[np.isclose(arrow.points[:, 0], 0.)][:, 1:], axis=1).max()
                   for arrow in arrows]
    np.testing.assert_allclose(shaft_radii, shaft_radii[0], atol=1e-7)
