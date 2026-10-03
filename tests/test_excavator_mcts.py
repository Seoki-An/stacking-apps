"""Native search and portable orientation policies for the excavator app."""

from dataclasses import replace
import json
from pathlib import Path
import random

import numpy as np
import pytest
import stacking_core as core
from scipy.spatial import ConvexHull
from scipy.spatial.transform import Rotation

from stacking_common.actions.orientation import PrincipalAxisOrientationProvider
from stacking_common.domain import State
from stacking_common.search import EvaluationFidelity, MCTSNode
from stacking_excavator import ExcavatorApplication

ROOT = Path(__file__).resolve().parents[1]


def test_principal_orientation_rolls_preserve_asset_axes_and_ignore_scan():
    application = ExcavatorApplication.from_repository(ROOT)
    initial = {55: np.r_[[0., 0., 1.], Rotation.from_euler("xyz", [.3, .7, .9]).as_quat()]}
    inventory = application.load_inventory([55], initial)
    provider = application.build_orientation_provider()
    assert isinstance(provider, PrincipalAxisOrientationProvider)
    assert application.build_orientation_filters(application.build_floor_policy(inventory)) == ()
    rotations = provider.quaternions(inventory, None, None, 0)
    points, _ = inventory[0].mesh_arrays()
    points = points[ConvexHull(points).vertices]
    covariance = np.cov(Rotation.from_quat(rotations[0]).apply(points), rowvar=False)
    np.testing.assert_allclose(covariance, np.diag(np.diag(covariance)), atol=1e-12)
    assert covariance[0, 0] > covariance[1, 1] > covariance[2, 2]
    relative = Rotation.from_quat(rotations[1]) * Rotation.from_quat(rotations[0]).inv()
    np.testing.assert_allclose(relative.as_matrix(), np.diag([1, -1, -1]), atol=1e-12)
    rotations[0][:] = 0.
    assert np.linalg.norm(provider.quaternions(inventory, None, None, 0)[0]) > .99


@pytest.mark.parametrize("posegen_backend", ["cpu", "cuda"])
@pytest.mark.parametrize("simulation_backend", ["cpu", "cuda"])
def test_native_excavator_mcts_selects_a_validated_recorded_object(posegen_backend, simulation_backend):
    capabilities = core.cuda_capabilities()
    if ((posegen_backend == "cuda" and not capabilities.get("posegen_solve"))
            or (simulation_backend == "cuda" and not capabilities.get("simulation_solve"))
            or ("cuda" in (posegen_backend, simulation_backend) and not capabilities.get("devices"))):
        pytest.skip("requires CUDA-enabled stacking-core and GPU")
    application = ExcavatorApplication.from_repository(ROOT)
    application = replace(application, mcts_profile=replace(application.mcts_profile,
        parallel=replace(application.mcts_profile.parallel,
            posegen_backend=posegen_backend, simulation_backend=simulation_backend)))
    reference = json.loads((ROOT / "tests/fixtures/legacy_planner/reference.json").read_text())
    ids = [56, 58, 8]
    inventory = application.load_inventory(ids, {
        i: np.concatenate(reference["pickup_poses"][str(i)]) for i in ids})
    profile = application.mcts_profile
    # The narrower ablation target needs more candidates than the old 8-sample smoke.
    application = replace(application, mcts_profile=replace(profile,
        tree=replace(profile.tree, root_max_children=12, final_validation_candidates=12),
        root_cem=replace(profile.root_cem, population=96, elite=24, iterations=1, keep=12)))
    request = replace(application.default_mcts_request(), iterations=2,
                      max_time_seconds=60., final_validation_reserve_seconds=20.)
    environment = application.build_environment(inventory,
        application.build_core_backend(inventory, EvaluationFidelity.SHORT))
    reset = environment.reset()
    search = application.build_mcts(inventory, rng=random.Random(0))
    from stacking_common.environment.core_backend import CorePlacementBackend
    from stacking_common.environment.cuda_backend import CudaPlacementBackend
    from stacking_common.search import StackingEnvironmentEvaluator

    expected_backend = CudaPlacementBackend if simulation_backend == "cuda" else CorePlacementBackend
    assert type(environment.transition.backend) is expected_backend
    assert isinstance(search.transition_evaluator, StackingEnvironmentEvaluator) == (simulation_backend == "cuda")
    assert len(search.action_generator.pose_solvers) == (
        1 if posegen_backend == "cuda" else application.mcts_profile.parallel.effective_workers
    )
    try:
        result = search.search(MCTSNode(state=reset.state, observation=reset.observation), request)
        assert result.generated_candidates > 0
        assert result.selected is not None
        assert result.selected.action.object_id in ids
        assert result.selected.selectable
        if posegen_backend == "cuda":
            assert result.selected.action.diagnostics["posegen_backend"] == "cuda"
        if simulation_backend == "cuda":
            assert result.selected.state.action_history[-1].diagnostics["simulation_backend"] == "cuda"
        assert not result.selected.state.failed
        assert len(result.selected.state.placed_object_indices) == 1
        # Search owns candidate states; the source scene has not advanced.
        assert environment.snapshot().placed_object_indices == []
    finally:
        search.close()
        environment.close()
