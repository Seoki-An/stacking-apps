"""CUDA adapter continuation, failure isolation, and native opt-in checks."""
from dataclasses import replace
from enum import Enum
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import stacking_core as core

from stacking_common.actions.core_solver import CorePlacementPoseSolver
from stacking_common.actions.cuda_solver import CudaPlacementPoseSolver
from stacking_common.actions.types import InitializedPlacement, PlacementSolveResult
from stacking_common.profiles.posegen import load_posegen_profile

ROOT = Path(__file__).resolve().parents[1]


class Status(Enum):
    converged = 0
    precision_limit = 1
    geometry_failure = 2
    capacity_overflow = 3
    unsupported_input = 4
    iteration_limit = 5
    numerical_failure = 6


def pose(x=0.):
    return np.array([x, 0., .1, 0., 0., 0., 1.])


def adapter(monkeypatch, failure=None, force_converged=True, failure_stage=2):
    solver = object.__new__(CudaPlacementPoseSolver)
    solver.profile = load_posegen_profile(ROOT / 'apps/excavator/config/solvers/posegen.yml')
    solver.core = SimpleNamespace(
        CudaPosegenBatchItem=lambda i, problem, config: SimpleNamespace(request_id=i, problem=problem, config=config),
        CudaPosegenStatus=Status,
    )
    solver._validate_request = lambda *args: None
    solver._prepare_problem_scene = lambda state: state
    def problem(state, index, p, *, scene_context):
        assert scene_context == state
        return state, index, p.copy()
    solver._problem = problem
    solver._scheduled_config = lambda t, gap, final: (t, gap, final)
    solver._native_mass_scale = lambda state, index: 1.0
    solver._placement_values = lambda r, native_mass_scale: dict(
        pose=r.optimal_pose, c_feq=0., c_gap=r.c_gap, converged=r.solver.converged,
        rejection_reason=None if r.solver.converged else 'posegen_nonconvergence',
        diagnostics={'posegen_pose_converged': r.solver.converged},
    )
    calls, retries = [], []

    def solve(items):
        calls.append(items)
        outputs = []
        for item in items:
            fail = len(calls) == failure_stage and item.request_id == 0
            status = failure if fail and failure else Status.converged
            result = SimpleNamespace(optimal_pose=item.problem[2] + np.array([.01, 0, 0, 0, 0, 0, 0]),
                                     c_gap=.001 * (item.request_id + 1) * len(calls), c_feq=0.,
                                     solver=SimpleNamespace(converged=status == Status.converged, force_solver=SimpleNamespace(
                                         converged=force_converged if fail else True)))
            outputs.append(SimpleNamespace(request_id=item.request_id, status=status,
                                           result=None if fail and failure == Status.geometry_failure else result,
                                           job=SimpleNamespace(resource=SimpleNamespace(name="bodies"), required=33, capacity=32)))
        return outputs[::-1]  # The adapter must use request IDs, not output slots.

    def cpu(self, inventory, state, index, initial):
        retries.append((state, index, initial))
        pytest.fail('CUDA candidate solving must not invoke the CPU pose solver')

    monkeypatch.setattr(CorePlacementPoseSolver, 'solve', cpu)
    solver.cuda_generator = SimpleNamespace(solve=solve)
    return solver, calls, retries


def requests(n=2):
    return tuple((f'state-{i}', i, InitializedPlacement(pose(i), 0.)) for i in range(n))


def test_cuda_schedule_propagates_pose_and_per_request_gap(monkeypatch):
    solver, calls, retries = adapter(monkeypatch)
    jobs = requests()
    results = solver.solve_batch(None, jobs)
    assert not retries
    assert len(calls) == 4
    for step, items in enumerate(calls):
        for item in items:
            assert item.problem[0] == f'state-{item.request_id}'
            assert item.problem[2][0] == pytest.approx(item.request_id + step * .01)
            assert item.config[1] == pytest.approx(.001 * (item.request_id + 1) * step)
            assert item.config[2] == (step == 3)
    assert [r.pose[0] for r in results] == pytest.approx([.04, 1.04])
    assert all(r.diagnostics['posegen_result_backend'] == 'cuda' for r in results)


@pytest.mark.parametrize('status', [s for s in Status if s not in (Status.converged, Status.precision_limit, Status.iteration_limit, Status.capacity_overflow)])
def test_failed_request_rejects_without_disrupting_other_stages(monkeypatch, status):
    solver, calls, retries = adapter(monkeypatch, failure=status)
    jobs = requests()
    results = solver.solve_batch(None, jobs)
    assert not retries
    assert [[i.request_id for i in items] for items in calls] == [[0, 1], [0, 1], [1], [1]]
    np.testing.assert_array_equal(results[0].pose, jobs[0][2].pose)
    assert not results[0].converged and not results[0].valid
    assert results[0].rejection_reason == f'posegen_{status.name}'
    assert results[0].diagnostics['posegen_cuda_failure_reason'] == status.name
    from stacking_common.actions.filtering import PosegenFeasibilityFilter
    assert not PosegenFeasibilityFilter(None, None, require_solver_convergence=False).evaluate(
        None, None, None, None, results[0]).accepted
    assert results[0].diagnostics['posegen_cuda_stage_statuses'] == ('converged', status.name)
    assert results[1].diagnostics['posegen_result_backend'] == 'cuda'


@pytest.mark.parametrize('stage', [1, 2, 3, 4])
@pytest.mark.parametrize('limit', [Status.precision_limit, Status.iteration_limit])
def test_limited_solve_keeps_gpu_pose_and_continuation(monkeypatch, stage, limit):
    from stacking_common.actions.filtering import PosegenFeasibilityFilter
    solver, calls, retries = adapter(monkeypatch, failure=limit, failure_stage=stage)
    results = solver.solve_batch(None, requests())
    assert not retries
    assert len(calls) == 4
    assert [r.pose[0] for r in results] == pytest.approx([.04, 1.04])
    if stage < 4:
        item = calls[stage][0]
        assert item.problem[2][0] == pytest.approx(stage * .01)
        assert item.config[1] == pytest.approx(stage * .001)
    result = results[0]
    assert result.diagnostics['posegen_cuda_stage_statuses'][stage - 1] == limit.name
    assert result.valid
    assert result.diagnostics['posegen_result_backend'] == 'cuda'
    assert result.converged == (stage != 4)
    assert result.diagnostics['posegen_pose_converged'] == (stage != 4)
    if stage == 4:
        assert result.rejection_reason == f'posegen_{limit.name}'
        def check(gap, equilibrium, require=False):
            return PosegenFeasibilityFilter(gap, equilibrium, require_solver_convergence=require).evaluate(
                None, None, None, None, result)
        assert check(.005, 1.).accepted  # Excavator accepts feasible nonconverged poses.
        assert not check(.005, 1., require=True).accepted
        assert check(.001, 1.).reason == 'posegen_gap'
        bad = replace(result, c_feq=2.)
        assert not PosegenFeasibilityFilter(.005, 1., require_solver_convergence=False).evaluate(
            None, None, None, None, bad).accepted


@pytest.mark.parametrize('limit', [Status.precision_limit, Status.iteration_limit])
def test_limited_solve_does_not_ignore_force_failure(monkeypatch, limit):
    solver, _, retries = adapter(monkeypatch, failure=limit, force_converged=False)
    result = solver.solve_batch(None, requests())[0]
    assert not retries and not result.valid
    assert result.diagnostics['posegen_cuda_failure_reason'] == 'force_nonconvergence'


def test_precision_limit_without_result_rejects(monkeypatch):
    solver, _, retries = adapter(monkeypatch)
    solver.cuda_generator = SimpleNamespace(solve=lambda items: [SimpleNamespace(
        request_id=item.request_id, status=Status.precision_limit, result=None) for item in items])
    result = solver.solve_batch(None, requests(1))[0]
    assert not retries and not result.valid
    assert result.diagnostics['posegen_cuda_failure_reason'] == 'missing_result'


def test_inner_force_nonconvergence_rejects(monkeypatch):
    solver, _, retries = adapter(monkeypatch, force_converged=False)
    results = solver.solve_batch(None, requests())
    assert not retries and not results[0].valid
    assert results[0].diagnostics['posegen_cuda_failure_reason'] == 'force_nonconvergence'


def test_empty_single_and_chunked_requests(monkeypatch):
    solver, calls, _ = adapter(monkeypatch)
    solver.batch_size = 2
    assert solver.solve_batch(None, ()) == ()
    assert not calls
    results = solver.solve_batch(None, requests(5))
    assert [r.pose[0] for r in results] == pytest.approx([.04, 1.04, 2.04, 3.04, 4.04])
    assert [len(c) for c in calls] == [2] * 8 + [1] * 4
    assert solver.solve(None, *requests(1)[0]).pose[0] == pytest.approx(.04)


def test_mismatched_gpu_result_ids_fail_explicitly(monkeypatch):
    solver, _, _ = adapter(monkeypatch)
    solver.cuda_generator = SimpleNamespace(solve=lambda items: [])
    with pytest.raises(RuntimeError, match='request IDs'):
        solver.solve_batch(None, requests())


@pytest.mark.parametrize('capabilities', [{}, {'posegen_solve': False}, {'posegen_solve': True, 'devices': []}])
def test_unavailable_cuda_does_not_silently_select_cpu(capabilities):
    with pytest.raises(RuntimeError, match='CUDA-enabled'):
        CudaPlacementPoseSolver(None, None, core_module=SimpleNamespace(cuda_capabilities=lambda: capabilities))


def test_posegen_rejects_old_backend_location(tmp_path):
    path = ROOT / 'apps/excavator/config/solvers/posegen.yml'
    configured = tmp_path / 'posegen.yml'
    configured.write_text(path.read_text() + '\nbackend: cuda\n')
    with pytest.raises(ValueError, match='parallel.posegen_backend'):
        load_posegen_profile(configured)


def test_native_cuda_excavator_schedule(monkeypatch):
    if not core.cuda_capabilities().get('posegen_solve') or not core.cuda_capabilities().get('devices'):
        pytest.skip('requires a CUDA-enabled stacking-core build and GPU')
    from test_excavator_action_parity import inputs
    app, inventory, state = inputs()
    gpu = CudaPlacementPoseSolver(inventory, app.posegen_profile)
    initial = InitializedPlacement(pose(), 0.)
    monkeypatch.setattr(CorePlacementPoseSolver, 'solve',
                        lambda *args: pytest.fail('unexpected CPU pose solve'))
    result = gpu.solve(inventory, state, 0, initial)
    assert np.isfinite(result.pose).all()
    assert result.diagnostics['posegen_backend'] == 'cuda'
    assert result.diagnostics['posegen_cuda_stage_statuses']
    assert not result.diagnostics['posegen_cpu_retry']
    assert result.diagnostics['posegen_result_backend'] == 'cuda'
    if result.valid:
        assert len(result.diagnostics['posegen_cuda_stage_statuses']) == 4
    else:
        assert result.rejection_reason and not result.converged


@pytest.mark.parametrize('field', ['optimal_pose', 'c_gap', 'c_feq'])
def test_nonfinite_results_reject_without_cpu(monkeypatch, field):
    solver, _, retries = adapter(monkeypatch)
    original = solver.cuda_generator.solve
    def bad(items):
        outputs = original(items)
        for out in outputs:
            if out.request_id == 0:
                setattr(out.result, field, np.full(7, np.nan) if field == 'optimal_pose' else np.nan)
        return outputs
    solver.cuda_generator.solve = bad
    result = solver.solve_batch(None, requests())[0]
    assert not retries and not result.valid
    assert result.rejection_reason == 'posegen_nonfinite_result'


def test_cuda_scene_context_lifetime_and_branch_isolation(monkeypatch):
    solver, _, _ = adapter(monkeypatch)
    first, second = object(), object()
    prepared = []
    solver._prepare_problem_scene = lambda state: prepared.append(state) or state
    jobs = ((first, 0, InitializedPlacement(pose(), 0.)),
            (first, 1, InitializedPlacement(pose(1.), 0.)),
            (second, 2, InitializedPlacement(pose(2.), 0.)))
    solver.solve_batch(None, jobs)
    assert prepared == [first, second]
    solver.solve_batch(None, jobs)
    assert prepared == [first, second, first, second]


@pytest.mark.parametrize('boundaries', [False, True])
def test_prepared_problem_preserves_native_scene_and_roles(boundaries):
    from test_excavator_action_parity import inputs
    app, inventory, state = inputs(placed=True)
    solver = CorePlacementPoseSolver(inventory, replace(app.posegen_profile, placed_as_boundaries=boundaries))
    context = solver._prepare_problem_scene(state)
    original = solver._problem(state, 1, pose())
    cached = solver._problem(state, 1, pose(), scene_context=context)
    assert cached.candidate == original.candidate
    assert cached.targets == original.targets
    assert cached.boundaries == original.boundaries
    assert cached.scene.frame_id == original.scene.frame_id
    for a, b in zip(cached.scene.bodies, original.scene.bodies, strict=True):
        assert (a.id, a.model_id, a.mobility) == (b.id, b.model_id, b.mobility)
        np.testing.assert_array_equal(a.pose, b.pose)
    placed_id = solver.scene_builder.entity_id(state.placed_object_indices[0])
    before = cached.scene.body_pose(placed_id)
    state.object_poses[state.object_ids[0]][0] += 1.
    fresh = solver._problem(state, 1, pose(), scene_context=solver._prepare_problem_scene(state))
    assert fresh.scene.body_pose(placed_id)[0] == pytest.approx(before[0] + 1.)
    np.testing.assert_array_equal(cached.scene.body_pose(placed_id), before)


def test_cuda_final_diagnostics_own_arrays_without_reconstruction(monkeypatch):
    solver, _, _ = adapter(monkeypatch)
    points = np.array([[1., 2., 3.]])
    native = SimpleNamespace(optimal_pose=pose(), c_gap=0., solver=SimpleNamespace(converged=True))
    original = solver._placement_values
    solver._placement_values = lambda result, scale: {**original(result, scale), 'diagnostics': {'points': points}}
    stages = ['converged', 'precision_limit']
    result = solver._placement_result(native, 1.0, stages, 'precision_limit')
    points[:] = 9.
    native.optimal_pose[:] = 9.
    stages.clear()
    np.testing.assert_array_equal(result.diagnostics['points'], [[1., 2., 3.]])
    np.testing.assert_array_equal(result.pose, pose())
    assert result.diagnostics['posegen_cuda_stage_statuses'] == ('converged', 'precision_limit')
    assert not result.converged and result.valid
    assert result.rejection_reason == 'posegen_precision_limit'


@pytest.mark.parametrize('stage', [1, 2, 4])
def test_capacity_failure_stops_batch_without_candidate_retries(monkeypatch, stage):
    from stacking_common.errors import BackendCapacityError
    solver, calls, retries = adapter(monkeypatch, failure=Status.capacity_overflow, failure_stage=stage)
    with pytest.raises(BackendCapacityError) as raised:
        solver.solve_batch(None, requests())
    assert len(calls) == stage
    assert not retries
    assert raised.value.diagnostics == {
        'backend': 'posegen', 'resource': 'bodies', 'required': 33, 'capacity': 32,
        'request_id': 0, 'stage': stage - 1, 'reason': 'posegen_capacity_overflow',
        'message': 'posegen capacity exceeded for bodies: required 33, configured capacity 32',
    }


def test_configured_capacity_covers_inventory_and_controls_chunking(monkeypatch):
    from stacking_common.actions.config import CudaPosegenCapacityConfig
    models = {i: SimpleNamespace(geometry_count=4) for i in range(70)}
    class Inventory:
        def __len__(self): return len(models)
    inventory = Inventory()
    inventory.models = models
    def base_init(self, inventory, profile, *, core_module):
        self.target_model = SimpleNamespace(geometry_count=1)
    monkeypatch.setattr(CorePlacementPoseSolver, '__init__', base_init)
    capacities = []
    native = SimpleNamespace(
        cuda_capabilities=lambda: {'posegen_solve': True, 'devices': [0]},
        CudaCapacity=SimpleNamespace,
        CudaPoseGenerator=lambda capacity: capacities.append(capacity),
    )
    solver = CudaPlacementPoseSolver(inventory, None, core_module=native,
        capacity=CudaPosegenCapacityConfig(batch=512, bodies=64, geometries=256))
    assert capacities[0].batch == 512
    assert capacities[0].bodies == 71
    assert capacities[0].geometries == 281
    assert capacities[0].pairs == 4096
    solver._validate_request = lambda *args: None
    chunks = []
    def chunk(inventory, jobs):
        chunks.append(len(jobs))
        return [job[1] for job in jobs]
    solver._solve_chunk = chunk
    assert solver.solve_batch(inventory, requests(513)) == tuple(range(513))
    assert chunks == [512, 1]
