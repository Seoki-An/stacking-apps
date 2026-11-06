"""Shape-aware, original-anchor boxes shared by CPU/CUDA continuation."""
from types import SimpleNamespace
import itertools
import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from stacking_common.actions.core_solver import CorePlacementPoseSolver
from stacking_common.actions.types import InitializedPlacement
from test_cuda_posegen import adapter, requests


def stone(points, volume=.216):
    return SimpleNamespace(volume=volume, mesh_arrays=lambda: (points, None))


def test_box_uses_rotated_asymmetric_shape_and_inscribed_translation_budget():
    points=np.array(list(itertools.product([-.1,.9],[-.2,.2],[-.1,.1])))
    initial=InitializedPlacement(np.r_[2.,-3.,1.,Rotation.from_euler('z',90,degrees=True).as_quat()],0.)
    solver=object.__new__(CorePlacementPoseSolver)
    solver.inventory=[stone(points)]
    solver.max_initial_xy_displacement_ratio=.5
    bounds=np.array(solver._initial_box_bounds(0,initial))
    world=Rotation.from_quat(initial.pose[3:]).apply(points)+initial.pose[:3]
    padding=.3/np.sqrt(2)
    np.testing.assert_allclose(bounds,[world[:,0].min()-padding,world[:,0].max()+padding,
                                     world[:,1].min()-padding,world[:,1].max()+padding])
    assert (bounds[1]-bounds[0]) != pytest.approx(bounds[3]-bounds[2])
    assert np.linalg.norm([padding,padding])==pytest.approx(.3)
    solver.max_initial_xy_displacement_ratio=None
    assert solver._initial_box_bounds(0,initial) is None


def test_cuda_keeps_original_box_across_stages_and_candidates(monkeypatch):
    solver,calls,_=adapter(monkeypatch)
    solver.inventory=[stone(np.array([[-.2,-.1,0.],[.2,.1,.3]])),
                      stone(np.array([[-.4,-.2,0.],[.4,.2,.2]]),.512)]
    solver.max_initial_xy_displacement_ratio=.5
    solver._scheduled_config=lambda *args,**kwargs: SimpleNamespace(objective=SimpleNamespace(box_bounds=None))
    jobs=requests()
    expected=[solver._initial_box_bounds(idx,initial) for _,idx,initial in jobs]
    results = solver.solve_batch(None,jobs)
    for result, bounds in zip(results, expected):
        np.testing.assert_array_equal(result.diagnostics['posegen_box_bounds'], bounds)
        assert result.diagnostics['posegen_box_reference'] == 'initial_footprint'
    assert len(calls)==4
    for stage in calls:
        for item in stage:
            np.testing.assert_array_equal(item.config.objective.box_bounds,expected[item.request_id])
    assert calls[-1][0].problem[2][0] > calls[0][0].problem[2][0]


@pytest.mark.parametrize('backend',['cpu','cuda'])
def test_native_explicit_box_moves_full_shape_and_validates_bounds(backend):
    import stacking_core as core
    if not hasattr(core.PosegenObjectiveConfig(),'box_bounds'):
        pytest.skip('requires rebuilt stacking-core box_bounds')
    if backend=='cuda' and not core.cuda_capabilities().get('devices'):
        pytest.skip('requires visible CUDA device')
    points=np.array(list(itertools.product([-.1,.1],repeat=3)))
    model=core.BodyModel.dsf(1,points,mass=1.,inertia=np.eye(3))
    scene=core.SceneSnapshot(1,[core.BodyInstance(1,model,np.array([2.,0.,1.,0.,0.,0.,1.]))])
    problem=core.PosegenProblem(scene,candidate=1)
    cfg=core.PosegenConfig();cfg.objective.gravity=0.;cfg.objective.k_potential=0.
    cfg.objective.k_wrench=np.zeros((6,6));cfg.objective.k_box=100.;cfg.objective.w_box=10.
    cfg.objective.box_bounds=[-1.,1.,-.5,.5]
    if backend=='cpu':
        solve=lambda: core.PoseGenerator(cfg).solve(problem)
    else:
        generator=core.CudaPoseGenerator(core.CudaCapacity())
        solve=lambda: generator.solve([core.CudaPosegenBatchItem(0,problem,cfg)])[0].result
    result=solve()
    assert result is not None
    assert result.optimal_pose[0] < 1.01  # Would remain at x=2 with the old centered square.
    cfg.objective.box_bounds=[1.,-1.,-.5,.5]
    with pytest.raises(ValueError,match='box bounds'):
        solve()
