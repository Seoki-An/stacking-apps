"""Early stops only shorten trajectories that cannot be accepted."""
import numpy as np
import pytest
import stacking_core as core
from stacking_common.environment.robustness import PlaceStabilityConfig, trial_outcome
from stacking_common.environment.core_backend import SimulationPhaseConfig


@pytest.fixture
def simulator():
    if not hasattr(core,'CudaExcursionStopping') or not core.cuda_capabilities().get('devices'):
        pytest.skip('requires rebuilt CUDA simulation and visible device')
    return core.CudaSimulator(core.CudaCapacity.small(8))


def problem(*, speed=1., angular=0., steps=100, quiet=10, min_steps=0, second=False):
    model=core.BodyModel.dsf(1,np.concatenate([np.eye(3),-np.eye(3)])*.05,
        mass=1.,inertia=np.eye(3),sharpness=2)
    def body(id,x,v,w):
        return core.BodyInstance(id,model,np.array([x,0.,1.,0.,0.,0.,1.]),
            np.array([v,0.,0.]),np.array([0.,0.,w]))
    bodies=[body(1,0.,0. if second else speed,0. if second else angular)]
    if second:
        bodies.append(body(2,2.,speed,angular))
    p=core.CudaSimulationProblem();p.request_id=0
    p.scene=core.SceneSnapshot(0,bodies)
    c=core.SimulationConfig();c.gravity=np.zeros(3);c.damping=0.;p.config=c
    phase=core.CudaSimulationPhase();phase.steps=steps;phase.dt=.01
    phase.min_steps=min_steps;phase.quiet_steps=quiet;phase.energy_threshold=1e-5
    phase.monitored=[b.id for b in bodies];phase.sample_stride=1;p.phases=[phase]
    return p


def enable(p,*,limit=None,rotation=.1,quiet=False):
    phase=p.phases[0]
    if limit is not None:
        rule=core.CudaExcursionStopping();rule.limit=limit;rule.rotation_weight=rotation
        phase.excursion=rule
    phase.stop_if_quiet_impossible=quiet;p.phases=[phase]


@pytest.mark.parametrize('second,angular',[(False,0.),(True,0.),(False,1.)])
def test_excursion_prefix_and_neighbor_motion(simulator,second,angular):
    p=problem(speed=0. if angular else 1.,angular=angular,second=second)
    full=simulator.run([p])[0]
    enable(p,limit=.025)
    early=simulator.run([p])[0]
    assert early.status.name=='excursion_limit'
    assert early.snapshot is not None
    assert early.phases[0].steps<full.phases[0].steps
    entity=2 if second else 1
    history=early.sample_poses(entity)
    np.testing.assert_array_equal(history,full.sample_poses(entity)[:len(history)])
    phase=SimulationPhaseConfig(.01,1e-5,100.,0.,1.,10)
    metric,reason=trial_outcome(p.scene.body_pose(entity),history,
        PlaceStabilityConfig(fail_displacement=.025,rotation_weight=.1),phase,energy_settled=False)
    assert reason=='excursion_limit'
    assert metric>.025
    assert len(history)==early.phases[0].steps


@pytest.mark.parametrize('steps',[31,32,33,100])
def test_safe_trajectories_exact_with_shortcuts(simulator,steps):
    p=problem(speed=0.,steps=steps,quiet=steps,min_steps=0)
    full=simulator.run([p])[0]
    enable(p,quiet=True)
    fast=simulator.run([p])[0]
    assert fast.status.name=='quiet'
    np.testing.assert_array_equal(full.sample_poses(1),fast.sample_poses(1))
    enable(p,limit=0.,quiet=False) # Equality is not excursion failure.
    fast=simulator.run([p])[0]
    assert fast.status.name=='quiet'
    np.testing.assert_array_equal(full.sample_poses(1),fast.sample_poses(1))


@pytest.mark.parametrize('minimum,quiet,expected',[(0,20,82),(90,20,1),(80,20,82)])
def test_impossible_quiet_window(simulator,minimum,quiet,expected):
    p=problem(steps=101,min_steps=minimum,quiet=quiet)
    full=simulator.run([p])[0]
    assert full.status.name=='complete' and full.phases[0].quiet_steps==0
    enable(p,quiet=True)
    fast=simulator.run([p])[0]
    assert fast.status.name=='quiet_impossible'
    assert fast.phases[0].steps==expected
    np.testing.assert_array_equal(fast.sample_poses(1),full.sample_poses(1)[:expected])


def test_mixed_batch_independent_stopping(simulator):
    a=problem(speed=1.);b=problem(speed=0.);b.request_id=1
    full=simulator.run([a,b])
    enable(a,limit=.05);enable(b,limit=.05)
    fast=simulator.run([a,b])
    assert [x.status.name for x in fast]==['excursion_limit','quiet']
    np.testing.assert_array_equal(full[1].sample_poses(1),fast[1].sample_poses(1))


def test_quiet_stop_cannot_truncate_intermediate_phase(simulator):
    p=problem();enable(p,quiet=True)
    p.phases=[*p.phases,core.CudaSimulationPhase()]
    with pytest.raises(ValueError,match='final structural'):
        simulator.run([p])


@pytest.mark.parametrize('limit,rotation',[(float('nan'),0.),(-1.,0.),(1.,-1.)])
def test_invalid_excursion_rules(simulator,limit,rotation):
    p=problem();enable(p,limit=limit,rotation=rotation)
    with pytest.raises(ValueError,match='finite nonnegative'):
        simulator.run([p])


@pytest.mark.parametrize('energy,minimum,maximum,required',[
    (0.,0.,.99,100), (1.,0.,1.,20), (0.,.9,1.,20),
    (0.,.1,.3,20), (1.,0.,.3,4),
])
def test_cpu_quiet_stop_preserves_acceptance(energy,minimum,maximum,required):
    from types import SimpleNamespace
    from stacking_common.environment.core_backend import CorePlacementBackend
    from stacking_common.domain import StackObjectTrajectory
    backend=object.__new__(CorePlacementBackend)
    backend.object_ids=[10]
    backend.scene_builder=SimpleNamespace(entity_id=lambda i:i)
    pose=np.array([0.,0.,0.,0.,0.,0.,1.])
    scene=SimpleNamespace(body_pose=lambda i:pose,
        body_linear_velocity=lambda i:np.zeros(3),body_angular_velocity=lambda i:np.zeros(3))
    backend.simulator=SimpleNamespace(step=lambda scene,dt:SimpleNamespace(snapshot=scene,contacts=()))
    backend._normalized_energy=lambda *args:energy
    phase=SimulationPhaseConfig(.01,1e-5,100.,minimum,maximum,required)
    outputs=[]
    for stop in (False,True):
        trajectory=StackObjectTrajectory(10)
        outputs.append(backend._simulate_phase(scene,[0],phase,{10:trajectory},{10:0.},0.,0,
            stop_if_quiet_impossible=stop))
    assert outputs[0][-2:]==outputs[1][-2:]
    if outputs[0][-2]:
        assert outputs[0][1:3]==outputs[1][1:3]
    else:
        assert outputs[1][1]<outputs[0][1]


def test_cpu_quiet_bound_with_resets_and_exact_deadline():
    from itertools import product
    from types import SimpleNamespace
    from stacking_common.environment.core_backend import CorePlacementBackend
    from stacking_common.domain import StackObjectTrajectory
    backend=object.__new__(CorePlacementBackend)
    backend.object_ids=[10]
    backend.scene_builder=SimpleNamespace(entity_id=lambda i:i)
    pose=np.array([0.,0.,0.,0.,0.,0.,1.])
    scene=SimpleNamespace(body_pose=lambda i:pose,
        body_linear_velocity=lambda i:np.zeros(3),body_angular_velocity=lambda i:np.zeros(3))
    backend.simulator=SimpleNamespace(step=lambda scene,dt:SimpleNamespace(snapshot=scene,contacts=()))
    phase=SimulationPhaseConfig(.01,1e-5,100.,.02,.06,3)
    for energies in product((0.,1.),repeat=7):
        results=[]
        for early in (False,True):
            samples=iter(energies)
            backend._normalized_energy=lambda *args:next(samples)
            results.append(backend._simulate_phase(scene,[0],phase,
                {10:StackObjectTrajectory(10)},{10:0.},0.,0,stop_if_quiet_impossible=early))
        assert results[0][-2:]==results[1][-2:]
        if results[0][-2]:
            assert results[0][1:3]==results[1][1:3]


def test_quiet_bound_preserves_continued_phase_budget(simulator):
    p=problem(speed=0.,steps=3,quiet=3)
    # Own the phases: replacing p.phases invalidates references into its vector.
    first=core.CudaSimulationPhase()
    first.steps=3;first.quiet_steps=3;first.energy_threshold=1e-5
    first.monitored=[1];first.sample_stride=1
    second=core.CudaSimulationPhase()
    second.dt=first.dt;second.steps=100;second.max_total_steps=8
    second.min_steps=5;second.quiet_steps=6
    second.monitored=first.monitored;second.energy_threshold=first.energy_threshold
    second.sample_stride=1;second.continue_previous=True
    p.phases=[first,second]
    full=simulator.run([p])[0]
    second.stop_if_quiet_impossible=True;p.phases=[first,second]
    fast=simulator.run([p])[0]
    assert full.status.name==fast.status.name=='quiet'
    assert [v.steps for v in fast.phases]==[3,5]
    np.testing.assert_array_equal(fast.sample_poses(1),full.sample_poses(1))
