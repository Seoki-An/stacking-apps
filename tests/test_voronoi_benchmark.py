"""Geometry certificates, asset interoperability and honest greedy selection."""

from dataclasses import replace
import json
from pathlib import Path

import numpy as np
import pytest

from stacking_common.benchmarks.voronoi import (
    _prism, export_puzzle, make_layered_voronoi, voronoi_cells,
)
from stacking_common.domain import Action, State
from stacking_common.domain.geometry import inverse_transform_points, transform_points
from stacking_common.profiles import load_inventory_profile, load_mcts_profile
from stacking_common.search import MCTSNode, MonteCarloTreeSearch
from benchmark_voronoi_mcts import greedy_profile, require_depth_finished, require_native_reference

ROOT=Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('cells,seed',[(3,0),(3,1),(4,0),(4,1)])
def test_reference_is_a_complete_nonoverlapping_tiling(cells,seed):
    puzzle=make_layered_voronoi(seed,cells_per_layer=cells)
    assert sum(p.geometry.volume for p in puzzle.pieces)==pytest.approx(puzzle.target_volume,rel=1e-12)
    assert sorted(p.object_id for p in puzzle.pieces)==list(range(1,2*cells+1))
    layers={p.object_id:p.layer for p in puzzle.pieces}
    assert [layers[i] for i in puzzle.reference_order]==sorted(layers.values())
    x=puzzle.width*((np.arange(43)+.371)/43-.5)
    y=puzzle.length*((np.arange(47)+.619)/47-.5)
    xx,yy=np.meshgrid(x,y)
    for layer in range(2):
        queries=np.column_stack([xx.ravel(),yy.ravel(),np.full(xx.size,(layer+.5)*puzzle.layer_height)])
        count=np.zeros(len(queries),dtype=int)
        for piece in puzzle.pieces:
            count+=piece.geometry.contains(inverse_transform_points(queries,piece.reference_pose))
            world=transform_points(piece.geometry.points,piece.reference_pose)
            assert world[:,2].min()==pytest.approx(piece.layer*puzzle.layer_height)
            assert world[:,2].max()==pytest.approx((piece.layer+1)*puzzle.layer_height)
            assert np.linalg.eigvalsh(piece.inertia).min()>0
        np.testing.assert_array_equal(count,np.ones(len(queries),dtype=int))


def test_prism_mass_inertia_and_yaw_match_analytic_box():
    rectangle=np.array([[-1.,-2.],[1.,-2.],[1.,2.],[-1.,2.]])
    geometry,centroid,inertia=_prism(rectangle,3.,0.)
    assert geometry.volume==pytest.approx(24.)
    np.testing.assert_allclose(centroid,0.,atol=1e-12)
    expected=np.diag([50.,26.,40.])
    np.testing.assert_allclose(inertia,expected,atol=1e-12)
    yaw=.43;c,s=np.cos(yaw),np.sin(yaw)
    rotation=np.array([[c,-s,0.],[s,c,0.],[0.,0.,1.]])
    _,_,rotated=_prism(rectangle,3.,yaw)
    np.testing.assert_allclose(rotated,rotation@expected@rotation.T,atol=1e-12)


def test_assets_load_with_the_existing_native_inventory_and_height_initializer(tmp_path):
    pytest.importorskip('stacking_core')  # Body-model construction only; no native solve/GPU work.
    from stacking_common.assets import ObjectAssetLoader
    from stacking_common.actions.ply_height import read_ply_vertices
    puzzle=make_layered_voronoi(2,cells_per_layer=3)
    path=tmp_path/'case';export_puzzle(puzzle,path)
    profile=load_inventory_profile(path/'inventory.yml')
    assert profile.target.geometry().volume==pytest.approx(puzzle.target_volume,rel=1e-12)
    inventory=ObjectAssetLoader(profile.model_directory,profile.density,profile.target.geometry()).load(
        profile.object_ids,{i:np.array([0.,0.,0.,0.,0.,0.,1.]) for i in profile.object_ids})
    for obj,piece in zip(inventory,puzzle.pieces):
        assert obj.mass==pytest.approx(1000.*piece.geometry.volume)
        np.testing.assert_allclose(obj.inertia,1000.*piece.inertia,rtol=1e-12,atol=1e-10)
        np.testing.assert_allclose(read_ply_vertices(path/'assets'/f'model_{obj.object_id}_mesh.ply'),
                                   piece.geometry.points,rtol=1e-12,atol=1e-12)
    reference=json.loads((path/'reference.json').read_text())
    assert reference['privileged'] and set(reference['poses'])=={str(i) for i in profile.object_ids}
    assert 'poses' not in json.loads((path/'case.json').read_text())
    with pytest.raises(FileExistsError):
        export_puzzle(puzzle,path)


def test_seed_is_reproducible_and_layers_are_not_copies():
    a=make_layered_voronoi(7);b=make_layered_voronoi(7)
    for x,y in zip(a.pieces,b.pieces):
        np.testing.assert_array_equal(x.geometry.points,y.geometry.points)
        np.testing.assert_array_equal(x.reference_pose,y.reference_pose)
    centers=[[tuple(p.reference_pose[:2]) for p in a.pieces if p.layer==layer] for layer in [0,1]]
    assert set(centers[0])!=set(centers[1])


def test_invalid_sites_and_dimensions_fail_before_asset_creation():
    with pytest.raises(ValueError):
        voronoi_cells([[0.,0.],[0.,0.]],2.,2.)
    with pytest.raises(ValueError):
        make_layered_voronoi(0,layer_height=0.)
    with pytest.raises(ValueError):
        make_layered_voronoi(0,cells_per_layer=1)


def test_pending_depth_benchmark_and_missing_reference_block_native_runs(tmp_path):
    status=tmp_path/'depth-status.json';status.write_text(json.dumps({'phase':'running'}))
    (tmp_path/'manifest.json').write_text(json.dumps({'depth_benchmark_status':str(status)}))
    with pytest.raises(RuntimeError,match='not complete'):
        require_depth_finished(tmp_path)
    with pytest.raises(RuntimeError,match='has not passed'):
        require_native_reference(tmp_path,'unknown')


def test_greedy_ranks_immediate_reward_even_with_conflicting_means_and_descendants():
    profile=greedy_profile(load_mcts_profile(ROOT/'apps/excavator/config/mcts_cuda.yml'))
    engine=MonteCarloTreeSearch(profile.tree,None,None,None,None)
    state=State([1,2,3],[],{},[],[])
    root=MCTSNode(state=state,observation=())
    pose=np.array([0.,0.,1.,0.,0.,0.,1.])
    immediate=MCTSNode(action=Action(0,pose,pose,object_id=1),state=state,observation=(),
                       reward=2.,visits=100,value_sum=-100.)
    optimistic=MCTSNode(action=Action(1,pose,pose,object_id=2),state=state,observation=(),
                        reward=1.,visits=1,value_sum=1000.)
    future=MCTSNode(action=Action(2,pose,pose,object_id=3),state=state,observation=(),
                   reward=100.)
    optimistic.attach(future);root.attach(immediate);root.attach(optimistic)
    failed=MCTSNode(action=Action(2,pose,pose,object_id=3),state=state,observation=(),
                   reward=1000.,failed=True,rejection_reason='simulation_failure')
    root.attach(failed)
    selected,ranked=engine._final_select(root)
    assert selected is immediate
    assert ranked==(immediate,optimistic)
    assert all(n.info['descendant_path_return']==n.reward for n in ranked)
    # In the depth2 max-path control, the feasible continuation changes the choice.
    engine.config=replace(profile.tree,max_depth=2)
    assert engine._final_select(root)[0] is optimistic
