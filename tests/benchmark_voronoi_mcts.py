"""Prepare and later run native greedy/MCTS comparisons on Voronoi prisms.

Preparation is CPU-only. Native validation and planning refuse to start while
the depth benchmark is unfinished. See docs/voronoi-search-benchmark.md.
"""

import argparse
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import subprocess
import sys
import time

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = ('packages/stacking-common/src', 'apps/excavator/src')


def bootstrap(dataset=None):
    source = ROOT if dataset is None else dataset / 'source'
    for part in SOURCE_PATHS:
        sys.path.insert(0, str(source / part))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def encode(value):
    from collections.abc import Mapping
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value))


def write_json(path, value):
    temporary = path.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value, indent=2, default=encode))
    temporary.replace(path)


def require_depth_finished(dataset):
    manifest = json.loads((dataset / 'manifest.json').read_text())
    path = Path(manifest['depth_benchmark_status'])
    if not path.is_file():
        raise RuntimeError(f'depth benchmark status is missing: {path}')
    status = json.loads(path.read_text())
    if status.get('phase') != 'complete':
        raise RuntimeError('depth benchmark is not complete; Voronoi native work remains blocked')
    if len(status.get('completed', [])) != 4:
        raise RuntimeError('depth benchmark must contain four finished full rollouts')
    if digest(__file__) != manifest['harness_sha256']:
        raise RuntimeError('benchmark harness changed; prepare a new dataset snapshot')
    import stacking_core as core
    if digest(core._native.__file__) != manifest['native_sha256']:
        raise RuntimeError('native library changed; prepare a new dataset snapshot before comparing')
    for group in ('asset_hashes', 'source_hashes', 'profile_hashes'):
        for path, expected in manifest[group].items():
            if digest(dataset / path) != expected:
                raise RuntimeError(f'prepared benchmark input changed: {path}')
    return manifest


def prepare(args):
    bootstrap()
    from stacking_common.benchmarks.voronoi import make_layered_voronoi, export_puzzle
    import stacking_core as core

    dataset = args.dataset
    dataset.mkdir(parents=True, exist_ok=False)
    for part in SOURCE_PATHS:
        shutil.copytree(ROOT / part, dataset / 'source' / part,
                        ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copytree(ROOT / 'apps/excavator/config', dataset / 'repository/apps/excavator/config')
    rows = []
    for cells in (3, 4):
        for seed in args.seeds:
            name = f'cells{cells}-seed{seed}'
            puzzle = make_layered_voronoi(seed, cells_per_layer=cells)
            export_puzzle(puzzle, dataset / 'cases' / name)
            rows.append(dict(name=name, cells_per_layer=cells, seed=seed, pieces=len(puzzle.pieces),
                             polyhedral_volume=sum(p.geometry.volume for p in puzzle.pieces),
                             target_volume=puzzle.target_volume, native_reference_status='not_run'))
    manifest = dict(schema_version=1, cases=rows, repository_root=str(ROOT),
        depth_benchmark_status=str(args.depth_status.resolve()),
        native_path=core._native.__file__, native_sha256=digest(core._native.__file__),
        harness_sha256=digest(__file__),
        asset_hashes={str(p.relative_to(dataset)): digest(p) for p in (dataset/'cases').rglob('*') if p.is_file()},
        source_hashes={str(p.relative_to(dataset)): digest(p) for p in (dataset/'source').rglob('*.py')},
        profile_hashes={str(p.relative_to(dataset)): digest(p) for p in (dataset/'repository').rglob('*.yml')},
        protocol=dict(reference_policy='Native validation is mandatory before comparison.',
            planner_inputs='Only inventory.yml, model OBJ/PLY assets and normal platform profiles; no reference poses.',
            reward='Unchanged platform multi-term reward; geometric fill1 is not a reward-optimality claim.',
            greedy='Depth1, root lookahead off, bootstrap0, final path-only score = immediate evaluated reward.',
            mcts='Current MCTS, bootstrap and tree reuse; same root CEM, pose optimization, physics and filters.',
            budgets='Same per-search wall cap; report actual native evaluations and total time for both.',
            native_status='not_run', cases_selected_by='Fixed seeds and geometry-only cell-area criterion; no performance screening.'))
    write_json(dataset / 'manifest.json', manifest)
    (dataset / 'PREPARED.txt').write_text('CPU geometry/assets prepared. Native reference validation and greedy/MCTS planning have NOT run.\n'
        'No automatic queue has been started. Native commands wait for the recorded depth benchmark to finish.\n')
    preview(dataset)
    print(f'Prepared {len(rows)} cases; no native solving or planning: {dataset}', flush=True)


def preview(dataset):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from stacking_common.assets import _parse_object_asset
    from stacking_common.domain.geometry import transform_points

    cases=json.loads((dataset/'manifest.json').read_text())['cases']
    fig=plt.figure(figsize=(12, 4 * ((len(cases)+2)//3)), layout='constrained')
    for i,case in enumerate(cases,1):
        path=dataset/'cases'/case['name'];reference=json.loads((path/'reference.json').read_text())
        ax=fig.add_subplot((len(cases)+2)//3,3,i,projection='3d')
        for object_id,pose in reference['poses'].items():
            asset=_parse_object_asset(path/'assets'/f'model_{object_id}.obj')
            for component in asset.components:
                geometry=component.geometry;world=transform_points(geometry.points,np.asarray(pose))
                ax.add_collection3d(Poly3DCollection(world[geometry.faces],
                    facecolor=plt.get_cmap('tab10')((int(object_id)-1)%10),edgecolor='black',linewidth=.4,alpha=.85))
        ax.set(xlim=(-1.25,1.25),ylim=(-.85,.85),zlim=(0,1.25),xlabel='x [m]',ylabel='y [m]',zlabel='z [m]')
        ax.set_box_aspect((2.4,1.6,1.2));ax.view_init(25,-55);ax.set_title(f"{case['name']}: {case['pieces']} pieces")
    fig.suptitle('Exact polyhedral reference arrangements — native DSF/simulation validation pending')
    fig.savefig(dataset/'reference-preview.png',dpi=160);plt.close(fig)


def application(dataset, case, seed, mcts_config=None):
    from stacking_excavator import ExcavatorApplication
    from stacking_common.profiles import load_inventory_profile

    app=ExcavatorApplication.from_repository(dataset/'repository',
        mcts_config=mcts_config or dataset/'repository/apps/excavator/config/mcts_cuda.yml')
    profile=replace(app.mcts_profile,root_cem=replace(app.mcts_profile.root_cem,seed=seed))
    return replace(app,repository_root=ROOT,pose_from_scan=False,mcts_profile=profile,
                   inventory_profile=load_inventory_profile(dataset/'cases'/case/'inventory.yml'))


def greedy_profile(profile):
    from stacking_common.search.mcts import FinalSelectionConfig
    # Root preparation/physical admission/CEM retries are identical. At depth1
    # path return is exactly the root reward; neither PUCT nor bootstrap can rank it.
    return replace(profile,tree=replace(profile.tree,max_depth=1,initial_depth1_steps=0,
        root_lookahead=False,final_selection=FinalSelectionConfig(max_weight=1.)),
        bootstrap_horizon=0,request=replace(profile.request,iterations=1))


def inventory_and_context(app, method, seed):
    from stacking_common.artifacts.planning_context import _values
    from stacking_common.planning import PickSceneSnapshot
    from stacking_common.profiles.native import load_native_solver_profile
    from stacking_excavator.planning.cli import planning_context

    ids=app.inventory_profile.object_ids
    poses={i:np.array([0.,0.,0.,0.,0.,0.,1.]) for i in ids}
    inventory=app.load_inventory(ids,poses)
    scene=PickSceneSnapshot(app.planning_profile.world_T_stack.parent_frame,poses,
        'Procedural catalog; no measured pickups',{'format':'synthetic_search_inventory'})
    solver=load_native_solver_profile(app.solver_profiles.grasp,app.solver_profiles.motion)
    context=planning_context(app,inventory,scene,solver,dict(search_only=True,seed=seed,
        benchmark_method=method,iterations=app.mcts_profile.request.iterations,
        max_search_time=app.mcts_profile.request.max_time_seconds))
    return inventory,scene,context


def require_native_reference(dataset, case):
    path=dataset/'validation'/case/'summary.json'
    if not path.is_file() or not json.loads(path.read_text()).get('passed',False):
        raise RuntimeError(f'{case}: native reference has not passed; run validate-reference first')


def run_native(args):
    dataset=args.dataset
    bootstrap(dataset)
    require_depth_finished(dataset)  # Before allocating any native solver/environment.
    if args.command=='run':
        require_native_reference(dataset,args.case)
    from stacking_common.actions import PlanarPlacement
    from stacking_common.artifacts import StackRunStore, PlanningCheckpoint
    from stacking_common.artifacts.planning_context import _values
    from stacking_common.domain import State
    from stacking_common.environment.overlap import VoxelOverlapEvaluator
    from stacking_common.search import MCTSNode,MCTSSession,EvaluationFidelity
    from stacking_common.visualization import MCTSDebugArtifact,target_display
    from stacking_common.visualization.mcts import project_mcts_step
    from stacking_common.workflows.mcts_reporting import search_statistics

    reference_mode=args.command=='validate-reference'
    method='reference' if reference_mode else args.method
    output=dataset/'validation'/args.case if reference_mode else dataset/'runs'/args.case/f'{method}-seed{args.seed}'
    output.mkdir(parents=True,exist_ok=False)
    app=application(dataset,args.case,args.seed,args.mcts_config)
    profile=replace(app.mcts_profile,request=replace(app.mcts_profile.request,
        max_time_seconds=args.seconds,iterations=args.iterations))
    if method=='greedy':
        profile=greedy_profile(profile)
    app=replace(app,mcts_profile=profile)
    inventory,scene,context=inventory_and_context(app,method,args.seed)
    observer,_=app._build_state_evaluators(inventory)
    state=State(list(inventory.object_ids),[],{},[],[])
    writer=StackRunStore.create_mcts(output/'run.stack',
        MCTSDebugArtifact('stack',(),target_display(inventory.target)),
        metadata={'platform':'excavator','mode':'search','benchmark':'voronoi',
                  'case':args.case,'method':method,'seed':args.seed,'reference_poses_used':reference_mode})
    writer.append_run_context(context)
    engine=None
    summary=dict(case=args.case,method=method,seed=args.seed,rows=[],outcome='running',
                 profile=_values(profile),reference_poses_used=reference_mode,
                 native_global_reward_optimum_certified=False)
    try:
        engine=app.build_mcts(inventory,rng=random.Random(args.seed))
        session=MCTSSession(engine,MCTSNode(state=state,observation=observer.build(state)))
        reference=(json.loads((dataset/'cases'/args.case/'reference.json').read_text()) if reference_mode else None)
        for k in range(len(inventory)+1):
            if session.root.terminal:
                summary['outcome']='inventory_exhausted';break
            step=k+1;root=session.root;start=time.perf_counter()
            if reference_mode:
                object_id=reference['order'][k];index=inventory.object_ids.index(object_id)
                generator=getattr(engine.root_action_generator,'generator',engine.action_generator)
                candidate,reason=generator.build_candidate(root.state,index,
                    PlanarPlacement(np.asarray(reference['poses'][str(object_id)])))
                if candidate is None:
                    summary['rows'].append(dict(step=step,object_id=object_id,accepted=False,
                        stage='posegen_or_admission',reason=reason,seconds=time.perf_counter()-start))
                    summary['outcome']='reference_generation_failed';break
                engine.transition_evaluator.begin_search(root.state)
                evaluation=engine.transition_evaluator.evaluate(root.state,candidate.action,EvaluationFidelity.LONG)
                node=MCTSNode(action=candidate.action,state=evaluation.state,observation=evaluation.observation,
                    reward=evaluation.reward,failed=evaluation.failed,rejection_reason=evaluation.rejection_reason,
                    info=evaluation.diagnostics,visits=1,value_sum=evaluation.reward)
                root.attach(node);selected=None if node.failed else node
                statistics=dict(step=step,reference_validation=True,elapsed_seconds=time.perf_counter()-start)
                row=dict(step=step,object_id=object_id,accepted=not node.failed,stage='transition',
                    reason=node.rejection_reason,seconds=statistics['elapsed_seconds'],reward=node.reward,
                    optimized_xy_shift=float(np.linalg.norm(candidate.action.pose[:2]-np.asarray(reference['poses'][str(object_id)])[:2])),
                    info=node.info)
            else:
                result=session.search(profile.request);selected=result.selected;root=result.root
                statistics=search_statistics(result,time.perf_counter()-start,engine,step=step)
                row=dict(step=step,accepted=selected is not None,reason=result.failure_reason,
                    seconds=statistics['elapsed_seconds'],statistics=statistics,
                    object_id=None if selected is None else selected.action.object_id,
                    reward=None if selected is None else selected.reward,
                    info={} if selected is None else selected.info)
                if method=='greedy' and selected is not None:
                    assert selected.reward==max(n.reward for n in root.evaluated_children())
            writer.append_mcts_step(project_mcts_step(root,selected,step=step,frame_id='stack'),
                statistics=statistics,checkpoint=None if selected is None else PlanningCheckpoint(
                    step,context.sha256,selected.state,scene,app.planning_profile.world_T_stack))
            summary['rows'].append(row);write_json(output/'summary.json',summary)
            print(json.dumps({k:row[k] for k in ['step','object_id','accepted','reason','seconds']},default=encode),flush=True)
            if selected is None:
                summary['outcome']='reference_transition_failed' if reference_mode else result.failure_reason;break
            state=selected.state.copy();session.promote(selected)
        summary['passed']=reference_mode and len(state.placed_object_indices)==len(inventory)
        summary['placed']=len(state.placed_object_indices)
        summary['total_search_seconds']=sum(row['seconds'] for row in summary['rows'])
        rewards=[row['reward'] for row in summary['rows'] if row.get('accepted') and 'reward' in row]
        summary['total_reward']=sum(rewards)
        summary['discounted_return']=sum(profile.tree.discount**i*r for i,r in enumerate(rewards))
        summary['evaluated_candidates']=sum(row.get('statistics',{}).get('evaluated_candidates',0)
                                            for row in summary['rows'])
        overlap=VoxelOverlapEvaluator(inventory.target,.05)
        occupied=np.zeros(len(overlap._grid_points),dtype=bool)
        for index in state.placed_object_indices:
            occupied[overlap._occupied_rows(inventory.objects[index],state.object_poses[inventory.object_ids[index]])]=True
        summary['native_pipeline_polyhedral_fill']=float(occupied.sum()*.05**3/overlap.target_volume)
        summary['geometry_note']='Mesh-union metric at native settled poses; not exact DSF occupied volume or a global reward optimum.'
        if any(row.get('stage')=='transition' for row in summary['rows']) or not reference_mode:
            from stacking_excavator.gui.operations.controller import OperationsController
            summary['viewer_verified_steps']=len(OperationsController(ROOT).load(output/'run.stack').steps)
        else:
            summary['viewer_verified_steps']=0
        write_json(output/'summary.json',summary)
    except BaseException as error:
        summary['outcome']='error';summary['error']=repr(error);write_json(output/'summary.json',summary);raise
    finally:
        writer.finish('complete' if summary['outcome']=='inventory_exhausted' else 'failed')
        if engine is not None:
            engine.close()


def suite(args):
    bootstrap(args.dataset);manifest=require_depth_finished(args.dataset)
    env=dict(os.environ,OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1',QT_QPA_PLATFORM='offscreen')
    jobs=[]
    for case in manifest['cases']:
        name=case['name']
        command=[sys.executable,'-u',__file__,'validate-reference','--dataset',str(args.dataset),'--case',name]
        if args.mcts_config:
            command+=['--mcts-config',str(args.mcts_config)]
        code=subprocess.call(command,env=env)
        path=args.dataset/'validation'/name/'summary.json'
        passed=not code and path.is_file() and json.loads(path.read_text()).get('passed',False)
        jobs.append(dict(case=name,reference_passed=passed,validation_exit_code=code))
        if passed:
            for seed in args.seeds:
                for method in (['greedy','mcts'] if seed%2==0 else ['mcts','greedy']):
                    command=[sys.executable,'-u',__file__,'run','--dataset',str(args.dataset),'--case',name,
                        '--method',method,'--seed',str(seed),'--seconds',str(args.seconds),
                        '--iterations',str(args.iterations)]
                    if args.mcts_config:
                        command+=['--mcts-config',str(args.mcts_config)]
                    jobs.append(dict(case=name,method=method,seed=seed,exit_code=subprocess.call(command,env=env)))
        write_json(args.dataset/'suite-results.json',dict(jobs=jobs))


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    for name in ['prepare','validate-reference','run','suite']:
        p=sub.add_parser(name);p.add_argument('--dataset',type=Path,required=True)
        if name=='prepare':
            p.add_argument('--seeds',type=int,nargs='+',default=[0,1,2])
            p.add_argument('--depth-status',type=Path,default=ROOT/'.stacks/benchmarks/mcts-full-rollout-20261101/status.json')
        else:
            p.add_argument('--seconds',type=float,default=60.)
            p.add_argument('--iterations',type=int,default=4096)
            p.add_argument('--mcts-config',type=Path)
            if name=='suite':
                p.add_argument('--seeds',type=int,nargs='+',default=[0,1])
            else:
                p.add_argument('--case',required=True);p.add_argument('--seed',type=int,default=0)
            if name=='run':
                p.add_argument('--method',choices=['greedy','mcts'],required=True)
    args=parser.parse_args(argv);args.dataset=args.dataset.resolve()
    try:
        {'prepare':prepare,'validate-reference':run_native,'run':run_native,'suite':suite}[args.command](args)
    except (RuntimeError,ValueError,FileExistsError,FileNotFoundError) as error:
        parser.exit(2,str(error)+'\n')


if __name__=='__main__':
    main()
