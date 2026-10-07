"""Evaluate a synthetic capture against initial/GT poses with stacking-core SceneID.

python apps/tabletop/scripts/test_sceneid_fixture.py FIXTURE_DIRECTORY --cuda
Add --solve to run the full joint pose/physics solver; --progress shows its iterations.
Add --visualize for Open3D, or --view-result to inspect a saved result without solving.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
import stacking_core as core
from stacking_common.planning.sceneid_fixture import load_fixture, pose_errors


def visualize_result(args, metadata, initial, truth, result):
    from stacking_common.rendering.sceneid import comparison_layers, show_comparison, write_comparison

    layers = comparison_layers(args.fixture, metadata, initial, truth, result)
    path = write_comparison(args.fixture/'visualization/overview.html', layers, result)
    print(f'SceneID comparison: {path.resolve()}', file=sys.stderr, flush=True)
    if args.visualize:
        show_comparison(layers, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('fixture', type=Path)
    parser.add_argument('--cuda', action='store_true', help='use SceneIdCudaScan for evaluation and optional solve')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--solve', action='store_true', help='run the full SceneID solver starting from perturbed poses')
    mode.add_argument('--view-result', action='store_true', help='visualize saved sceneid_result.json without solving or overwriting it')
    parser.add_argument('--progress', action='store_true', help='stream native solver iterations and stage timings to stderr')
    parser.add_argument('--visualize', action='store_true', help='also open the comparison in Open3D (requires a graphical desktop)')
    args = parser.parse_args()

    def progress(message):
        if args.progress:
            print(f'[fixture] {message}', file=sys.stderr, flush=True)

    progress('loading capture and models')
    problem, metadata, initial, truth = load_fixture(args.fixture)
    if args.view_result:
        path = args.fixture/'sceneid_result.json'
        if not path.is_file():
            parser.error(f'no saved result at {path}; run with --solve first')
        visualize_result(args, metadata, initial, truth, json.loads(path.read_text()))
        return
    config = core.SceneIdConfig()
    if args.progress:
        if not hasattr(config, 'verbose'):
            parser.error('--progress requires stacking-core rebuilt with SceneIdConfig.verbose support')
        config.verbose = True
    config.scan_sigma = max(.001, metadata['axial_depth_noise_std_m'])
    config.outlier_cutoff = .015
    config.reference_scan_points = max(1, metadata['point_count']//len(metadata['stones']))
    start_poses = np.array([s['pose'] for s in initial['stones']])
    truth_poses = np.array([s['pose'] for s in truth['stones']])
    progress(f"preparing {'CUDA' if args.cuda else 'CPU'} scan evaluation")
    resident = core.SceneIdCudaScan(problem, config) if args.cuda else None
    def evaluate(poses):
        return resident.evaluate(poses) if resident is not None else core.evaluate_sceneid_scan(problem, config, poses)
    started = time.perf_counter()
    progress('evaluating initial and ground-truth poses')
    before, ground_truth = evaluate(start_poses), evaluate(truth_poses)
    result = dict(kind='synthetic_sceneid_evaluation', frame=metadata['frame'],
        backend='cuda' if args.cuda else 'cpu', initial_scan_loss=before.loss,
        ground_truth_scan_loss=ground_truth.loss, initial_errors=pose_errors(start_poses, truth_poses),
        ground_truth_fit=[dict(entity_id=s.entity_id, inliers=s.inliers, rmse_m=s.rmse) for s in ground_truth.stones],
        evaluation_s=time.perf_counter()-started, solve_requested=args.solve)
    if args.solve:
        progress('starting joint SceneID solve')
        started = time.perf_counter()
        solved = resident.solve() if resident is not None else core.SceneIdentifier(config).solve(problem)
        by_entity = {s['entity_id']: s['pose'] for s in truth['stones']}
        result.update(status=str(solved.status), accepted_snapshot=solved.snapshot is not None,
            solve_s=time.perf_counter()-started, support_warning=solved.support_warning,
            proposed_poses=[dict(entity_id=e, pose=np.asarray(p).tolist()) for e,p in zip(solved.entity_ids,solved.proposed_poses)],
            diagnostics=dict(icp_status=str(solved.icp.status), physical_status=str(solved.physical.status),
                consensus_iterations=solved.consensus.iterations,
                consensus_primal_residual=solved.consensus.primal_residual,
                consensus_dual_residual=solved.consensus.dual_residual,
                nonpenetrating=solved.gap_screen.nonpenetrating, worst_gap_m=solved.gap_screen.worst_gap))
        if solved.proposed_poses:
            result['proposed_errors'] = pose_errors(np.array(solved.proposed_poses),
                                                    np.array([by_entity[e] for e in solved.entity_ids]))
    destination = args.fixture/'sceneid_result.json'
    destination.write_text(json.dumps(result, indent=2)+'\n')
    progress(f'result saved to {destination}')
    print(json.dumps(result, indent=2), flush=True)
    visualize_result(args, metadata, initial, truth, result)


if __name__ == '__main__':
    main()
