"""Isolated stacked-scene reconstruction followed by live SceneID."""

import argparse
import json
import os
from pathlib import Path
import time

import numpy as np

from .hand_eye_capture import atomic_json, digest


def _sceneid_record(candidate, *, backend, elapsed):
    result = candidate.result
    record = {
        'kind': 'tabletop_stack_scene_identification',
        'status': str(result.status),
        'backend': backend,
        'accepted_snapshot': result.snapshot is not None,
        'support_warning': bool(result.support_warning),
        'elapsed_s': elapsed,
        'entity_ids': [int(value) for value in result.entity_ids],
        'proposed_poses': [np.asarray(value).tolist()
                           for value in result.proposed_poses],
        'evidence': [{
            'entity_id': int(value.entity_id),
            'prior_carried': bool(value.prior_carried),
            'inliers': int(value.inliers),
            'rmse_m': float(value.rmse),
            'min_information': float(value.min_information),
            'translation_change_m': float(value.translation_change),
            'rotation_change_rad': float(value.rotation_change),
            'initial_coverage': float(value.initial_coverage),
        } for value in result.evidence],
        'diagnostics': {
            'icp_status': str(result.icp.status),
            'physical_status': str(result.physical.status),
            'consensus_iterations': int(result.consensus.iterations),
            'consensus_primal_residual': float(result.consensus.primal_residual),
            'consensus_dual_residual': float(result.consensus.dual_residual),
            'nonpenetrating': bool(result.gap_screen.nonpenetrating),
            'worst_gap_m': float(result.gap_screen.worst_gap),
            'max_normalized_wrench': float(result.max_normalized_wrench),
            'max_separated_force': float(result.max_separated_force),
        },
        'source_hashes': dict(candidate.source_hashes),
    }
    return record


def identify_saved_stack(saved, repository, capture, step):
    """Solve SceneID after reconstruction, using the recorded planning state."""
    import stacking_core as core
    from stacking_common.artifacts import StackRunStore
    from stacking_common.reconstruction import load_point_cloud
    from stacking_tabletop.planning.scene_identification import (
        prepare_saved_stack_candidate,
    )

    status = json.loads((Path(capture)/'capture.json').read_text())
    points = load_point_cloud(status['pcd']).points
    placed_count = len(StackRunStore.load_planning_checkpoint(
        saved.path, step
    ).state.placed_object_indices)
    config = core.SceneIdConfig()
    config.scan_sigma = .005
    config.outlier_cutoff = .025
    config.reference_scan_points = max(1., len(points)/placed_count)
    capabilities = core.cuda_capabilities() if hasattr(core, 'cuda_capabilities') else {}
    cuda_scan = bool(capabilities.get('sceneid_scan') and capabilities.get('devices'))
    started = time.monotonic()
    candidate = prepare_saved_stack_candidate(
        saved, repository, capture, config, step,
        scan_voxel_size=.001, mesh_voxel_size=.001, cuda_scan=cuda_scan,
    )
    record = _sceneid_record(
        candidate, backend='cuda' if cuda_scan else 'cpu',
        elapsed=time.monotonic()-started,
    )
    return record


def process_stack_scene(run, repository, capture, step, *, reconstruct=None,
                        identify=identify_saved_stack):
    """Reconstruct the captured views, then solve dependent SceneID."""
    from stacking_tabletop.execution.saved_plan import load_execution_run

    if reconstruct is None:
        from .hand_eye_fast_stereo import reconstruct_with_runtime
        reconstruct = reconstruct_with_runtime
    run, repository, capture = Path(run), Path(repository).resolve(), Path(capture).resolve()
    report_path = capture/'processing.json'
    report = {
        'kind': 'stacked_scene_processing', 'status': 'running',
        'step': int(step), 'started_ns': time.time_ns(),
    }
    atomic_json(report_path, report)
    try:
        saved = load_execution_run(run, repository)
        status_path = capture/'capture.json'
        status = json.loads(status_path.read_text())
        if (status.get('kind') != 'stacked_scene_wrist_capture'
                or status.get('step') != int(step)
                or status.get('context_sha256') != saved.context.sha256):
            raise ValueError('capture does not match the saved execution step')
        vendor = repository/'.data/vendor/Fast-FoundationStereo'
        checkpoint = vendor/'weights/23-36-37/model_best_bp2_serialize.pth'
        status['stage'] = 'reconstruction'
        atomic_json(status_path, status)
        reconstruct(
            capture/'session', None, None, capture/'rgb_dense',
            repository=vendor, checkpoint=checkpoint,
            region_file=capture/'region.json',
        )
        pcd = capture/'rgb_dense/refined/rgb_dense_voxel_1mm.pcd'
        if not pcd.is_file():
            raise ValueError('stack reconstruction produced no PCD')
        status.update(
            status='complete', stage='scene_identification', pcd=str(pcd),
            pcd_sha256=digest(pcd),
            viewer=str(capture/'rgb_dense/rgb_only/overview.html'),
        )
        atomic_json(status_path, status)
        sceneid = identify(saved, repository, capture, int(step))
        sceneid_path = capture/'sceneid_result.json'
        atomic_json(sceneid_path, sceneid)
        status.update(
            stage='complete', scene_identification={
                'path': str(sceneid_path), 'sha256': digest(sceneid_path),
                'status': sceneid['status'],
                'accepted_snapshot': sceneid['accepted_snapshot'],
            },
        )
        atomic_json(status_path, status)
        report.update(
            status='complete', completed_ns=time.time_ns(),
            pcd=str(pcd), scene_identification=str(sceneid_path),
        )
    except BaseException as error:
        report.update(status='failed', completed_ns=time.time_ns(),
                      error=f'{type(error).__name__}: {error}')
        if (capture/'capture.json').is_file():
            status = json.loads((capture/'capture.json').read_text())
            status.update(status='failed', error=report['error'])
            atomic_json(capture/'capture.json', status)
        raise
    finally:
        atomic_json(report_path, report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--repository', type=Path, required=True)
    parser.add_argument('--capture', type=Path, required=True)
    parser.add_argument('--step', type=int, required=True)
    # Keep the 100 Hz robot-streaming process favored while perception shares
    # the return-home interval. Child reconstruction inherits this niceness.
    try:
        os.nice(10)
    except OSError:
        pass
    process_stack_scene(**vars(parser.parse_args()))


if __name__ == '__main__':
    main()
