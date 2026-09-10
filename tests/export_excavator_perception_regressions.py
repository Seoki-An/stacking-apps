"""Export portable recorded perception evidence; no legacy runtime imports."""
import argparse
import json
from pathlib import Path
import numpy as np
import open3d as o3d
from scipy.spatial.transform import Rotation
from stacking_excavator.acquisition.storage import file_hash


def export(legacy, output):
    output.mkdir(parents=True, exist_ok=True)
    field = legacy / 'logs/260707/plan_260707_7/exec_232318_1_nuc/step_44/field_scan'
    initial = np.eye(4)
    initial[:3, 3] = np.load(field / 'field_pose_init_pos.npy', allow_pickle=False)
    initial[:3, :3] = Rotation.from_quat(np.load(field / 'field_pose_init_rot.npy', allow_pickle=False)).as_matrix()
    def points(path):
        return np.asarray(o3d.io.read_point_cloud(str(path)).points)
    def save(name, arrays, sources, **metadata):
        path = output / (name + '.npz')
        np.savez_compressed(path, **arrays)
        path.with_suffix('.json').write_text(json.dumps({**metadata,
            'purpose': 'Recorded software regression; no independent physical pose ground truth',
            'timestamp_source': 'recorded_unknown',
            'sources': {str(p.relative_to(legacy)): file_hash(p) for p in sources},
            'sha256': file_hash(path)}, indent=2) + '\n')
    save('field_recovery', dict(points=points(field / 'field_scene_cropped.ply'), initial=initial,
        expected=np.load(field / 'field_pose_opt_T.npy', allow_pickle=False)),
        [field / n for n in ('field_scene_cropped.ply', 'field_pose_init_pos.npy',
                            'field_pose_init_rot.npy', 'field_pose_opt_T.npy')], object_id=17)
    arrays, sources = {}, []
    sessions = ['logs/260623/plan_260623_1/exec_121628_1/step_17',
                'logs/260623/plan_260623_1/exec_120500_1/step_17']
    for s, relative in enumerate(sessions):
        for index in (0, 4, 8):
            path = legacy / relative / f'pcd_{index}.ply'
            arrays[f'inhand_{s}_{index}'] = points(path)
            sources.append(path)
    scenes = ['logs/260706/plan_260706_4/exec_164224_1_nuc/step_39/scene_scan',
              'logs/260623/plan_260623_1/exec_121628_1/step_17/scene_scan']
    for s, relative in enumerate(scenes):
        path = legacy / relative / 'scene_base_iv_points21_swing0.ply'
        # Preserve meaningful target-region evidence, reduce fixture size.
        p = points(path)
        p = p[(p[:, 0] < -3) & (p[:, 0] > -9) & (p[:, 1] > 0) & (p[:, 1] < 6)]
        arrays[f'scene_{s}'] = p
        sources.append(path)
    save('perception_views', arrays, sources)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('legacy', type=Path)
    parser.add_argument('--output', type=Path, default=Path('tests/fixtures/legacy_planner'))
    args = parser.parse_args()
    export(args.legacy, args.output)
