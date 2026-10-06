"""Manual Open3D inspection of saved RGB reconstructions; no camera/robot access.

From the repository root:
    python apps/tabletop/scripts/view_rgb_scene.py
    python apps/tabletop/scripts/view_rgb_scene.py --stones-only
    python apps/tabletop/scripts/view_rgb_scene.py --scene stack
    python apps/tabletop/scripts/view_rgb_scene.py /path/to/colored_cloud.ply

Drag to rotate, scroll to zoom, Shift+drag to pan; Q closes the viewer.
Use --check-only to validate a cloud without opening a desktop window.
"""
import argparse
from pathlib import Path

import numpy as np
import open3d as o3d


REVIEW = (Path(__file__).resolve().parents[1] / 'data/experiments/hand_eye_rgb_review_001')
SCENES = {
    'real': 'pick_scene_004_fast_real_001',
    'stack': 'stacked_scene_sim_003',
}


def load_rgb_cloud(path):
    if not path.is_file():
        raise ValueError(f'Point cloud not found: {path}. Supply an existing colored PCD or PLY path.')
    cloud = o3d.io.read_point_cloud(str(path))
    if cloud.is_empty():
        raise ValueError(f'Point cloud is empty or unreadable: {path}')
    if not cloud.has_colors():
        raise ValueError('This cloud has no RGB attributes. Select scene_rgb.pcd or stones_rgb.ply '
                         'from the RGB review exports, rather than the original XYZ-only reconstruction.')
    if not np.isfinite(np.asarray(cloud.points)).all() or not np.isfinite(np.asarray(cloud.colors)).all():
        raise ValueError('Point coordinates and RGB values must be finite.')
    return cloud


def show(cloud, title):
    viewer = o3d.visualization.Visualizer()
    try:
        if not viewer.create_window(window_name=title, width=1280, height=800):
            raise RuntimeError('Open3D could not create a window. Run in a graphical desktop session; '
                               'use --check-only when running without a display.')
        if not viewer.add_geometry(cloud):
            raise RuntimeError('Open3D could not add the point cloud.')
        options = viewer.get_render_option()
        options.background_color = np.array([.08, .09, .11])
        options.point_size = 2.5
        options.point_color_option = o3d.visualization.PointColorOption.Color
        view = viewer.get_view_control()
        view.set_lookat(cloud.get_center())
        view.set_front([0., -1., 1.])
        view.set_up([0., 0., 1.])
        view.set_zoom(.7)
        print('Drag: rotate | Wheel: zoom | Shift+drag: pan | Q: close', flush=True)
        viewer.run()
    finally:
        viewer.destroy_window()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('path', nargs='?', type=Path, help='explicit colored PCD/PLY instead of a saved-scene preset')
    parser.add_argument('--scene', choices=SCENES, default=None, help='saved scene: real (default) or synthetic stack')
    parser.add_argument('--stones-only', action='store_true', help='open the preset stone-only export, hiding the table')
    parser.add_argument('--check-only', action='store_true', help='load and validate RGB data without a window')
    args = parser.parse_args(argv)
    if args.path is not None and (args.scene is not None or args.stones_only):
        parser.error('use either an explicit path or the --scene/--stones-only preset options')
    scene = args.scene or 'real'
    path = (args.path if args.path is not None else REVIEW/SCENES[scene]/
            ('stones_rgb.ply' if args.stones_only else 'scene_rgb.ply')).expanduser().resolve()
    try:
        cloud = load_rgb_cloud(path)
        print(f'{path}\n{len(cloud.points):,} points with RGB; coordinates preserved.', flush=True)
        if not args.check_only:
            title = path.name if args.path else ('Real wrist capture' if scene == 'real' else 'Synthetic stacked scene')
            show(cloud, title + ' — RGB point cloud')
    except (ValueError, RuntimeError) as error:
        parser.exit(1, f'RGB viewer: {error}\n')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
