"""Overlapping translated and oblique wrist views in physical camera/tool frames."""

from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np

from stacking_tabletop.execution.motion import UR_JOINT_NAMES, validate_joint_path, validate_pose
from .calibration import BASE_LINK_T_UR_BASE


def kinematic_urdf(description):
    """Snapshot the driver's actual kinematics without external mesh dependencies."""
    root = ET.fromstring(description)
    if root.tag != 'robot':
        raise ValueError('robot_description must contain a URDF robot')
    active = [joint for joint in root.findall('joint') if joint.get('type') != 'fixed']
    if len(active) != 6 or {joint.get('name') for joint in active} != set(UR_JOINT_NAMES):
        raise ValueError('robot_description must have the canonical six UR joints')
    for link in root.findall('link'):
        for child in list(link):
            link.remove(child)
    # Native FK needs only links and joints; remove ros2_control/transmissions.
    for child in list(root):
        if child.tag not in ('link', 'joint'):
            root.remove(child)
    for joint in active:
        root.remove(joint)
    by_name = {joint.get('name'): joint for joint in active}
    for name in UR_JOINT_NAMES:
        root.append(by_name[name])
    return ET.tostring(root, encoding='unicode') + '\n'


def camera_pose(kinematics, calibration, joints):
    # UR5eKinematics.pose always returns base_T_tool, irrespective of capture TF.
    return BASE_LINK_T_UR_BASE @ kinematics.pose(joints) @ calibration.tool_T_color


SIDE_ANGLE_DEG = 30.
EXTENDED_SIDE_ANGLES = {'side_left_45': 45., 'side_right_45': 45.,
                        'side_left_60': 60., 'side_right_60': 60.}


def _view_specs(pattern, offset, oblique_offset):
    if pattern == 'legacy5':
        return [('center', (0., 0.), 'overview'), ('left', (-offset, 0.), 'near'),
                ('right', (offset, 0.), 'near'), ('up', (0., -offset), 'near'),
                ('down', (0., offset), 'near')]
    if pattern not in ('coverage9', 'coverage13', 'coverage17'):
        raise ValueError('unsupported wrist viewpoint pattern')
    diagonal = oblique_offset / np.sqrt(2)
    views = [('center', (0., 0.), 'overview'), ('left', (-offset, 0.), 'near'),
             ('right', (offset, 0.), 'near'), ('up', (0., -offset), 'near'),
             ('down', (0., offset), 'near'),
             ('oblique_left_up', (-diagonal, -diagonal), 'oblique'),
             ('oblique_right_up', (diagonal, -diagonal), 'oblique'),
             ('oblique_right_down', (diagonal, diagonal), 'oblique'),
             ('oblique_left_down', (-diagonal, diagonal), 'oblique')]
    if pattern in ('coverage13', 'coverage17'):
        views += [('side_left', (-1., 0.), 'side'), ('side_right', (1., 0.), 'side'),
                  ('side_up', (0., -1.), 'side'), ('side_down', (0., 1.), 'side')]
    if pattern == 'coverage17':
        views += [(label, (-1., 0.) if 'left' in label else (1., 0.), 'side')
                  for label in EXTENDED_SIDE_ANGLES]
    return views


def build_views(kinematics, calibration, seed, *, distance, offset=.04,
                oblique_offset=.08, pattern='coverage9', check=lambda: None, labels=None):
    """Overlapping views about one seed, with explicit reverse-path returns.

    IK and joint/cable limits are checked; this does not check collisions.
    The target lies distance metres along the seed camera's optical +Z axis.
    ``offset`` is the radius of the four near views. ``oblique_offset`` is the
    larger radial translation of four diagonal views that look at the same target.
    """
    if not np.isfinite(distance) or not .1 <= distance <= 1.:
        raise ValueError('target distance must be between 0.1 and 1 metre')
    if not np.isfinite(offset) or not 0 < offset <= min(.12, distance / 2):
        raise ValueError('offset must be positive and at most 0.12 m and half the target distance')
    if pattern in ('coverage9', 'coverage13', 'coverage17') and (not np.isfinite(oblique_offset)
            or not offset < oblique_offset <= min(.12, distance / 2)):
        raise ValueError('oblique offset must exceed the near offset and be at most 0.12 m and half the target distance')
    seed = validate_joint_path([seed])[0]
    reference = camera_pose(kinematics, calibration, seed)
    target = reference[:3, 3] + distance * reference[:3, 2]
    views = []
    specs = _view_specs(pattern, offset, oblique_offset)
    if labels is not None:
        if not labels or not set(labels).issubset({item[0] for item in specs}):
            raise ValueError('unknown or empty view subset')
        specs = [item for item in specs if item[0] in labels]
    for label, xy, role in specs:
        q = seed.copy()
        path = [q.copy()]
        side = role == 'side'
        angle_deg = EXTENDED_SIDE_ANGLES.get(label, SIDE_ANGLE_DEG) if side else 0.
        angle = np.deg2rad(angle_deg)
        steps = max(1, int(np.ceil((distance * angle if side else np.linalg.norm(xy)) / .005)))
        for fraction in np.linspace(0, 1, steps + 1)[1:]:
            check()
            pose = reference.copy()
            if side:
                theta = fraction * angle
                pose[:3, 3] = (target - distance * np.cos(theta) * reference[:3, 2]
                               + distance * np.sin(theta) * (reference[:3, :2] @ xy))
            else:
                pose[:3, 3] += reference[:3, :2] @ (fraction * np.asarray(xy))
            z = target - pose[:3, 3]
            z /= np.linalg.norm(z)
            x = reference[:3, 0] - z * (z @ reference[:3, 0])
            x /= np.linalg.norm(x)
            pose[:3, :3] = np.column_stack((x, np.cross(z, x), z))
            if label != 'center':
                base_T_tool = BASE_LINK_T_UR_BASE @ pose @ np.linalg.inv(calibration.tool_T_color)
                next_q = kinematics.solve(base_T_tool, q)
                if np.max(np.abs(next_q - q)) > np.deg2rad(10):
                    raise ValueError(f'{label}: IK changed joint branch or exceeded the local step limit')
                q = next_q
                path.append(q.copy())
        path = validate_joint_path(path)
        if np.any(path < kinematics.lower) or np.any(path > kinematics.upper):
            raise ValueError(f'{label}: path exceeds robot joint limits')
        reached = camera_pose(kinematics, calibration, q)
        if not np.allclose(reached, pose, atol=2e-4, rtol=0):
            raise ValueError(f'{label}: camera target FK does not match IK')
        radial_offset = distance * np.sin(angle) if side else float(np.linalg.norm(xy))
        offset_xy = (radial_offset * np.asarray(xy)).tolist() if side else list(xy)
        views.append(dict(label=label, role=role, camera_offset_xy_m=offset_xy,
                          view_angle_deg=float(angle_deg if side else np.rad2deg(np.arctan2(radial_offset, distance))),
                          path=path.tolist(), base_link_T_color=reached.tolist()))
    return dict(schema_version={'coverage17': 4, 'coverage13': 3}.get(pattern, 2),
                kind='tabletop_wrist_scene_plan' if labels is None else 'tabletop_wrist_scene_subset_plan', output_frame='base_link',
                seed_joints=seed.tolist(), target=target.tolist(), distance_m=float(distance),
                viewpoint_pattern=pattern, offset_m=float(offset),
                oblique_offset_m=float(oblique_offset),
                side_angle_deg=SIDE_ANGLE_DEG if pattern in ('coverage13', 'coverage17') else None,
                views=views, collision_checked=False,
                return_policy='reverse_each_leg_to_seed', held_object=False)


def validate_views(plan, kinematics, calibration):
    """Validate the complete saved route before any hardware command."""
    if (plan.get('schema_version') not in (1, 2, 3, 4) or plan.get('kind') != 'tabletop_wrist_scene_plan'
            or plan.get('output_frame') != 'base_link' or plan.get('held_object') is not False
            or plan.get('return_policy') != 'reverse_each_leg_to_seed'):
        raise ValueError('unsupported scene scan plan')
    seed = validate_joint_path([plan['seed_joints']])[0]
    legacy = plan['schema_version'] == 1
    pattern = 'legacy5' if legacy else plan.get('viewpoint_pattern')
    if {'legacy5': 1, 'coverage9': 2, 'coverage13': 3, 'coverage17': 4}.get(pattern) != plan['schema_version']:
        raise ValueError('scene scan schema and viewpoint pattern disagree')
    expected_count = {'legacy5': 5, 'coverage9': 9, 'coverage13': 13, 'coverage17': 17}[pattern]
    if len(plan['views']) != expected_count:
        raise ValueError(f'scene scan requires the {expected_count} generated viewpoints')
    expected = build_views(kinematics, calibration, seed, distance=plan['distance_m'],
                           offset=plan['offset_m'],
                           oblique_offset=plan.get('oblique_offset_m', .08), pattern=pattern)
    if pattern in ('coverage13', 'coverage17') and plan.get('side_angle_deg') != SIDE_ANGLE_DEG:
        raise ValueError('saved side angle differs from the generated route')
    if not np.allclose(plan['target'], expected['target'], atol=1e-9, rtol=0):
        raise ValueError('saved target disagrees with the seed view')
    for view, original in zip(plan['views'], expected['views']):
        path = validate_joint_path(view['path'])
        target = validate_pose(view['base_link_T_color'])
        metadata_differs = (not legacy and
            (view.get('role') != original['role']
             or not np.allclose(view.get('camera_offset_xy_m'), original['camera_offset_xy_m'], atol=1e-9, rtol=0)
             or not np.isclose(view.get('view_angle_deg'), original['view_angle_deg'], atol=1e-9, rtol=0)))
        if (view['label'] != original['label'] or metadata_differs or path.shape != np.shape(original['path'])
                or not np.allclose(path, original['path'], atol=1e-6, rtol=0)
                or not np.allclose(target, original['base_link_T_color'], atol=1e-6, rtol=0)):
            raise ValueError('saved path differs from the generated, calibrated route')


def render_preview(plan, kinematics, calibration, output):
    """Offline Plotly preview of every leg, camera axis and robot link skeleton."""
    import plotly.graph_objects as go
    figure = go.Figure()
    def line(points, name, **kwargs):
        points = np.asarray(points)
        figure.add_trace(go.Scatter3d(x=points[:, 0], y=points[:, 1], z=points[:, 2],
                                     mode='lines+markers', name=name, **kwargs))
    names = ('base_link', 'shoulder_link', 'upper_arm_link', 'forearm_link',
             'wrist_1_link', 'wrist_2_link', 'wrist_3_link', 'tool0')
    role_colors = {'overview': '#24334a', 'near': '#18a999', 'oblique': '#ef8354', 'side': '#7953c6'}
    for view in plan['views']:
        path = np.asarray(view['path'])
        centers = [camera_pose(kinematics, calibration, q)[:3, 3] for q in path]
        color = role_colors.get(view.get('role', 'near'), '#8797a5')
        line(centers, view['label'] + ' camera path', line=dict(color=color))
        if view.get('return_path'):
            line([camera_pose(kinematics, calibration, q)[:3, 3] for q in view['return_path']],
                 view['label'] + ' return', opacity=.4, line=dict(color=color, dash='dot'))
        line([centers[-1], plan['target']], view['label'] + ' optical axis',
             opacity=.35, line=dict(color=color))
        points = []
        for name in names:
            link = kinematics.model.kinematics.find_link(name)
            if link is not None:
                points.append((BASE_LINK_T_UR_BASE @ kinematics._pose(path[-1], link))[:3, 3])
        if points:
            line(points, view['label'] + ' robot links', opacity=.4)
    for key, label in (('entry_path', 'Entry from execution retreat'), ('return_path', 'Return to execution retreat')):
        if key in plan:
            line([camera_pose(kinematics, calibration, q)[:3, 3] for q in plan[key]], label,
                 line=dict(color='#db5151', dash='dash'))
    target = np.asarray(plan['target'])
    figure.add_trace(go.Scatter3d(x=[target[0]], y=[target[1]], z=[target[2]],
                                 mode='markers', name='shared look-at target',
                                 marker=dict(size=7, color='#db5151', symbol='diamond')))
    figure.update_layout(title=f'Wrist scan preview — {len(plan["views"])} views; link skeletons; not collision checked',
                         scene=dict(aspectmode='data', xaxis_title='base_link X (m)',
                                    yaxis_title='Y (m)', zaxis_title='Z (m)'),
                         annotations=[dict(text='Gray: overview · teal: near · orange: oblique · purple: side · review paths and return legs; verify clearance',
                                           x=0, y=-.08, xref='paper', yref='paper', showarrow=False)])
    figure.write_html(str(Path(output)), include_plotlyjs=True)
