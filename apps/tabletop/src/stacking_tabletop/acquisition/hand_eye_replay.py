"""Portable wrist sessions and calibrated-only offline depth replay."""

import json
from pathlib import Path
import shutil

import numpy as np

from stacking_common.pointcloud import validate_rigid_transform
from stacking_common.pointcloud.depth import optical_rays, backproject_depth
from .hand_eye import HandEyeCalibration
from .hand_eye_capture import atomic_json, digest, inspect_capture
from .storage import write_xyz_pcd


def read_json(path):
    return json.loads(Path(path).read_text())


def inside(root, relative):
    path = (Path(root) / relative).resolve()
    if not path.is_relative_to(Path(root).resolve()):
        raise ValueError('session path escapes its directory')
    return path


def file_hashes(root):
    root = Path(root)
    result = {}
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('portable inputs cannot contain symbolic links')
        if path.is_file():
            result[str(path.relative_to(root))] = digest(path)
    return result


def copy_capture(source, destination):
    expected = file_hashes(source)
    shutil.copytree(source, destination)
    if file_hashes(source) != expected or file_hashes(destination) != expected:
        raise ValueError('input changed during assembly')


def view_interval(root):
    manifest = read_json(root / 'manifest.json')
    stamps = [read_json(root / (stem + '.json'))[name]['timestamp_ns']
              for stem in manifest['frames'] for name in manifest.get('streams', ('color', 'depth'))]
    if not stamps:
        raise ValueError('wrist view contains no recorded exposures')
    return [min(stamps), max(stamps)]


def native_depth_rays(geometry):
    model = geometry['distortion_model'].removeprefix('distortion.')
    coefficients = np.asarray(geometry['distortion'], dtype=float)
    # Zero coefficients reduce these known RealSense models to pinhole rays.
    if model in ('inverse_brown_conrady', 'modified_brown_conrady') and not np.any(coefficients):
        model = 'none'
    return optical_rays(geometry['width'], geometry['height'], geometry['K'], coefficients, model=model)


def view_identity(root):
    inspect_capture(root)
    manifest, device = read_json(root / 'manifest.json'), read_json(root / 'device.json')
    if manifest.get('output_frame') != 'base_link':
        raise ValueError('wrist replay requires recorded base_link output')
    origin = manifest.get('data_origin', 'hardware')
    if origin not in ('synthetic', 'hardware') or device.get('data_origin', 'hardware') != origin:
        raise ValueError('inconsistent wrist data origin')
    if 'depth' in device['streams']:
        native_depth_rays(device['streams']['depth'])
    calibration = HandEyeCalibration.load(root / 'calibration/profile.yml')
    return dict(origin=origin, serial=device['device']['serial_number'],
                calibration={name: getattr(calibration.profile, name).sha256 for name in calibration.artifacts},
                streams=device['streams'], depth_metres_per_unit=device.get('depth_metres_per_unit'),
                color_T_depth=device.get('color_T_depth'))


def femto_reference(root):
    """Accept completed fixed-field FEMTO captures, never in-hand/tool clouds."""
    metadata = read_json(root / 'field_scan.json')
    if (read_json(root / 'capture_status.json')['status'] != 'complete'
            or metadata.get('status') != 'complete' or metadata.get('output_frame') != 'base_link'
            or metadata.get('fused_source') != 'organized_depth_burst'
            or metadata.get('fused_depth_corrected') is not True
            or len(metadata.get('shots', [])) != 7):
        raise ValueError('FEMTO reference must be a complete native base_link field capture')
    for kind in ('camera', 'depth'):
        path = inside(root, metadata[kind + '_calibration_file'])
        if digest(path) != metadata[kind + '_calibration_sha256']:
            raise ValueError('FEMTO reference calibration hash mismatch')
    if not inside(root, metadata['fused_file']).is_file():
        raise ValueError('missing FEMTO reference cloud')
    organized = read_json(root / 'organized_depth_manifest.json')
    info = read_json(inside(root, organized['camera_info']))
    if (organized.get('values_are_raw') is not True or len(organized['views']) != 1
            or len(organized['views'][0]['depth_frames']) != 7
            or organized['depth_mode'] != metadata['depth_mode']
            or info['frame_id'] != metadata['cloud_frame']):
        raise ValueError('inconsistent FEMTO organized-depth evidence')
    for shot in metadata['shots']:
        validate_rigid_transform(shot['base_T_cloud'])
        if (not inside(root, shot['organized_depth_filename']).is_file()
                or not inside(root, 'raw_pcd/' + shot['filename']).is_file()):
            raise ValueError('missing FEMTO native depth evidence')
        depth = np.load(inside(root, shot['organized_depth_filename']), allow_pickle=False)
        if depth.shape != (info['height'], info['width']) or depth.dtype.kind not in 'uf':
            raise ValueError('invalid FEMTO native depth shape/type')
    return metadata


def assemble_session(directory, views, *, femto_before=None, femto_after=None, reference=None):
    """Copy verified inputs into a new immutable-input session with relative paths."""
    directory = Path(directory).expanduser().resolve()
    views = [Path(path).expanduser().resolve() for path in views]
    if not views or len(set(views)) != len(views):
        raise ValueError('a session needs distinct wrist view directories')
    if (femto_before is None) != (femto_after is None):
        raise ValueError('supply both FEMTO before and after captures, or neither')
    identity = view_identity(views[0])
    previous_end = 0
    intervals = []
    for view in views:
        if view_identity(view) != identity:
            raise ValueError('wrist views have incompatible calibration, geometry or data origin')
        interval = view_interval(view)
        if interval[0] <= previous_end:
            raise ValueError('wrist views must be ordered and must not overlap in source time')
        previous_end = interval[1]
        intervals.append(interval)
    references = {}
    for name, path in (('femto_before', femto_before), ('femto_after', femto_after)):
        if path is not None:
            root = Path(path).expanduser().resolve()
            metadata = femto_reference(root)
            if metadata.get('data_origin', 'hardware') != identity['origin']:
                raise ValueError('cannot mix synthetic and hardware sensor evidence')
            stamps = [shot['stamp_ns'] for shot in metadata['shots']]
            if not stamps or (name == 'femto_before' and max(stamps) >= intervals[0][0]
                              or name == 'femto_after' and min(stamps) <= intervals[-1][1]):
                raise ValueError('FEMTO reference does not bracket wrist acquisition times')
            references[name] = root
    if references:
        before, after = [femto_reference(references[name]) for name in ('femto_before', 'femto_after')]
        if any(before[key] != after[key] for key in ('camera_serial', 'camera_calibration_sha256',
                                                    'depth_calibration_sha256', 'depth_mode')):
            raise ValueError('FEMTO before/after references have incompatible identity/calibration')
    inputs = views + list(references.values())
    if any(directory.is_relative_to(path) or path.is_relative_to(directory) for path in inputs):
        raise ValueError('session output and capture inputs must be separate directories')
    directory.mkdir(parents=True, exist_ok=False)
    manifest = dict(schema_version=1, kind='hand_eye_session', status='assembling',
        data_origin=identity['origin'], output_frame='base_link', identity=identity,
        views=[], references=list(references), files={}, accuracy_1mm_verified=False)
    atomic_json(directory / 'session.json', manifest)
    try:
        for index, (view, interval) in enumerate(zip(views, intervals)):
            relative = f'views/{index:03d}'
            copy_capture(view, directory / relative)
            if view_identity(directory / relative) != identity or view_interval(directory / relative) != interval:
                raise ValueError('wrist input changed during assembly')
            manifest['views'].append(dict(path=relative, source_interval_ns=interval))
        for name, source in references.items():
            copy_capture(source, directory / name)
        if reference is not None:
            # A reference is evidence supplied by the experiment, never inferred from ICP.
            atomic_json(directory / 'reference.json', read_json(reference))
        manifest['files'] = {name: value for name, value in file_hashes(directory).items() if name != 'session.json'}
        manifest['status'] = 'complete'
        atomic_json(directory / 'session.json', manifest)
    except BaseException as error:
        manifest.update(status='incomplete', error=f'{type(error).__name__}: {error}')
        atomic_json(directory / 'session.json', manifest)
        raise
    return manifest


def inspect_session(directory):
    directory = Path(directory).resolve()
    manifest = read_json(directory / 'session.json')
    if (manifest.get('schema_version') != 1 or manifest.get('kind') != 'hand_eye_session'
            or manifest.get('status') != 'complete' or manifest.get('output_frame') != 'base_link'
            or not manifest['views'] or manifest['data_origin'] != manifest['identity']['origin']
            or manifest['references'] not in ([], ['femto_before', 'femto_after'])):
        raise ValueError('unsupported or incomplete wrist session')
    actual = {name: value for name, value in file_hashes(directory).items() if name != 'session.json'}
    if actual != manifest['files']:
        raise ValueError('session input files/hash mismatch')
    previous_end = 0
    for index, view in enumerate(manifest['views']):
        interval = view_interval(inside(directory, view['path']))
        if (view['path'] != f'views/{index:03d}' or interval != view['source_interval_ns']
                or interval[0] <= previous_end):
            raise ValueError('session view order/time mismatch')
        previous_end = interval[1]
        if view_identity(inside(directory, view['path'])) != manifest['identity']:
            raise ValueError('session view identity mismatch')
    for name in manifest['references']:
        femto_reference(inside(directory, name))
    return manifest


def replay_session(session, output, *, visualize=True):
    """Export each native exposure and an aggregate with exact pixel provenance."""
    session, output = Path(session).resolve(), Path(output).resolve()
    manifest = inspect_session(session)
    if 'depth' not in manifest['identity']['streams']:
        raise ValueError('RGB-only session has no native depth to replay; use RGB stereo reconstruction')
    if output.is_relative_to(session) or session.is_relative_to(output):
        raise ValueError('replay output must be separate from immutable session inputs')
    output.mkdir(parents=True, exist_ok=False)
    report = dict(schema_version=1, status='running', data_origin=manifest['data_origin'],
        output_frame='base_link', session_sha256=digest(session / 'session.json'),
        input_hashes=manifest['files'], processing='native axial depth; calibrated transforms; concatenate',
        distortion_implementation_sha256=digest(Path(optical_rays.__code__.co_filename)),
        replay_source_sha256=digest(__file__), numpy_version=np.__version__,
        frames=[], references={}, accuracy_1mm_verified=False)
    atomic_json(output / 'replay.json', report)
    layers, poses, all_points, all_pixels, all_frames = {}, {}, [], [], []
    try:
        for view_index, view in enumerate(manifest['views']):
            root = inside(session, view['path'])
            device = read_json(root / 'device.json')
            rays = native_depth_rays(device['streams']['depth'])
            view_points = []
            for stem in read_json(root / 'manifest.json')['frames']:
                record = read_json(root / (stem + '.json'))
                with np.load(root / (stem + '.npz'), allow_pickle=False) as data:
                    cloud, pixels = backproject_depth(data['depth'], rays, device['depth_metres_per_unit'])
                points = cloud.transformed(record['base_link_T_depth']).points
                frame_index = len(report['frames'])
                filename = f'frame_{frame_index:04d}.pcd'
                write_xyz_pcd(output / filename, points)
                sidecar = dict(view_index=view_index, source=view['path'] + '/' + stem,
                    timestamp_ns=record['depth']['timestamp_ns'], point_count=len(points),
                    base_link_T_depth=record['base_link_T_depth'], output_frame='base_link',
                    valid_pixel_fraction=len(points)/rays.shape[0]/rays.shape[1],
                    pixels='aggregate.npz: pixel_index where frame_index matches this frame', frame_index=frame_index)
                atomic_json(output / filename.replace('.pcd', '.json'), sidecar)
                report['frames'].append(dict(filename=filename, **sidecar))
                view_points.append(points)
                all_points.append(points)
                all_pixels.append(pixels)
                all_frames.append(np.full(len(points), frame_index, dtype=np.int32))
                poses.setdefault(f'view {view_index+1}', record['base_link_T_depth'])
            layers[f'view {view_index+1}'] = np.concatenate(view_points)
        points = np.concatenate(all_points)
        if not len(points):
            raise ValueError('session has no valid depth pixels')
        np.savez_compressed(output / 'aggregate.npz', points=points,
                            pixel_index=np.concatenate(all_pixels), frame_index=np.concatenate(all_frames))
        write_xyz_pcd(output / 'aggregate.pcd', points)
        for name in manifest['references']:
            from stacking_common.reconstruction.io import load_point_cloud
            metadata = femto_reference(session / name)
            cloud = load_point_cloud(inside(session / name, metadata['fused_file']))
            write_xyz_pcd(output / (name + '.pcd'), cloud.points)
            layers[name.replace('_', ' ')] = cloud.points
            report['references'][name] = dict(point_count=len(cloud), output_frame='base_link',
                processing='existing FEMTO calibrated/cropped/filtered field cloud; no wrist registration')
        if visualize:
            from stacking_common.rendering.pointcloud import render_clouds
            title = ('SYNTHETIC · ' if manifest['data_origin'] == 'synthetic' else '') + 'Wrist scene observations'
            render_clouds(output, layers, poses, title=title)
        report.update(status='complete', point_count=len(points),
                      output_hashes={name: value for name, value in file_hashes(output).items() if name != 'replay.json'})
        atomic_json(output / 'replay.json', report)
    except BaseException as error:
        report.update(status='incomplete', error=f'{type(error).__name__}: {error}')
        atomic_json(output / 'replay.json', report)
        raise
    return report
