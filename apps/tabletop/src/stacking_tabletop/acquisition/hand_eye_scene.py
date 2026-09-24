"""Portable scene assembly of FEMTO reference and labelled wrist RGB evidence."""
from pathlib import Path
import inspect
import shutil
import numpy as np

from stacking_common.multiview.scene import RGB_LAYERS, rgb_layer_indices
from stacking_common.rendering.scene_layers import render_scene_layers
from .hand_eye_capture import atomic_json, digest
from .hand_eye_replay import inspect_session, inside, read_json, file_hashes
from .storage import write_xyz_pcd


STYLE = {
    'femto_before': ('FEMTO reference', '#8c97a5', True),
    'femto_after': ('FEMTO after capture', '#548cfa', False),
    'rgb_background': (RGB_LAYERS['rgb_background'], '#88bbaa', False),
    'rgb_overlap': (RGB_LAYERS['rgb_overlap'], '#18a999', True),
    'rgb_unresolved': (RGB_LAYERS['rgb_unresolved'], '#e9a328', True),
    'rgb_extensions': (RGB_LAYERS['rgb_extensions'], '#9a65cc', True),
    'rgb_rejected': (RGB_LAYERS['rgb_rejected'], '#db5252', False),
}


def load_npz(path):
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def region_groups(points, region):
    labels = np.full(len(points), -1, np.int16)
    plane = np.asarray(region['support_plane'])
    upper = points[:, 2] > points[:, :2] @ plane[:2] + plane[2] + .008
    for group, (low, high) in enumerate(region['stone_boxes_xy']):
        labels[upper & np.all((points[:, :2] >= low) & (points[:, :2] <= high), axis=1)] = group
    return labels


def validate_rgb(data, summary):
    indices = rgb_layer_indices(data)
    if (len(data['points']) != summary['input_points']
            or int(data['keep'].sum()) != summary['point_count']
            or int(np.count_nonzero(data['coverage_class'])) != summary['retained_stone_points']):
        raise ValueError('RGB provenance counts differ from filter summary')
    if not np.array_equal(region_groups(data['points'], summary['region']), data['stone']):
        raise ValueError('RGB region labels differ from filter region')
    valid_views = set()
    for view in summary['view_sources']:
        index = view['view']
        if index in valid_views:
            raise ValueError('duplicate source view')
        valid_views.add(index)
        pixels = data['observation_pixels'][data['observation_views'] == index]
        if np.any(pixels >= np.prod(view['image_shape'])):
            raise ValueError('RGB source pixel outside recorded image')
    if not set(np.unique(data['observation_views'])).issubset(valid_views):
        raise ValueError('unknown RGB source view')
    for group in summary['groups']:
        selected = data['stone'] == group['stone']
        for label, key in [(1, 'reference_overlap_points'), (2, 'unresolved_surface_difference_points'), (3, 'candidate_extension_points')]:
            if np.count_nonzero(selected & (data['coverage_class'] == label)) != group[key]:
                raise ValueError('RGB coverage counts differ from filter summary')
    return indices


def assemble_scene(session, filtered, output, *, visualize=True):
    """Preserve sources and partition indices; no merged production geometry."""
    import open3d as o3d
    session, filtered, output = [Path(p).resolve() for p in (session, filtered, output)]
    if any(output.is_relative_to(p) or p.is_relative_to(output) for p in (session, filtered)):
        raise ValueError('scene output must be separate from its inputs')
    source = inspect_session(session)
    summary = read_json(filtered/'summary.json')
    if (source['references'] != ['femto_before', 'femto_after']
            or summary.get('status') != 'complete' or summary.get('kind') != 'dense_rgb_confidence'
            or summary.get('output_frame') != 'base_link'
            or summary.get('session_sha256') != digest(session/'session.json')
            or summary.get('data_origin') != source['data_origin']):
        raise ValueError('complete filtered RGB and both FEMTO references must belong to this session')
    rgb = load_npz(filtered/'points_and_provenance.npz')
    indices = validate_rgb(rgb, summary)
    output.mkdir(parents=True, exist_ok=False)
    (output/'sources').mkdir()
    (output/'layers').mkdir()
    report = dict(schema_version=1, kind='layered_capture_scene', status='running',
                  output_frame='base_link', units='m', data_origin=source['data_origin'],
                  accuracy_1mm_verified=False, production_fusion_accepted=False,
                  scene_stability='not_evaluated_by_assembly', region=summary['region'],
                  source_sha256={'adapter': digest(__file__), 'partition': digest(inspect.getfile(rgb_layer_indices))},
                  layers=[], reference_sources={})
    atomic_json(output/'scene.json', report)
    try:
        for src, name in [(session/'session.json', 'session.json'), (filtered/'summary.json', 'filter_summary.json'),
                          (filtered/'points_and_provenance.npz', 'rgb_observations.npz')]:
            shutil.copyfile(src, output/'sources'/name)
            if digest(src) != digest(output/'sources'/name):
                raise ValueError('source changed while copying scene')
        region = summary['region']
        for name in ('femto_before', 'femto_after'):
            metadata = read_json(session/name/'field_scan.json')
            path = inside(session/name, metadata['fused_file'])
            points = np.asarray(o3d.io.read_point_cloud(str(path)).points)
            report['reference_sources'][name] = dict(session_file=str(path.relative_to(session)), sha256=digest(path))
            selected = np.flatnonzero(np.all((points >= region['minimum']) & (points <= region['maximum']), axis=1))
            if not len(selected):
                raise ValueError('FEMTO reference has no points in scene ROI')
            np.savez_compressed(output/'sources'/f'{name}.npz', points=points, group_ids=region_groups(points, region))
            shutil.copyfile(session/name/'field_scan.json', output/'sources'/f'{name}_metadata.json')
            indices[name] = selected
        for name in STYLE:
            is_rgb = name.startswith('rgb_')
            source_file = 'sources/rgb_observations.npz' if is_rgb else f'sources/{name}.npz'
            points = rgb['points'] if is_rgb else load_npz(output/source_file)['points']
            selected = indices[name]
            np.save(output/'layers'/f'{name}_indices.npy', selected, allow_pickle=False)
            write_xyz_pcd(output/'layers'/f'{name}.pcd', points[selected])
            report['layers'].append(dict(name=name, source=source_file, indices=f'layers/{name}_indices.npy',
                                        pcd=f'layers/{name}.pcd', point_count=len(selected),
                                        role='operational_reference' if name == 'femto_before' else 'diagnostic_evidence'))
        report.update(status='complete', point_count=int(summary['point_count']),
                      retained_stone_points=int(summary['retained_stone_points']),
                      provenance='Layer indices address source rows. RGB rows use the copied CSR view/pixel provenance; FEMTO rows address the original fused cloud.',
                      limitations=['Sources are separate evidence layers; no coordinate changes, averaging or accepted fusion.',
                                   'Both FEMTO brackets are preserved; assembly does not certify scene stability.',
                                   'RGB source pixels identify undistorted dense images. Images/raw captures remain in the original session/dense experiment.',
                                   'Source JSON files are provenance snapshots; their historical file paths are not bundle dependencies.',
                                   'Region IDs are spatial masks, not catalog stone IDs. Candidate extensions remain unverified.'])
        # Hash all portable numeric sources and exports, excluding the manifest/viewer.
        report['files'] = {str(p.relative_to(output)): digest(p)
                           for folder in ('sources', 'layers') for p in sorted((output/folder).iterdir())}
        atomic_json(output/'scene.json', report)
        inspect_scene(output)
        if visualize:
            replay_scene(output, output/'viewer')
        return report
    except BaseException as error:
        report.update(status='failed', error=str(error))
        atomic_json(output/'scene.json', report)
        raise


def inspect_scene(directory):
    """Verify a standalone assembly, its source mapping and all export hashes."""
    import open3d as o3d
    directory = Path(directory).resolve()
    report = read_json(directory/'scene.json')
    if (report.get('schema_version') != 1 or report.get('kind') != 'layered_capture_scene'
            or report.get('status') != 'complete' or report.get('output_frame') != 'base_link'
            or report.get('units') != 'm' or report.get('accuracy_1mm_verified') is not False
            or report.get('production_fusion_accepted') is not False):
        raise ValueError('unsupported or incomplete layered scene')
    actual = {f'{folder}/{name}': value for folder in ('sources', 'layers')
              for name, value in file_hashes(directory/folder).items()}
    if actual != report['files']:
        raise ValueError('scene file/hash mismatch')
    summary = read_json(directory/'sources/filter_summary.json')
    if summary['session_sha256'] != digest(directory/'sources/session.json'):
        raise ValueError('scene session binding mismatch')
    rgb = load_npz(directory/'sources/rgb_observations.npz')
    expected = validate_rgb(rgb, summary)
    if report['region'] != summary['region']:
        raise ValueError('scene region differs from filter')
    if [layer['name'] for layer in report['layers']] != list(STYLE):
        raise ValueError('unexpected scene layers')
    for layer in report['layers']:
        name = layer['name']
        is_rgb = name.startswith('rgb_')
        source_file = 'sources/rgb_observations.npz' if is_rgb else f'sources/{name}.npz'
        if (layer['source'] != source_file or layer['indices'] != f'layers/{name}_indices.npy'
                or layer['pcd'] != f'layers/{name}.pcd'):
            raise ValueError('invalid scene source mapping')
        data = rgb if is_rgb else load_npz(directory/source_file)
        points = data['points']
        if not is_rgb:
            expected[name] = np.flatnonzero(np.all((points >= report['region']['minimum']) & (points <= report['region']['maximum']), axis=1))
            if not np.array_equal(data['group_ids'], region_groups(points, report['region'])):
                raise ValueError('invalid reference region labels')
        ids = np.load(inside(directory, layer['indices']), allow_pickle=False)
        if not np.array_equal(ids, expected[name]) or len(ids) != layer['point_count']:
            raise ValueError('scene layer indices differ from evidence partition')
        exported = np.asarray(o3d.io.read_point_cloud(str(inside(directory, layer['pcd']))).points) if len(ids) else np.empty((0, 3))
        if exported.shape != (len(ids), 3) or not np.array_equal(exported, points[ids].astype(np.float32).astype(float)):
            raise ValueError('scene layer coordinates changed')
    return report


def replay_scene(directory, output):
    directory, output = Path(directory).resolve(), Path(output).resolve()
    report = inspect_scene(directory)
    if output == directory or directory.is_relative_to(output) or any(output.is_relative_to(directory/f) for f in ('sources', 'layers')):
        raise ValueError('viewer output would overlap scene data')
    output.mkdir(parents=True, exist_ok=False)
    sources = {}
    layers = []
    for layer in report['layers']:
        source = layer['source']
        if source not in sources:
            sources[source] = load_npz(inside(directory, source))
        data = sources[source]
        ids = np.load(inside(directory, layer['indices']), allow_pickle=False)
        label, color, visible = STYLE[layer['name']]
        groups = data['stone'] if layer['name'].startswith('rgb_') else data['group_ids']
        layers.append(dict(points=data['points'][ids], source_indices=ids, group_ids=groups[ids],
                           label=label, color=color, visible=visible))
    render_scene_layers(output/'overview.html', layers)
    return report
