import json
import shutil
import numpy as np
import pytest
from stacking_common.multiview.scene import rgb_layer_indices
from stacking_tabletop.acquisition import hand_eye_scene as scene
from stacking_tabletop.acquisition.hand_eye_capture import atomic_json, digest
from stacking_tabletop.acquisition.storage import write_xyz_pcd


def observations():
    return dict(points=np.array([[0, 0, 0], [0, 0, .1], [.01, 0, .1], [.02, 0, .1], [.03, 0, .1]]),
                keep=np.array([True, True, True, True, False]), stone=np.array([-1, 0, 0, 0, 0]),
                coverage_class=np.array([0, 1, 2, 3, 0]), observation_offsets=np.array([0, 2, 3, 4, 5, 6]),
                observation_views=np.array([0, 1, 0, 0, 1, 1]), observation_pixels=np.arange(6))


def test_partition_preserves_rejected_and_unresolved_evidence():
    data = observations()
    result = rgb_layer_indices(data)
    assert {key: ids.tolist() for key, ids in result.items()} == {
        'rgb_background': [0], 'rgb_overlap': [1], 'rgb_unresolved': [2], 'rgb_extensions': [3], 'rgb_rejected': [4]}
    data['coverage_class'][4] = 3
    with pytest.raises(ValueError, match='exactly'):
        rgb_layer_indices(data)
    data = observations()
    data['observation_offsets'][-1] = 7
    with pytest.raises(ValueError, match='provenance'):
        rgb_layer_indices(data)


def make_inputs(tmp_path, monkeypatch):
    session = tmp_path/'session'; filtered = tmp_path/'filtered'
    session.mkdir(); filtered.mkdir()
    manifest = dict(references=['femto_before', 'femto_after'], data_origin='synthetic')
    atomic_json(session/'session.json', manifest)
    monkeypatch.setattr(scene, 'inspect_session', lambda p: manifest)
    for name in manifest['references']:
        (session/name).mkdir()
        write_xyz_pcd(session/name/'cloud.pcd', np.array([[0, 0, 0], [0, 0, .1], [1, 1, 1]]))
        atomic_json(session/name/'field_scan.json', dict(fused_file='cloud.pcd'))
    data = observations()
    np.savez_compressed(filtered/'points_and_provenance.npz', **data)
    summary = dict(status='complete', kind='dense_rgb_confidence', output_frame='base_link',
                   data_origin='synthetic', session_sha256=digest(session/'session.json'),
                   input_points=5, point_count=4, retained_stone_points=3,
                   region=dict(minimum=[-.2, -.2, -.01], maximum=[.2, .2, .2], support_plane=[0, 0, 0],
                               stone_boxes_xy=[[[-.1, -.1], [.1, .1]]]),
                   view_sources=[dict(view=0, image_shape=[3, 3]), dict(view=1, image_shape=[3, 3])],
                   groups=[dict(stone=0, reference_overlap_points=1, unresolved_surface_difference_points=1,
                                candidate_extension_points=1)])
    atomic_json(filtered/'summary.json', summary)
    return session, filtered


def test_scene_portability_exact_coordinates_and_tamper_detection(tmp_path, monkeypatch):
    session, filtered = make_inputs(tmp_path, monkeypatch)
    output = tmp_path/'assembled'
    report = scene.assemble_scene(session, filtered, output, visualize=False)
    assert report['accuracy_1mm_verified'] is False
    assert report['production_fusion_accepted'] is False
    assert [x['point_count'] for x in report['layers']] == [2, 2, 1, 1, 1, 1, 1]
    copied = tmp_path/'relocated'
    shutil.move(output, copied)
    shutil.rmtree(session); shutil.rmtree(filtered)
    assert scene.inspect_scene(copied)['status'] == 'complete'
    scene.replay_scene(copied, tmp_path/'viewer')
    assert (tmp_path/'viewer/overview.html').is_file()
    # Even if file hashes are refreshed, partition corruption cannot relabel a disagreement.
    path = copied/'layers/rgb_extensions_indices.npy'
    np.save(path, np.array([2]))
    with pytest.raises(ValueError, match='hash'):
        scene.inspect_scene(copied)
    report['files']['layers/rgb_extensions_indices.npy'] = digest(path)
    atomic_json(copied/'scene.json', report)
    with pytest.raises(ValueError, match='partition'):
        scene.inspect_scene(copied)


def test_wrong_session_binding_fails_before_output(tmp_path, monkeypatch):
    session, filtered = make_inputs(tmp_path, monkeypatch)
    summary = json.loads((filtered/'summary.json').read_text())
    summary['session_sha256'] = 'wrong'
    atomic_json(filtered/'summary.json', summary)
    with pytest.raises(ValueError, match='belong'):
        scene.assemble_scene(session, filtered, tmp_path/'output', visualize=False)
    assert not (tmp_path/'output').exists()
