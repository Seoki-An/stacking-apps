import numpy as np
import pytest
from scipy.spatial.transform import Rotation
from stacking_common.multiview.dense import write_known_cameras, read_colmap_depth, backproject_colmap_depth, depth_bounds_from_roi


def test_known_pose_export_pixel_convention_and_transform(tmp_path):
    pose = np.eye(4); pose[:3, :3] = Rotation.from_euler('xyz', [.1, -.2, .3]).as_matrix(); pose[:3, 3] = [.1, .2, .5]
    k = np.array([[900., 0, 640], [0, 910., 360], [0, 0, 1]])
    write_known_cameras(tmp_path/'model', k, np.array([.1,.2,.01,.02,.3]), [1280,720], [pose], ['frame.png'])
    tokens = (tmp_path/'model/cameras.txt').read_text().split()
    np.testing.assert_allclose(np.array(tokens[4:],float), [900,910,640.5,360.5,.1,.2,.01,.02,.3,0,0,0])
    image = (tmp_path/'model/images.txt').read_text().split()
    q = np.array(image[1:5], float); t = np.array(image[5:8], float)
    r = Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
    np.testing.assert_allclose(r@pose[:3,:3], np.eye(3), atol=1e-12)
    np.testing.assert_allclose(r@pose[:3,3]+t,0,atol=1e-12)


def test_depth_binary_orientation_invalid_pixels_and_metric_backprojection(tmp_path):
    depth = np.array([[.3,0,.4],[np.nan,.5,.6]],dtype='<f4')
    path=tmp_path/'depth.bin';path.write_bytes(b'3&2&1&'+depth.T.tobytes(order='F'))
    recovered=read_colmap_depth(path);np.testing.assert_array_equal(recovered,depth)
    k=np.array([[100.,0,1.5],[0,100.,.5],[0,0,1.]])
    pose=np.eye(4);pose[:3,3]=[.2,.3,.4]
    points,pixels=backproject_colmap_depth(recovered,k,pose)
    np.testing.assert_array_equal(pixels,[0,2,4,5])
    np.testing.assert_allclose(points[0], [.197,.3,.7],atol=1e-7)
    path.write_bytes(b'3&2&1&'+b'1234')
    with pytest.raises(ValueError,match='payload'):read_colmap_depth(path)


def test_roi_depth_bounds_and_camera_plane_rejection():
    assert depth_bounds_from_roi([np.eye(4)],[-.1,-.1,.3],[.1,.1,.5]) == pytest.approx((.28,.52))
    with pytest.raises(ValueError,match='in front'):
        depth_bounds_from_roi([np.eye(4)],[-.1,-.1,-.1],[.1,.1,.5])


def test_crop_preserves_metric_rays_and_clips_image_bounds():
    from stacking_common.multiview.dense import projected_roi_crop
    matrix = np.array([[100., 0, 50.5], [0, 100., 40.5], [0, 0, 1.]])
    boxes = [[[-.1, -.1, .4], [.1, .1, .5]]]
    bounds, cropped_matrix = projected_roi_crop(matrix, np.eye(4), (100, 80), boxes, padding=4)
    x0, y0, x1, y1 = bounds
    depth = np.full((80, 100), .45)
    cropped = depth[y0:y1, x0:x1]
    full_points, _ = backproject_colmap_depth(depth, matrix, np.eye(4))
    crop_points, pixels = backproject_colmap_depth(cropped, cropped_matrix, np.eye(4))
    ys, xs = np.unravel_index(pixels, cropped.shape)
    np.testing.assert_allclose(crop_points, full_points[(ys+y0)*100+xs+x0], atol=1e-12)
    np.testing.assert_array_equal(cropped_matrix[:2, :2], matrix[:2, :2])
    whole, _ = projected_roi_crop(matrix, np.eye(4), (100, 80),
                                 [[[-1., -1., .4], [1., 1., .5]]])
    np.testing.assert_array_equal(whole, [0, 0, 100, 80])
    with pytest.raises(ValueError, match='outside'):
        projected_roi_crop(matrix, np.eye(4), (100, 80), [[[1., 1., .4], [2., 2., .5]]])
    with pytest.raises(ValueError, match='in front'):
        projected_roi_crop(matrix, np.eye(4), (100, 80), [[[-.1, -.1, -.1], [.1, .1, .5]]])


def test_cropped_camera_export_roundtrips_distinct_intrinsics(tmp_path):
    pycolmap = pytest.importorskip('pycolmap')
    from stacking_common.multiview.dense import write_pinhole_cameras
    matrices = [np.array([[900., 0, 230.5], [0, 910., 190.5], [0, 0, 1.]]),
                np.array([[900., 0, 250.5], [0, 910., 210.5], [0, 0, 1.]])]
    poses = [np.eye(4), np.eye(4)]
    poses[1][:3, :3] = Rotation.from_euler('y', .15).as_matrix()
    poses[1][:3, 3] = [.05, -.02, .01]
    sizes = [(500, 400), (600, 420)]
    write_pinhole_cameras(tmp_path/'model', matrices, sizes, poses, ['a.png', 'b.png'])
    reconstruction = pycolmap.Reconstruction(tmp_path/'model')
    (tmp_path/'binary').mkdir()
    reconstruction.write(tmp_path/'binary')
    reconstruction = pycolmap.Reconstruction(tmp_path/'binary')
    for index, image in enumerate(sorted(reconstruction.images.values(), key=lambda i:i.name)):
        camera = reconstruction.cameras[image.camera_id]
        np.testing.assert_allclose(camera.calibration_matrix(), matrices[index])
        assert (camera.width, camera.height) == sizes[index]
        np.testing.assert_allclose(image.cam_from_world().matrix(), np.linalg.inv(poses[index])[:3], atol=1e-12)


def test_source_selection_uses_parallax_and_preserves_region_support():
    from stacking_common.multiview.dense import select_source_views
    matrix = np.array([[900., 0, 640.5], [0, 900., 360.5], [0, 0, 1.]])
    poses = np.repeat(np.eye(4)[None], 9, axis=0)
    poses[1:, :2, 3] = [[.04, 0], [-.04, 0], [0, .04], [0, -.04],
                        [.08, .08], [-.08, .08], [.08, -.08], [-.08, -.08]]
    boxes = [[[-.08, -.03, .35], [-.02, .03, .45]],
             [[.02, -.03, .35], [.08, .03, .45]]]
    graph = select_source_views([matrix]*9, [(1280, 720)]*9, poses, boxes)
    for i, sources in enumerate(graph):
        assert len(sources) == len(set(sources)) == 6
        assert i not in sources
        assert any(poses[j, 0, 3] > 0 for j in sources)
        assert any(poses[j, 0, 3] < 0 for j in sources)
    poses[1] = poses[0]
    graph = select_source_views([matrix]*9, [(1280, 720)]*9, poses, boxes)
    assert 1 not in graph[0] and 0 not in graph[1]
    with pytest.raises(ValueError, match='parallax'):
        select_source_views([matrix]*3, [(1280, 720)]*3, [np.eye(4)]*3, boxes)


def test_source_selection_rejects_a_region_without_shared_visibility():
    from stacking_common.multiview.dense import select_source_views
    matrix = np.array([[600., 0, 320.5], [0, 600., 240.5], [0, 0, 1.]])
    poses = np.repeat(np.eye(4)[None], 5, axis=0)
    poses[:, 0, 3] = [0, -.1, -.12, -.14, -.16]
    boxes = [[[-.15, -.02, .4], [-.13, .02, .42]],
             [[.13, -.02, .4], [.15, .02, .42]]]
    with pytest.raises(ValueError, match='every ROI'):
        select_source_views([matrix]*5, [(640, 480)]*5, poses, boxes, maximum_sources=3)
