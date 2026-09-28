import numpy as np
import pytest
from stacking_common.multiview.pose_inputs import stereo_pose_input


def test_metric_depth_mask_uses_camera_transform_and_table():
    d = np.full((5, 5), 10., dtype=np.float32)
    k = np.array([[100., 0, 2], [0, 100., 2], [0, 0, 1]])
    pose = np.eye(4); pose[:3, 3] = [1., 2., -.9]
    valid = np.ones_like(d, bool); valid[2, 2] = False
    depth, mask = stereo_pose_input(d, valid, k, pose, .1, [[.985,1.985],[1.015,2.015]], [0,0,.08])
    np.testing.assert_allclose(depth[valid],1.)
    assert depth[2,2] == 0 and not mask[2,2]
    assert mask.sum() == 8
    assert not mask[0,:].any()
    _, on_table = stereo_pose_input(d, valid, k, pose, .1, [[0,0],[3,3]], [0,0,.095])
    assert not on_table.any()


def test_invalid_disparity_stays_unknown():
    d = np.array([[10.,0.,-1.,np.nan]],dtype=np.float32)
    depth,mask=stereo_pose_input(d,np.ones_like(d,bool),np.eye(3),np.eye(4),.1,[[-10,-10],[10,10]],[0,0,0],0)
    assert depth[0,0] == pytest.approx(.01)
    assert not np.any(depth[0,1:])
    assert mask.tolist()==[[True,False,False,False]]


def test_zero_baseline_rejected():
    with pytest.raises(ValueError,match='geometry'):
        stereo_pose_input(np.ones((2,2)),np.ones((2,2),bool),np.eye(3),np.eye(4),0,[[0,0],[1,1]],[0,0,0])
