import numpy as np
import pytest
from stacking_common.pointcloud.pose_metrics import pose_errors


def test_identical_and_translated_pose():
    points=np.array([[0,0,0],[.01,0,0],[0,.02,0],[0,0,.03]])
    truth=np.eye(4)
    same=pose_errors(points,truth,truth)
    assert same['translation_mm']==same['rotation_deg']==same['add_mean_mm']==0
    estimate=truth.copy();estimate[2,3]=.004
    moved=pose_errors(points,estimate,truth)
    assert moved['translation_mm']==pytest.approx(4)
    assert moved['add_mean_mm']==pytest.approx(4)


def test_symmetric_surface_distance_does_not_hide_rotation_metric():
    points=np.array([[-.01,0,0],[.01,0,0],[0,-.02,0],[0,.02,0]])
    flipped=np.diag([-1.,-1.,1.,1.])
    result=pose_errors(points,flipped,np.eye(4))
    assert result['rotation_deg']==pytest.approx(180)
    assert result['add_mean_mm']==pytest.approx(30)
    assert result['adds_mean_mm']==pytest.approx(0)


def test_common_frame_change_preserves_errors():
    points=np.array([[0,0,0],[.01,0,0],[0,.02,0]])
    truth=np.eye(4); estimate=np.eye(4);estimate[0,3]=.004
    frame=np.diag([-1.,-1.,1.,1.]);frame[:3,3]=[.5,-.4,.003]
    original=pose_errors(points,estimate,truth)
    transformed=pose_errors(points,frame@estimate,frame@truth)
    assert transformed['translation_mm']==pytest.approx(original['translation_mm'])
    assert transformed['add_mean_mm']==pytest.approx(original['add_mean_mm'])
