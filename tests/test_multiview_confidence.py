import numpy as np
import pytest
from stacking_common.multiview.confidence import voxel_observations, filter_rgb_agreement, classify_reference_coverage


def test_filter_rejects_single_pair_hallucination_and_mutually_supported_isolate():
    x,y=np.meshgrid(np.arange(8)*.001,np.arange(8)*.001)
    surface=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,.1)))
    hallucination=np.array([[.04,0,.1]]); isolate=np.array([[.07,0,.1]])
    candidates=np.concatenate((surface,hallucination,isolate))
    a=candidates.copy();b=np.concatenate((surface+[0,0,.0004],isolate))
    result=filter_rgb_agreement(candidates,[a,b])
    assert result['keep'][:64].all()
    assert not result['pair_agreement'][64]
    assert result['pair_agreement'][65] and not result['local_support'][65]
    assert not result['keep'][64:].any()
    np.testing.assert_array_equal(candidates,np.concatenate((surface,hallucination,isolate)))


def test_reference_offset_is_not_certified_as_new_coverage():
    x,y=np.meshgrid(np.arange(8)*.001,np.arange(8)*.001)
    reference=np.column_stack((x.ravel(),y.ravel(),np.full(x.size,.1)))
    p=np.array([[.003,.003,.1005],[.003,.003,.107],[.02,.003,.1]])
    result=classify_reference_coverage(p,reference)
    np.testing.assert_array_equal(result['labels'],[1,2,3])
    assert result['candidate_xy_cells']==1
    assert result['candidate_projected_area_mm2']==pytest.approx(4)


def test_voxel_provenance_preserves_every_observation_and_view():
    points=np.array([[.0002,0,.1002],[.0012,0,.1002],[.0004,0,.1004]])
    r=voxel_observations(points,np.array([0,1,2]),np.array([12,25,37]))
    np.testing.assert_allclose(r['points'],[[.0003,0,.1003],[.0012,0,.1002]])
    np.testing.assert_array_equal(r['observation_offsets'],[0,2,3])
    np.testing.assert_array_equal(r['observation_views'],[0,2,1])
    np.testing.assert_array_equal(r['observation_pixels'],[12,37,25])
    with pytest.raises(ValueError,match='nonempty'):
        filter_rgb_agreement(points,[points,np.empty((0,3))])
