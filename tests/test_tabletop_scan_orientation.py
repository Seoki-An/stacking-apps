"""Tabletop scan orientation selection preserves pickup poses and old contexts."""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import yaml

from stacking_common.actions.orientation import ScanAlignedOrientationProvider, PrincipalAxisOrientationProvider
from stacking_common.artifacts.planning_context import _values
from stacking_common.profiles.action import load_action_generation_profile as load_common
from stacking_tabletop import TabletopApplication
from stacking_tabletop.config.action import load_action_generation_profile, resolve_pose_from_scan

ROOT = Path(__file__).resolve().parents[1]


def test_profile_records_scored_settings_and_scan_override(tmp_path):
    path = ROOT / 'apps/tabletop/config/action_generation.yml'
    kwargs = dict(max_poses_per_object=8, banned_object_ids=())
    current = load_action_generation_profile(path, **kwargs)
    assert current.pose_from_scan is False
    recorded = _values(current)
    settings = recorded.pop('sampling_settings')
    assert recorded.pop('pose_from_scan') is False
    assert recorded == _values(load_common(path, **kwargs))
    assert settings['contact_gap_tolerance'] == .02
    assert settings['score']['fill_area'] == 4.
    data = yaml.safe_load(path.read_text())
    data['orientation']['pose_from_scan'] = True
    alternate = tmp_path / 'action.yml'
    alternate.write_text(yaml.safe_dump(data))
    on = load_action_generation_profile(alternate, **kwargs)
    assert on.pose_from_scan is True
    assert resolve_pose_from_scan(on.pose_from_scan, None) is True
    assert _values(on)['sampling_settings'] == settings
    data['orientation'].pop('pose_from_scan')
    alternate.write_text(yaml.safe_dump(data))
    assert load_action_generation_profile(alternate, **kwargs).pose_from_scan is True
    data['orientation']['pose_from_scan'] = 'false'
    alternate.write_text(yaml.safe_dump(data))
    with pytest.raises(TypeError, match='must be boolean'):
        load_action_generation_profile(alternate, **kwargs)


@pytest.mark.parametrize('configured', [True, False])
def test_override_and_resume_precedence(configured):
    assert resolve_pose_from_scan(configured, None) is configured
    assert resolve_pose_from_scan(configured, True) is True
    assert resolve_pose_from_scan(configured, False) is False
    for recorded in (True, False):
        assert resolve_pose_from_scan(configured, None, recorded) is recorded
        assert resolve_pose_from_scan(configured, recorded, recorded) is recorded
        with pytest.raises(ValueError, match='resume preserves'):
            resolve_pose_from_scan(configured, not recorded, recorded)


def test_off_uses_model_orientations_without_replacing_measured_pickup():
    app = TabletopApplication.from_repository(ROOT)
    app = replace(app, action_generation_profile=replace(app.action_generation_profile, pose_from_scan=True))
    off = replace(app, action_generation_profile=replace(app.action_generation_profile, pose_from_scan=False))
    measured = np.array([.1, -.7, .2, np.sin(.4), 0., 0., np.cos(.4)])
    inventory = off.load_inventory([1], {1: measured})
    np.testing.assert_array_equal(inventory.initial_pose(1), measured)
    assert isinstance(app.build_orientation_provider(), ScanAlignedOrientationProvider)
    assert app.build_orientation_filters(app.build_floor_policy(inventory))
    provider = off.build_orientation_provider()
    assert isinstance(provider, PrincipalAxisOrientationProvider)
    assert off.build_orientation_filters(off.build_floor_policy(inventory)) == ()
    quaternions = provider.quaternions(inventory, None, None, 0)
    assert len(quaternions) == 2
    other = off.load_inventory([1], {1: np.array([0., 0., .2, 0., 0., 0., 1.])})
    np.testing.assert_allclose(quaternions, provider.quaternions(other, None, None, 0))
