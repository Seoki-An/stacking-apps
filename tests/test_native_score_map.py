"""Native score maps reproduce the Python reference bit for bit."""

import numpy as np
import pytest

from stacking_common.actions import _scored_floor as floor
from stacking_common.actions._support_maps import BottomHeightKernel
from test_composite_target_sampling import pose, sampler, state
from test_previous_action_connectivity import setup

NATIVE = floor.native_score_xy_map


def assert_identical(native, reference, path="map"):
    if isinstance(native, floor.CandidateColumns):
        native = native.columns
    if isinstance(reference, floor.CandidateColumns):
        reference = reference.columns
    if isinstance(reference, dict):
        assert list(native) == list(reference), path
        for key in reference:
            assert_identical(native[key], reference[key], f"{path}.{key}")
    elif isinstance(reference, np.ndarray):
        native = np.asarray(native)
        assert native.dtype == reference.dtype and native.shape == reference.shape, path
        assert native.tobytes() == reference.tobytes(), path  # NaN and -0.0 included
    else:
        assert type(native) is type(reference), path
        assert native == reference or (native != native and reference != reference), path


def recorded_maps(monkeypatch, connectivity, enabled, placed):
    inv, current, observation = setup(False)
    sample = sampler(inv)
    sample.settings["connectivity"] = connectivity
    sample.connectivity_enabled = enabled
    calls = []

    def record(*args, **kwargs):
        calls.append((args, kwargs))
        return NATIVE(*args, **kwargs)

    monkeypatch.setattr(floor, "native_score_xy_map", record)
    scene = current if placed else state(inv)
    sample.score_map(scene, observation, 2)
    # A CEM mutation lookup reuses the fresh map's normalization.
    sample.score_at_object_pose(observation, pose(.1, .2, yaw=.4), 2)
    return calls


@pytest.mark.parametrize("placed", [False, True])
@pytest.mark.parametrize("connectivity, enabled", [
    ("previous_action_soft", True), ("previous_action_soft", False), ("active_floor", False),
])
def test_native_maps_match_python_reference(monkeypatch, connectivity, enabled, placed):
    calls = recorded_maps(monkeypatch, connectivity, enabled, placed)
    assert calls
    for args, kwargs in calls:
        native = NATIVE(*args, **kwargs)
        assert len(native["candidates"])
        assert_identical(native, floor.python_score_xy_map(*args, **kwargs))


def test_maps_without_anchors_or_support_match_python_reference(monkeypatch):
    (args, kwargs), *_ = recorded_maps(monkeypatch, "previous_action_soft", True, True)
    kernels = kwargs["support_kernels"]
    # Footprints wider than the target leave no anchor; kernels off the grid
    # leave anchors without support.
    cases = [dict(kwargs, footprint_bounds=100. * np.asarray(kwargs["footprint_bounds"])),
             dict(kwargs, support_kernels=[BottomHeightKernel(
                 y_offsets=np.array([1000]), x_offsets=np.array([0]), bottom_height=np.zeros(1))
                 for _ in kernels])]
    for case in cases:
        native = NATIVE(*args, **case)
        assert not len(native["candidates"])
        assert_identical(native, floor.python_score_xy_map(*args, **case))
    assert not np.any(NATIVE(*args, **cases[0])["valid"])
    assert np.any(NATIVE(*args, **cases[1])["valid"])


def test_hard_connectivity_stays_on_python_path(monkeypatch):
    assert not recorded_maps(monkeypatch, "active_floor", True, True)
