"""Human-readable degree profiles preserve canonical runtime coordinates."""

from pathlib import Path

import numpy as np
import pytest
import yaml

from stacking_common.artifacts.planning_context import _values
from stacking_common.profiles.angles import resolve_degree_fields
from stacking_common.profiles.native import load_native_solver_profile
from stacking_common.profiles.planning import load_planning_profile
from stacking_excavator.execution.simulation import SimulatedDevice
from stacking_excavator.perception.pipeline import DesktopPerception


ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "apps/excavator/config"


def test_degree_planning_and_legacy_radian_profiles_resolve_identically(tmp_path):
    values = yaml.safe_load((CONFIG / "planning.yml").read_text())
    expected_home = np.deg2rad(values["robot"]["home_positions_degrees"])
    for section, fields in (("robot", ("home_positions", "gripper_opening")),
                            ("gripper", ("opening_offset", "opening_limits"))):
        values[section] = resolve_degree_fields(values[section], fields)
    radians = tmp_path / "planning.yml"
    radians.write_text(yaml.safe_dump(values))
    profile = load_planning_profile(CONFIG / "planning.yml")
    assert _values(profile) == _values(load_planning_profile(radians))
    np.testing.assert_array_equal(profile.robot.home_positions, expected_home)
    np.testing.assert_array_equal(SimulatedDevice(ROOT).positions, expected_home)
    np.testing.assert_array_equal(profile.world_T_stack.parent_T_child,
                                  [-6., 3., 0., 0., 0., 0., 1.])
    np.testing.assert_array_equal(profile.gripper.opening_direction, [1., 1.])


def test_native_degree_settings_resolve_before_artifact_serialization():
    profile = load_native_solver_profile(CONFIG / "solvers/grasp.yml", CONFIG / "solvers/motion.yml")
    assert profile.grasp["sampling"]["spin_step"] == np.pi / 6
    assert profile.motion["direct"]["target_rot_tol"] == np.pi / 18
    assert profile.motion["regrasp"]["target_rot_tol"] == np.pi / 18
    assert profile.motion["regrasp"]["max_handoff_orientation_distance"] == 3 * np.pi / 4
    assert profile.motion["motion"]["joint_limit_margin"] == .001
    assert profile.grasp["trust_region"]["radius_init"] == .1
    assert "spin_step_degrees" not in _values(profile)["grasp"]["sampling"]


def test_perception_opening_thresholds_keep_calibrated_radian_values():
    opening = DesktopPerception(ROOT).profile["opening"]
    assert opening["limits"] == [-1.14, 0.]
    assert opening["maximum_measured_difference"] == .2
    raw = yaml.safe_load((CONFIG / "perception.yml").read_text())["opening"]
    assert opening["coefficients"] == raw["coefficients"]


@pytest.mark.parametrize("section,name", [("robot", "home_positions"), ("gripper", "opening_limits")])
def test_planning_rejects_conflicting_angle_units(tmp_path, section, name):
    values = yaml.safe_load((CONFIG / "planning.yml").read_text())
    values[section][name] = [0.] * len(values[section][f"{name}_degrees"])
    path = tmp_path / "planning.yml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match=f"specify only one of {name}"):
        load_planning_profile(path)


def test_native_rejects_conflicting_angle_units(tmp_path):
    values = yaml.safe_load((CONFIG / "solvers/grasp.yml").read_text())
    values["sampling"]["spin_step"] = .5
    path = tmp_path / "grasp.yml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="specify only one of spin_step"):
        load_native_solver_profile(path, CONFIG / "solvers/motion.yml")


@pytest.mark.parametrize("value", [float("nan"), [0., float("inf")]])
def test_degree_conversion_rejects_nonfinite_values(value):
    with pytest.raises(ValueError, match="finite angles"):
        resolve_degree_fields({"angle_degrees": value}, ("angle",))
