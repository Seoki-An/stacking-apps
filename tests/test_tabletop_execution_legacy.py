"""Characterization cases ported from legacy scripts/test/test_ur5e_execution.py.

Retains the legacy expected numbers and sample sequences; requires no legacy
checkout, ROS, Dynamixel SDK, or physical devices.
"""

import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from stacking_tabletop.execution.gripper import (
    GRIPPER_CLOSED_POSITION, GRIPPER_OPEN_POSITION, GRIPPER_GRASP_CLOSE_RATE,
    GRIPPER_GRASP_COMMAND_RATE_HZ, GRIPPER_GRASP_CLOSING_MARGIN,
    GripperContactDetector, GripperHoldGuard,
    gripper_position_for_opening_angle, gripper_position_ramp,
)
from stacking_tabletop.execution.motion import (
    CartesianStreamLimits, canonical_joint_positions, pose_error, sample_joint_path,
)
from stacking_tabletop.execution.ur5e import CONTACT_ADMITTANCE_GAINS, RIGID_ADMITTANCE_GAINS


def test_admittance_gains_match_legacy_source_not_stale_documentation():
    # Baseline: legacy planning/ur5e_control.py (linear M/B/K; angular M/B/K).
    np.testing.assert_array_equal(np.asarray(CONTACT_ADMITTANCE_GAINS).reshape(6, 3),
                                  [[8]*3, [600]*3, [1200]*3, [5]*3, [100]*3, [200]*3])
    np.testing.assert_array_equal(np.asarray(RIGID_ADMITTANCE_GAINS).reshape(6, 3),
                                  [[10]*3, [650]*3, [2000]*3, [5]*3, [150]*3, [300]*3])

class _Kinematics:
    @staticmethod
    def pose(q):
        pose = np.eye(4)
        pose[:3, :3] = Rotation.from_rotvec([0.0, 0.0, q[1]]).as_matrix()
        pose[0, 3] = q[0]
        return pose


class TestLegacyExecution(unittest.TestCase):
    def test_default_execution_limits_use_accelerated_bounded_profile(self):
        limits = CartesianStreamLimits()
        self.assertEqual(limits.max_joint_speed, 0.40)
        self.assertEqual(limits.max_tcp_speed, 0.08)
        self.assertEqual(limits.max_angular_speed, 0.70)

    def test_gripper_position_tracks_planned_opening_angle(self):
        self.assertEqual(
            gripper_position_for_opening_angle(0.0), GRIPPER_CLOSED_POSITION
        )
        self.assertEqual(
            gripper_position_for_opening_angle(0.9), GRIPPER_OPEN_POSITION
        )
        self.assertEqual(gripper_position_for_opening_angle(0.45), 796)
        self.assertEqual(
            gripper_position_for_opening_angle(
                0.45,
                closing_margin=GRIPPER_GRASP_CLOSING_MARGIN,
            ),
            169,
        )
        self.assertEqual(
            gripper_position_for_opening_angle(
                0.02,
                closing_margin=GRIPPER_GRASP_CLOSING_MARGIN,
            ),
            GRIPPER_CLOSED_POSITION,
        )
        self.assertEqual(
            gripper_position_for_opening_angle(-0.1),
            GRIPPER_CLOSED_POSITION,
        )

    def test_gripper_position_rejects_angle_outside_planned_range(self):
        for opening_angle in (-0.11, 0.91, np.nan):
            with self.subTest(opening_angle=opening_angle):
                with self.assertRaisesRegex(ValueError, "opening angle"):
                    gripper_position_for_opening_angle(opening_angle)

        for closing_margin in (-0.01, 0.91, np.nan):
            with self.subTest(closing_margin=closing_margin):
                with self.assertRaisesRegex(ValueError, "closing margin"):
                    gripper_position_for_opening_angle(
                        0.45,
                        closing_margin=closing_margin,
                    )

    def test_gripper_grasp_ramp_is_bounded_and_reaches_goal(self):
        max_step = GRIPPER_GRASP_CLOSE_RATE / GRIPPER_GRASP_COMMAND_RATE_HZ
        positions = gripper_position_ramp(1500.0, 169, max_step)

        self.assertEqual(positions[-1], 169)
        self.assertTrue(all(type(position) is int for position in positions))
        deltas = np.diff([1500, *positions])
        self.assertTrue(np.all(deltas <= 0))
        self.assertTrue(np.any(deltas == 0))
        self.assertLessEqual(np.max(np.abs(deltas)), np.ceil(max_step))
        midpoint = len(deltas) // 2
        self.assertLess(abs(deltas[0]), abs(deltas[midpoint]))
        self.assertLess(abs(deltas[-1]), abs(deltas[midpoint]))

    def test_gripper_grasp_ramp_rejects_invalid_values(self):
        for start, goal, max_step in (
            (np.nan, 169, 10.0),
            (1500.0, 50, 10.0),
            (1500.0, 169, 0.0),
        ):
            with self.subTest(start=start, goal=goal, max_step=max_step):
                with self.assertRaises(ValueError):
                    gripper_position_ramp(start, goal, max_step)

    def test_gripper_contact_uses_only_distinct_samples(self):
        detector = GripperContactDetector(
            1500.0,
            7,
            movement_counts=10.0,
            stable_band_counts=5.0,
            stable_samples=3,
        )
        self.assertFalse(detector.observe(1200.0, 8))
        self.assertFalse(detector.observe(1200.0, 8))
        self.assertFalse(detector.observe(1201.0, 9))
        self.assertFalse(detector.observe(1200.0, 10))
        self.assertTrue(detector.observe(1201.0, 11))

    def test_gripper_contact_resets_after_stone_motion(self):
        detector = GripperContactDetector(
            1500.0,
            0,
            movement_counts=10.0,
            stable_band_counts=5.0,
            stable_samples=3,
        )
        # Pairwise motion is small, but cumulative motion leaves the fixed
        # stability band and must begin a new candidate window.
        for sequence, position in enumerate((1200, 1202, 1204, 1206), start=1):
            self.assertFalse(detector.observe(position, sequence))
        self.assertFalse(detector.observe(1207.0, 5))
        self.assertFalse(detector.observe(1208.0, 6))
        self.assertTrue(detector.observe(1207.0, 7))

    def test_gripper_contact_requires_actual_closing_motion(self):
        detector = GripperContactDetector(
            1500.0,
            0,
            movement_counts=10.0,
            stable_band_counts=5.0,
            stable_samples=2,
        )
        self.assertFalse(detector.observe(1499.0, 1))
        self.assertFalse(detector.observe(1500.0, 2))
        self.assertFalse(detector.observe(1498.0, 3))

    def test_gripper_contact_rejects_opening_motion(self):
        detector = GripperContactDetector(
            500.0,
            0,
            movement_counts=10.0,
            stable_band_counts=5.0,
            stable_samples=2,
        )
        self.assertFalse(detector.observe(800.0, 1))
        self.assertFalse(detector.observe(801.0, 2))
        self.assertFalse(detector.observe(800.0, 3))

    def test_gripper_hold_requires_sustained_opening(self):
        guard = GripperHoldGuard(opening_limit_counts=50.0, opening_samples=3)
        guard.arm(500.0)
        self.assertFalse(guard.observe(560.0))
        self.assertFalse(guard.observe(555.0))
        self.assertFalse(guard.observe(520.0))
        self.assertFalse(guard.observe(570.0))
        self.assertFalse(guard.observe(565.0))
        self.assertTrue(guard.observe(560.0))

    def test_gripper_hold_reference_follows_only_additional_closing(self):
        guard = GripperHoldGuard(opening_limit_counts=50.0, opening_samples=1)
        guard.arm(500.0)
        self.assertFalse(guard.observe(480.0))
        self.assertEqual(guard.reference_position, 480.0)
        self.assertTrue(guard.observe(531.0))

    def test_canonical_joint_order_accepts_prefixes(self):
        names = [
            "arm/wrist_3_joint",
            "arm/shoulder_lift_joint",
            "arm/elbow_joint",
            "arm/wrist_1_joint",
            "arm/shoulder_pan_joint",
            "arm/wrist_2_joint",
        ]
        actual = canonical_joint_positions(names, [6, 2, 3, 4, 1, 5])
        np.testing.assert_allclose(actual, [1, 2, 3, 4, 5, 6])

    def test_joint_path_sampling_respects_all_reference_limits(self):
        limits = CartesianStreamLimits(
            rate_hz=20.0,
            max_joint_speed=0.2,
            max_tcp_speed=0.1,
            max_angular_speed=0.25,
        )
        samples = sample_joint_path(
            [np.zeros(6), np.array([0.3, 0.4, 0.0, 0.0, 0.0, 0.0])],
            _Kinematics(),
            limits,
        )
        for (q_a, pose_a), (q_b, pose_b) in zip(samples[:-1], samples[1:]):
            self.assertLessEqual(
                float(np.max(np.abs(q_b - q_a))),
                limits.max_joint_speed / limits.rate_hz + 1e-12,
            )
            position_error, rotation_error = pose_error(pose_a, pose_b)
            self.assertLessEqual(
                position_error, limits.max_tcp_speed / limits.rate_hz + 1e-12
            )
            self.assertLessEqual(
                rotation_error, limits.max_angular_speed / limits.rate_hz + 1e-12
            )
