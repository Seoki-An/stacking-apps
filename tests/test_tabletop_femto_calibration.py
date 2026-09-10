"""Ported legacy calibration parity tests; no legacy imports at runtime."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation


from stacking_tabletop.acquisition import calibration as loader


def pose(translation, rotation_vector):
    result = np.eye(4)
    result[:3, :3] = Rotation.from_rotvec(rotation_vector).as_matrix()
    result[:3, 3] = translation
    return result


class CameraCalibrationLoaderTest(unittest.TestCase):
    def test_loads_frame_aware_poseid_result(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "camera.npz"
            np.savez(
                path,
                T_base_to_camera=np.eye(4),
                parent_frame="base_link",
                child_frame="camera_link",
                camera="FEMTO_BOLT",
                method="d435_femto_simultaneous_charuco",
                ee_pose_frame="base_link",
            )
            loaded = loader.load_camera_calibration(path)

        self.assertEqual(loaded.parent_frame, "base_link")
        self.assertEqual(loaded.child_frame, "camera_link")
        self.assertEqual(loaded.camera, "FEMTO_BOLT")
        self.assertEqual(loaded.correction, "")
        np.testing.assert_allclose(loaded.transform, np.eye(4))

    def test_legacy_poseid_ur_base_is_converted_to_base_link(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "legacy_poseid.npz"
            transform = pose([0.1, -0.8, 0.6], [0.1, 0.2, 0.3])
            np.savez(
                path,
                T_base_to_camera=transform,
                parent_frame="base_link",
                child_frame="camera_link",
                camera="FEMTO_BOLT",
                method="d435_femto_simultaneous_charuco",
            )

            loaded = loader.load_camera_calibration(path)

        expected = np.diag([-1.0, -1.0, 1.0, 1.0]) @ transform
        np.testing.assert_allclose(loaded.transform, expected)
        self.assertEqual(loaded.correction, "legacy_ur_base_to_base_link")

    def test_relocates_color_only_mount_to_depth_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            camera_path = Path(directory) / "camera.npz"
            depth_path = Path(directory) / "depth.npz"
            base_T_camera = pose(
                [-0.08, 0.96, 0.70],
                [0.05, -0.03, 1.2],
            )
            base_T_color = pose(
                [-0.08, 0.96, 0.70],
                [-1.3, 0.2, -1.4],
            )
            depth_T_color = pose(
                [0.0325, 0.0009, -0.0019],
                [0.11, 0.002, 0.004],
            )
            np.savez(
                camera_path,
                T_base_to_camera=base_T_camera,
                T_base_to_observation=base_T_color,
                parent_frame="base_link",
                child_frame="camera_link",
                observation_frame="camera_color_optical_frame",
                camera="FEMTO_BOLT",
                method="d435_femto_simultaneous_charuco_v2",
                ee_pose_frame="base_link",
            )
            np.savez(
                depth_path,
                capture_depth_T_color=np.repeat(
                    depth_T_color[None, :, :], 8, axis=0
                ),
                depth_frame="camera_depth_optical_frame",
                color_frame="camera_color_optical_frame",
            )

            loaded = loader.load_camera_calibration(
                camera_path,
                depth_path,
            )

        expected_base_T_depth = base_T_color @ np.linalg.inv(depth_T_color)
        np.testing.assert_allclose(
            loaded.transform[:3, 3],
            expected_base_T_depth[:3, 3],
        )
        np.testing.assert_allclose(
            loaded.transform[:3, :3],
            base_T_camera[:3, :3],
        )
        self.assertEqual(
            loaded.correction,
            "femto_color_to_depth_origin",
        )
        self.assertEqual(
            loaded.depth_frame,
            "camera_depth_optical_frame",
        )

    def test_rejects_color_frame_mismatch_during_depth_relocation(self):
        with tempfile.TemporaryDirectory() as directory:
            camera_path = Path(directory) / "camera.npz"
            depth_path = Path(directory) / "depth.npz"
            np.savez(
                camera_path,
                T_base_to_camera=np.eye(4),
                T_base_to_observation=np.eye(4),
                parent_frame="base_link",
                child_frame="camera_link",
                observation_frame="camera_color_optical_frame",
            )
            np.savez(
                depth_path,
                capture_depth_T_color=np.eye(4)[None, :, :],
                depth_frame="camera_depth_optical_frame",
                color_frame="unexpected_color_frame",
            )

            with self.assertRaisesRegex(ValueError, "differ"):
                loader.load_camera_calibration(camera_path, depth_path)

    def test_rejects_pose_and_depth_from_different_camera_serials(self):
        with tempfile.TemporaryDirectory() as directory:
            camera_path = Path(directory) / "camera.npz"
            depth_path = Path(directory) / "depth.npz"
            np.savez(
                camera_path,
                T_base_to_camera=np.eye(4),
                T_base_to_observation=np.eye(4),
                parent_frame="base_link",
                child_frame="camera_link",
                observation_frame="camera_color_optical_frame",
                camera_serial="CAMERA_A",
            )
            np.savez(
                depth_path,
                capture_depth_T_color=np.eye(4)[None, :, :],
                depth_frame="camera_depth_optical_frame",
                color_frame="camera_color_optical_frame",
                camera_serial="CAMERA_B",
            )

            with self.assertRaisesRegex(ValueError, "serials differ"):
                loader.load_camera_calibration(camera_path, depth_path)

    def test_camera_artifact_is_compared_after_composing_cloud_frame(self):
        base_T_camera = pose([0.2, 0.8, 0.7], [0.03, -0.02, 0.01])
        camera_T_cloud = pose([0.01, -0.02, 0.03], [0.0, 0.0, 0.04])
        published = base_T_camera @ camera_T_cloud

        difference = loader.camera_cloud_transform_difference(
            base_T_camera,
            camera_T_cloud,
            published,
        )

        self.assertAlmostEqual(difference.translation_m, 0.0, places=12)
        self.assertAlmostEqual(difference.rotation_deg, 0.0, places=9)

    def test_camera_artifact_comparison_reports_translation_and_rotation(self):
        published = pose([0.001, 0.0, 0.0], [0.0, 0.0, np.deg2rad(0.2)])

        difference = loader.camera_cloud_transform_difference(
            np.eye(4),
            np.eye(4),
            published,
        )

        self.assertAlmostEqual(difference.translation_m, 0.001, places=12)
        self.assertAlmostEqual(difference.rotation_deg, 0.2, places=9)


if __name__ == "__main__":
    unittest.main()
