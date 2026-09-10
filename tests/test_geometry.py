import numpy as np

from stacking_common.environment._geometry import quaternion_matrix


def test_scalar_last_quaternion_rotates_about_z_axis() -> None:
    half_sqrt = np.sqrt(0.5)
    quaternion_xyzw = np.array([0.0, 0.0, half_sqrt, half_sqrt])

    rotated = quaternion_matrix(quaternion_xyzw) @ np.array([1.0, 0.0, 0.0])

    assert np.allclose(rotated, [0.0, 1.0, 0.0], atol=1e-12)
