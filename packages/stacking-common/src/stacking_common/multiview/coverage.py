"""Estimated RGB footprints on an explicitly chosen reference plane."""
import numpy as np


def plane_coverage(camera_poses, matrix, distortion, image_size, origin, basis):
    """Intersect calibrated image-border rays with a plane, ignoring occlusion.

    basis columns are the reference plane's orthonormal X, Y, normal axes.
    A view crossing the plane horizon has no bounded footprint on that plane.
    """
    import cv2

    width, height = image_size
    corners = np.array([[0., 0.], [width-1., 0.], [width-1., height-1.], [0., height-1.]])
    pixels = np.concatenate([a + np.linspace(0, 1, 33)[:, None]*(b-a)
                             for a, b in zip(corners, np.roll(corners, -1, axis=0))])
    normalized = cv2.undistortPoints(pixels[:, None], np.asarray(matrix, dtype=float),
                                    np.asarray(distortion, dtype=float)).reshape(-1, 2)
    rays = np.column_stack((normalized, np.ones(len(normalized))))
    origin, basis = np.asarray(origin), np.asarray(basis)
    footprints = []
    for label, pose in camera_poses.items():
        pose = np.asarray(pose)
        directions = rays @ pose[:3, :3].T
        denominator = directions @ basis[:, 2]
        numerator = (origin-pose[:3, 3]) @ basis[:, 2]
        if (not np.isfinite(directions).all() or abs(numerator) < 1e-9 or
                np.any(denominator * np.sign(numerator) <= 1e-8)):
            footprints.append(dict(label=label, bounded=False))
            continue
        points = pose[:3, 3] + directions*(numerator/denominator)[:, None]
        xy = (points-origin) @ basis[:, :2]
        footprints.append(dict(label=label, bounded=True, boundary=points.tolist(),
                               plane_xy_m=xy.tolist(), span_m=np.ptp(xy, axis=0).tolist()))
    bounded = [f for f in footprints if f['bounded']]
    span = None
    if len(bounded) == len(footprints) and bounded:
        span = np.ptp(np.concatenate([f['plane_xy_m'] for f in bounded]), axis=0).tolist()
    return dict(plane_origin=origin.tolist(), plane_basis=basis.tolist(),
                footprints=footprints, combined_bounding_span_m=span,
                limitation='Estimated field of view on the reference plane, not guaranteed reconstructed '
                           'coverage. Occlusion, stereo overlap, texture, and reconstruction crops reduce usable coverage.')
