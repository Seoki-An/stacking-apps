"""SIFT matching and conservative tracks, independent of geometric acceptance."""

import numpy as np


def extract_features(rgb, matrix, distortion):
    import cv2
    image = np.asarray(rgb)
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('expected native RGB8 image')
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    keypoints, descriptors = cv2.SIFT_create().detectAndCompute(gray, None)
    pixels = np.array([k.pt for k in keypoints], dtype=float).reshape(-1, 2)
    normalized = (cv2.undistortPoints(pixels[:, None], matrix, distortion).reshape(-1, 2)
                  if len(pixels) else np.empty((0, 2)))
    return dict(pixels=pixels, normalized=normalized, descriptors=descriptors,
                sharpness=float(cv2.Laplacian(gray, cv2.CV_64F).var()))


def match_features(first, second, ratio=.75):
    """Mutual nearest matches passing a two-neighbour ratio test both ways."""
    import cv2
    if first is None or second is None or min(len(first), len(second)) < 2:
        return []
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    def accepted(a, b):
        return {pair[0].queryIdx: pair[0].trainIdx
                for pair in matcher.knnMatch(a, b, k=2)
                if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance}
    forward, backward = accepted(first, second), accepted(second, first)
    return [(a, b) for a, b in sorted(forward.items()) if backward.get(b) == a]


def build_tracks(edges):
    """Connected components; reject whole components with duplicate-view nodes.

    Nodes are (view_index, feature_index). No camera pose or FEMTO reference
    participates in track assembly, keeping unused-view checks meaningful.
    """
    adjacency = {}
    for a, b in edges:
        adjacency.setdefault(a, set()).add(b)
        adjacency.setdefault(b, set()).add(a)
    seen, tracks, conflicts = set(), [], 0
    for node in sorted(adjacency):
        if node in seen:
            continue
        stack, component = [node], []
        while stack:
            current = stack.pop()
            if current in seen:
                continue
            seen.add(current)
            component.append(current)
            stack.extend(sorted(adjacency[current] - seen))
        if len({v for v, _ in component}) != len(component):
            conflicts += 1
        else:
            tracks.append(sorted(component))
    return tracks, conflicts
