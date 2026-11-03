"""Orthographic original-mesh surfaces for fully vector scientific figures."""

import numpy as np


def orthographic_basis(direction):
    """Camera right, camera up, and scene-to-eye unit vectors; world Z is up."""
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) == 0:
        raise ValueError("camera direction must be a finite nonzero three-vector")
    eye = direction / np.linalg.norm(direction)
    right = np.cross([0., 0., 1.], eye)
    if np.linalg.norm(right) < 1e-12:
        raise ValueError("camera direction must not be parallel to world Z")
    right /= np.linalg.norm(right)
    return np.array([right, np.cross(eye, right), eye])


def vector_mesh_faces(triangles, basis, *, vertex_normals=None, color=(.50, .50, .50)):
    """Cull back faces, shade, and globally sort triangles from far to near.

    Retain every front-facing original triangle without simplifying geometry.
    Painter sorting is intended for opaque, nonintersecting stone surfaces.
    Vertex-normal lighting is averaged over each face for vector solid fills.
    """
    triangles = np.asarray(triangles, dtype=float)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3) or not np.isfinite(triangles).all():
        raise ValueError("triangles must be finite Nx3x3 world coordinates")
    normals = np.cross(triangles[:, 1]-triangles[:, 0], triangles[:, 2]-triangles[:, 0])
    keep = normals @ basis[2] > 1e-12
    triangles, normals = triangles[keep], normals[keep]
    if vertex_normals is None:
        normals /= np.linalg.norm(normals, axis=1)[:, None]
        normals = normals[:, None, :]
    else:
        vertex_normals = np.asarray(vertex_normals, dtype=float)
        if vertex_normals.shape != keep.shape + (3, 3) or not np.isfinite(vertex_normals).all():
            raise ValueError("vertex normals must match the original triangles")
        normals = vertex_normals[keep]
    key = np.array([-.4, .55, .75]) @ basis
    key /= np.linalg.norm(key)
    fill = np.array([.6, .1, .8]) @ basis
    fill /= np.linalg.norm(fill)
    intensity = (.48 + .50*np.maximum(normals @ key, 0.)
                 + .12*np.maximum(normals @ fill, 0.)).mean(axis=1)
    colors = np.clip(intensity[:, None]*np.asarray(color), 0., 1.)
    depth = triangles.mean(axis=1) @ basis[2]
    order = np.argsort(depth, kind="stable")
    return (triangles @ basis[:2].T)[order], colors[order]
