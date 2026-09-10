"""Optional PyVista mesh adapters for canonical visualization geometry."""

import numpy as np

from stacking_common.visualization.contracts import TargetVolume
from stacking_common.visualization.dsf import dsf_support_points, read_dsf_components


def read_dsf_surface(path):
    """Return a display mesh of each curved component at icosphere level 4."""
    import pyvista as pv
    from scipy.spatial import ConvexHull

    directions = np.asarray(pv.Icosphere(nsub=4).points, dtype=float)
    pieces = []
    for nodes, sharpness in read_dsf_components(path):
        surface = dsf_support_points(nodes, directions, sharpness)
        hull = ConvexHull(surface)
        faces = hull.simplices.copy()
        triangles = surface[faces]
        normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
        reverse = np.einsum("ij,ij->i", normals, hull.equations[:, :3]) < 0
        faces[reverse] = faces[reverse, ::-1]
        mesh = pv.PolyData(surface, np.column_stack((np.full(len(faces), 3), faces)))
        pieces.append(mesh.clean().compute_normals(
            consistent_normals=True, auto_orient_normals=True, split_vertices=False))
    # Keep concave multi-component models separate; never hull the entire body.
    result = pieces[0] if len(pieces) == 1 else pieces[0].append_polydata(*pieces[1:])
    result.field_data["dsf_surface"] = [1]
    return result


def open_target_shell(target: TargetVolume):
    """Build tapered walls and ceiling without a ground-coplanar bottom face."""
    import pyvista as pv

    faces = np.asarray(
        (
            4, 4, 5, 6, 7,
            4, 0, 1, 5, 4,
            4, 1, 2, 6, 5,
            4, 2, 3, 7, 6,
            4, 3, 0, 4, 7,
        ),
        dtype=np.int64,
    )
    return pv.PolyData(target.points, faces)
