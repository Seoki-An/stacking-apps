"""Optional PyVista mesh adapters for canonical visualization geometry."""

import numpy as np

from stacking_common.visualization.contracts import CompositeTargetVolume, TargetDisplay
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


def open_target_shell(target: TargetDisplay):
    """Build tapered walls and ceiling without a ground-coplanar bottom face."""
    import pyvista as pv

    if isinstance(target, CompositeTargetVolume):
        points, faces, labels, offset = [], [], [], 0
        for index, component in enumerate(target.components):
            geometry = component.world_geometry
            _, normals, _ = geometry.face_data()
            # Omit horizontal downward caps as for the old open-bottom shell.
            # Sloping/rotated side faces remain intact. Never hull or weld pieces.
            kept = geometry.faces[normals[:, 2] > -1. + 1e-9]
            points.append(geometry.points + target.structure_offset)
            faces.append(kept + offset)
            labels.extend([index] * len(kept))
            offset += len(geometry.points)
        faces = np.concatenate(faces)
        mesh = pv.PolyData(np.concatenate(points), np.c_[np.full(len(faces), 3), faces])
        mesh.cell_data["target_component_index"] = np.asarray(labels, dtype=int)
        mesh.field_data["target_component_ids"] = np.asarray([item.component_id for item in target.components])
        return mesh

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
