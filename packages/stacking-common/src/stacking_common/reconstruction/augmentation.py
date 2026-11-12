"""Synthetic mesh deformation followed by fresh convex decomposition and DSF fitting."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import numpy as np

from .convex_decomposition import ConvexDecomposer, build_dsf_point_sets
from .dsf_fitting import DsfFitter
from .geometry import DsfExporter, dsf_support_points, estimate_mesh_inertial, load_dsf_components
from .types import InertialMetadata, TriangleMeshData


def deform_surface(mesh, seed):
    """Apply a smooth, bounded warp and anisotropic scale in principal axes.

    Gaussian displacements have a global Jacobian norm below 0.30, so their
    continuous map is injective. Triangle topology is retained; the resulting
    discrete mesh is separately checked for self intersections before fitting.
    """
    rng = np.random.default_rng(seed)
    center = mesh.vertices.mean(axis=0)
    _, axes = np.linalg.eigh(np.cov((mesh.vertices-center).T))
    axes = axes[:, ::-1]
    for column in range(3):
        if axes[np.argmax(abs(axes[:, column])), column] < 0:
            axes[:, column] *= -1
    if np.linalg.det(axes) < 0:
        axes[:, -1] *= -1
    local = (mesh.vertices-center) @ axes
    radius = np.ptp(local, axis=0)/2
    if np.min(radius) <= 0:
        raise ValueError("source mesh must have positive extent along all principal axes")
    coordinates = local/radius
    scales = rng.uniform(.82, 1.18, 3) * rng.uniform(.94, 1.06)
    centers = rng.uniform(-.65, .65, (4, 3))
    widths = rng.uniform(.65, 1.05, 4)
    weights = rng.normal(0., .055, (4, 3))
    bound = float(np.sum(np.linalg.norm(weights, axis=1)/widths)*np.exp(-.5))
    if bound > .30:
        weights *= .30/bound
        bound = .30
    squared = np.sum((coordinates[:, None, :]-centers[None, :, :])**2, axis=2)
    displacement = np.exp(-squared/(2*widths**2)) @ weights
    vertices = ((coordinates+displacement)*radius*scales) @ axes.T
    return TriangleMeshData(vertices, mesh.triangles), dict(
        seed=int(seed), principal_axes=axes.tolist(), normalization_half_extents_m=radius.tolist(),
        source_vertex_mean_m=center.tolist(), axis_scales=scales.tolist(),
        gaussian_centers=centers.tolist(), gaussian_widths=widths.tolist(),
        gaussian_weights=weights.tolist(), warp_jacobian_norm_bound=bound)


def fit_augmented_stone(source_mesh, source_obj, output_directory, object_id, seed,
                        convex_profile, fitting_profile, export_profile):
    """Build one new original surface/DSF pair, retaining fitting provenance.

    Geometry is re-centered at its integrated center of mass. OBJ inertial
    metadata is at unit density, as expected by ObjectAssetLoader. Callers own
    platform profiles, object IDs, job scheduling, and catalog publication.
    """
    import coacd
    import open3d as o3d

    source_mesh, source_obj = Path(source_mesh), Path(source_obj)
    destination = Path(output_directory)
    destination.mkdir(parents=True, exist_ok=True)
    surface = o3d.io.read_triangle_mesh(str(source_mesh))
    mesh = TriangleMeshData(np.asarray(surface.vertices), np.asarray(surface.triangles))
    deformed, deformation = deform_surface(mesh, seed)
    inertial = estimate_mesh_inertial(deformed)
    centered = TriangleMeshData(deformed.vertices-inertial.center_of_mass, deformed.triangles)
    runtime_inertial = InertialMetadata(inertial.mass, inertial.inertia, np.zeros(3))
    # CoACD expects outward faces. Unit-density integration accepts either
    # consistent orientation, so normalize it explicitly at this boundary.
    triangles = centered.vertices[centered.triangles]
    signed_volume = float(np.einsum("ij,ij->i", triangles[:, 0],
        np.cross(triangles[:, 1], triangles[:, 2])).sum()/6)
    if signed_volume < 0:
        centered = TriangleMeshData(centered.vertices, centered.triangles[:, ::-1])
    coacd.set_log_level("warn")
    parts = ConvexDecomposer(replace(convex_profile, seed=int(seed))).decompose(centered)
    point_sets, _ = build_dsf_point_sets(centered.vertices, parts,
        fit_source=fitting_profile.fit_source,
        inside_tolerance=fitting_profile.split_inside_tolerance,
        minimum_points=fitting_profile.split_minimum_points)
    friction_values = {component.friction for component in load_dsf_components(source_obj)}
    if len(friction_values) != 1:
        raise ValueError("source component friction must agree before inheriting its material")
    exporter = DsfExporter(replace(export_profile, friction=friction_values.pop()))
    # Independently sample directional support errors after integer sharpness
    # commissioning, rather than treating optimizer termination as fit quality.
    directions = np.asarray(o3d.geometry.TriangleMesh.create_sphere(resolution=20).vertices)
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    span = float(np.ptp(centered.vertices, axis=0).max())
    attempts = [(1, fitting_profile.loss), (2, fitting_profile.loss)]
    if fitting_profile.loss != "linear":
        attempts.append((2, "linear"))
    for retry, (multiplier, loss) in enumerate(attempts):
        effective_profile = replace(fitting_profile,
            total_vertices=fitting_profile.total_vertices*multiplier,
            maximum_iterations=fitting_profile.maximum_iterations*multiplier, loss=loss)
        fits = DsfFitter(effective_profile).fit(point_sets)
        components = tuple(exporter.component_from_vertices(fit.vertices,
            sharpness=max(2, int(fit.sharpness))) for fit in fits)
        errors = []
        for points, component, fit in zip(point_sets, components, fits):
            target = np.quantile(points @ directions.T, fitting_profile.support_quantile, axis=0)
            boundary = dsf_support_points(component.nodes, component.sharpness, directions)
            actual = np.einsum("ij,ij->i", boundary, directions)
            error = abs(actual-target)
            errors.append(dict(support_rmse_m=float(np.sqrt(np.mean(error**2))),
                support_p95_m=float(np.quantile(error, .95)), support_max_m=float(error.max()),
                support_p95_fraction_of_stone_extent=float(np.quantile(error, .95)/span),
                cost=fit.cost, evaluations=fit.evaluations, fitted_sharpness=float(fit.sharpness),
                commissioned_sharpness=component.sharpness, nodes=len(component.nodes)))
        if max(error["support_p95_fraction_of_stone_extent"] for error in errors) <= .05:
            break
    else:
        values = [round(error["support_p95_fraction_of_stone_extent"], 5) for error in errors]
        raise ValueError(f"stone {object_id}: DSF support p95 fractions {values} exceed 5% of stone extent")
    obj_path = destination / f"model_{object_id}.obj"
    ply_path = destination / f"model_{object_id}_mesh.ply"
    exporter.write(obj_path, components, runtime_inertial)
    surface = o3d.geometry.TriangleMesh(o3d.utility.Vector3dVector(centered.vertices.copy()),
                                       o3d.utility.Vector3iVector(centered.triangles.copy()))
    surface.compute_vertex_normals()
    if not o3d.io.write_triangle_mesh(str(ply_path), surface):
        raise RuntimeError(f"could not write augmented mesh: {ply_path}")
    record = dict(object_id=int(object_id), kind="synthetic", source_mesh=str(source_mesh),
        source_mesh_sha256=hashlib.sha256(source_mesh.read_bytes()).hexdigest(),
        source_obj=str(source_obj), source_obj_sha256=hashlib.sha256(source_obj.read_bytes()).hexdigest(),
        deformation=deformation, center_of_mass_shift_m=inertial.center_of_mass.tolist(),
        volume_m3=inertial.mass, unit_density_inertia=runtime_inertial.inertia.tolist(),
        surface_extents_m=np.ptp(centered.vertices, axis=0).tolist(),
        surface_vertices=len(centered.vertices), surface_triangles=len(centered.triangles),
        watertight=True, self_intersecting=False, convex_components=len(parts),
        convex_profile=asdict(replace(convex_profile, seed=int(seed))),
        fitting_profile=asdict(effective_profile), fitting_retry_count=retry, support_validation=errors,
        files={path.name: hashlib.sha256(path.read_bytes()).hexdigest() for path in (obj_path, ply_path)})
    metadata_path = destination / f"model_{object_id}.json"
    temporary = metadata_path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(record, indent=2)+"\n")
    temporary.replace(metadata_path)
    return record
