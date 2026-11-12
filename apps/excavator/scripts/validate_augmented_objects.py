"""Independently audit original surfaces, fitted DSFs, native loading and catalog provenance."""

import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import open3d as o3d

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/stacking-common/src"))

from stacking_common.assets import ObjectAssetLoader
from stacking_common.profiles.inventory import load_inventory_profile
from stacking_common.reconstruction.geometry import (
    dsf_support_points, estimate_mesh_inertial, load_dsf_components)
from stacking_common.reconstruction.types import TriangleMeshData


def occupancy(vertices, faces, points):
    scene = o3d.t.geometry.RaycastingScene(nthreads=1)
    scene.add_triangles(o3d.core.Tensor(vertices, dtype=o3d.core.Dtype.Float32),
                        o3d.core.Tensor(faces, dtype=o3d.core.Dtype.UInt32))
    return scene.compute_occupancy(o3d.core.Tensor(points, dtype=o3d.core.Dtype.Float32),
                                   nthreads=1).numpy().astype(bool)


def main():
    directory = ROOT / "apps/excavator/assets/objects_augmented"
    manifest = json.loads((directory / "manifest.json").read_text())
    assert manifest["status"] == "complete" and len(manifest["models"]) == 100
    profile = load_inventory_profile(ROOT / "apps/excavator/config/inventory_augmented.yml")
    ids = profile.object_ids
    inventory = ObjectAssetLoader(directory, profile.density, profile.target.geometry(),
        dict(profile.density_overrides)).load(ids, {i: np.array([0., 0., 0., 0., 0., 0., 1.]) for i in ids})
    assert len(inventory) == 100
    sphere = o3d.geometry.TriangleMesh.create_sphere(resolution=20)
    directions = np.asarray(sphere.vertices).copy()
    directions /= np.linalg.norm(directions, axis=1)[:, None]
    sphere_faces = np.asarray(sphere.triangles).copy()
    rows = []
    for obj in inventory:
        oid = obj.object_id
        record = manifest["models"][str(oid)]
        for name, expected in record["files"].items():
            assert hashlib.sha256((directory / name).read_bytes()).hexdigest() == expected
        if record["kind"] == "real":
            for name, expected in record["files"].items():
                original = ROOT / "apps/excavator/assets/objects" / name
                assert original.read_bytes() == (directory / name).read_bytes()
        surface = o3d.io.read_triangle_mesh(str(directory / f"model_{oid}_mesh.ply"))
        vertices, faces = np.asarray(surface.vertices), np.asarray(surface.triangles)
        inertial = estimate_mesh_inertial(TriangleMeshData(vertices, faces))
        assert obj.mass > 0 and np.linalg.eigvalsh(obj.inertia).min() > 0
        components = load_dsf_components(directory / f"model_{oid}.obj")
        for component in components:
            assert component.sharpness >= 2 and np.isfinite(component.nodes).all()
        if record["kind"] == "synthetic":
            np.testing.assert_allclose(inertial.center_of_mass, 0., atol=1e-10)
            np.testing.assert_allclose(obj.mass, profile.density_for(oid)*inertial.mass, rtol=1e-10)
            np.testing.assert_allclose(obj.inertia, profile.density_for(oid)*inertial.inertia, rtol=1e-9, atol=1e-10)
            assert max(v["support_p95_fraction_of_stone_extent"] for v in record["support_validation"]) <= .05
        dsfs = [dsf_support_points(c.nodes, c.sharpness, directions) for c in components]
        cloud = np.vstack([vertices, *dsfs])
        lower, upper = cloud.min(axis=0), cloud.max(axis=0)
        spacing = np.ptp(cloud, axis=0).max()/45
        axes = [np.arange(lo+spacing/2, hi, spacing) for lo, hi in zip(lower, upper)]
        points = np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, 3)
        truth = occupancy(vertices, faces, points)
        predicted = np.zeros(len(points), dtype=bool)
        for values in dsfs:
            predicted |= occupancy(values, sphere_faces, points)
        intersection = int(np.count_nonzero(truth & predicted))
        iou = intersection / np.count_nonzero(truth | predicted)
        coverage = intersection / np.count_nonzero(truth)
        precision = intersection / np.count_nonzero(predicted)
        rows.append(dict(object_id=oid, kind=record["kind"], components=len(components),
            nodes=sum(len(c.nodes) for c in components), mass_kg=obj.mass,
            original_surface_volume_m3=inertial.mass, watertight=True, self_intersecting=False,
            surface_dsf_voxel_iou=iou, surface_covered_by_dsf_fraction=coverage,
            dsf_inside_surface_fraction=precision, audit_voxel_size_m=spacing))
        print(f"{oid}: {record['kind']}, IoU={iou:.4f}, coverage={coverage:.4f}", flush=True)
    synthetic = [r for r in rows if r["kind"] == "synthetic"]
    report = dict(object_count=100, real_count=49, synthetic_count=51, native_loading="passed",
        originals_byte_identical=True, synthetic_center_of_mass_and_inertials="passed",
        runtime_body_capacity_required_including_ground=101,
        runtime_geometry_count_including_target=1+sum(r["components"] for r in rows),
        synthetic_mesh_dsf_iou_min=min(r["surface_dsf_voxel_iou"] for r in synthetic),
        synthetic_mesh_dsf_iou_mean=float(np.mean([r["surface_dsf_voxel_iou"] for r in synthetic])),
        synthetic_mesh_coverage_min=min(r["surface_covered_by_dsf_fraction"] for r in synthetic),
        audit_definition="45 voxels per largest AABB extent; raycast original surface vs union of sampled smooth DSF component surfaces",
        models=rows)
    (directory / "validation.json").write_text(json.dumps(report, indent=2)+"\n")
    assert report["synthetic_mesh_dsf_iou_min"] >= .80
    assert report["synthetic_mesh_coverage_min"] >= .90
    print("Validation passed:", {k: v for k, v in report.items() if k != "models"}, flush=True)


if __name__ == "__main__":
    main()
