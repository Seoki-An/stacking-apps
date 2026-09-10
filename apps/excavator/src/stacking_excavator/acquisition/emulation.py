"""Desktop-only field, scene and stone_grab capture exercises. No ROS or NUC."""

import argparse
import copy
from pathlib import Path

import numpy as np

from stacking_common.pointcloud.raycast import simulate_lidar
from stacking_common.profiles.planning import load_planning_profile
from stacking_excavator.planning.trajectory import ExcavatorRobotTrajectoryProjector, pose_matrix

from .capture import CaptureFrame, CropRequest, GRIPPER_FRAME, digest
from .storage import atomic_json, create_capture, load_product, prepare_crop, receive_local


def look_at(position, target):
    """A declared emulated sensor pose, never a commissioned calibration."""
    forward = np.asarray(target) - position
    forward = forward / np.linalg.norm(forward)
    left = np.cross([0., 0., 1.], forward)
    left /= np.linalg.norm(left)
    transform = np.eye(4)
    transform[:3, :3] = np.column_stack((forward, left, np.cross(forward, left)))
    transform[:3, 3] = position
    return transform


def demonstration(repository, kind, *, opening=-.45):
    """Use real PLY/stone_grab geometry with explicit synthetic sensor placement."""
    import open3d as o3d

    root = Path(repository)
    profile = load_planning_profile(root / "apps/excavator/config/planning.yml")
    projector = ExcavatorRobotTrajectoryProjector(profile, root)
    joints = tuple(profile.robot.home_positions)
    parts = projector.project(np.array(joints), opening)
    body = next(part for part in parts if part.part_id.startswith("gripper/grip_body/"))
    world_T_gripper = pose_matrix(body.pose)
    stone = o3d.io.read_triangle_mesh(str(root / "apps/excavator/assets/objects/model_10_mesh.ply"))
    meshes = []
    if kind == "inhand":
        for part in parts:
            mesh = o3d.io.read_triangle_mesh(str(part.mesh_path))
            mesh.vertices = o3d.utility.Vector3dVector(np.asarray(mesh.vertices) * part.mesh_scale)
            mesh.transform(pose_matrix(part.pose))
            meshes.append(mesh)
        reference_T_box = np.eye(4)
        reference_T_box[:3, 3] = [0., .65, 0.]
        stone.translate(-stone.get_center()).translate([0., .65, 0.]).transform(world_T_gripper)
        reference_frame = GRIPPER_FRAME
        half_extents = (1., 1., .7)
        world_T_box = world_T_gripper @ reference_T_box
    else:
        stone.translate(-stone.get_center())
        stone.translate([3., 0., -stone.get_min_bound()[2]])
        reference_T_box = np.eye(4)
        reference_T_box[:3, 3] = stone.get_center()
        reference_frame = "world"
        half_extents = (.8, .8, .8) if kind == "field" else (1.4, 1., 1.2)
        world_T_box = reference_T_box
        if kind == "scene":
            overhang = copy.deepcopy(stone).translate([.65, .15, .55])
            meshes.append(overhang)
        # Distractor retained in raw data, intentionally outside the purpose ROI.
        meshes.append(copy.deepcopy(stone).translate([0., 2.5, 0.]))
    meshes.append(stone)
    center = world_T_box[:3, 3]
    sensors = [look_at(center + offset, center) for offset in
               (np.array([-3., -1.5, 1.2]), np.array([-2.5, 1.5, 1.5]))]
    calibration = digest({"provenance": "emulated", "sensors": [s.tolist() for s in sensors],
                          "world_T_gripper": world_T_gripper.tolist(), "opening": opening})
    crop = CropRequest(kind, reference_frame, reference_T_box, half_extents,
                       (.15, .15, .15), calibration, 1, (10,))
    return meshes, sensors, world_T_gripper, joints, crop


def run_demonstration(repository, output, *, kinds=("field", "inhand", "scene")):
    """Capture → crop → interrupt/resume → verify → merge; keep every artifact."""
    import open3d as o3d

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    report = {"mode": "desktop_emulation", "commissioned_calibration": False, "purposes": {}}
    for kind in kinds:
        meshes, sensors, gripper, joints, crop = demonstration(repository, kind)
        frames = []
        for index, sensor in enumerate(sensors):
            points, ray_ids = simulate_lidar(meshes, sensor)
            frames.append(CaptureFrame(f"frame-{index}", f"emulated-lidar-{index}",
                1_000_000_000 + index * 100_000_000, 1_000_000_000 + index * 100_000_000,
                sensor, gripper, crop.calibration_sha256, points, joints, -.45,
                attributes={"ray_index": ray_ids}))
        raw, product, received = (output / kind / name for name in ("raw", "crop-v1", "desktop"))
        capture = create_capture(raw, kind + "-capture", kind, 1, frames)
        cropped = prepare_crop(raw, product, kind + "-crop-v1", crop, maximum_timestamp_skew_ns=0)
        interrupted = not receive_local(product, received, byte_budget=1024)
        if not interrupted or (received / "manifest.json").exists():
            raise RuntimeError("interruption did not leave an unpublished partial transfer")
        receive_local(product, received)
        _, verified = load_product(received, expected_revision=1)
        if kind in {"field", "inhand"}:
            reference = o3d.io.read_triangle_mesh(str(Path(repository) /
                "apps/excavator/assets/objects/model_10_mesh.ply"))
            prior = np.eye(4)
            prior[:3, 3] = crop.reference_T_box[:3, 3] - reference.get_center()
            atomic_json(output / kind / "priors.json", {"scene_revision": 1,
                "ground_height": 0., "poses": {"10": prior.tolist()}})
        for name, values in (("raw-world", frames), ("crop-world", verified)):
            world = np.concatenate([f.points @ f.world_T_sensor[:3, :3].T + f.world_T_sensor[:3, 3]
                                    for f in values])
            cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(world))
            if not len(world) or not o3d.io.write_point_cloud(str(output / kind / (name + ".ply")), cloud):
                raise RuntimeError(f"{kind}: empty cloud or PLY export failed")
        raw_bytes = sum(f["bytes"] for f in capture["frames"])
        crop_bytes = sum(f["bytes"] for f in cropped["frames"])
        report["purposes"][kind] = {"raw_points": sum(len(f.points) for f in frames),
            "cropped_points": sum(len(f.points) for f in verified),
            "raw_bytes": raw_bytes, "cropped_bytes": crop_bytes,
            "byte_fraction": crop_bytes / raw_bytes, "interrupted_then_resumed": interrupted,
            "manifest_sha256": cropped["sha256"]}
    atomic_json(output / "report.json", report)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repository", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, required=True, help="new artifact directory (must not exist)")
    parser.add_argument("--purpose", choices=("all", "field", "inhand", "scene"), default="all")
    args = parser.parse_args(argv)
    kinds = ("field", "inhand", "scene") if args.purpose == "all" else (args.purpose,)
    try:
        report = run_demonstration(args.repository, args.output, kinds=kinds)
    except (ValueError, OSError, RuntimeError, ImportError) as error:
        parser.exit(1, f"PCD emulation error: {error}\n")
    for kind, values in report["purposes"].items():
        print(f"{kind}: {values['raw_points']} → {values['cropped_points']} points; "
              f"{values['byte_fraction']:.1%} of raw bytes; interrupted transfer resumed and verified")
    print(f"Artifacts: {args.output.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
