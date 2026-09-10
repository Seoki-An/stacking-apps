"""Verified crop products to reviewable field, in-hand and scene pose proposals."""

from itertools import product
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation
import yaml

from stacking_common.pointcloud.identification import identify_pose
from stacking_common.pointcloud.open3d_registration import Open3dRegistrationBackend
from stacking_common.pointcloud.processing import voxel_downsample
from stacking_common.pointcloud.registration import MultiscaleRegistrationConfig, RegistrationQualityConfig, evaluate_registration
from stacking_common.pointcloud.types import PointCloudData, validate_rigid_transform
from stacking_common.profiles.angles import resolve_degree_fields
from stacking_excavator.acquisition.capture import CropRequest, GRIPPER_FRAME, digest, integer
from stacking_excavator.acquisition.storage import load_product

from .models import ExcavatorModels


def orientation_seeds(kind, initial):
    initial = validate_rigid_transform(initial)
    if kind == "field":
        # Legacy field recovery ignores an unreliable prior rotation.
        rotations = [Rotation.from_euler("xyz", angles, degrees=True).as_matrix()
                     for angles in product((0, 90, 180, 270), repeat=3)]
    elif kind == "inhand":
        rotations = [Rotation.from_euler("x", angle, degrees=True).as_matrix() @ initial[:3, :3]
                     for angle in (0, 90, 180, 270)]
    else:
        rotations = [initial[:3, :3]]
    result = []
    for rotation in rotations:
        seed = initial.copy()
        seed[:3, :3] = rotation
        result.append(seed)
    return result


class DesktopPerception:
    def __init__(self, repository):
        self.root = Path(repository)
        self.profile = yaml.safe_load((self.root / "apps/excavator/config/perception.yml").read_text())
        self.profile["opening"] = resolve_degree_fields(
            self.profile["opening"], ("limits", "maximum_measured_difference"))
        self.quality = RegistrationQualityConfig(**self.profile["quality"])
        self.models = ExcavatorModels(self.root)

    def estimate_opening(self, cloud, measured):
        """Legacy width estimate, separately gated by measured opening and jaw evidence."""
        cfg = self.profile["opening"]
        points = cloud.points
        if len(points) < self.profile["minimum_points"]:
            return {"estimated": None, "measured": measured, "valid": False,
                    "reasons": ["insufficient_opening_evidence"]}
        width = float(np.ptp(points[:, 0]))
        estimate = float(np.polyval(cfg["coefficients"], width))
        reasons = []
        if not cfg["limits"][0] <= estimate <= cfg["limits"][1]:
            reasons.append("opening_outside_calibrated_range")
        if measured is not None and abs(estimate - measured) > cfg["maximum_measured_difference"]:
            reasons.append("opening_disagrees_with_measured")
        if not reasons:
            jaw = self.models.gripper(estimate).points
            distances = cKDTree(jaw).query(points)[0]
            supported = points[distances < cfg["gripper_removal_distance"]]
            if (np.count_nonzero(supported[:, 0] < -.2) < cfg["minimum_jaw_points"] or
                    np.count_nonzero(supported[:, 0] > .2) < cfg["minimum_jaw_points"]):
                reasons.append("insufficient_bilateral_jaw_evidence")
        return {"estimated": estimate, "measured": measured, "measured_valid": measured is not None, "valid": not reasons,
                "reasons": reasons, "x_extent": width}

    def identify(self, directory, priors, *, expected_revision, ground_height=0.):
        """Priors are object-to-world matrices, or object-to-gripper for in-hand.

        No accepted scene is mutated. Empty/occluded objects retain unknown status.
        """
        integer(expected_revision, "expected scene revision")
        if not np.isfinite(ground_height):
            raise ValueError("ground height must be finite")
        manifest, frames = load_product(directory, expected_revision=expected_revision)
        crop = CropRequest(**manifest["crop"])
        kind = manifest["kind"]
        if kind not in {"field", "inhand", "scene"}:
            raise ValueError("diagnostic captures do not identify objects")
        checked = {}
        for object_id, pose in priors.items():
            integer(object_id, "object ID")
            checked[object_id] = validate_rigid_transform(pose, "initial object pose")
        if not checked or set(checked) != set(crop.object_ids):
            raise ValueError("prior object IDs must exactly match the crop request")
        clouds = []
        openings = []
        alignments = []
        for frame in frames:
            # Revalidate metadata even when a product was received a long time ago.
            crop.select(frame, maximum_timestamp_skew_ns=self.profile["maximum_timestamp_skew_ns"])
            transform = (np.linalg.inv(frame.world_T_gripper) @ frame.world_T_sensor
                         if kind == "inhand" else frame.world_T_sensor)
            cloud = PointCloudData(frame.points).transformed(transform)
            if kind == "inhand":
                p = cloud.points
                cloud = PointCloudData(p[(np.abs(p[:, 0]) < 2) & (p[:, 1] > 0) &
                                        (p[:, 1] < 2) & (np.abs(p[:, 2]) < 2)])
                cloud = voxel_downsample(cloud, .003)
                opening = self.estimate_opening(cloud, frame.gripper_opening)
                openings.append(opening)
                # Never use an invalid estimate to form accepted attachment geometry.
                angle = opening["estimated"] if opening["valid"] else frame.gripper_opening
                if angle is not None:
                    cloud, alignment = self.align_gripper(cloud, angle)
                    alignment["frame_id"] = frame.frame_id
                    alignments.append(alignment)
                    distances = cKDTree(self.models.gripper(angle).points).query(cloud.points)[0]
                    cloud = PointCloudData(cloud.points[distances >= self.profile["opening"]["gripper_removal_distance"]])
            elif kind == "scene" or self.profile["field_ground_removal"]:
                cloud = PointCloudData(cloud.points[cloud.points[:, 2] > ground_height + self.profile["ground_clearance"]])
            clouds.append(cloud.points)
        points = np.concatenate(clouds)
        if kind == "field":
            # Legacy recovery crops before downsampling and retains ground.
            center = next(iter(checked.values()))[:3, 3]
            points = points[np.all((points >= center - [.8, .8, .5]) &
                                   (points <= center + [.8, .8, 1.]), axis=1)]
        merged = voxel_downsample(PointCloudData(points), self.profile["voxel_size"])
        models = {i: self.models.stone(i) for i in checked}
        # Each scene point is evidence for at most one expected object. A prior
        # outside this region needs a wider crop/manual reinitialization.
        distances = np.array([cKDTree(models[i][0].transformed(pose).points).query(merged.points)[0]
                              for i, pose in checked.items()])
        assignment = distances.argmin(axis=0)
        results = []
        cfg = self.profile["inhand" if kind == "inhand" else "field"]
        backend = Open3dRegistrationBackend(MultiscaleRegistrationConfig(
            **cfg, correspondence_distance_scale=1.5, evaluation_distance=.03,
            minimum_points=self.profile["minimum_points"]))
        for index, (object_id, initial) in enumerate(checked.items()):
            model, model_hash = models[object_id]
            lower = model.transformed(initial).points.min(axis=0) - .35
            upper = model.transformed(initial).points.max(axis=0) + .35
            mask = (np.ones(len(merged), dtype=bool) if kind == "field" else
                    np.all((merged.points >= lower) & (merged.points <= upper), axis=1))
            if kind == "scene":
                mask &= assignment == index
            target = PointCloudData(merged.points[mask])
            identified = identify_pose(model, target, orientation_seeds(kind, initial), backend,
                self.quality, minimum_points=self.profile["minimum_points"], **self.profile["ambiguity"])
            reasons = list(identified.reasons)
            if kind == "inhand" and not all(item["valid"] for item in openings):
                reasons.append("unvalidated_opening")
            if kind == "inhand" and (len(alignments) != len(frames) or
                                     not all(item["valid"] for item in alignments)):
                reasons.append("unvalidated_gripper_alignment")
            result = identified.result
            if result is not None and result.correspondences < 20:
                reasons.append("insufficient_correspondences")
            results.append({"object_id": object_id, "model_sha256": model_hash,
                "initial_pose": initial.tolist(), "pose": None if result is None else result.transform.tolist(),
                "valid": not reasons, "reasons": reasons, "target_points": len(target),
                "candidate_count": identified.candidate_count, "score": identified.score,
                "metrics": None if result is None else {"target_coverage": result.target_coverage,
                    "source_coverage": result.source_coverage,
                    "rmse": result.rmse if np.isfinite(result.rmse) else None,
                    "correspondences": result.correspondences}})
        proposal = {"schema_version": 1, "kind": kind, "scope": "desktop_replay",
            "frame": GRIPPER_FRAME if kind == "inhand" else "world",
            "capture_id": manifest["capture_id"], "product_id": manifest["product_id"],
            "source_manifest_sha256": manifest["sha256"], "calibration_sha256": crop.calibration_sha256,
            "crop": manifest["crop"],
            "scene_revision": expected_revision, "profile_sha256": digest(self.profile),
            "timestamp_sources": sorted({frame.timestamp_source for frame in frames}),
            "ground_height": ground_height, "opening": openings,
            "gripper_alignment": alignments,
            "objects": results, "valid": all(item["valid"] for item in results),
            "next_action": "review" if all(item["valid"] for item in results) else "recrop_or_reinitialize"}
        proposal["sha256"] = digest(proposal)
        return proposal, merged

    def align_gripper(self, cloud, opening):
        """Correct each view against stone_grab before removing its surface.

        A rejected correction is reported and never applied. Thresholds bound
        desktop replay corrections; they are not a calibration certificate.
        """
        backend = Open3dRegistrationBackend(MultiscaleRegistrationConfig(
            **self.profile["inhand"], correspondence_distance_scale=1.5,
            evaluation_distance=.03, minimum_points=self.profile["minimum_points"]))
        try:
            result = backend.register(cloud, self.models.gripper(opening), np.eye(4))
            decision = evaluate_registration(result, np.eye(4),
                RegistrationQualityConfig(**self.profile["gripper_alignment_quality"]))
        except ValueError:
            return cloud, {"valid": False, "reasons": ["insufficient_alignment_evidence"], "transform": None}
        return (cloud.transformed(result.transform) if decision.accepted else cloud), {
            "valid": decision.accepted, "reasons": list(decision.reasons),
            "transform": result.transform.tolist(), "rmse": result.rmse if np.isfinite(result.rmse) else None,
            "source_coverage": result.source_coverage, "target_coverage": result.target_coverage,
            "translation_m": decision.correction_translation,
            "rotation_degrees": decision.correction_rotation_degrees}
