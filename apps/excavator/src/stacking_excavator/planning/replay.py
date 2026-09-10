"""Recorded excavator profile and asset verification for replay consumers."""
import hashlib
from pathlib import Path
import xml.etree.ElementTree as ET
from stacking_common.planning import FrameTransform
from stacking_common.profiles.planning import (
    PlanningProfile, RobotPlanningProfile, GripperPlanningProfile, PlanningObstacle,
)


def recorded_planning_profile(context):
    values = dict(context.profiles["planning_profile"])
    return PlanningProfile(**{**values,
        "world_T_stack": FrameTransform(**values["world_T_stack"]),
        "robot": RobotPlanningProfile(**values["robot"]),
        "gripper": GripperPlanningProfile(**values["gripper"]),
        "obstacles": tuple(PlanningObstacle(**item) for item in values["obstacles"])})



def verify_replay_assets(repository, context, planning):
    """Replay uses recorded geometry; current solver tuning is irrelevant."""
    root = Path(repository)
    paths = {}
    for role, relative in (("robot", planning.robot.urdf), ("gripper", planning.gripper.urdf)):
        urdf = root / relative
        paths[f"{role}/urdf"] = urdf
        for element in ET.parse(urdf).iter():
            filename = element.get("filename")
            if filename is not None:
                path = root / filename
                paths[f"{role}/{filename}"] = path if path.is_file() else urdf.parent / filename
    directory = root / context.profiles["inventory_profile"]["model_directory"]
    for key in context.asset_hashes:
        if key.startswith("object/"):
            paths[key] = directory / f"model_{int(key.split('/')[1])}.obj"
        elif key.startswith("object_surface/"):
            paths[key] = directory / f"model_{int(key.split('/')[1])}_mesh.ply"
    for key, path in paths.items():
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if context.asset_hashes.get(key) != digest:
            raise ValueError(f"Recorded replay asset differs: {key}")
    return {int(key.split('/')[1]): path for key, path in paths.items() if key.startswith("object/")}

