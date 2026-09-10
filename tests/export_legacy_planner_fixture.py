"""Export trusted local planner records; never invoked by runtime code or tests.

Run with the project Python and a trusted stacking-planner checkout. Pickle is
used only at this explicit compatibility boundary; the fixtures are JSON/NPZ.
"""

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import yaml


def plain(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(key): plain(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [plain(item) for item in value]
    return value


def export(root, output):
    session = root / "sessions/260701/plan_260701_5"
    names = ("planning_params.pkl", "pose_data.pkl", "action_sequence.pkl",
             "motion_result_sequence.pkl")
    records = {name: pickle.loads((session / name).read_bytes()) for name in names}
    params = records["planning_params.pkl"]
    cases, arrays = [], {}
    for index, mode in ((38, "direct"), (39, "regrasp")):
        action = records["action_sequence.pkl"][index]
        motion = records["motion_result_sequence.pkl"][index]
        paths = motion["q_path_sequence"]
        assert len(paths) == (4 if mode == "direct" else 8)
        prefix = f"step_{index + 1}"
        for section, (joints, target) in enumerate(zip(paths, motion["target_path_sequence"])):
            arrays[f"{prefix}_joints_{section}"] = np.asarray(joints)
            arrays[f"{prefix}_target_{section}"] = np.asarray(target)
        cases.append({"step": index + 1, "mode": mode, "action": plain(action),
                      "section_count": len(paths), "grasp_sequence": plain(motion["grasp_sequence"]),
                      "score": plain(motion["scores"]), "feasible": bool(motion["is_feasible"]),
                      "failure_stage": motion["failure_stage"]})
    sources = [session / name for name in names]
    sources += [session / "config.yml", root / "agent/configs/config.yml",
                root / "planning/planning.py", root / "scripts/desktop/generate_sequence.py"]
    data = {
        "schema_version": 1,
        "source_session": str(session.relative_to(root)),
        "source_hashes": {str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
                          for path in sources},
        "production_config": yaml.safe_load((root / "agent/configs/config.yml").read_text()),
        "session_config": yaml.safe_load((session / "config.yml").read_text()),
        "planning": {key: plain(params[key]) for key in (
            "q_home", "target_structure_offset", "n_move", "n_grasp", "n_opening_angle",
            "regrasp_xy_candidates", "pick_plane_height", "place_plane_height", "place_z_offset")},
        "pickup_poses": plain(records["pose_data.pkl"]),
        "placed_prefix": plain(params["resume_scene_poses"]),
        "cases": cases,
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output / "motion.npz", **arrays)
    data["motion_sha256"] = hashlib.sha256((output / "motion.npz").read_bytes()).hexdigest()
    (output / "reference.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("legacy_root", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    export(args.legacy_root, args.output)
