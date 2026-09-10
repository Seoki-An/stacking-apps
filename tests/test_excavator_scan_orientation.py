"""Scan-mode proposals and support footprints captured from legacy excavator."""

from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from stacking_common.actions.types import PlacementSolveResult
from stacking_common.domain import State
from stacking_excavator import ExcavatorApplication

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests/fixtures/legacy_planner/scan_sampling.npz"
META = json.loads(FIXTURE.with_suffix(".json").read_text())


@pytest.mark.parametrize("label", ["empty", "placed"])
def test_scan_support_maps_and_seeded_placements_match_legacy(label):
    application = replace(ExcavatorApplication.from_repository(ROOT), pose_from_scan=True)
    inventory = application.load_inventory(META["object_ids"], {
        object_id: np.array(META["pick_pose"]) for object_id in META["object_ids"]})
    placed_index = META["placed_index"]
    state = State(META["object_ids"], [placed_index] if label == "placed" else [],
        {META["object_ids"][placed_index]: np.array(META["placed_pose"])}
        if label == "placed" else {}, [], [])
    solver_inputs = []

    class Solver:
        def solve(self, inventory, state, object_index, initial):
            solver_inputs.append(initial.pose.copy())
            return PlacementSolveResult(initial.pose, 0., 0.)

    generator = application.build_action_generator(inventory, Solver())
    sampler = generator.planar_sampler
    sampler.rng.seed(META["seed"])
    index = META["candidate_index"]
    with np.load(FIXTURE) as recorded:
        observation = SimpleNamespace(scene_height_map=recorded[label + "_height"],
            height_map_xlim=np.array(META["xlim"]), height_map_ylim=np.array(META["ylim"]))
        rotations = Rotation.from_quat(sampler.quaternions(state, observation, index))
        expected_rotations = Rotation.from_quat(recorded["quaternions"])
        np.testing.assert_allclose(rotations.as_matrix(), expected_rotations.as_matrix(), atol=1e-12)

        scores = sampler.score_map(state, observation, index)
        for key in ("scores", "candidate_mask", "placement_height", "support_likelihood", "orientation_index"):
            np.testing.assert_allclose(scores[key], recorded[label + "_" + key],
                atol=1e-12, rtol=0., equal_nan=True)
        kernels, = sampler._kernels.values()
        for orientation_index, kernel in enumerate(kernels):
            for key in ("y_offsets", "x_offsets", "bottom_height"):
                np.testing.assert_allclose(getattr(kernel, key),
                    recorded[f"kernel_{orientation_index}_{key}"], atol=1e-12)

        proposals = sampler.sample(inventory, state, observation, index, 32, "lower_floor_first")
        np.testing.assert_allclose([proposal.pose for proposal in proposals],
            recorded[label + "_poses"], atol=1e-12)
        # Height initialization and the builder must not compose the scan again.
        for proposal in proposals:
            candidate, reason = generator.build_candidate(state, index, proposal)
            assert reason is None
            assert candidate is not None
            np.testing.assert_allclose(candidate.action.init_pose[3:], proposal.pose[3:], atol=1e-12)
            np.testing.assert_allclose(candidate.action.pose[3:], proposal.pose[3:], atol=1e-12)
        np.testing.assert_allclose(np.asarray(solver_inputs)[:, 3:],
            recorded[label + "_poses"][:, 3:], atol=1e-12)
