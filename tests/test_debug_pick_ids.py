"""Missing pickup IDs are a CLI warning, never fabricated pickup poses."""

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from stacking_common.planning import PickSceneSnapshot


ROOT = Path(__file__).resolve().parents[1]


def _runner(monkeypatch, requested):
    spec = importlib.util.spec_from_file_location("debug_pick_ids", ROOT / "apps/tabletop/scripts/debug/mcts.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(sys, "argv", [
        "mcts.py", "--config", str(ROOT / "apps/tabletop/config/mcts_cpu.yml"),
        "--pick-pose-data", "unused.pkl", "--object-ids", *map(str, requested),
    ])
    scene = PickSceneSnapshot("world", {
        3: [0.1, -0.7, 0.12, 0, 0, 0, 1],
        4: [0.2, -0.7, 0.12, 0, 0, 0, 1],
    }, "recorded.pkl")
    monkeypatch.setattr(module, "load_legacy_pick_scene", lambda *args, **kwargs: scene)
    return module, scene


def test_missing_requested_ids_are_skipped_before_inventory_build(monkeypatch, capsys):
    module, scene = _runner(monkeypatch, [4, 1, 3, 2])
    monkeypatch.setattr(module, "infer_legacy_pick_support", lambda *args: SimpleNamespace(height=0.1, bottom_by_object={}))

    class InventoryReached(Exception):
        pass

    def load_inventory(self, object_ids, initial_poses):
        assert object_ids == (4, 3)
        assert tuple(initial_poses) == (4, 3)
        np.testing.assert_array_equal(initial_poses[4], scene.pose(4))
        raise InventoryReached

    monkeypatch.setattr(module.TabletopApplication, "load_inventory", load_inventory)
    with pytest.raises(InventoryReached):
        module.main()
    assert "[WARN] Skipping requested object IDs absent from pick-pose-data: [1, 2]" in capsys.readouterr().out


def test_no_matching_requested_ids_exit_without_initializing_planning(monkeypatch, capsys):
    module, _ = _runner(monkeypatch, [1, 2])

    def unexpected(*args, **kwargs):
        pytest.fail("no-match run must stop before support inference or native setup")

    monkeypatch.setattr(module, "infer_legacy_pick_support", unexpected)
    monkeypatch.setattr(module.TabletopApplication, "load_inventory", unexpected)
    assert module.main() is None
    output = capsys.readouterr().out
    assert "[1, 2]" in output
    assert "nothing to plan" in output


def test_common_pick_scene_selection_stays_strict():
    scene = PickSceneSnapshot("world", {3: [0, 0, 0, 0, 0, 0, 1]}, "recorded")
    with pytest.raises(ValueError, match="does not contain"):
        scene.select((1, 3))


def test_planning_log_reports_cem_even_when_zero_candidates_survive(monkeypatch, capsys):
    from stacking_common.actions.cem import CEMStatistics, CEMRoundStatistics
    from stacking_common.workflows.planning_sequence import SequenceSearchAttempt
    module, _ = _runner(monkeypatch, [3])
    cem = CEMStatistics(rounds=(CEMRoundStatistics(0, 96, 0, 96, 0, 0, fresh_attempts=96),),
        rejection_counts={"active_floor_above_active_layer": 96},
        next_fresh_offset=96, fresh_exhausted=False)
    attempt = SequenceSearchAttempt(1, 1.5, 200, 0, 0, 0, None, (cem,), 3,
                                    tree_nodes=1, tree_leaves=1)
    step = SimpleNamespace(step=4, search_attempts=(attempt,), planning_attempts=(), elapsed_seconds=1.5)
    module._print_sequence_step(step)
    output = capsys.readouterr().out
    assert "attempts=96, accepted=0, rejected=96" in output
    assert "active_floor_above_active_layer=96" in output
    assert "fresh_exhausted=False" in output
    assert "tree_nodes=1, leaves=1" in output
    reporter = module._ProgressReporter()
    reporter.update("search_completed", dict(step=4, elapsed_seconds=1.5,
        generated_candidates=0, evaluated_candidates=0, ranked_candidates=0,
        cem_statistics=(cem.to_dict(),)))
    assert "CEM rejection reasons" in capsys.readouterr().out


def test_planning_log_distinguishes_execution_filters_and_candidate_skips(monkeypatch, capsys):
    from stacking_common.actions.cem import CEMStatistics
    module, _ = _runner(monkeypatch, [3])
    statistics = CEMStatistics(rejection_counts={"execution_rejected_action": 4})
    module._print_cem_statistics(statistics.to_dict())
    assert "execution_filtered_before_keep=4" in capsys.readouterr().out
    reporter = module._ProgressReporter()
    reporter.update("candidate_skipped", dict(
        step=4, candidate_rank=3, candidate_count=7,
        candidate_id="step-4-candidate-3", reason="simulation_unsettled",
    ))
    assert "candidate 3/7 skipped (id=step-4-candidate-3, reason=simulation_unsettled)" in capsys.readouterr().out
