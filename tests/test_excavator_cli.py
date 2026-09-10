"""CLI selection tolerates missing poses without changing the requested inventory."""

import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from stacking_common.planning import PickSceneSnapshot
from stacking_common.artifacts import StackRunStore
from stacking_common.search import MonteCarloTreeSearch, MCTSSearchResult
from stacking_excavator import ExcavatorApplication
from stacking_excavator.planning.cli import main, select_pick_scene, search_runtime, resolve_pose_from_scan


def scene():
    return PickSceneSnapshot("world", {
        55: [1., 0., .4, 0., 0., 0., 1.],
        50: [2., 0., .4, 0., 0., 0., 1.],
    }, "recorded pickup", {"sha256": "source hash"})


def test_invalid_grasp_profile_stops_before_search_or_run_creation(tmp_path, monkeypatch, capsys):
    import yaml

    root = Path(__file__).resolve().parents[1]
    application = ExcavatorApplication.from_repository(root)
    grasp = yaml.safe_load(application.solver_profiles.grasp.read_text())
    grasp["trust_region"].update(radius_init=1.0, radius_max=0.1)
    invalid = tmp_path / "grasp.yml"
    invalid.write_text(yaml.safe_dump(grasp))
    application = replace(application, solver_profiles=replace(
        application.solver_profiles, grasp=invalid))
    monkeypatch.setattr(ExcavatorApplication, "from_repository", lambda *a, **k: application)
    monkeypatch.setattr(ExcavatorApplication, "build_mcts",
                        lambda *a, **k: pytest.fail("MCTS must not start with invalid grasp settings"))
    pickup = tmp_path / "pick.json"
    pickup.write_text(json.dumps({str(k): v.tolist() for k, v in scene().object_poses.items()}))
    output = tmp_path / "invalid.stack"
    result = main(["--worker", "--repository", str(root), "--pick-pose-data", str(pickup),
                   "--save-run", str(output)])
    assert result == 2
    assert not output.exists()
    assert "radius_init=1.0, radius_max=0.1" in capsys.readouterr().err


def test_missing_ids_warn_and_preserve_present_selection_order(capsys):
    source = scene()
    selected = select_pick_scene(source, [50, 56, 55])
    assert selected.object_ids == (50, 55)
    assert selected.source == source.source
    assert selected.metadata == source.metadata
    assert source.object_ids == (55, 50)
    assert "[WARN]" in (warning := capsys.readouterr().err)
    assert "[56]" in warning


def test_omitted_selection_uses_scene_without_warning(capsys):
    source = scene()
    assert select_pick_scene(source, None) is source
    assert capsys.readouterr().err == ""


def test_matching_selection_has_no_warning(capsys):
    assert select_pick_scene(scene(), [50]).object_ids == (50,)
    assert capsys.readouterr().err == ""


def test_duplicate_selection_remains_invalid():
    with pytest.raises(ValueError, match="unique"):
        select_pick_scene(scene(), [55, 55])


def test_no_matching_ids_warn_and_exit_without_creating_run(tmp_path, capsys):
    pickup = tmp_path / "pick.json"
    pickup.write_text(json.dumps({str(k): v.tolist() for k, v in scene().object_poses.items()}))
    output = tmp_path / "empty.stack"
    result = main(["--worker", "--repository", str(Path(__file__).resolve().parents[1]),
                   "--pick-pose-data", str(pickup), "--object-ids", "56", "58", "8",
                   "--search-only", "--save-run", str(output)])
    assert result == 0
    assert not output.exists()
    warning = capsys.readouterr().err
    assert "[56, 58, 8]" in warning
    assert "nothing to plan" in warning
    assert "planning error" not in warning


@pytest.mark.parametrize("search_only,scan,object_ids,message", [
    (False, False, [], "motion planning requires --pick-pose-data"),
    (True, True, [], "pose_from_scan requires --pick-pose-data"),
    (True, False, [55, 55], "selected catalog object IDs must be unique"),
    (True, False, [9999], "objects outside the excavator catalog"),
])
def test_missing_pickup_guards(tmp_path, monkeypatch, capsys, search_only, scan, object_ids, message):
    root = Path(__file__).resolve().parents[1]
    application = replace(ExcavatorApplication.from_repository(root), pose_from_scan=scan)
    monkeypatch.setattr(ExcavatorApplication, "from_repository", lambda *a, **k: application)
    output = tmp_path / "run.stack"
    args = ["--worker", "--repository", str(root), "--save-run", str(output)]
    if search_only:
        args.append("--search-only")
    if object_ids:
        args += ["--object-ids", *map(str, object_ids)]
    assert main(args) == 2
    assert message in capsys.readouterr().err
    assert not output.exists()


def test_search_without_pickup_defaults_to_entire_catalog(tmp_path, monkeypatch):
    root_path = Path(__file__).resolve().parents[1]
    expected_ids = ExcavatorApplication.from_repository(root_path).inventory_profile.object_ids
    searched = []

    def search(self, root, request):
        searched.append(root.state.object_ids)
        assert root.state.placed_object_indices == []
        return MCTSSearchResult(root=root, selected=None, completed_iterations=0,
            expansion_calls=0, generated_candidates=0, evaluated_candidates=0,
            failed_evaluations=0, effective_max_depth=1, timed_out=False)

    monkeypatch.setattr(MonteCarloTreeSearch, "search", search)
    output = tmp_path / "catalog.stack"
    assert main(["--worker", "--repository", str(root_path), "--search-only",
                 "--save-run", str(output)]) == 1  # Controlled search exhaustion.
    assert len(searched) == 1
    assert tuple(searched[0]) == expected_ids
    context, = StackRunStore.load_run_contexts(output)
    assert context.profiles["pick_scene"]["format"] == "synthetic_search_inventory"
    assert len(context.profiles["pick_scene"]["object_poses"]) == len(expected_ids)


@pytest.mark.parametrize("name,value", [("seed", 1), ("iterations", 201), ("max_search_time", 241.)])
def test_resume_rejects_changed_search_controls(name, value):
    recorded = SimpleNamespace(profiles={
        "runtime": {"seed": 0, "iterations": None, "max_search_time": None, "search_only": False},
        "mcts_profile": {"request": {"iterations": 200, "max_time_seconds": 240.}},
    })
    args = SimpleNamespace(seed=None, iterations=None, max_search_time=None)
    setattr(args, name, value)
    with pytest.raises(ValueError, match="resume requires recorded --" + name.replace("_", "-")):
        search_runtime(args, recorded)


def test_resume_repeated_effective_values_preserve_null_overrides_and_context():
    runtime = {"seed": 0, "iterations": None, "max_search_time": None, "search_only": False}
    recorded = SimpleNamespace(profiles={"runtime": runtime,
        "mcts_profile": {"request": {"iterations": 200, "max_time_seconds": 240.}}})
    args = SimpleNamespace(seed=0, iterations=200, max_search_time=240.)
    assert search_runtime(args, recorded) == runtime
    assert search_runtime(SimpleNamespace(seed=None, iterations=None, max_search_time=None), recorded) == runtime


def test_new_run_search_controls_are_overrides():
    assert search_runtime(SimpleNamespace(seed=3, iterations=10, max_search_time=120., search_only=True)) == {
        "seed": 3, "iterations": 10, "max_search_time": 120., "search_only": True}


@pytest.mark.parametrize("enabled", [False, True])
def test_scan_override_is_used_logged_and_saved(tmp_path, monkeypatch, capsys, enabled):
    root = Path(__file__).resolve().parents[1]
    application = replace(ExcavatorApplication.from_repository(root), pose_from_scan=not enabled)
    monkeypatch.setattr(ExcavatorApplication, "from_repository", lambda *a, **k: application)
    original = ExcavatorApplication.build_mcts
    def build(self, *args, **kwargs):
        assert self.pose_from_scan is enabled
        return original(self, *args, **kwargs)
    monkeypatch.setattr(ExcavatorApplication, "build_mcts", build)
    monkeypatch.setattr(MonteCarloTreeSearch, "search", lambda self, root, request:
        MCTSSearchResult(root=root, selected=None, completed_iterations=0,
            expansion_calls=0, generated_candidates=0, evaluated_candidates=0,
            failed_evaluations=0, effective_max_depth=1, timed_out=False))
    pickup = tmp_path / "pick.json"
    pickup.write_text(json.dumps({str(k): v.tolist() for k, v in scene().object_poses.items()}))
    output = tmp_path / "run.stack"
    flag = "--pose-from-scan" if enabled else "--no-pose-from-scan"
    assert main(["--worker", "--repository", str(root), "--search-only", flag,
                 "--pick-pose-data", str(pickup), "--save-run", str(output)]) == 1
    context, = StackRunStore.load_run_contexts(output)
    assert context.profiles["pose_from_scan"] is enabled
    assert f"pose_from_scan: {str(enabled).lower()} (CLI override)" in capsys.readouterr().out


@pytest.mark.parametrize("recorded", [False, True])
def test_scan_resume_preserves_recorded_mode(recorded):
    assert resolve_pose_from_scan(not recorded, None, recorded) is recorded
    assert resolve_pose_from_scan(not recorded, recorded, recorded) is recorded
    with pytest.raises(ValueError, match="resume preserves recorded pose_from_scan"):
        resolve_pose_from_scan(recorded, not recorded, recorded)


def test_scan_override_requires_recorded_pickup_even_for_search(tmp_path, capsys):
    assert main(["--worker", "--search-only", "--pose-from-scan",
                 "--save-run", str(tmp_path / "run.stack")]) == 2
    assert "pose_from_scan requires --pick-pose-data" in capsys.readouterr().err
