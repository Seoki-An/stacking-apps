"""Compute selection belongs to MCTS, independently of solver physics."""

from dataclasses import replace
from pathlib import Path

import pytest
import yaml

from stacking_common.artifacts.planning_context import _values
from stacking_common.profiles.mcts import load_mcts_profile
from stacking_excavator import ExcavatorApplication

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "apps/excavator/config"


@pytest.mark.parametrize("posegen", ["cpu", "cuda"])
@pytest.mark.parametrize("simulation", ["cpu", "cuda"])
def test_mcts_compute_selection_round_trip(tmp_path, posegen, simulation):
    values = yaml.safe_load((CONFIG / "mcts_cuda.yml").read_text())
    values["parallel"].update(posegen_backend=posegen, simulation_backend=simulation)
    path = tmp_path / "mcts.yml"
    path.write_text(yaml.safe_dump(values))
    application = ExcavatorApplication.from_repository(ROOT, mcts_config=path)
    parallel = application.mcts_profile.parallel
    assert parallel.posegen_backend == posegen
    assert parallel.simulation_backend == simulation
    assert not hasattr(application.posegen_profile, "backend")
    assert not hasattr(application.simulation_profile, "backend")
    for key, value in (("posegen_backend", posegen), ("simulation_backend", simulation)):
        assert _values(parallel).get(key, "cpu") == value
        with pytest.raises(ValueError, match=key):
            replace(parallel, **{key: "gpu_typo"})


def test_older_mcts_profiles_default_to_cpu(tmp_path):
    values = yaml.safe_load((CONFIG / "mcts_cuda.yml").read_text())
    values["parallel"].pop("posegen_backend")
    values["parallel"].pop("simulation_backend")
    path = tmp_path / "mcts.yml"
    path.write_text(yaml.safe_dump(values))
    parallel = load_mcts_profile(path).parallel
    assert parallel.posegen_backend == parallel.simulation_backend == "cpu"
    assert "posegen_backend" not in _values(parallel)
    assert "simulation_backend" not in _values(parallel)


def test_explicit_cpu_cuda_profiles_select_their_backends():
    cpu = load_mcts_profile(CONFIG / "mcts_cpu.yml")
    cuda = load_mcts_profile(CONFIG / "mcts_cuda.yml")
    assert cpu.parallel.posegen_backend == cpu.parallel.simulation_backend == "cpu"
    assert cuda.parallel.posegen_backend == cuda.parallel.simulation_backend == "cuda"


def test_excavator_default_uses_cpu_profile():
    application = ExcavatorApplication.from_repository(ROOT)
    assert application.mcts_profile == load_mcts_profile(CONFIG / "mcts_cpu.yml")


@pytest.mark.parametrize('field', ['batch', 'bodies', 'geometries', 'pairs', 'contacts', 'incidences'])
def test_posegen_capacity_fields_are_loaded_recorded_and_validated(tmp_path, field):
    values = yaml.safe_load((CONFIG / 'mcts_cuda.yml').read_text())
    values['parallel']['posegen_capacity'][field] = 777
    path = tmp_path / 'mcts.yml'
    path.write_text(yaml.safe_dump(values))
    profile = load_mcts_profile(path)
    assert getattr(profile.parallel.posegen_capacity, field) == 777
    assert _values(profile.parallel)['posegen_capacity'][field] == 777
    for invalid in (0, -1, 1.5, True):
        values['parallel']['posegen_capacity'][field] = invalid
        path.write_text(yaml.safe_dump(values))
        with pytest.raises(ValueError, match=field):
            load_mcts_profile(path)


@pytest.mark.parametrize("key", ["validate_descendant_path", "validate_descendant_depth",
                                "require_validated_descendant", "mode"])
def test_removed_validation_flags_fail_with_migration_message(tmp_path, key):
    values = yaml.safe_load((CONFIG / "mcts_cuda.yml").read_text())
    values["final_selection"][key] = 1
    path = tmp_path / "mcts.yml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="descendant_depth"):
        load_mcts_profile(path)


def test_new_profiles_have_one_full_simulation_after_settling():
    from stacking_common.profiles.simulation import load_simulation_profile
    from stacking_common.search import EvaluationFidelity
    for platform in ("tabletop", "excavator"):
        config = ROOT / "apps" / platform / "config"
        profile = load_simulation_profile(config / "solvers/simulation.yml")
        assert profile.placement.short_gate is None
        assert profile.placement.settle_steps == 400
        assert profile.placement.structural.freeze_radius is None
        assert profile.placement.structural.min_time == 1.0
        assert profile.placement.structural.max_time == 10.0
        assert profile.placement.structural.final_extra_steps == 200
        for backend in ("cpu", "cuda"):
            mcts = load_mcts_profile(config / f"mcts_{backend}.yml")
            assert mcts.request.evaluation_fidelity is EvaluationFidelity.LONG


def test_short_structural_runtime_configuration_is_not_silently_ignored(tmp_path):
    from stacking_common.profiles.simulation import load_simulation_profile
    values = yaml.safe_load((CONFIG / "solvers/simulation.yml").read_text())
    values["phases"]["short_structural"] = dict(values["phases"]["structural"])
    path = tmp_path / "simulation.yml"
    path.write_text(yaml.safe_dump(values))
    with pytest.raises(ValueError, match="short_structural was removed"):
        load_simulation_profile(path)
