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
