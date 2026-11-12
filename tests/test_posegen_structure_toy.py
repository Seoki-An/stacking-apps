"""Structure equilibrium must reject placements that only balance the candidate.

See tests/posegen_structure_toy.py for the scene and the full sweep.
"""

import pytest

pytest.importorskip("stacking_core")

from posegen_structure_toy import PLANK, StructureToy, tilt_degrees


@pytest.fixture(scope="module")
def toy():
    return StructureToy()


def test_base_stack_stands_without_candidate(toy):
    assert toy.placed[PLANK][0] == pytest.approx(0.0, abs=1e-3)
    assert tilt_degrees(toy.placed[PLANK]) < 1.0


@pytest.mark.parametrize("x0", [-0.35, 0.35])
def test_single_stone_equilibrium_accepts_overhang_that_topples(toy, x0):
    row = toy.place("single", x0)

    assert row["accepted"]
    assert not row["stands"]


@pytest.mark.parametrize("x0", [-0.35, 0.35])
def test_structure_equilibrium_moves_candidate_over_support(toy, x0):
    row = toy.place("structure", x0)

    assert abs(row["solved"].pose[0]) < 0.1
    assert row["accepted"]
    assert row["stands"]


@pytest.mark.parametrize("mode", ["single", "structure"])
def test_equilibrium_metrics_do_not_depend_on_mass_units(toy, mode):
    heavy = StructureToy(mass_factor=50.0).place(mode, 0.35)["solved"]
    light = toy.place(mode, 0.35)["solved"]

    assert heavy.pose == pytest.approx(light.pose, abs=1e-3)
    assert heavy.c_feq == pytest.approx(light.c_feq, rel=1e-2)
    key = "posegen_equilibrium_residual_normalized"
    assert heavy.diagnostics[key] == pytest.approx(light.diagnostics[key], rel=1e-2)
