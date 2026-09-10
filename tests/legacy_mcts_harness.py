"""Run small policy scenarios against the sibling legacy MCTS source.

This module deliberately executes in a subprocess. It stubs heavyweight legacy
environment imports, but loads the actual legacy ``mcts.py`` and ``node.py``
implementations for the policies exercised by the migration regressions.
"""

from __future__ import annotations

import importlib.util
import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

class _Config(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as error:
            raise AttributeError(name) from error

    def __setattr__(self, name, value):
        self[name] = value


class _Action:
    def __init__(self, stone_id: int):
        self.stone_id = int(stone_id)
        self.stone_idx = int(stone_id)


def _module(name: str, **values):
    module = types.ModuleType(name)
    module.__dict__.update(values)
    sys.modules[name] = module
    return module


def _package(name: str):
    module = _module(name)
    module.__path__ = []
    return module


def _identity_decorator(*args, **kwargs):
    del kwargs
    if args and callable(args[0]):
        return args[0]
    return lambda function: function


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load legacy module: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_stubs() -> None:
    _package("agent")
    _package("agent.env")
    _package("agent.env.components")
    _package("agent.mcts")
    _package("loader")
    _package("utils")

    root_policy = SimpleNamespace(root_keep=lambda fallback: int(fallback))
    _module(
        "agent.config_views",
        mcts_action_generation_config=lambda cfg: SimpleNamespace(
            attempt_multiplier=float(cfg.get("attempt_multiplier", 1.0))
        ),
        mcts_root_proposal_config=lambda cfg: root_policy,
        mcts_validation_config=lambda cfg: SimpleNamespace(max_candidates=None),
        mcts_sampling_config=lambda cfg: SimpleNamespace(),
        support_config=lambda cfg: SimpleNamespace(),
    )
    _module(
        "agent.env.components.action",
        Action=_Action,
        ActionList=list,
    )
    _module(
        "agent.env.components.state",
        State=object,
        Observation=object,
    )
    _module(
        "agent.env.components.action.support",
        ground_placement_connected_ok=lambda *args, **kwargs: True,
    )
    _module(
        "loader.heightmap_feasibility",
        stone_heightmaps_from_mesh=lambda *args, **kwargs: (),
    )
    _module("omegaconf", OmegaConf=_Config)
    _module("gymnasium", Env=object)
    _module(
        "torch",
        nn=SimpleNamespace(Module=object),
        no_grad=lambda: _NullContext(),
    )
    _module(
        "utils._phase_timer",
        phase=_identity_decorator,
        timed=_identity_decorator,
        reset=lambda: None,
        format_summary=lambda: "",
    )
    _module("agent.mcts.cem", CEMRootMixin=type("CEMRootMixin", (), {}))
    _module(
        "agent.mcts.root_lookahead",
        RootLookaheadMixin=type("RootLookaheadMixin", (), {}),
    )
    _module(
        "agent.mcts.utils",
        best_sequence_from_node=lambda *args, **kwargs: [],
        place_plane_gap_failure_reason=lambda *args, **kwargs: None,
        final_support_ok=lambda *args, **kwargs: True,
        place_scene_gap_failure_reason=lambda *args, **kwargs: None,
        refined_action_from_state=lambda *args, **kwargs: None,
        settled_action_pose=lambda *args, **kwargs: None,
        target_containment_failure_reason=lambda *args, **kwargs: None,
    )


class _NullContext:
    def __enter__(self):
        return self

    def __exit__(self, *args):
        del args
        return False


def _config() -> _Config:
    return _Config(
        max_depth=3,
        initial_depth1_steps=3,
        max_children_num=3,
        max_first_children_num=3,
        attempt_multiplier=1.0,
        widening=_Config(enabled=False),
        confidence=1.0,
        reward=_Config(
            discount=0.5,
            mean=2.5,
            value_estimate=_Config(horizon=4),
        ),
    )


def _state(placed_count: int):
    return SimpleNamespace(
        stone_set=[0, 1, 2, 3],
        stone_seq=list(range(placed_count)),
        failed=False,
    )


def _evaluated(node, *, reward: float, placed_count: int, done=False) -> None:
    node.update_state(
        _state(placed_count),
        object(),
        reward,
        done,
        False,
        {},
    )
    node.set_is_simulated(True)


def run(legacy_root: Path) -> dict[str, object]:
    _install_stubs()
    node_module = _load(
        "agent.mcts.node",
        legacy_root / "agent" / "mcts" / "node.py",
    )
    mcts_module = _load(
        "agent.mcts.mcts",
        legacy_root / "agent" / "mcts" / "mcts.py",
    )
    config = _config()
    coordinator = object.__new__(mcts_module.MonteCarloTreeSearch)
    coordinator.cfg = config

    depth = {}
    for placed_count in (0, 2, 3):
        root = SimpleNamespace(state=_state(placed_count))
        depth[str(placed_count)] = coordinator._effective_max_depth(root, 3)

    selection_root = node_module.MCTS_Node(config)
    _evaluated(selection_root, reward=0.0, placed_count=0)
    selection_root.visits = 16
    selection_root.set_action_generation_exhausted(True)
    selection_children = []
    for stone_id, prior, value in ((101, -2.0, 3.0), (102, 2.0, 0.0)):
        child = node_module.MCTS_Node(
            config,
            action=_Action(stone_id),
            q_value_init=prior,
            parent=selection_root,
            depth=1,
        )
        _evaluated(child, reward=0.0, placed_count=1)
        child.visit(value)
        child.set_action_generation_exhausted(True)
        selection_children.append(child)
    selection_root.children = selection_children
    selected = selection_root.select_child(4.0, epsilon=0.0)

    root = node_module.MCTS_Node(config)
    _evaluated(root, reward=0.0, placed_count=0)
    child = node_module.MCTS_Node(
        config,
        action=_Action(10),
        parent=root,
        depth=1,
    )
    _evaluated(child, reward=2.0, placed_count=1)
    leaf = node_module.MCTS_Node(
        config,
        action=_Action(20),
        parent=child,
        depth=2,
    )
    _evaluated(leaf, reward=3.0, placed_count=2)
    leaf.value_init = 4.0
    root.children = [child]
    child.children = [leaf]
    coordinator._backpropagate(leaf)

    return {
        "effective_depth": depth,
        "puct_selected_id": selected.action.stone_id,
        "backup": {
            "root": [root.visits, root.q_value],
            "child": [child.visits, child.q_value],
            "leaf": [leaf.visits, leaf.q_value],
        },
    }


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: legacy_mcts_harness.py LEGACY_REPOSITORY")
    print(json.dumps(run(Path(sys.argv[1]).resolve()), sort_keys=True))


if __name__ == "__main__":
    main()
