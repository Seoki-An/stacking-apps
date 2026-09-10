from dataclasses import dataclass, field

import numpy as np

from stacking_common.domain import Action, State
from stacking_common.environment import StackingEnvironment, StepResult


def state(*, terminated: bool = False, failed: bool = False) -> State:
    return State([], [], {}, [], [], terminated=terminated, failed=failed)


def action() -> Action:
    pose = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0])
    return Action(0, pose, pose.copy())


@dataclass
class FakeTransition:
    current: State = field(default_factory=state)
    applied_actions: list[Action] = field(default_factory=list)
    closed: bool = False

    def reset(self) -> State:
        self.current = state()
        return self.current

    def step(self, candidate: Action) -> State:
        self.applied_actions.append(candidate)
        self.current = state(terminated=True)
        return self.current

    def restore(self, restored: State) -> None:
        self.current = restored

    def snapshot(self) -> State:
        return self.current

    def close(self) -> None:
        self.closed = True


class FakeObservationBuilder:
    def build(self, current: State) -> tuple[bool, bool]:
        return current.terminated, current.failed


class FakeRewardEvaluator:
    def compute(self, current: State) -> tuple[float, dict[str, object]]:
        return 2.5, {"terminated": current.terminated}


def environment() -> tuple[
    StackingEnvironment[tuple[bool, bool]], FakeTransition
]:
    transition = FakeTransition()
    return (
        StackingEnvironment(
            transition,
            FakeObservationBuilder(),
            FakeRewardEvaluator(),
        ),
        transition,
    )


def test_reset_returns_state_and_observation() -> None:
    env, _ = environment()

    result = env.reset()

    assert result.state == state()
    assert result.observation == (False, False)


def test_step_returns_explicit_transition_result() -> None:
    env, transition = environment()
    candidate = action()

    result = env.step(candidate)

    assert transition.applied_actions == [candidate]
    assert result.observation == (True, False)
    assert result.reward == 2.5
    assert result.info == {"terminated": True}
    assert result.done is True


def test_failed_state_is_done() -> None:
    env, transition = environment()
    transition.current = state(failed=True)

    result = StepResult(
        state=transition.snapshot(),
        observation=env.observation(),
        reward=0.0,
    )

    assert result.observation == (False, True)
    assert result.done is True


def test_restore_snapshot_and_close_are_delegated() -> None:
    env, transition = environment()
    restored = state(failed=True)

    env.restore(restored)

    assert env.snapshot() is restored
    env.close()
    assert transition.closed is True
