"""Composition of state transition, reward, and observation services."""

from typing import Generic, TypeVar

from stacking_common.domain import Action, State

from .contracts import ObservationBuilder, RewardEvaluator, TransitionModel
from .result import ResetResult, StepResult


observation_t = TypeVar("observation_t")


class StackingEnvironment(Generic[observation_t]):
    """Run stacking episodes without depending on Gym or a simulator API."""

    def __init__(
        self,
        transition: TransitionModel,
        observation_builder: ObservationBuilder[observation_t],
        reward_evaluator: RewardEvaluator,
    ) -> None:
        self.transition = transition
        self.observation_builder = observation_builder
        self.reward_evaluator = reward_evaluator

    def reset(self) -> ResetResult[observation_t]:
        state = self.transition.reset()
        return ResetResult(
            state=state,
            observation=self.observation_builder.build(state),
        )

    def step(self, action: Action) -> StepResult[observation_t]:
        state = self.transition.step(action)
        return self._step_result(state)

    def _step_result(self, state):
        reward, info = self.reward_evaluator.compute(state)
        return StepResult(
            state=state,
            observation=self.observation_builder.build(state),
            reward=float(reward),
            info=dict(info),
        )

    def step_batch(self, states, actions):
        """Batch physics where supported; retain the usual reward/observation path."""
        if len(states) != len(actions):
            raise ValueError("environment batch states and actions must align")
        backend = getattr(self.transition, "backend", None)
        if callable(getattr(backend, "place_batch", None)):
            return tuple(self._step_result(state) for state in self.transition.step_batch(states, actions))
        results = []
        for state, action in zip(states, actions):
            self.restore(state)
            results.append(self.step(action))
        return tuple(results)

    def restore(self, state: State) -> None:
        self.transition.restore(state)

    def snapshot(self) -> State:
        return self.transition.snapshot()

    def observation(self) -> observation_t:
        return self.observation_builder.build(self.snapshot())

    def close(self) -> None:
        self.transition.close()
