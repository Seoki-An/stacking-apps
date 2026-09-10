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
        reward, info = self.reward_evaluator.compute(state)
        return StepResult(
            state=state,
            observation=self.observation_builder.build(state),
            reward=float(reward),
            info=dict(info),
        )

    def restore(self, state: State) -> None:
        self.transition.restore(state)

    def snapshot(self) -> State:
        return self.transition.snapshot()

    def observation(self) -> observation_t:
        return self.observation_builder.build(self.snapshot())

    def close(self) -> None:
        self.transition.close()
