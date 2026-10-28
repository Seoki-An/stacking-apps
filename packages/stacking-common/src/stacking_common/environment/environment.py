"""Composition of state transition, reward, and observation services."""

from concurrent.futures import ThreadPoolExecutor, wait
from collections import deque
from collections.abc import Sequence
from typing import Generic, TypeVar

from stacking_common.domain import Action, State

from .contracts import ObservationBuilder, RewardEvaluator, TransitionModel
from .result import ResetResult, StepResult


observation_t = TypeVar("observation_t")


class StackingEnvironment(Generic[observation_t]):
    """Run stacking episodes without depending on Gym or a simulator API."""

    # CPU work granularity is independent of the backend's placement batches.
    result_batch_size = 32

    def __init__(
        self,
        transition: TransitionModel,
        observation_builder: ObservationBuilder[observation_t],
        reward_evaluator: RewardEvaluator,
        *,
        defer_observations: bool = False,
        worker_evaluators: Sequence[tuple[ObservationBuilder[observation_t], RewardEvaluator]] = (),
    ) -> None:
        self.defer_observations = defer_observations
        self.transition = transition
        self.observation_builder = observation_builder
        self.reward_evaluator = reward_evaluator
        self._evaluators = ((observation_builder, reward_evaluator), *worker_evaluators)
        self._result_executor = None
        self._pipeline_executor = None

    @property
    def result_workers(self) -> int:
        return len(self._evaluators)

    def reset(self) -> ResetResult[observation_t]:
        state = self.transition.reset()
        return ResetResult(
            state=state,
            observation=self.observation_builder.build(state),
        )

    def step(self, action: Action) -> StepResult[observation_t]:
        prepare = getattr(self.reward_evaluator, "prepare_reward_action", None)
        if callable(prepare):
            action = prepare(self.snapshot(), action)
        state = self.transition.step(action)
        return self._step_result(state)

    def _step_result(self, state, evaluators=None, geometry=None):
        observation_builder, reward_evaluator = evaluators or self._evaluators[0]
        reward, info = (reward_evaluator.compute(state) if geometry is None else
                        reward_evaluator.compute_prepared(state, geometry))
        observe = observation_builder.build
        if self.defer_observations:
            observe = getattr(observation_builder, "defer", observe)
        return StepResult(
            state=state,
            observation=observe(state),
            reward=float(reward),
            info=dict(info),
        )

    def step_batch(self, states, actions):
        """Overlap chunked physics/results, preserving the input result order."""
        if len(states) != len(actions):
            raise ValueError("environment batch states and actions must align")
        backend = getattr(self.transition, "backend", None)
        if callable(getattr(backend, "place_batch", None)) or callable(
                getattr(backend, "place_batch_with_preparation", None)):
            def prepare_action(evaluators, item):
                parent, action = item
                prepare = getattr(evaluators[1], "prepare_reward_action", None)
                return prepare(parent, action) if callable(prepare) else action

            actions = self._map_evaluators(tuple(zip(states, actions)), prepare_action, start_index=0)
        if (self.result_workers > 1
                and callable(getattr(backend, "place_batch_with_preparation", None))
                and callable(getattr(self.transition, "iter_prepared_step_batches", None))
                and all(callable(getattr(reward, "prepare_geometry", None))
                        and callable(getattr(reward, "compute_prepared", None))
                        for _, reward in self._evaluators)):
            return self._prepared_step_batch(states, actions)
        if callable(getattr(backend, "place_batch", None)):
            batches = getattr(self.transition, "iter_step_batches", None)
            chunk_size = getattr(backend, "placement_batch_size", len(states))
            if callable(batches) and self.result_workers > 1 and len(states) > chunk_size:
                return self._pipelined_step_results(batches(states, actions))
            evaluated_states = tuple(self.transition.step_batch(states, actions))
            return self._step_result_chunks(evaluated_states)
        results = []
        for state, action in zip(states, actions):
            self.restore(state)
            results.append(self.step(action))
        return tuple(results)

    def _prepared_step_batch(self, states, actions):
        """Prepare nominal geometry beside robustness, then publish final results.

        One coordinator owns evaluator scheduling. Each finite batch retains
        scalar prepared values, rather than depending on reward-cache eviction.
        """
        if self._pipeline_executor is None:
            self._pipeline_executor = ThreadPoolExecutor(max_workers=1)
        pending = []

        def prepare(start, parents, candidates, poses):
            pending.append(self._pipeline_executor.submit(
                self._prepare_geometries, parents, candidates, poses, start_index=start))

        results = []
        start_index = 0
        try:
            for batch in self.transition.iter_prepared_step_batches(states, actions, prepare):
                geometry = tuple(value for future in pending for value in future.result())
                if len(geometry) != len(batch):
                    raise RuntimeError("nominal preparation does not match placement results")
                pending.clear()
                results.extend(self._step_result_chunks(
                    batch, start_index=start_index, prepared=geometry))
                start_index += len(batch)
            return tuple(results)
        finally:
            # The backend drains native work before propagating failures;
            # drain CPU work too before any evaluator can be reused/closed.
            for future in pending:
                future.cancel()
            wait(pending)

    def _prepare_geometries(self, parents, candidates, poses, *, start_index):
        def prepare(evaluators, item):
            parent, candidate, nominal = item
            if nominal is None:
                return None  # Native failure: ordinary final evaluation handles it.
            return evaluators[1].prepare_geometry(
                [*parent.placed_object_indices, candidate.object_index],
                {**parent.object_poses, **nominal})

        return self._map_evaluators(tuple(zip(parents, candidates, poses, strict=True)),
                                    prepare, start_index=start_index)

    def _step_result_chunks(self, states, *, start_index=0, prepared=None):
        """Process small CPU chunks with stable evaluator assignment."""
        if self.result_batch_size <= 0:
            raise ValueError("result batch size must be positive")
        results = []
        for start in range(0, len(states), self.result_batch_size):
            kwargs = ({} if prepared is None else
                      {"prepared": prepared[start:start + self.result_batch_size]})
            results.extend(self._step_results(
                states[start:start + self.result_batch_size],
                start_index=start_index + start, **kwargs))
        return tuple(results)

    def _pipelined_step_results(self, batches):
        """Queue CPU chunks independently of GPU batches, with bounded backlog.

        One coordinator serializes result groups to protect evaluator caches.
        At most two groups are submitted, plus the physics batch on the caller.
        A slow CPU group therefore permits one extra GPU batch of lookahead.
        """
        if self._pipeline_executor is None:
            self._pipeline_executor = ThreadPoolExecutor(max_workers=1)
        pending = deque()
        results = []
        start_index = 0

        def collect():
            # Keep the future tracked until result() returns, including when
            # the waiting caller is interrupted before the worker finishes.
            results.extend(pending[0].result())
            pending.popleft()

        try:
            for states in batches:
                # The iterator runs physics on the caller while queued CPU work
                # proceeds. Consume completed groups in original input order.
                while pending and pending[0].done():
                    collect()
                if len(pending) == 2:
                    collect()
                pending.append(self._pipeline_executor.submit(
                    self._step_result_chunks, states, start_index=start_index))
                start_index += len(states)
            while pending:
                collect()
            return tuple(results)
        finally:
            # Cancel unstarted groups after errors and drain any active work
            # before callers can reuse evaluator caches or close the backend.
            for future in pending:
                future.cancel()
            wait(pending)

    def _step_results(self, states, *, start_index=0, prepared=None):
        geometry = (None,) * len(states) if prepared is None else prepared
        return self._map_evaluators(tuple(zip(states, geometry, strict=True)),
            lambda evaluators, item: self._step_result(item[0], evaluators, item[1]),
            start_index=start_index)

    def _map_evaluators(self, items, operation, *, start_index):
        """Preserve worker/cache ownership and input ordering across both phases."""
        if self.result_workers == 1 or len(items) <= 1:
            return tuple(operation(self._evaluators[(start_index + index) % self.result_workers], item)
                         for index, item in enumerate(items))
        if self._result_executor is None:
            self._result_executor = ThreadPoolExecutor(max_workers=self.result_workers)
        buckets = [[] for _ in self._evaluators]
        for index, item in enumerate(items, start=start_index):
            buckets[index % self.result_workers].append((index, item))

        def evaluate(evaluators, jobs):
            return [(index, operation(evaluators, item)) for index, item in jobs]

        futures = [self._result_executor.submit(evaluate, evaluators, jobs)
                   for evaluators, jobs in zip(self._evaluators, buckets) if jobs]
        # Every worker must finish before errors propagate or caches are reused.
        wait(futures)
        results = [item for future in futures for item in future.result()]
        results.sort(key=lambda item: item[0])
        return tuple(result for _, result in results)

    def restore(self, state: State) -> None:
        self.transition.restore(state)

    def snapshot(self) -> State:
        return self.transition.snapshot()

    def observation(self) -> observation_t:
        return self.observation_builder.build(self.snapshot())

    def close(self) -> None:
        if self._pipeline_executor is not None:
            self._pipeline_executor.shutdown(wait=True)
        if self._result_executor is not None:
            self._result_executor.shutdown(wait=True)
        self.transition.close()
