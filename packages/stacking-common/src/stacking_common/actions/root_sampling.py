"""Root exploration mixes local continuation with global placement proposals."""

import numpy as np

from .scored import ScoredPlanarPoseSampler
from .types import PlanarPlacement


class RootMixturePlanarSampler(ScoredPlanarPoseSampler):
    """Use normal scoring in both pools; connectivity changes allocation only."""

    def __init__(self, inventory, config, orientation_provider, settings, *, seed=0, floor_policy=None):
        super().__init__(inventory, config, orientation_provider, settings,
                         seed=seed, floor_policy=floor_policy, connectivity_enabled=False)
        self.root_global_fraction = float(settings.get("root_global_fraction", .2))
        if not 0. < self.root_global_fraction <= 1.:
            raise ValueError("root mixture fraction must be in (0, 1]")
        self.connected = ScoredPlanarPoseSampler(inventory, config, orientation_provider, settings,
                                                seed=seed, floor_policy=floor_policy)
        self._global_credit = 0.
        self._root_key = None

    def sample(self, inventory, state, observation, object_index, count, sampling_priority):
        key = (tuple(state.placed_object_indices),
               tuple((oid, np.asarray(pose).tobytes()) for oid, pose in sorted(state.object_poses.items())))
        if key != self._root_key:
            self._root_key = key
            self._global_credit = 0.
        if not state.placed_object_indices:
            return super().sample(inventory, state, observation, object_index, count, sampling_priority)
        # Carry fractional slots across stones; rounding every small allocation
        # separately would erase global exploration (or over-allocate it).
        self._global_credit += count * self.root_global_fraction
        global_count = min(count, int(self._global_credit + 1e-12))
        self._global_credit -= global_count
        self.connected.relax_corner_seed = self.relax_corner_seed
        local = (self.connected.sample(inventory, state, observation, object_index,
                                      count - global_count, sampling_priority)
                 if count > global_count else ())
        # Empty/undersized connected pools spend their unused slots globally.
        needed = count - len(local)
        global_proposals = (super().sample(inventory, state, observation, object_index,
                                          needed, sampling_priority) if needed else ())
        # Global scores normalize every root proposal, including local ones.
        self.score_map(state, observation, object_index)
        selected = []
        for group, proposals in (("connected", local), ("global", global_proposals)):
            for proposal in proposals:
                raw, normalized = self.score_at_object_pose(observation, proposal.pose, object_index)
                selected.append(PlanarPlacement(proposal.pose, raw, diagnostics={
                    **proposal.diagnostics, "root_sampling_group": group,
                    "height_map_score": raw, "height_map_score_normalized": normalized}))
        return tuple(selected)
