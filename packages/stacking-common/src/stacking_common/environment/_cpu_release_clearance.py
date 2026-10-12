"""Read native DSF contacts at a noisy release pose without evolving its scene."""

import numpy as np


class CpuReleaseQueries:
    """Use the simulator's input contacts until core exposes a CPU query API.

    Native ``step`` computes contacts before integration and never mutates its
    input snapshot. The isolated probe has no gravity or penetration correction;
    its returned snapshot and constraint-solver statistics are discarded.
    """

    def __init__(self, backend):
        self.backend = backend
        self.core = backend.core
        config = self.core.SimulationConfig()
        config.gravity = np.zeros(3)
        config.contact.model = self.core.ContactModel.SINGLE_POINT
        config.contact.error_reduction_ratio = 0.
        config.contact.detection_margin = .001
        config.solver.max_iters = 1
        self.simulator = self.core.Simulator(config)

    def query(self, rows, poses):
        contacts, errors = [], {}
        for trial_index, ((state, action), pose) in enumerate(zip(rows, poses, strict=True)):
            trial = action.copy()
            trial.pose = np.array(pose, dtype=float, copy=True)
            indices = [*state.placed_object_indices, action.object_index]
            entity = self.backend.scene_builder.entity_id(action.object_index)
            try:
                scene = self.backend._scene(state, trial, indices, {action.object_index})
                self.simulator.clear_warm_start()
                # A positive dt is required; only the pre-integration contacts
                # are used, so the value cannot affect the queried geometry.
                result = self.simulator.step(scene, .01)
                found = []
                for contact in result.contacts:
                    first, second = contact['entity_first'], contact['entity_second']
                    if entity not in (first, second):
                        continue
                    gap = float(contact['gap'])
                    normal = np.asarray(contact['normal'], dtype=float)
                    difference = (np.asarray(contact['point_second'], dtype=float)
                                  - np.asarray(contact['point_first'], dtype=float))
                    if (not np.isfinite(gap) or normal.shape != (3,)
                            or difference.shape != (3,)
                            or not np.all(np.isfinite(normal))
                            or not np.all(np.isfinite(difference))
                            or abs(np.linalg.norm(normal) - 1.) > 1e-6):
                        raise ValueError('invalid release contact')
                    # The CPU direction solver does not expose convergence.
                    # Its stationarity test is this tangential support-point
                    # residual, with a native default tolerance of 1e-6 m.
                    tangent = difference - normal * float(difference @ normal)
                    if np.linalg.norm(tangent) > 1e-6 + 1e-12:
                        raise ValueError('unconverged release contact')
                    derivative = float(normal[2]) * (-1. if first == entity else 1.)
                    found.append((trial_index, gap, derivative))
                contacts.extend(found)
            except (RuntimeError, ValueError, FloatingPointError):
                errors[trial_index] = 'initial_clearance_query_failure'
        return contacts, errors
