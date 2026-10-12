"""Batched native DSF queries for noisy release initialization."""

import numpy as np


class CudaReleaseQueries:
    def __init__(self, backend):
        self.backend = backend
        self.core = backend.core
        self.context = self.core.CudaContext()
        self.workspaces = {}

    def _workspace(self, count):
        # Cache powers of two: total cached capacity stays below twice the
        # largest batch. Small batches avoid full-workspace upload/download.
        count = 1 << (count - 1).bit_length()
        if count not in self.workspaces:
            cap = self.core.CudaCapacity.small(count)
            cap.bodies = max(cap.bodies, len(self.backend.models) + 1)
            cap.geometries = max(cap.geometries, 1 + sum(
                model.geometry_count for model in self.backend.models.values()))
            cap.pairs = max(cap.pairs, cap.geometries * (cap.geometries - 1) // 2)
            self.workspaces[count] = self.context.create_workspace(cap)
        return self.workspaces[count]

    def query(self, rows, poses):
        contacts, errors = [], {}
        # Tabletop can request more than four noise trials per candidate.
        size = self.backend.batch_size
        for start in range(0, len(rows), size):
            found, failed = self._query_chunk(rows[start:start + size], poses[start:start + size])
            contacts.extend((i + start, gap, dz) for i, gap, dz in found)
            errors.update((i + start, reason) for i, reason in failed.items())
        return contacts, errors

    def _query_chunk(self, rows, poses):
        core, backend = self.core, self.backend
        workspace = self._workspace(len(rows))
        items, layouts, queries, errors = [], [], [], {}
        for job, ((state, action), pose) in enumerate(zip(rows, poses, strict=True)):
            indices = [*state.placed_object_indices, action.object_index]
            trial = action.copy()
            trial.pose = pose
            scene = backend._scene(state, trial, indices, {action.object_index})
            # Explicit view order: ground, previously placed bodies, candidate.
            view = core.SceneView(scene, [backend._ground_entity_id] + [
                backend.scene_builder.entity_id(i) for i in indices])
            items.append(core.CudaSceneItem(job, view, [2] + [0] * (len(indices) - 1) + [1]))
            layout = [(0, 0)]
            for body, index in enumerate(indices, 1):
                count = backend.models[state.object_ids[index]].geometry_count
                layout.extend((body, geometry) for geometry in range(count))
                if body == len(indices):
                    for geometry in range(count):
                        q = core.CudaQuery()
                        q.job, q.kind, q.first = job, core.CudaQueryKind.ground, body
                        q.geometry_first, q.bound = geometry, backend.config.ground_z
                        q.direction = np.array([0., 0., 1.])
                        queries.append((q, 2))
            layouts.append((layout, len(indices)))
        for i, status in enumerate(workspace.upload(items)):
            if status.status != core.CudaStatus.ready:
                errors[i] = 'initial_clearance_query_failure'
        if any(state.placed_object_indices for state, _ in rows):
            pairs = workspace.detect_pairs(.001, excluded_tags=2, candidate_tag=1)
            for job, status in enumerate(pairs.jobs[:len(rows)]):
                if status.status != core.CudaStatus.ready:
                    errors[job] = 'initial_clearance_query_failure'
            for job, job_pairs in enumerate(pairs.pairs[:len(rows)]):
                layout, candidate = layouts[job]
                for pair in job_pairs:
                    first, ga = layout[pair.first]
                    second, gb = layout[pair.second]
                    q = core.CudaQuery()
                    q.job, q.kind = job, core.CudaQueryKind.contact
                    q.first, q.second = first, second
                    q.geometry_first, q.geometry_second = ga, gb
                    queries.append((q, 2 if first == candidate else 8))
        # Do not launch queries for jobs whose broad phase/upload failed.
        queries = [(q, column) for q, column in queries if q.job not in errors]
        results = workspace.evaluate([q for q, _ in queries])
        contacts = []
        for (q, column), result in zip(queries, results, strict=True):
            if (result.status != core.CudaQueryStatus.success
                    or (q.kind == core.CudaQueryKind.contact
                        and result.termination != core.CudaTermination.converged)):
                errors[q.job] = 'initial_clearance_query_failure'
            else:
                contacts.append((q.job, float(result.gap),
                                 float(np.asarray(result.d_gap).reshape(-1)[column])))
        return contacts, errors
