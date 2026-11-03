"""Tabletop replay uses embedded objects while execution verifies live assets."""

from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from stacking_common.artifacts import StackRunStore
from stacking_common.artifacts.object_assets import materialize_object_assets
from stacking_common.visualization.operations import (
    OperationsController as CommonOperationsController, OperationsFrame, RenderObject,
)


class OperationsController(CommonOperationsController):
    def __init__(self, *args, repository=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.repository = repository
        self._object_paths = {}
        self._asset_directory = None

    def load(self, path):
        paths = {}
        assets = TemporaryDirectory(prefix='stack-replay-')
        if self.repository is not None and Path(path).suffix.lower() == '.stack':
            # Old documents without embedded objects keep their legacy rendering.
            if any(item.role == 'object_model' for item in StackRunStore.attachments(path)):
                for context in StackRunStore.load_run_contexts(path):
                    resolved = materialize_object_assets(path, context, self.repository, assets.name)
                    paths[context.sha256] = {int(role.split('/')[1]): value
                        for role, value in resolved.items() if role.startswith('object/')}
        artifact = super().load(path)
        self._object_paths, self._asset_directory = paths, assets
        return artifact

    def _recorded_objects(self, frame):
        if not self._object_paths:
            return frame
        record = self.current_plan
        digest = record.context_sha256 if record is not None else next(iter(self._object_paths))
        return replace(frame, object_mesh_paths=self._object_paths.get(digest))

    def frame(self, **kwargs):
        return self._recorded_objects(super().frame(**kwargs))

    def planning_frame(self):
        return self._recorded_objects(super().planning_frame())
