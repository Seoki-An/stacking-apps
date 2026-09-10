"""Open3D adapter for the app-owned tabletop PCD catalog."""

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from stacking_common.pointcloud import PointCloudData


class TabletopPcdCatalog:
    """Resolve and load measured PCD models for configured tabletop objects."""

    def __init__(
        self,
        model_directory: str | Path,
        object_ids: Sequence[int],
    ) -> None:
        self.model_directory = Path(model_directory)
        if not self.model_directory.is_dir():
            raise FileNotFoundError(
                f"missing PCD model directory: {self.model_directory}"
            )
        self.object_ids = tuple(int(object_id) for object_id in object_ids)
        if not self.object_ids:
            raise ValueError("PCD catalog must not be empty")
        if len(set(self.object_ids)) != len(self.object_ids):
            raise ValueError("PCD catalog object IDs must be unique")
        missing = [
            object_id
            for object_id in self.object_ids
            if not self.path(object_id).is_file()
        ]
        if missing:
            raise FileNotFoundError(
                f"missing PCD models for object IDs: {missing}"
            )

    def path(self, object_id: int) -> Path:
        """Return the configured PCD path for one object."""

        object_id = int(object_id)
        if object_id not in self.object_ids:
            raise KeyError(f"object {object_id} is not in the PCD catalog")
        return self.model_directory / f"model_{object_id}.pcd"

    def load(self, object_id: int) -> PointCloudData:
        """Load one model into the platform-neutral array representation."""

        try:
            import open3d as o3d
        except ImportError as error:
            raise RuntimeError(
                "Open3D is required to load tabletop PCD assets"
            ) from error
        path = self.path(object_id)
        cloud = o3d.io.read_point_cloud(str(path))
        points = np.asarray(cloud.points, dtype=np.float64)
        if len(points) == 0:
            raise ValueError(f"PCD model is empty: {path}")
        if not np.all(np.isfinite(points)):
            raise ValueError(f"PCD model contains non-finite points: {path}")
        return PointCloudData(points)

    def load_many(
        self, object_ids: Sequence[int]
    ) -> dict[int, PointCloudData]:
        """Load a selected catalog subset in caller-provided order."""

        return {
            int(object_id): self.load(object_id)
            for object_id in object_ids
        }
