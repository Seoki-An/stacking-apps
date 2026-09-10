"""Raw in-hand capture preprocessing before global reconstruction."""

import json
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from stacking_common.pointcloud import (
    IncrementalRegistration,
    MultiViewVoxelMerge,
    Open3dRegistrationBackend,
    PointCloudData,
    RegistrationDecision,
    RegistrationResult,
    VoxelMergeConfig,
)
from stacking_tabletop.config import TabletopPerceptionProfile

from .capture import LoadedCaptureView, TabletopCaptureView, TabletopRawCapture
from .extraction import InteractiveObjectSelector, ManualObjectExtractor
from .organized_depth import DepthFilterConfig
from .pose_graph import TabletopPoseGraphRefiner
from .reconstruction import write_point_cloud


@dataclass(frozen=True)
class AlignedCaptureView:
    metadata: TabletopCaptureView
    cloud: PointCloudData
    correction: np.ndarray
    decision: RegistrationDecision
    selected: bool
    registration: RegistrationResult | None


@dataclass(frozen=True)
class InHandIcpInspection:
    """One candidate overlay presented before multi-view consensus merging."""

    metadata: TabletopCaptureView
    source: PointCloudData
    reference: PointCloudData | None
    candidate: PointCloudData
    correction: np.ndarray
    decision: RegistrationDecision
    registration: RegistrationResult | None
    index: int
    total: int


class InHandIcpInspector(Protocol):
    def inspect_icp(self, case: InHandIcpInspection) -> str: ...


@dataclass(frozen=True)
class PreparedCapture:
    paths: tuple[Path, ...]
    capture_indices: tuple[int, ...]
    viewpoints: tuple[np.ndarray, ...]
    merged_path: Path
    report_path: Path


class _PrepareProgress:
    """Map every preparation phase onto one monotonic progress timeline."""

    _PHASE_COUNT = 7

    def __init__(
        self,
        view_count: int,
        callback: Callable[[int, int, str], None] | None,
    ) -> None:
        if view_count <= 0:
            raise ValueError("preparation progress requires at least one view")
        self.view_count = view_count
        self.callback = callback
        self.completed = 0
        self.total = 5 * view_count + 2

    def update(self, phase: int, completed: int, message: str) -> None:
        capacity = self._capacity(phase)
        offset = sum(self._capacity(index) for index in range(phase))
        value = offset + min(max(completed, 0), capacity)
        self.completed = max(self.completed, value)
        if self.callback is not None:
            self.callback(self.completed, self.total, message)

    def finish(self, phase: int, message: str) -> None:
        self.update(phase, self._capacity(phase), message)

    def _capacity(self, phase: int) -> int:
        if not 0 <= phase < self._PHASE_COUNT:
            raise ValueError(f"invalid preparation phase: {phase}")
        return self.view_count if phase in {0, 1, 2, 3, 5} else 1


class IncrementalInHandAligner:
    """Align each tool-frame view to a bounded accumulated GICP reference."""

    def __init__(self, profile: TabletopPerceptionProfile) -> None:
        self.registration = profile.in_hand_registration
        self.quality = profile.in_hand_quality
        self.reference_voxel_size = profile.in_hand_capture.reference_voxel_size
        self.backend = Open3dRegistrationBackend(
            self.registration,
            generalized=True,
        )

    def align(
        self,
        views: tuple[LoadedCaptureView, ...],
        *,
        inspector: InHandIcpInspector | None = None,
        progress: Callable[[int, int], None] | None = None,
    ) -> tuple[AlignedCaptureView, ...]:
        if not views:
            raise ValueError("incremental alignment requires at least one view")
        alignment = IncrementalRegistration(
            self.backend, self.quality, self.reference_voxel_size,
        )
        results = []
        index = 0
        completed = 0
        while index < len(views):
            view = views[index]
            proposal = alignment.propose(view.cloud)
            correction = proposal.correction
            registration = proposal.registration
            decision = proposal.decision
            aligned = proposal.cloud
            selected = decision.accepted
            if inspector is not None:
                action = inspector.inspect_icp(
                    InHandIcpInspection(
                        view.metadata,
                        view.cloud,
                        alignment.reference,
                        aligned,
                        correction,
                        decision,
                        registration,
                        index,
                        len(views),
                    )
                )
                if action == "back":
                    if index == 0:
                        continue
                    results.pop()
                    index -= 1
                    alignment.undo()
                    continue
                if action not in {"accept", "reject"}:
                    raise ValueError(f"unsupported ICP inspection action: {action}")
                selected = action == "accept" and decision.accepted
            alignment.commit(proposal, selected=selected)
            results.append(
                AlignedCaptureView(
                    view.metadata,
                    aligned,
                    correction,
                    decision,
                    selected,
                    registration,
                )
            )
            index += 1
            if index > completed:
                completed = index
                if progress is not None:
                    progress(completed, len(views))
        return tuple(results)


class TabletopRawCapturePreprocessor:
    """Convert raw Femto bursts into aligned, object-only tool-frame views."""

    def __init__(self, profile: TabletopPerceptionProfile) -> None:
        self.profile = profile

    def prepare(
        self,
        capture_directory: str | Path,
        output_directory: str | Path,
        selector: InteractiveObjectSelector,
        *,
        progress: Callable[[int, int, str], None] | None = None,
        icp_inspector: InHandIcpInspector | None = None,
    ) -> PreparedCapture:
        capture = TabletopRawCapture.load(capture_directory)
        prepare_progress = _PrepareProgress(len(capture.views), progress)
        prepare_progress.update(0, 0, "Loading and filtering raw capture views")
        capture_profile = self.profile.in_hand_capture
        depth_filter = DepthFilterConfig(
            minimum_temporal_support=capture_profile.minimum_temporal_support,
            temporal_tolerance=capture_profile.temporal_tolerance,
            enable_boundary_filter=capture_profile.enable_boundary_filter,
            discontinuity_threshold=capture_profile.discontinuity_threshold,
            discontinuity_radius_pixels=capture_profile.discontinuity_radius_pixels,
            boundary_dilation_pixels=capture_profile.boundary_dilation_pixels,
        )
        loaded = capture.load_tool_views(
            depth_filter,
            capture_profile.crop_min,
            capture_profile.crop_max,
            progress=lambda completed, _total: prepare_progress.update(
                0,
                completed,
                "Loading and filtering raw capture views",
            ),
        )
        prepare_progress.finish(0, "Raw capture views loaded and filtered")
        aligned = IncrementalInHandAligner(self.profile).align(
            loaded,
            inspector=icp_inspector,
            progress=lambda completed, _total: prepare_progress.update(
                1,
                completed,
                "Aligning in-hand views with incremental ICP",
            ),
        )
        prepare_progress.finish(1, "Incremental in-hand alignment complete")
        accepted = tuple(view for view in aligned if view.selected)
        if len(accepted) < self.profile.reconstruction.minimum_view_support:
            raise ValueError("too few incrementally aligned views for reconstruction")
        preliminary = MultiViewVoxelMerge(
            VoxelMergeConfig(
                self.profile.reconstruction.merge_voxel_size,
                self.profile.reconstruction.minimum_view_support,
            )
        ).merge(
            tuple(view.cloud for view in accepted),
            progress=(
                lambda completed, _total: prepare_progress.update(
                    2,
                    completed,
                    "Building preliminary multi-view merge",
                )
            ),
        )
        prepare_progress.finish(2, "Preliminary multi-view merge complete")
        if not len(preliminary.cloud):
            raise ValueError("incremental alignment produced an empty consensus merge")
        selection = selector.select(preliminary.cloud)
        extractor = ManualObjectExtractor(selection)

        root = Path(output_directory).resolve()
        view_directory = root / "object_views_tool0"
        view_directory.mkdir(parents=True, exist_ok=True)
        for stale in view_directory.glob("*.pcd"):
            stale.unlink()
        paths, indices, viewpoints = [], [], []
        object_clouds = []
        records = []
        saved_records = []
        for view_index, view in enumerate(aligned):
            if not view.selected:
                records.append(
                    {
                        "filename": view.metadata.filename,
                        "capture_index": view.metadata.capture_index,
                        "status": (
                            "operator_rejected"
                            if view.decision.accepted
                            else "icp_rejected"
                        ),
                        "input_points": len(view.cloud),
                        "object_points": 0,
                        "correction": view.correction.tolist(),
                        "registration_reasons": list(view.decision.reasons),
                        "operator_selected": False,
                    }
                )
                prepare_progress.update(
                    3,
                    view_index + 1,
                    "Extracting the selected object from aligned views",
                )
                continue
            object_cloud = extractor.extract(view.cloud, view.metadata)
            status = "saved" if len(object_cloud) >= self.profile.in_hand_registration.minimum_points else "too_few_object_points"
            record = {
                "filename": view.metadata.filename,
                "capture_index": view.metadata.capture_index,
                "status": status,
                "input_points": len(view.cloud),
                "object_points": len(object_cloud),
                "correction": view.correction.tolist(),
                "registration_reasons": list(view.decision.reasons),
                "operator_selected": True,
            }
            if status == "saved":
                path = write_point_cloud(view_directory / view.metadata.filename, object_cloud)
                paths.append(path)
                indices.append(view.metadata.capture_index)
                viewpoints.append(view.metadata.tool_T_camera)
                object_clouds.append(object_cloud)
                record["path"] = str(path)
                saved_records.append(record)
            records.append(record)
            prepare_progress.update(
                3,
                view_index + 1,
                "Extracting the selected object from aligned views",
            )
        prepare_progress.finish(3, "Object extraction complete")
        if len(paths) < self.profile.reconstruction.minimum_view_support:
            raise ValueError("manual object selection retained too few views")
        prepare_progress.update(4, 0, "Optimizing the prepared-view pose graph")
        graph = TabletopPoseGraphRefiner(
            self.profile.reconstruction.pose_graph
        ).refine(
            tuple(object_clouds),
            capture_indices=tuple(indices),
            viewpoints=tuple(viewpoints),
        )
        prepare_progress.finish(4, "Prepared-view pose graph optimized")
        refined_clouds = tuple(
            cloud.transformed(transform)
            for cloud, transform in zip(object_clouds, graph.transforms)
        )
        object_merge = MultiViewVoxelMerge(
            VoxelMergeConfig(
                self.profile.reconstruction.merge_voxel_size,
                self.profile.reconstruction.minimum_view_support,
            )
        ).merge(
            refined_clouds,
            progress=(
                lambda completed, _total: prepare_progress.update(
                    5,
                    completed,
                    "Building final object multi-view merge",
                )
            ),
        )
        prepare_progress.finish(5, "Final object multi-view merge complete")
        if not len(object_merge.cloud):
            raise ValueError("prepared object consensus merge is empty")
        merged_path = write_point_cloud(
            root / "object_merged_tool0.pcd",
            object_merge.cloud,
        )
        for record, transform in zip(saved_records, graph.transforms):
            record["pose_graph_transform"] = transform.tolist()
        rejected = [view for view in aligned if not view.selected]
        report = {
            "status": "prepared",
            "capture_directory": str(capture.directory),
            "coordinate_frame": "tool0",
            "input_views": len(loaded),
            "missing_transform_views": capture.missing_transform_count,
            "incremental_alignment": {
                "method": "multiscale_generalized_icp_to_accumulated_reference",
                "registration": asdict(self.profile.in_hand_registration),
                "quality": asdict(self.profile.in_hand_quality),
                "accepted_views": len(accepted),
                "rejected_views": len(rejected),
                "operator_inspection": icp_inspector is not None,
            },
            "depth_filter": capture.depth.report(depth_filter),
            "preliminary_merge": asdict(preliminary.stats),
            "object_selection": {
                "method": "interactive_box_and_radius_removal",
                "bounds": selection.bounds.tolist(),
                "removal_centers": selection.removal_centers.tolist(),
                "removal_distance": selection.removal_distance,
                "maximum_remove_ratio": selection.maximum_remove_ratio,
            },
            "output_views": len(paths),
            "pose_graph": {
                "optimized": graph.optimized,
                "reason": graph.reason,
                "edge_count": len(graph.edges),
                "accepted_edge_count": sum(
                    edge.decision.accepted for edge in graph.edges
                ),
            },
            "object_merge": asdict(object_merge.stats),
            "object_merged_path": str(merged_path),
            "views": records,
        }
        report_path = root / "preprocessing.json"
        report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
        prepare_progress.finish(6, "Prepared capture exported")
        return PreparedCapture(
            tuple(paths),
            tuple(indices),
            tuple(viewpoints),
            merged_path,
            report_path,
        )

