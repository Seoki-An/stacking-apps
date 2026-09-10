"""ROS-independent tabletop capture sequencing."""

from stacking_common.pointcloud import CapturedPointCloud
from stacking_tabletop.config import CameraProfile, CaptureProfile


class TabletopCaptureBuffer:
    """Apply warm-up, timing, identity, and count rules to camera frames."""

    def __init__(
        self,
        camera: CameraProfile,
        profile: CaptureProfile,
    ) -> None:
        self.camera = camera
        self.profile = profile
        self.seen_frames = 0
        self.frames: list[CapturedPointCloud] = []

    @property
    def complete(self) -> bool:
        return len(self.frames) >= self.profile.frame_count

    def accept(self, frame: CapturedPointCloud) -> bool:
        if frame.sensor_id != self.camera.serial:
            raise ValueError("captured cloud camera serial mismatch")
        if self.complete:
            return False
        self.seen_frames += 1
        if self.seen_frames <= self.profile.warmup_frames:
            return False
        if self.frames:
            if frame.timestamp_ns <= self.frames[-1].timestamp_ns:
                raise ValueError("captured cloud timestamps must increase")
            elapsed = (frame.timestamp_ns - self.frames[-1].timestamp_ns) * 1e-9
            if elapsed < self.profile.period:
                return False
        self.frames.append(frame)
        return True
