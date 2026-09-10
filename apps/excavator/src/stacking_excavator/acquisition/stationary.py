"""Identified stationary acquisition with one source owner and bounded frames."""

import json

from .capture import digest
from .storage import create_capture, load_manifest


class StationaryCaptureService:
    """Source yields capture-time CaptureFrame values. It cannot command joints.

    A failed/interrupted job retains its reservation and files for inspection;
    resubmitting the same ID never triggers a second physical capture.
    """
    def __init__(self, spool, source):
        self.spool = spool
        self.source = source

    def capture(self, job_id, crop, *, maximum_bytes, maximum_frames):
        from .capture import integer
        integer(maximum_frames, "maximum capture frames", 1)
        identity = digest({"crop": crop.values(), "maximum_frames": maximum_frames})
        state = self.spool.reserve(job_id, identity, maximum_bytes=maximum_bytes)
        raw = self.spool.root / job_id / "raw"
        if state == "retained":
            return load_manifest(raw)
        if state != "reserved":
            raise ValueError("interrupted capture requires reconciliation, not automatic recapture")
        self.spool.start(job_id)

        def bounded_frames():
            used = 0
            for index, frame in enumerate(self.source.frames(job_id)):
                # Unknown recording clocks cannot establish a stationary live scan.
                if frame.timestamp_source not in {"sensor", "emulated"}:
                    raise ValueError("stationary acquisition requires sensor timing or explicit emulation")
                crop.select(frame, maximum_timestamp_skew_ns=50_000_000)
                used += (frame.points.nbytes + 4096 + 2*len(json.dumps(frame.metadata())) +
                         sum(v.nbytes + 1024 + 4*len(k) for k, v in frame.attributes.items()))
                if index >= maximum_frames or used > maximum_bytes:
                    raise ValueError("capture exceeded its declared frame/storage bound")
                yield frame

        create_capture(raw, job_id, crop.kind, crop.scene_revision, bounded_frames())
        return self.spool.finish(job_id)
