"""Process isolation for offline planning; never a hardware-stop mechanism."""

import math
import os
from pathlib import Path
import signal
import subprocess
import time


def stop_planning_process(process, grace_seconds=2.0):
    """Terminate only the worker's dedicated process group, then reap it."""
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    try:
        process.wait(timeout=grace_seconds)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # Also remove descendants if the group leader exited before its workers.
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()


def supervise_planning(command, *, destination=None, max_seconds=None):
    """Keep Ctrl+C and an optional wall deadline responsive during native work.

    SQLite commits remain durable; the interrupted attempt is not a committed
    placement. Restart branches from the last checkpoint with fresh workers.
    """
    if max_seconds is not None and (not math.isfinite(max_seconds) or max_seconds <= 0):
        raise ValueError("--max-run-time must be positive and finite")
    destination = None if destination is None else Path(destination)
    if destination is not None and destination.exists():
        raise FileExistsError("choose a new --save-run path; existing runs are preserved")
    process = subprocess.Popen(command, start_new_session=True)
    started = time.monotonic()
    interrupted = False
    status = 0
    previous = None
    try:
        while process.poll() is None:
            if max_seconds is not None and time.monotonic() - started >= max_seconds:
                interrupted = True
                status = 124
                print("[WARN] Planning wall limit reached; stopping the isolated worker.", flush=True)
                break
            time.sleep(0.05)
    except KeyboardInterrupt:
        interrupted = True
        status = 130
        print("[WARN] Planning interrupted; stopping the isolated worker.", flush=True)
    finally:
        if interrupted:
            # Repeated Ctrl+C must not interrupt termination/reaping or marking
            # the journal. Restore the caller's signal handler afterwards.
            previous = signal.signal(signal.SIGINT, signal.SIG_IGN)
            try:
                stop_planning_process(process)
            except BaseException:
                signal.signal(signal.SIGINT, previous)
                raise
    try:
        if not interrupted:
            status = process.wait()
        if (interrupted or status != 0) and destination is not None and destination.is_file():
            from stacking_common.artifacts import StackRunStore
            with StackRunStore.open_append(destination) as writer:
                writer.finish("cancelled" if interrupted else "failed")
            print(f"[INFO] Committed records preserved: {destination}", flush=True)
        return status
    finally:
        if previous is not None:
            signal.signal(signal.SIGINT, previous)
