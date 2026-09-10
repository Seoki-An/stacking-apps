"""Bounded rsync pull into staging, followed by desktop integrity verification."""

from pathlib import Path
import json
import re
import subprocess
import time

from .capture import integer
from .storage import atomic_json, load_manifest, receive_local


def pull_product(host, remote_directory, staging, destination, *, bandwidth_kib,
                 control_fresh, spawn=subprocess.Popen):
    """A caller schedules one transfer at a time and pauses it on stale control.

    SSH keys/host verification are provisioned outside the application. This
    adapter never transfers raw captures or interprets transfer as scan success.
    """
    integer(bandwidth_kib, "bulk transfer KiB/s", 1)
    if not control_fresh():
        raise ValueError("bulk transfer suspended: control feedback is stale")
    if not re.fullmatch(r"[A-Za-z0-9_][A-Za-z0-9_.@-]*", host):
        raise ValueError("invalid SSH host alias")
    if not re.fullmatch(r"/[A-Za-z0-9_./-]+", remote_directory) or ".." in Path(remote_directory).parts:
        raise ValueError("remote product must be an absolute safe path")
    staging, destination = Path(staging), Path(destination)
    if staging.resolve() == destination.resolve():
        raise ValueError("network staging must be separate from verified products")
    staging.mkdir(parents=True, exist_ok=True)
    # BatchMode never prompts for credentials; standard SSH host checking stays on.
    base = ["rsync", "--recursive", "--compress", "--partial", "--partial-dir=.rsync-partial",
            "--timeout=30", f"--bwlimit={bandwidth_kib}", "--max-size=256m",
            "-e", "ssh -o BatchMode=yes -o ConnectTimeout=10"]
    remote = host + ":" + remote_directory.rstrip("/") + "/"
    deadline = time.monotonic() + 300

    def transfer(includes, target):
        args = base + includes + ["--exclude=*", remote, str(target) + "/"]
        _run_transfer(args, spawn, control_fresh, deadline)

    # Validate purpose and identity before downloading any potentially raw XYZ.
    candidate = staging / "_manifest"
    candidate.mkdir(exist_ok=True)
    transfer(["--include=/manifest.json"], candidate)
    manifest = load_manifest(candidate)
    if "product_id" not in manifest:
        raise ValueError("bulk transfer requires an identified crop product, not raw capture")
    if any(item["bytes"] > 256*1024*1024 for item in manifest["frames"]):
        raise ValueError("product exceeds the per-frame transfer bound")
    binding = staging / "binding.json"
    identity = {"product_id": manifest["product_id"], "sha256": manifest["sha256"]}
    if binding.exists() and json.loads(binding.read_text()) != identity:
        raise ValueError("staging is bound to another immutable product")
    atomic_json(binding, identity)
    atomic_json(staging / "manifest.json", manifest)
    transfer(["--include=/*.npz"], staging)
    if not control_fresh():
        raise ValueError("bulk transfer verification deferred: control feedback is stale")
    return receive_local(staging, destination)


def _run_transfer(args, spawn, control_fresh, deadline):
    process = spawn(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        while True:
            if not control_fresh():
                raise ValueError("bulk transfer suspended: control feedback is stale")
            if time.monotonic() > deadline:
                raise TimeoutError("bulk transfer attempt exceeded its time bound")
            try:
                _, stderr = process.communicate(timeout=.25)
                break
            except subprocess.TimeoutExpired:
                continue
        if process.returncode:
            raise RuntimeError(f"rsync failed ({process.returncode}): {stderr.decode(errors='replace')[-2000:]}")
    finally:
        if process.poll() is None:
            process.terminate()
            try:
                process.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
                process.communicate()
