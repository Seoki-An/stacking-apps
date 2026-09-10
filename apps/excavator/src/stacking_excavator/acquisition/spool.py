"""Durable capture reservations: outages must not silently evict pending data."""

from contextlib import contextmanager
from pathlib import Path
import shutil
import sqlite3

from .capture import identifier, integer
from .storage import load_manifest


class CaptureSpool:
    def __init__(self, root, *, capacity_bytes):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.capacity = integer(capacity_bytes, "spool capacity", 1)
        with self._db() as db:
            db.execute("CREATE TABLE IF NOT EXISTS jobs (id TEXT PRIMARY KEY, identity TEXT, reserved INTEGER, state TEXT, manifest TEXT)")

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.root / "spool.db", timeout=5)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def reserve(self, job_id, identity, *, maximum_bytes):
        identifier(job_id)
        maximum_bytes = integer(maximum_bytes, "capture reservation", 1)
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT identity, state, reserved FROM jobs WHERE id=?", (job_id,)).fetchone()
            if existing:
                if existing[0] != identity or existing[2] != maximum_bytes:
                    raise ValueError("job ID was reused for a different capture request")
                return existing[1]
            used, = db.execute("SELECT COALESCE(SUM(reserved), 0) FROM jobs").fetchone()
            if used + maximum_bytes > self.capacity or maximum_bytes > shutil.disk_usage(self.root).free:
                raise ValueError("capture deferred: spool capacity unavailable")
            if (self.root / job_id).exists():
                raise ValueError("untracked capture directory requires reconciliation")
            db.execute("INSERT INTO jobs VALUES (?, ?, ?, 'reserved', NULL)",
                       (job_id, identity, maximum_bytes))
        return "reserved"

    def start(self, job_id):
        with self._db() as db:
            result = db.execute("UPDATE jobs SET state='capturing' WHERE id=? AND state='reserved'", (job_id,))
            if result.rowcount != 1:
                raise ValueError("capture cannot restart automatically; inspect the durable job")

    def finish(self, job_id):
        identifier(job_id)
        directory = self.root / job_id
        manifest = load_manifest(directory / "raw")
        used = sum(p.stat().st_size for p in directory.rglob("*") if p.is_file())
        with self._db() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT reserved, state FROM jobs WHERE id=?", (job_id,)).fetchone()
            if row is None or row[1] != "capturing":
                raise ValueError("capture is not active")
            if used > row[0]:
                raise ValueError("capture exceeded its storage reservation")
            db.execute("UPDATE jobs SET state='retained', manifest=? WHERE id=?",
                       (manifest["sha256"], job_id))
        return manifest

    def status(self, job_id):
        with self._db() as db:
            row = db.execute("SELECT state, reserved, manifest FROM jobs WHERE id=?", (job_id,)).fetchone()
        if row is None:
            raise ValueError("unknown capture job")
        return dict(zip(("state", "reserved_bytes", "manifest_sha256"), row))
