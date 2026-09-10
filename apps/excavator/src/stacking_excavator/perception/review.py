"""Durable desktop replay review with revision-checked, one-use decisions.

This store has no hardware authority. Accepted poses belong to its replay
session; a future live runtime must additionally enforce commissioned health.
"""

import json
from pathlib import Path
import sqlite3
from contextlib import contextmanager

from stacking_common.pointcloud.types import validate_rigid_transform
from stacking_excavator.acquisition.capture import GRIPPER_FRAME, digest, integer


class ReplayReviewStore:
    def __init__(self, path, *, initial_revision=0):
        integer(initial_revision, "initial revision")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self._connect() as db:
            db.execute("CREATE TABLE IF NOT EXISTS scene (id INTEGER PRIMARY KEY CHECK(id=1), revision INTEGER, state TEXT)")
            db.execute("CREATE TABLE IF NOT EXISTS proposals (token TEXT PRIMARY KEY, body TEXT, decision TEXT)")
            db.execute("INSERT OR IGNORE INTO scene VALUES (1, ?, ?)",
                (initial_revision, json.dumps({"mode": "replay", "poses": {}, "attachment": None})))

    @contextmanager
    def _connect(self):
        db = sqlite3.connect(self.path, timeout=5)
        try:
            db.execute("PRAGMA synchronous=FULL")
            with db:
                yield db
        finally:
            db.close()

    def stage(self, proposal):
        proposal = dict(proposal)
        token = proposal.pop("sha256")
        if digest(proposal) != token or proposal["schema_version"] != 1:
            raise ValueError("proposal integrity/version mismatch")
        proposal["sha256"] = token
        integer(proposal["scene_revision"], "scene revision")
        if proposal["kind"] not in {"field", "scene", "inhand"}:
            raise ValueError("unsupported proposal purpose")
        if proposal["frame"] != (GRIPPER_FRAME if proposal["kind"] == "inhand" else "world"):
            raise ValueError("proposal uses the wrong reference frame")
        ids = []
        for obj in proposal["objects"]:
            integer(obj["object_id"], "object ID")
            ids.append(obj["object_id"])
            if obj["pose"] is not None:
                validate_rigid_transform(obj["pose"])
        if not ids or len(ids) != len(set(ids)):
            raise ValueError("proposal requires unique objects")
        if proposal["kind"] in {"field", "inhand"} and len(ids) != 1:
            raise ValueError("single-object identification requires one object")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            revision, = db.execute("SELECT revision FROM scene WHERE id=1").fetchone()
            if revision != proposal["scene_revision"]:
                raise ValueError("stale scene revision")
            db.execute("INSERT OR IGNORE INTO proposals VALUES (?, ?, NULL)",
                       (token, json.dumps(proposal, allow_nan=False)))
        return token

    def decide(self, token, *, accept, expected_revision, released_object_id=None):
        if not isinstance(accept, bool):
            raise ValueError("review decision must be a bool")
        integer(expected_revision, "expected revision")
        with self._connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("SELECT body, decision FROM proposals WHERE token=?", (token,)).fetchone()
            if row is None:
                raise ValueError("unknown decision token")
            if row[1] is not None:
                raise ValueError("decision token was already consumed")
            revision, state = db.execute("SELECT revision, state FROM scene WHERE id=1").fetchone()
            proposal, state = json.loads(row[0]), json.loads(state)
            if released_object_id is not None:
                integer(released_object_id, 'released object ID')
                if proposal['kind'] != 'scene' or released_object_id not in {o['object_id'] for o in proposal['objects']}:
                    raise ValueError('release verification requires a scene observation of the released stone')
            if expected_revision != revision or proposal["scene_revision"] != revision:
                raise ValueError("stale scene revision")
            if accept:
                if (proposal["valid"] is not True or any(obj["valid"] is not True or obj["reasons"]
                        or obj["pose"] is None for obj in proposal["objects"])):
                    raise ValueError("cannot accept an invalid perception proposal")
                if proposal["kind"] == "inhand":
                    if not proposal["opening"] or not all(o["valid"] is True for o in proposal["opening"]):
                        raise ValueError("attachment requires validated opening")
                    state["attachment"] = {"object": proposal["objects"][0],
                                           "opening": proposal["opening"], "proposal": token}
                else:
                    for obj in proposal["objects"]:
                        state["poses"][str(obj["object_id"])] = {"pose": obj["pose"], "proposal": token}
                    if (released_object_id is not None and state['attachment'] is not None
                            and state['attachment']['object']['object_id'] == released_object_id):
                        state['attachment'] = None
                revision += 1
                db.execute("UPDATE scene SET revision=?, state=? WHERE id=1",
                           (revision, json.dumps(state, allow_nan=False)))
            db.execute("UPDATE proposals SET decision=? WHERE token=?", ("accepted" if accept else "rejected", token))
        return self.snapshot()

    def snapshot(self):
        with self._connect() as db:
            revision, state = db.execute("SELECT revision, state FROM scene WHERE id=1").fetchone()
        return {"revision": revision, **json.loads(state)}

    def proposals(self):
        """Durable review history, including proposals awaiting a decision."""
        with self._connect() as db:
            rows = db.execute("SELECT body, decision FROM proposals ORDER BY rowid").fetchall()
        return [{"proposal": json.loads(body), "decision": decision} for body, decision in rows]
