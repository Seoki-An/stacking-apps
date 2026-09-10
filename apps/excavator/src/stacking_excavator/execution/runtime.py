"""Single-owner, journaled desktop simulation coordinated with perception review.

All mutation methods run on the owning thread. A process lock prevents two
consoles from controlling the same session. Restart never resumes a path.
"""
from contextlib import contextmanager
import fcntl
import json
from pathlib import Path
import sqlite3
import time
import numpy as np

from stacking_common.artifacts.execution import encode_execution_command, encode_execution_event, decode_execution_command
from stacking_common.execution import ExecutionEvent, ExecutionStatus, ExecutionFailure
from stacking_excavator.acquisition.capture import digest, integer
from stacking_excavator.perception.review import ReplayReviewStore
from .simulation import SimulatedDevice


class SimulationRuntime:
    def __init__(self, repository, directory, *, device=None, clock=time.monotonic):
        self.directory = Path(directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self._lock = (self.directory / 'owner.lock').open('a')
        try:
            fcntl.flock(self._lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            self._lock.close()
            raise ValueError('this simulation session already has an owner') from None
        self.closed = False
        self.clock = clock
        self.device = device or SimulatedDevice(repository, clock=clock)
        if not isinstance(self.device, SimulatedDevice):
            self._lock.close()
            raise TypeError('this runtime only accepts the simulated device')
        self.review = ReplayReviewStore(self.directory / 'review.db')
        self.path = self.directory / 'execution.db'
        self.command = None
        self.pending = False
        self._lease_until = clock()
        self._issued_at = clock()
        self._feedback_sequence = -1
        self._settled = 0
        self._next_device_sequence = self.device.accepted_sequence + 1
        with self._db() as db:
            db.execute('CREATE TABLE IF NOT EXISTS commands (id TEXT PRIMARY KEY, body TEXT, digest TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS events (id INTEGER PRIMARY KEY, body TEXT)')
            db.execute('CREATE TABLE IF NOT EXISTS state (id INTEGER PRIMARY KEY CHECK(id=1), body TEXT)')
            row = db.execute('SELECT body FROM state WHERE id=1').fetchone()
        self.state = json.loads(row[0]) if row else dict(mode='simulation', status='idle',
            command_id=None, event_sequence=0, segment_index=0, sample_index=0,
            scene_revision=self.review.snapshot()['revision'], needs_replan=False, attachment=False)
        if self.state['status'] in ('running', 'held'):
            with self._db() as db:
                body, sha = db.execute('SELECT body, digest FROM commands WHERE id=?',
                                      (self.state['command_id'],)).fetchone()
            values = json.loads(body)
            if digest(values) != sha:
                self.device.stop()
                self._lock.close()
                raise ValueError('simulation command journal integrity mismatch')
            self.command = decode_execution_command(values)
            self._finish('failed', ExecutionFailure.INTERRUPTED, 'owner restarted; reconcile and replan')
        self.device.stop()
        self._persist()

    @contextmanager
    def _db(self):
        db = sqlite3.connect(self.path)
        try:
            db.execute('PRAGMA synchronous=FULL')
            with db:
                yield db
        finally:
            db.close()

    def _persist(self, event=None, command=None):
        try:
            with self._db() as db:
                if command is not None:
                    values = encode_execution_command(command)
                    db.execute('INSERT INTO commands VALUES (?, ?, ?)',
                               (command.command_id, json.dumps(values), digest(values)))
                db.execute('INSERT OR REPLACE INTO state VALUES (1, ?)', (json.dumps(self.state),))
                if event is not None:
                    db.execute('INSERT INTO events (body) VALUES (?)', (json.dumps(event),))
        except BaseException:
            self.device.stop()
            raise

    def _event(self, status, reason=None, failure=None, *, command=None, **diagnostics):
        self.state['status'] = status
        self.state['event_sequence'] += 1
        event = ExecutionEvent(self.state['command_id'], self.state['event_sequence'], time.time_ns(),
            ExecutionStatus(status), self.state['segment_index'], self.state['sample_index'], failure, reason,
            {'mode': 'simulation', 'scene_revision': self.state['scene_revision'],
             'measured': self.device.snapshot(), **diagnostics})
        self._persist(encode_execution_event(event), command)

    def _active(self):
        return self.state['status'] in ('running', 'held')

    def _ensure_open(self):
        if self.closed:
            raise ValueError('simulation owner is closed')

    def heartbeat(self):
        self._ensure_open()
        # A late heartbeat cannot revive an expired lease.
        if self._active() and self.clock() >= self._lease_until:
            self._finish('failed', ExecutionFailure.TIMEOUT, 'operator lease expired')
            return
        self._lease_until = self.clock() + 1.
        self.device.grant_lease(1.)

    def start(self, command, *, expected_revision):
        self._ensure_open()
        integer(expected_revision, 'scene revision')
        if self._active() or self.state['needs_replan']:
            raise ValueError('finish or reconcile the previous attempt before starting')
        if expected_revision != self.review.snapshot()['revision']:
            raise ValueError('stale scene revision')
        if command.joint_names != self.device.joint_names or command.frame_id != 'world':
            raise ValueError('command must use canonical excavator joints in world')
        if self.clock() - self.device.feedback_time > .5 or not self.device.gripper_known:
            raise ValueError('fresh measured joints and known gripper state required')
        for segment in command.segments:
            if np.any(segment.positions < self.device.lower) or np.any(segment.positions > self.device.upper):
                raise ValueError('path outside VDK23 joint limits')
            if segment.openings is not None and (np.any(segment.openings < -1.14) or np.any(segment.openings > 0)):
                raise ValueError('path outside stone_grab opening limits')
        attached = False
        for event in command.grasp_events:
            if not -1.14 <= event.opening <= 0 or (event.event == 'acquire') == attached:
                raise ValueError('invalid acquire/release sequence or opening')
            attached = event.event == 'acquire'
        if not command.grasp_events or attached:
            raise ValueError('placement requires explicit acquire and release events')
        if np.max(np.abs(command.segments[0].positions[0] - self.device.positions)) > .02:
            raise ValueError('measured start differs from planned start; replan')
        with self._db() as db:
            if db.execute('SELECT 1 FROM commands WHERE id=?', (command.command_id,)).fetchone():
                raise ValueError('command ID was already used')
        self.command = command
        self.pending = False
        self.state.update(status='running', command_id=command.command_id, event_sequence=0,
            segment_index=0, sample_index=0, scene_revision=expected_revision, attachment=False)
        self._lease_until = self.clock() + 1.
        self.device.grant_lease(1.)
        self._event('running', command=command, transition='started')

    def tick(self):
        self._ensure_open()
        self.device.advance()
        if not self._active():
            return self.snapshot()
        if self.review.snapshot()['revision'] != self.state['scene_revision']:
            self._finish('failed', ExecutionFailure.PERCEPTION_FAILED, 'accepted perception changed the scene; needs_replan')
        elif self.clock() >= self._lease_until:
            self._finish('failed', ExecutionFailure.TIMEOUT, 'operator lease expired')
        elif self.clock() - self.device.feedback_time > .5:
            self._finish('failed', ExecutionFailure.TIMEOUT, 'measured feedback stale')
        elif not self.device.gripper_known:
            self._finish('failed', ExecutionFailure.GRIPPER_FAULT, 'gripper state unknown')
        elif self.state['status'] == 'running':
            self._advance_path()
        return self.snapshot()

    def _advance_path(self):
        si, sample = self.state['segment_index'], self.state['sample_index']
        segment = self.command.segments[si]
        event = next((e for e in self.command.grasp_events if (e.segment_index, e.sample_index) == (si, sample)), None)
        target = segment.positions[sample]
        opening = (event.opening if event is not None else
                   float(segment.openings[sample]) if segment.openings is not None else self.device.opening)
        if not self.pending:
            self.device.send(self._next_device_sequence, target, opening)
            self._next_device_sequence += 1
            self.pending = True
            self._issued_at = self.clock()
            self._feedback_sequence = self.device.sequence
            self._settled = 0
            travel = max(float(np.max(np.abs(target - self.device.positions) / self.device.velocity)),
                         abs(opening - self.device.opening) / .2)
            self._deadline = self.clock() + max(5., 3. * travel)
            return
        if self.clock() >= self._deadline:
            self._finish('failed', ExecutionFailure.TRACKING_ERROR, 'sample did not reach measured target')
            return
        # Path times are minimum dwell constraints; URDF speeds bound simulation.
        dwell = 0. if segment.times is None or sample == 0 else segment.times[sample] - segment.times[sample-1]
        reached = (np.max(np.abs(target - self.device.positions)) <= .002 and
                   abs(opening - self.device.opening) <= .005 and self.clock() - self._issued_at >= dwell)
        if self.device.sequence != self._feedback_sequence:
            self._settled = self._settled + 1 if reached else 0
            self._feedback_sequence = self.device.sequence
        if self._settled < 2:
            return
        if event is not None:
            self.state['attachment'] = event.event == 'acquire'
        self._event('running', transition='sample_completed', grasp=None if event is None else event.event)
        self.pending = False
        if sample + 1 < len(segment.positions):
            self.state['sample_index'] += 1
        elif si + 1 < len(self.command.segments):
            self.state.update(segment_index=si+1, sample_index=0)
        else:
            self._finish('succeeded')
            return
        self._persist()

    def hold(self):
        self._ensure_open()
        if self.state['status'] != 'running':
            raise ValueError('only a running attempt can be held')
        self.device.stop()
        self.pending = False
        self._event('held', 'operator hold')

    def resume(self):
        self.tick()
        if self.state['status'] != 'held' or self.state['needs_replan']:
            raise ValueError('only a healthy held attempt can resume')
        self._event('running', transition='resumed')

    def cancel(self):
        self._ensure_open()
        if not self._active():
            raise ValueError('no active attempt to cancel')
        self._finish('cancelled', ExecutionFailure.CANCELLED, 'operator cancelled; reconcile before retry')

    def _finish(self, status, failure=None, reason=None):
        self.device.stop()
        self.pending = False
        self.state['needs_replan'] = status != 'succeeded'
        self._event(status, reason, failure)

    def reconcile(self, *, empty_gripper=False):
        """Explicit simulation-only acknowledgement; never resumes an old path."""
        self._ensure_open()
        if self._active():
            raise ValueError('stop the active attempt before reconciliation')
        if empty_gripper is not True:
            raise ValueError('explicit confirmation of empty simulated gripper required')
        self.device.advance()
        if self.clock() - self.device.feedback_time > .5 or not self.device.gripper_known:
            raise ValueError('fresh simulation feedback and known gripper required')
        self.state.update(needs_replan=False, attachment=False, scene_revision=self.review.snapshot()['revision'])
        self._persist({'kind': 'simulation_reconciled', 'timestamp_ns': time.time_ns(),
                       'scene_revision': self.state['scene_revision'], 'measured': self.device.snapshot()})

    def decide(self, token, *, accept, expected_revision):
        self._ensure_open()
        # Stop before committing an observation; a concurrent/crashed decision
        # cannot leave motion running under an obsolete scene revision.
        if accept and self._active():
            self._finish('failed', ExecutionFailure.PERCEPTION_FAILED, 'perception review invalidated motion; needs_replan')
        result = self.review.decide(token, accept=accept, expected_revision=expected_revision)
        self._persist({'kind': 'perception_review', 'token': token, 'accepted': accept,
                       'scene_revision': result['revision'], 'timestamp_ns': time.time_ns()})
        return result

    def snapshot(self):
        return {**self.state, 'measured': self.device.snapshot(), 'accepted_scene': self.review.snapshot(),
                'planned': None if self.command is None else
                    self.command.segments[self.state['segment_index']].positions[self.state['sample_index']].tolist()}

    def history(self):
        with self._db() as db:
            return [json.loads(row[0]) for row in db.execute('SELECT body FROM events ORDER BY id')]

    def close(self):
        if not self.closed:
            try:
                if self._active():
                    self._finish('failed', ExecutionFailure.INTERRUPTED, 'owner closed; reconcile and replan')
            finally:
                self.device.stop()
                self.closed = True
                self._lock.close()
