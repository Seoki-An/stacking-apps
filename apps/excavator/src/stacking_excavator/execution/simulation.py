"""Deterministic VDK23/stone_grab device emulator with an independent lease.

No ROS imports or hardware command path. Device acknowledgement does not
imply measured completion. Call advance regularly, including when the owner
has stopped, to emulate the device-side watchdog.
"""
import time
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
from stacking_common.profiles.planning import load_planning_profile


class SimulatedDevice:
    def __init__(self, repository, *, clock=time.monotonic):
        root = Path(repository)
        profile = load_planning_profile(root / 'apps/excavator/config/planning.yml')
        joints = [j for j in ET.parse(root / profile.robot.urdf).getroot().findall('joint')
                  if j.get('type') != 'fixed']
        self.joint_names = tuple(j.get('name') for j in joints)
        self.lower = np.array([float(j.find('limit').get('lower')) for j in joints])
        self.upper = np.array([float(j.find('limit').get('upper')) for j in joints])
        self.velocity = np.array([float(j.find('limit').get('velocity')) for j in joints])
        self.positions = profile.robot.home_positions.copy()
        self.opening = 0.
        self.target = self.positions.copy()
        self.target_opening = self.opening
        self.clock = clock
        self.last_tick = clock()
        self.feedback_time = self.last_tick
        self.lease_until = self.last_tick
        self.stopped = True
        self.stalled = False
        self.feedback_enabled = True
        self.gripper_known = True
        self.sequence = 0
        self.accepted_sequence = 0
        self.acknowledgements = []

    def grant_lease(self, duration=1.):
        if not np.isfinite(duration) or duration <= 0:
            raise ValueError('lease duration must be positive')
        self.lease_until = self.clock() + duration

    def send(self, sequence, positions, opening):
        if sequence != self.accepted_sequence + 1:
            raise ValueError('out-of-order device command')
        positions = np.asarray(positions, dtype=float)
        if (positions.shape != self.positions.shape or not np.all(np.isfinite(positions)) or
                np.any(positions < self.lower) or np.any(positions > self.upper)):
            raise ValueError('joint target outside VDK23 limits')
        if opening is None or not np.isfinite(opening) or not -1.14 <= opening <= 0:
            raise ValueError('unknown or invalid stone_grab opening')
        if self.clock() >= self.lease_until:
            raise ValueError('device lease expired')
        self.target, self.target_opening = positions.copy(), float(opening)
        self.accepted_sequence = sequence
        self.acknowledgements.append(sequence)
        self.stopped = False

    def stop(self):
        self.stopped = True
        self.target = self.positions.copy()
        self.target_opening = self.opening

    def advance(self):
        now = self.clock()
        dt = now - self.last_tick
        if dt < 0:
            raise ValueError('simulation clock went backwards')
        self.last_tick = now
        # Expiry is evaluated before movement; never integrate a disconnected gap.
        if now >= self.lease_until:
            self.stop()
        if not self.stopped and not self.stalled:
            self.positions += np.clip(self.target - self.positions, -self.velocity * dt, self.velocity * dt)
            self.opening += float(np.clip(self.target_opening - self.opening, -.2 * dt, .2 * dt))
        if self.feedback_enabled:
            self.feedback_time = now
            self.sequence += 1

    def snapshot(self):
        return dict(positions=self.positions.tolist(), opening=self.opening if self.gripper_known else None,
                    feedback_time=self.feedback_time, sequence=self.sequence,
                    acknowledged=self.accepted_sequence, stopped=self.stopped)
