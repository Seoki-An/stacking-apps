# Tabletop UR5e, Dynamixel, and saved execution

The device adapters live in `stacking_tabletop.execution`; the ROS message and
node boundary is `stacking_tabletop.ros.hardware.TabletopRosTransport`. They
connect to the existing `~/ros2_ws/src/ur5e_admittance` controller and gripper
node. They neither launch those drivers nor access the Dynamixel serial bus.
The saved-sequence coordinator and Operations Console controls compose these
adapters. [FEMTO acquisition and automatic wrist/pivot scans](tabletop-femto.md)
are integrated at the held-object placement review gate, with preview approval
and an explicit return to placement review. Live registration remains later work.

## Composition

Install `apps/tabletop[execution]` in the Python environment used by the
application, alongside stacking-common and stacking-core. Source ROS Jazzy and
the tabletop ROS workspace to make `rclpy` and the standard messages/services
available. ROS and the Dynamixel SDK are supplied by the ROS environment, not
installed as PyPI dependencies of stacking-common.

```python
from stacking_tabletop.execution import (
    AdapterRuntime, DynamixelAdapter, UR5eAdapter, UR5eKinematics,
)
from stacking_tabletop.ros.hardware import TabletopRosTransport

# Select the robot model verified against the physical robot's calibration.
kinematics = UR5eKinematics("/path/to/verified/ur5e.urdf")
runtime = AdapterRuntime()
with TabletopRosTransport() as transport:
    gripper = DynamixelAdapter(transport, runtime=runtime)
    arm = UR5eAdapter(transport, kinematics, gripper=gripper, runtime=runtime)
    arm.wait_until_ready()
    gripper.wait_until_ready()
    # Readiness checks send no motion or gripper commands.
```

The transport owns a private ROS context/node/executor by default. Alternatively,
pass an existing `node=`; its owner must spin it. Closing the transport removes
only its subscriptions/publishers/client from that node. It never shuts down a
caller's ROS context or sends a shutdown motion/release command.

Both adapters share a runtime and transport. One worker serializes commands and
kinematics access; callbacks update thread-safe feedback. Another thread can
request cancellation with `runtime.cancel_event.set()`. Do not concurrently call
arm/gripper movement methods. Construction does not arm, open, close, or move
devices. Starting the external gripper driver itself still enables torque and
opens the gripper, and its shutdown disables torque. Mock UR hardware does not
mock that external serial driver.

## Arm operations

- `validate_start(q)` checks the saved start against measured joints within
  2 degrees (raw wrist-3 branch), and model/controller FK within 15 mm / 5 degrees.
- `set_rigid_tracking()` and `set_contact_compliance()` preserve the legacy
  commissioned gain arrays. Contact switching includes the 1.6-second gain-ramp
  wait; the sequence coordinator selects the final descent/contact interval.
- `execute_path(path, attached=False, ...)` samples physical **tool0 in base**
  at 100 Hz, bounded to 0.40 rad/s joint speed, 0.08 m/s TCP translation, and
  0.70 rad/s TCP rotation. Native FK is reevaluated to check every interval.
  The controller remains responsible for command acceleration/slew limits.
- Paths must contain six finite joints, stay inside the wrist-3 ±300-degree
  cable envelope, and avoid raw wrist branch jumps. These checks do not establish
  collision freedom or validate the complete saved sequence/home return.
- `attached=True` requires `gripper.require_hold()` throughout lift/carry. The
  calling sequence coordinator must set this from verified attachment semantics.
- Readiness, connectivity, joint/tool freshness, and gripper faults are checked
  during streaming. Endpoints require 10 mm / 5 degrees for 0.25 seconds.
- Any streaming/profile failure or cancellation requests rigid hold at the last
  fresh actual tool pose and preserves the gripper command. `hold_actual_pose()`
  reports whether publication succeeded; it does not acknowledge a physical stop.
  Stale pose or a stopped transport makes that request fail explicitly (`False`).

Progress callbacks report **commanded** samples, separately from measured
`transport.feedback.arm_state()`. Timeouts cover sampling, streaming and settling.
Slow callbacks stretch playback; the adapter never sends a catch-up burst.

The kinematics adapter computes `inverse(root_T_base) * root_T_tool0`, avoiding
both the virtual grasp-frame rotation and a second base/base_link rotation.
The repository's planning URDF still contains nominal factory-independent
kinematics. Selecting it alone does not commission physical execution; reconcile
it with `~/ros2_ws/ur5e_calibration.yaml` and verify the tool mounting first.

## Gripper operations

- `close(opening_angle)` subtracts the legacy **0.40 rad** closing margin and
  maps `0..0.9 rad` to raw encoder `91..1500`; negative model openings down to
  `-0.1 rad` map to closed. The mapping denominator is not the mechanical opening
  limit. Physical calibration/retuning remains separate work.
- `close_to_position(raw)` supports an explicit closing goal. The 10 Hz
  minimum-jerk ramp retains the 500 counts/s bound and 20-count endpoint tolerance.
- Contact detection requires actual closing, distinct telemetry samples and a
  fixed 5-count candidate band; slow cumulative movement resets the window.
- Completion requires a **fresh hold clear**, successful
  `/gripper/hold_current_position` response, and **new positive hold acknowledgement**.
  A cached acknowledgement, repeated feedback or service success alone cannot
  authorize lift. The 15-second default timeout includes the ramp and hold setup.
- `require_hold()` checks the acknowledged latch, fresh telemetry, driver faults,
  and the five-sample / 50-count opening guard. Faults latch; reconnecting or
  retrying is a coordinator/reconciliation decision, not an automatic reset.
- `open()` is the explicit release operation and waits for a new valid endpoint
  sample. Cancellation/failure never automatically opens the gripper.

The transport retains `/grip_flag` (0 CLOSE, 1 MIDDLE, 2 OPEN),
`/grip_position_goal`, `/gripper/state`, `/gripper/fault`, and
`/gripper/contact_hold_goal`. Joint/gripper telemetry uses sensor-data QoS;
readiness, faults and hold acknowledgements use reliable transient-local QoS.
Gripper JointState positions remain **raw encoder counts**, not radians.

## Migration baseline and verification

The pure sampler/mapping/contact tests were ported from legacy
`scripts/test/test_ur5e_execution.py`. Adapter integration tests use simulated
time, fake telemetry, service results, and publishers. ROS transport tests cover
the exact graph/QoS, frame validation, cleanup and private/shared node ownership.
Native FK tests verify physical tool0 against the current robot asset.

The baseline is source-backed: legacy code uses linear contact stiffness **1200**
and gripper close speed **500 counts/s**, despite older documentation listing
800 and 400. These values were retained. Added checks reject non-rigid/nonfinite
poses, duplicate joint identities, readiness loss, and stale feedback throughout
motion. Legacy files and external ROS packages remain unchanged.

## Supervised saved-plan execution

Open a newly recorded motion-planning `.stack` run in the Operations Console and
select **Execution**. Loading is passive. **Start supervised execution** connects
to the existing controller graph; no drivers are launched. The worker checks
readiness, measured home joints, model/controller FK, and saved grasp-event FK
before presenting the first gate. An existing driver hold acknowledgement blocks
startup until the previous grasp has been reconciled.

The executor uses the resolved profile recorded in the run. It verifies the
context and plan hashes, canonical joint order, finite table bounds, all recorded
asset bytes and URDF dependencies, home branch, segment continuity and wrist
limits. Direct plans must contain these five segments:

| Segment | Attachment | Operator action |
| --- | --- | --- |
| `pick_approach` | Free | Review pickup before OPEN; approve closing at its endpoint |
| `pick_retreat` | Attached | Requires a fresh acknowledged gripper hold |
| `transfer` | Attached | Review the held object and saved placement before carrying |
| `place_approach` | Attached | Approve release at its endpoint |
| `place_retreat` | Free | Return to recorded home after OPEN completes |

Native acquire/release events and attachment modes now survive planning and
journal serialization. Acquisition belongs to the last `pick_approach` sample;
release belongs to the last `place_approach` sample. Older files without these
semantics remain viewable but must be regenerated for execution. Regrasp plans
are not supported. Recorded `grasp_steps` splits `pick_approach` into rigid travel
and the final compliant contact portion. `transfer` already reaches the
placement approach pose, so all of `place_approach` uses contact compliance;
that segment may contain exactly `grasp_steps` samples. Pickup requires more
than `grasp_steps` samples, while placement requires at least that many.
After release, rigid tracking is restored for retreat. Arm references are
retimed by the adapter rather than following the viewer's playback speed.

At the close/release gates, joint corrections use 0.5-degree increments, limited
to ±5 degrees per joint, with a Reset control. Corrections taper to zero over the
following retreat. These are supervised corrections, without a new collision
planning pass. Every gate has a unique token so stale clicks cannot approve the
next phase. There is no automatic approval.

**Hold** requests rigid actual-pose hold and confirms stationary measured joints
before reporting held. **Resume** rechecks the retained joint reference and
restores contact compliance when applicable. Supervised pause time does not use
up motion deadlines; feedback freshness still uses elapsed real time. Hold is
deferred during a bounded gripper operation. **Abort** cancels at the next device
check, requests actual-pose hold, and never opens a held gripper. A hold request
is not a hardware emergency stop acknowledgement. Closing the console waits for
the worker to exit; changing runs or exporting video is blocked while it runs.

The Execution viewport renders recorded collision boxes and hashed object models,
measured robot joints and raw gripper opening. A solid object pose is estimated
from the measured grasp transform; the translucent blue pose is the saved
placement. The release estimate is retained after retreat and in successful
journal results. This estimate does not measure object slip or confirm support;
the operator must inspect the actual grasp and placement.

Each command is queued durably before motion. Events and terminal outcomes are
written on the worker thread. Successful steps are skipped on a later start;
failed, cancelled, queued or interrupted attempts block further execution.
**Record physical reconciliation** requires operator notes plus confirmation that
the previous session has stopped, the saved pickup/stack scene has been restored,
and the gripper/object have been checked. It records evidence only; it sends no
commands. A retry starts the whole failed step at home, never an inferred sample.
Journal failures also stop execution and leave durable state for reconciliation.

Run the offline checks with a Python environment containing pytest, SciPy and
stacking-core:

```bash
python -m pytest -q tests/test_tabletop_execution_legacy.py \
  tests/test_tabletop_hardware_adapters.py tests/test_tabletop_ros_hardware.py \
  tests/test_architecture.py tests/test_execution_artifacts.py \
  tests/test_tabletop_saved_execution.py tests/test_tabletop_execution_console.py \
  tests/test_tabletop_scanning.py
```

Physical commissioning of the integrated sequence remains pending. Offline tests
use fake devices; they do not move the UR5e or operate the gripper. FEMTO
acquisition and automatic scan trajectories are implemented. Live in-hand
registration remains next.
