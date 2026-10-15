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

While the Execution tab is visible, the console opens a separate read-only ROS
telemetry monitor and renders the measured UR5e configuration. Leaving the tab
closes that monitor, and starting execution closes it before the command-owning
worker connects. The preview does not publish arm, gain, wrench, or gripper
commands. Missing gripper telemetry does not suppress the measured arm display.

Preflight accepts a measured start within 2 degrees of the saved home directly.
A 2–5 degree mismatch is corrected automatically with a slow rigid alignment
move. For a mismatch above 5 degrees, the console displays the saved home as a
cyan robot and requires explicit approval before making the same slow move. The
operator must confirm that the swept path is clear because this current-to-home
segment is not collision replanned. Joint validation remains active, and wrist 3
uses its literal cable-routing branch rather than a wrapped equivalent.

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

At the first motion-review gate, the viewport also draws the complete saved tool
path in green. **Play saved motion** and its slider move a translucent green robot
through every saved segment while the measured robot remains visible. This is a
read-only preview: scrubbing or playback never publishes a hardware command, and
the preview disappears when the review gate closes.

Each command is queued durably before motion. Events and terminal outcomes are
written on the worker thread. Successful steps are skipped on a later start;
failed, cancelled, queued or interrupted attempts block further execution.
**Record physical reconciliation** requires operator notes plus confirmation that
the previous session has stopped, the saved pickup/stack scene has been restored,
and the gripper/object have been checked. It records evidence only; it sends no
commands. A retry starts the whole failed step at home, never an inferred sample.
Journal failures also stop execution and leave durable state for reconciliation.

A post-release acquisition failure is different because restoring the pickup
scene may be impractical. When the journal contains durable `release_completed`
evidence and no later attempt entered pickup motion, Execution enables two
guarded choices.

**Retry stacked-scene capture…** is the preferred recovery after a camera or
reconstruction failure. The operator confirms that the robot is stopped, the
stone remains at the saved placement, and the gripper is empty. Execution then
reconnects at the measured robot pose, shows the cyan capture route for review,
runs only the stacked-scene capture, returns to the saved home pose, marks the
released step complete, and continues with the next step. It does not repeat
pickup, placement, or release.

**Accept already released placement…** skips the missing observation. It marks
the failed step physically complete and makes the next step pending without
sending a robot command. The ordinary reconciliation button retains whole-step
retry behavior and should be used only when the pickup scene has been restored.

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

## Previewing the pick-scene capture region

In Operations → Scene capture, select either wrist capture workflow (with FEMTO
or wrist RGB only), choose the view pattern, then click **Preview capture region
(no motion)**. Compact 9 is the default.
Bringup must supply current joint feedback. This only plans and opens an HTML
footprint plot; it does not move the robot or start either camera. The preview
uses a separate output folder, so the selected capture destination stays unused.
**Start** records the then-current robot pose for the approach and final return.
The capture center is fixed by `apps/tabletop/config/scene_capture.yml`:

```yaml
q_capture_init_deg: [90.0, -80.0, 60.0, -70.0, -90.0, 0.0]
```

All wrist viewpoints are relative to this configured pose. The sequence is:
FEMTO before → open wrist camera → move from the recorded start to the capture
center → capture views, returning to the center between them → close wrist
camera → reverse the approach back to the recorded start → FEMTO after.
In wrist-only mode the FEMTO before/after phases are omitted.
A valid robot-route preview includes both approach and return. These paths use joint-space
interpolation and check joint/cable limits; they are not collision checked.
Cancellation or failure holds the measured pose rather than initiating a return.
Previously saved plans retain their original behavior; regenerate plans to use
the configured capture center.

The plot shows calibrated RGB image footprints on a plane perpendicular to the
central camera axis at the workflow's assumed 40 cm target distance. It reports
the central footprint width/height and the bounding span of all views. This is
not a measured table plane, the bounding span may include gaps, and stereo
overlap, occlusion, texture and reconstruction crops reduce usable PCD coverage.
Image-border distortion is included. Horizon-crossing views are labelled
unbounded rather than assigned a misleading finite size.

Normal capture planning also prints these dimensions and saves `coverage.json`
and `coverage.html` alongside `capture_plan/preview.html`. If route IK fails,
the footprint viewer still opens as **desired coverage only**, with the error
shown and no executable plan produced. FEMTO's crop is
configured separately; this estimate describes the wrist RGB views.

## Stacked-scene wrist PCD after each placement

In **Execution**, **Collect stacked-scene PCD with wrist RGB after each placement**
is enabled by default. Choose it before **Start supervised execution**. This
uses the existing robot/gripper connection and the installed wrist calibration
and Fast-FoundationStereo environment. Keep UR bringup running; no separate pick
capture command or FEMTO capture is needed for the place scene.

After each release, execution first follows only the contact-to-pre-place portion
of the saved retreat. It scans directly from that offset instead of moving home
and then returning toward the stack:

1. Generate at most seven stack-facing views: center/left/right overview views,
   plus two low-side stereo pairs. Saved placed-object extents plus a 60-mm
   margin define the capture region; those models are not substituted for
   observed points. A central pose is accepted only if its two nearby views are
   reachable too. Unreachable side pairs are omitted together and listed in the
   saved plan and review message; the three overview views are required.
2. At the new route-review gate, scrub the cyan robot preview, including entry
   and return. Joint limits and endpoint kinematics are checked; these paths
   are not collision checked. **Continue** starts capture.
   Entry to and return from the reviewed stacked-scene route use free-space
   limits of 0.60 rad/s per joint, 0.12 m/s TCP translation, and 1.20 rad/s
   tool rotation (4× the conservative general capture limits). The shorter
   between-view motions use 0.90 rad/s, 0.18 m/s, and 1.80 rad/s (6× the
   general capture limits). Capture settling and pick/place motion limits are
   unchanged.
3. Record three timestamped RGB-D frames per viewpoint, return to the pre-place
   offset, and close the camera. The capture is assembled and an isolated,
   lower-priority perception process starts. While that process reconstructs
   the PCD and then runs SceneID, execution follows the remaining saved retreat
   to home. SceneID necessarily follows reconstruction because it consumes the
   reconstructed cloud, but neither stage waits for the home motion. The RGB
   images and recorded calibrated robot poses feed Fast-FoundationStereo;
   RealSense depth is retained as auxiliary evidence and is not used as
   reconstructed geometry.
4. At home, wait for perception if it has not completed, then display the
   resulting scene PCD in cyan in the live viewport and show the SceneID result
   path/status. **Open stacked-scene PCD** opens its HTML viewer. Review it and
   press **Continue** before the next placement starts.

Outputs are stored beside the run:

```text
<run-name>_scene_captures/step_<number>_<command-id>/
  plan.json, region.json, robot.urdf, calibration/, preview.html
  capture.json
  wrist/                 # Timestamped original views
  session/               # Portable, hashed wrist-only inputs
  rgb_dense/refined/rgb_dense_voxel_1mm.pcd
  rgb_dense/rgb_only/overview.html
  sceneid_result.json, processing.json
```

The execution journal records the capture location, PCD hash, and SceneID
result. Captures use
`base_link`; the viewport converts them to its recorded UR-base world frame.
The point cloud and SceneID proposal are separate observation layers. They do
not update catalog poses, obstacle geometry, or the saved plan automatically;
operator acceptance and online replanning remain separate work. A completed
but evidence-rejected SceneID solve is saved and shown for review; only a
technical perception failure stops execution.

Hold/Abort remain active during capture and GPU reconstruction. An interrupted
view is not accepted. Camera, motion, or reconstruction failure stops before the
next pickup and retains partial data. A pre-release failure requires whole-step
reconciliation. A post-release failure can use the guarded released-placement
recovery described above. Neither path automatically repeats a placement or
initiates a return motion after capture failure. Review gates remain active for
every scan.
One librealsense `Frame didn't arrive within 1000` gap is treated as transient;
the recorder retries up to five consecutive one-second waits. Five failures stop
the stream-start check, closes the pipeline, waits one second, and opens it once
more before any capture-route motion. The D435 is opened only after route
approval, so a long operator review cannot leave an unread stream running. A
second failure stops the step with a D435 power/USB/competing-process diagnostic
rather than saving an incomplete view. Mid-scan failures are not silently
restarted because resetting the camera invalidates the frame-counter and
timestamp continuity guards.

Repeated stream stalls indicate a transport problem, not a reason to add
unbounded retries. Connect the D435 directly to a USB 3 root port (or a powered
industrial hub), avoid sharing its downstream hub with another streaming video
camera, use a short secured USB 3 cable, and disable USB autosuspend for the
camera during commissioning. Stop RealSense viewers and ROS camera drivers
before execution.

Offline checks cover placement ordering, portable session assembly, failed
capture retention, child-process cancellation, and scene-frame conversion. The
post-placement hardware route still needs commissioning. No 1-mm accuracy is
claimed; this path currently uses recorded hand-eye poses without RGB camera
pose refinement or a FEMTO anchor.

### Offline stacked-scene capture simulation

Render a placed-object snapshot from a synthesized MCTS run and reconstruct its
wrist RGB views without a robot or camera connection:

```bash
python -m stacking_tabletop.acquisition.stack_scene_simulation \
  --run .stacks/tabletop_plan1-pending-bbcb79bae7da4682bab6e15f0a96a454.stack \
  --output apps/tabletop/data/experiments/stacked_scene_sim_002
```

Choose your saved tabletop search run. The default is its last nonempty scene;
`--step N` selects a specific snapshot. Snapshot N contains the stones already
placed before that search step. The example run above produces a five-stone,
two-layer scene with IDs 3, 7, 6, 10 and 9.

The simulator reuses the production stack region and viewpoint generator,
Operations tables and robot/gripper meshes, and saved catalog triangle surfaces.
If a straight-down central camera pose is unreachable, the route tries central
tilts of 25° then 40° toward the robot, preserving the same target and crop.
The route prefers low-side pairs at ±45° azimuth relative to the robot-facing
side, tilted 60°/66° from vertical. Their close spacing supplies stereo overlap
while observing more exposed sides. If either mate cannot be reached on a
continuous IK branch, alternative azimuths/shallower tilts are tried; a failed
pair is omitted together. The ROI is fitted into the camera image throughout
these side arcs. Each pair is visited on one excursion: center → first mate →
second mate → first mate → center. The short transition between mates interpolates
joint positions; it is included in the preview and is not collision checked.
Central fallback, joint-limit checks, and live route review
remain active; paths are not collision checked.

`overview.html` links the robot/camera scene, rendered RGB images, RGB-only PCD,
and ground-truth overlay. `captures/session` is a portable wrist-only session;
no synthetic FEMTO anchor is added. Ground-truth depth is retained separately
from the RGB reconstruction inputs. `metrics.json` reports unsigned distances
to exact stone triangles and reconstructed coverage of surface samples visible
from at least two useful viewpoints. No ICP alignment is applied.

The `stacked_scene_sim_001` experiment used seven views with a 25° central tilt,
producing 34,984 stack points. Median/p95 mesh distance was 0.54/1.83 mm. Rendering
and capture serialization took 8.43 s; RGB reconstruction plus its viewer took
17.27 s. Those times exclude real robot travel and sensor acquisition. Images
have ideal camera poses and procedural texture, so these values do not establish
real-world accuracy. Surface visibility ranged from 22% to 59%; hidden/contact
surfaces remain unobserved despite multi-view capture.

The revised `stack_paired7` route in `stacked_scene_sim_003` used the same seven
captures, a 25° center, and both 60°/66° side pairs. The most occluded stone's
visibility improved from 22.4% to 35.2%, and two-view surface support from 13.4%
to 29.1%. It produced 34,845 stack points with median/p95 triangle distance
0.50/1.66 mm. Reconstruction plus viewer took 8.68 s in this run, versus
17.27 s for the previous route; model loading is included, physical travel and
exposure are excluded. Lower camera paths may require more robot travel despite
the unchanged view count. These are single-run, ideal-simulation measurements.
Contact faces and surfaces hidden behind other stones remain unobserved; the
capture does not fill them using catalog geometry.

### Synthetic PCD fixtures for stacking-core SceneID

Generate an occluded, colored stacked-stone capture and separate ground-truth
poses from a saved tabletop stack. This path renders seven reachable wrist views
without running FoundationStereo or connecting to hardware:

```bash
python -m stacking_tabletop.acquisition.stack_scene_fixture \
  --run .stacks/tabletop_plan1-pending-bbcb79bae7da4682bab6e15f0a96a454.stack \
  --output apps/tabletop/data/experiments/sceneid_fixture_002
```

Use `--step N` to select a nonempty search snapshot. As with the stack simulator,
snapshot N contains objects already placed before that search step. Alternatively,
reuse existing rendered views with `--simulation PATH` instead of `--run`.
The example source above has five stones, IDs 3, 7, 6, 10, and 9. The sibling
`<output>_render` directory retains RGB frames, ideal depths, the camera route,
and the robot/camera scene viewer. Output directories must be new.

The fixture contains:

```text
fixture.json         # Frame, entity/catalog mapping, models, camera poses, hashes
observation.pcd      # Colored, unsegmented, visible stone observations
observation.npz      # points: N×3 metres; colors: N×3 in [0,1]
ground_truth.json    # Exact rendering poses; evaluation only
initial_poses.json   # Starting estimates, deliberately perturbed
models/              # Visual meshes and portable native DSF model arrays
views/               # Individual visible PCDs and source RGB images
```

All observations, ground-truth poses, priors, and the ground plane use
`base_link`, in metres, with poses `[x,y,z,qx,qy,qz,qw]`. Model surface samples
and DSF components remain in the body/catalog frame. Entity IDs 1…N map explicitly
to catalog stone IDs; entity/geometry 0 is the fixed tabletop plane. The final
stone in snapshot order is designated `placed_entity_id` for this fixture.
The local numeric frame ID is 1. Ground height comes from the rendered place
table, not the robot floor.

The scan uses first-hit depth pixels: hidden, underside, and contact faces are
not inserted as observations. Renderer labels provide an ideal foreground mask
removing table/robot points; instance labels are not supplied to SceneID.
By default, independent axial Gaussian noise has standard deviation 0.5 mm
(`--noise-mm 0` gives ideal depth), followed by 1 mm voxel averaging. `--seed`
controls reproducibility. Priors have 3 mm translation and 3° rotation errors.
This tests local pose correction with known identities, not blind recognition.

Run the installed core scan evaluator against the priors and the truth:

```bash
python apps/tabletop/scripts/test_sceneid_fixture.py \
  apps/tabletop/data/experiments/sceneid_fixture_002 --cuda

# Also run the full joint pose/physics solve and score proposed poses:
python apps/tabletop/scripts/test_sceneid_fixture.py \
  apps/tabletop/data/experiments/sceneid_fixture_002 --cuda --solve --progress

# Inspect the observation with captured RGB colors:
python apps/tabletop/scripts/view_rgb_scene.py \
  apps/tabletop/data/experiments/sceneid_fixture_002/observation.pcd
```

Omit `--cuda` for CPU. Results are written to `sceneid_result.json`, including
scan losses, translation/rotation errors and, when solving, native status and
whether an accepted snapshot exists. `stacking_common.planning.sceneid_fixture.load_fixture`
returns `(problem, metadata, initial_poses, ground_truth)` for direct use in core
experiments. The directory is portable: model arrays include mass, inertia,
component friction/sharpness, and visual surface samples; loading requires no
original simulation, catalog, ROS, or platform configuration. File hashes are
checked before native model construction.

Ground truth describes the synthesized rendering geometry. It does **not**
certify a stable DSF stack: visual meshes and physical collision models may
differ, and SceneID may correctly reject the physical configuration. The
simulated scan is an ideal raycast with optional noise, not an RGB MVS estimate
or a calibrated real-camera noise model. An accepted native snapshot still
requires the application's review/rollout checks before any replanning use.

SceneID's `--progress` option streams native ICP iterations, physical correction,
consensus residuals, final statuses, and stage timings to stderr. It requires a
rebuilt stacking-core with `SceneIdConfig.verbose` support; logging is off by
default. To retain the live log alongside the final JSON:

```bash
python -u apps/tabletop/scripts/test_sceneid_fixture.py \
  apps/tabletop/data/experiments/sceneid_fixture_from_run_001 \
  --cuda --solve --progress 2>&1 | tee /tmp/sceneid-progress.log
```

The test runner also saves `visualization/overview.html`: an interactive RGB
capture comparison with initial (orange), proposed (blue), and ground-truth
(green) model surface samples. Coordinates are preserved in the fixture frame;
no additional alignment is applied. A proposal remains labelled unaccepted
when the solver did not produce an accepted snapshot.

Add `--visualize` to open an Open3D window after solving. To inspect an existing
result without rerunning the solver or overwriting its JSON:

```bash
python apps/tabletop/scripts/test_sceneid_fixture.py \
  apps/tabletop/data/experiments/sceneid_fixture_from_run_001 \
  --view-result --visualize
```

Open3D keys: **0** RGB only, **1** initial poses, **2** proposed poses,
**3** ground truth, **C** toggle captured RGB, **G** toggle the GT overlay,
**Q** close. Drag to rotate and scroll to zoom. Without a graphical desktop,
omit `--visualize` and open the saved HTML, whose buttons and legend toggle
pose groups and individual stones. Model samples include hidden surfaces;
the captured PCD contains only observed surfaces.
