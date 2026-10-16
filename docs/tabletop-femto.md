# FEMTO acquisition

The tabletop app now owns FEMTO calibration ingestion, ROS acquisition,
synchronized depth bursts, and capture persistence. It uses the existing
`orbbec_camera` driver and `orbbec_camera_msgs` in `~/ros2_ws`. Calibration
producers remain in PoseID; their installation-specific NPZ files are consumed
and snapshotted, not copied into repository defaults.

The capture CLI supplies fixed-field capture and already-settled in-hand views
without commanding the robot. The Operations Console also composes acquisition
with automatic wrist/pivot motion at the saved execution's held-object review
gate. Fixed-field stone identification is available through the combined CLI
and console workflow below. A separate pick-scene mode moves through a reviewed
hand-eye plan, reconstructs wrist RGB geometry and validates the FEMTO pickup
poses before publishing them; see
[wrist RGB and FEMTO scene capture](tabletop-rgb-scene.md#integrated-pick-scene-identification).
Live in-hand registration and online replanning remain later work.

## Field capture and identification in the Operations Console

Install the entry points and perception/GUI dependencies, then source ROS in
the terminal used to start the console:

```bash
python -m pip install -e \
  'apps/tabletop[acquisition,hand-eye,perception,operations,hand-eye-rgb,hand-eye-mvs]'
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
stacking-tabletop-operations .
```

Choose **File → Field scan / identify…**, or the same action in the Planning
toolbar. No saved `.stack` file is needed.

The first two workflow choices retain the fixed FEMTO-only behavior described
below. **Capture wrist RGB + FEMTO and identify** generates the standard
nine-view route from the robot's current central view and moves the UR5e. It
publishes planning poses only after the FEMTO stability and RGB MVS gates pass.

1. Select **Capture and identify** and **Primary**, **Secondary**, or **Both
   FEMTO cameras**. Choose a new output directory and verify the pickup table
   height. Measured `model_<id>.pcd` files come from the inventory profile's
   model directory by default; a different PCD directory can be selected.
2. Stop external camera drivers first. The workflow starts and stops its own
   driver for each camera, sequentially for two cameras. For a single already
   running driver, select **Use an already running camera driver** instead.
   The selected serial and calibrations must still match.
3. Click **Start**. Each capture uses the migrated seven-frame synchronized
   burst and existing depth correction/filtering. In two-camera mode, bounded
   GICP registers secondary to primary; registration failures stop the workflow.
4. Assign a unique stone ID to each cluster, or mark it Skip, in the ordering
   window. **Assign stone IDs automatically** uses the orientation-bank matcher
   instead. Optional manual alignment opens after assignment; its seeds still
   pass through ICP and the configured quality gates.
5. Click **New plan using these poses** after success. The new planning dialog
   has the pickup file and identified IDs filled in. Planning starts only when
   you click **Start planning** there.

Ordering and alignment windows run in the identification subprocess; the
console stays responsive. **Stop** interrupts the workflow and waits for its
owned camera launch to exit. Failed/cancelled data is retained. Active hardware
execution and field capture cannot be started together from the console.
This fixed-field workflow sends no robot or gripper commands.

To retry without recapturing, select **Identify existing capture** and choose a
completed camera capture or `merged` directory. The workflow root and its
`pose_identification` results folder are also accepted: the GUI resolves them
to the capture containing `field_scan.json` before starting. Choose a new output
directory. Previous captures and identification results are preserved.

If identification rejects an assigned stone, the console reports its ID,
cluster, rejection reasons, and measured quality values against their limits.
The saved `pose_identification/stone_poses.json` also retains rejected candidate
transforms and metrics for diagnosis. Field identification checks coverage and
fitting error; it imposes no translation or rotation correction limit relative
to the coarse initial pose. Older captures rejected only for
`large_rotation_correction` or `large_translation_correction` can be retried
without recapturing. For poor coverage or fitting error, verify the assigned ID,
then retry **Identify existing capture** with
**Manually align assigned models before ICP** enabled. Adjust the model in the
alignment window and press **C** to confirm the seeds. Quality checks still
apply, and no pickup file is exported while an assigned stone is rejected.

## Combined field CLI

From the repository root, with the same environment sourced:

```bash
stacking-tabletop-scan-field --visualize
stacking-tabletop-scan-field --both-cameras --visualize
```

Before reinstalling entry points, use the equivalent module command:

```bash
python -m stacking_tabletop.acquisition.field_workflow --both-cameras --visualize
```

Useful flags include `--output-dir PATH`, `--camera secondary`,
`--use-running-camera` (single camera), `--stone-directory PATH`,
`--pick-table-height 0.10`, `--automatic-ordering`, and
`--manual-init-stones 3 9`. The default per-camera burst is seven exposures;
the old wrapper's capture-count and relaxed-ICP overrides are not used.
Segmentation, registration, and fusion settings come from
`apps/tabletop/config/perception.yml` (`--perception-config`). Camera identities
and calibration paths come from `apps/tabletop/config/femto.yml` (`--femto-config`).

The output root contains `field_workflow.json`, per-camera driver logs and
`primary/` or `secondary/` captures. Two-camera runs also contain `merged/`.
The selected camera directory, or `merged/`, receives `pose_identification/`
with cropped clouds, clusters, aligned models and `stone_poses.json`, plus
`pose_data.pkl` alongside it. The terminal and workflow manifest report the
exact pickup-file path, which the planning GUI accepts directly.

For an existing corrected capture, with a fresh result directory:

```bash
stacking-tabletop-identify-field .data/field_pcd/RUN/primary \
  --identification-output .data/field_pose_retry --visualize
```

Identification rotates `base_link` into planning-world axes, fits and aligns the
pickup table height, segments separated stones, and registers known models.
Raw captures are unchanged and corrected depth is not corrected again.
Stone-pose ICP fits the observed cluster to the complete model, then inverts
the result to obtain the model pose in planning world, matching the direction
used by legacy `stacking-planner`. Hidden model surfaces therefore do not act
as source points seeking matches in the partial scan. The table guides initial
height and candidate scoring; the solved pose is not shifted onto the table
after ICP. Earlier tabletop runs applied that shift, which could visibly move
a fitted model away from the observed surface. Re-identify an existing capture
to update those results. Coverage and RMSE gates measure fit quality but do not
prove the correct orientation of a symmetric or partly occluded stone; inspect
the final overlay and use manual initialization when needed.

The September 11 field-capture comparison found that the former tabletop
registration reproduced legacy `stacking-tabletop` results, while legacy
`stacking-planner/scripts/nuc/poseid_field_stone.py` uses observation-to-model
ICP (`POSEID_MODE = 0`) and does not snap its result onto the table. The migrated
field matcher now uses that registration direction and preserves the solved
transform. Its tabletop scales and support-based candidate scoring remain
platform-specific; it is not an exact copy of excavator candidate selection.
On capture `field_20260911_201936_828bc0b5`, median observed-point distance to
the model decreased for all six assigned stones. Stone 10 improved from
4.87 to 1.93 mm, stone 4 from 3.15 to 1.04 mm, and stone 13 from 3.82 to
1.52 mm. Reprocessed poses and `icp_comparison.png`/`icp_comparison.json` are
under that capture's `identification_observed_to_model/` directory. These
distances measure cloud fit, not ground-truth pose error. Stone 7 retains a
support-versus-fit ambiguity: its highest-fitness alternative puts the model's
first-percentile bottom about 14 mm below the reference plane. Review the
assigned model, table reference, and overlay before treating that alternative
as a correct pose.

If an assigned stone fails the quality gates, diagnostics are saved but no new
`pose_data.pkl` is published. Skipped/unassigned clusters are not exported.
The export retains the legacy `{stone_id: (position_xyz, quaternion_xyzw)}`
format in planning-world coordinates. Touching/stacked stones are not jointly
identified by this workflow.

## Start the camera and capture

Use the project Python 3.12 environment with the app's acquisition extra and the
ROS workspace sourced. The `python` prefix below makes the ROS launcher use the
same interpreter that contains the editable stacking packages.

```bash
python -m pip install -e 'apps/tabletop[acquisition]'
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash

python /opt/ros/jazzy/bin/ros2 launch apps/tabletop/launch/femto.launch.py camera:=primary
```

This explicit launch starts the existing Orbbec driver with the selected serial
and 1024×1024, 15 Hz WFOV unbinned depth. It validates both calibration files
before launching the driver. It does not start robot controllers. Camera
identities, calibration paths, namespace and field crop are in
`apps/tabletop/config/femto.yml`:

| Selection | Serial |
| --- | --- |
| primary | CL83C53005L |
| secondary | CL8956200P5 |

In another terminal with the same environment, from the repository root:

```bash
python -m stacking_tabletop.acquisition.cli --camera primary --output .data/field-primary
```

After editable installation, `stacking-tabletop-capture-femto` is the equivalent
entry point. Output must be a new directory. Ctrl+C cancels capture, closes the
owned capture ROS context, and retains incomplete data with failed status. The
separately launched camera driver remains owned by its launch terminal.

For the secondary camera, stop the first driver and launch with
`camera:=secondary`, then capture with `--camera secondary` into a different
output directory. Both retain the legacy `/camera` graph and frame names, so
run them sequentially. Duplicate depth/cloud publishers cause capture to fail.
The existing `DualFemtoFusion.fuse_recorded()` accepts the resulting directories.
No online replanning or excavator/NUC/LiDAR workflows are involved.

## Automatic scans in the Operations Console

Start the selected FEMTO driver as above, along with the commissioned UR5e and
Dynamixel controllers, then open a validated saved run in the Operations Console.
Use **Execution → Start supervised execution** and follow the pickup gates.
At the **place** review gate, the object is held and pickup retreat has finished:

1. Select **wrist** or **pivot** and **primary** or **secondary**. Click
   **Prepare held-object scan…** and select a parent output directory. The console
   creates a uniquely named capture directory; existing captures are not reused.
2. Preparation computes and validates the entire joint path against the recorded
   execution URDF and checks camera identity, exact-time TF and held robot state.
   The preflight burst is discarded. No scan motion starts during preparation.
3. At **scan_review**, scrub the cyan robot preview to inspect entry, scan and
   return poses alongside the measured robot. Review physical object and camera
   clearance as well: this is predefined joint motion without a new collision
   planning pass. **Start scan: move to Q_SCAN and capture** starts the scan;
   **Abort** cancels it. Q_SCAN is `[82, -73, 95, -182, -86, 0]` degrees,
   the same pose used by the legacy stone-model PCD collection workflow.
4. The worker moves, settles within 0.25 degrees for 0.25 seconds, and captures a
   fresh seven-exposure burst at each view. Progress and the output path appear
   in Execution. Arm readiness, fresh feedback and acknowledged grip remain
   required throughout. Hold/Resume and Abort use the same execution controls.
5. On success, the robot returns to the exact joint branch at the original gate.
   A **new placement review gate** requires another Continue before placement.
   The scan approval never also approves placement.

The wrist scan preserves 36 views at wrist-3 angles −180° through +170°, in 10°
steps. The pivot scan preserves 76 captured views, including approach/return
views, two opposite-side ±50° local-X sweeps and a complete relative wrist turn.
Its fixed point is 0.20 m along physical tool0 Z. Neutral wrist rotation starts
at −180° and finishes at +180° instead of the legacy 0° to +360° branch; this
retains full orientation coverage within the adapter's ±300° cable limit.
Pivot IK uses the same recorded URDF as execution, requires joint steps ≤30°,
waypoint pivot error ≤0.1 mm, and sampled interpolation drift ≤4 mm. Entry and
return pass through the legacy Q_SCAN pose. Reference speed caps are 0.25 rad/s
for wrist-only motion, otherwise 0.40 rad/s, with 0.08 m/s and 0.70 rad/s Cartesian
caps. These are reference limits; physical tracking requires commissioning.

The output uses the settled in-hand format below, plus `scan_plan.json` containing
the requested motions, return anchor, robot-model/context hashes and execution
command ID. Execution events record scan approval, motion/view progress and
completion. A burst crossing Hold/Resume is discarded and acquired again.
Abort, grip loss, missing camera data or persistence failure stops the command;
it never automatically returns or releases. Partial capture directories retain
failed/incomplete status, and the saved run requires physical reconciliation.
Camera subscriptions are closed when the scan exits; the independently launched
driver remains running. Use Prepare Capture and reconstruction for subsequent
processing; the console does not yet display registered live point clouds.

## Calibration and synchronization

In-hand preparation records `scan_prepare` and `scan_preflight` progress before
the preview gate. A missing folder dialog means preparation has not been
submitted: the button is enabled only at the held-object placement review
after pickup retreat. Camera preflight sends no motion; approving the preview
starts the move to Q_SCAN.

The October 16 failure delivered 814 fresh FEMTO depth/cloud pairs but captured
zero views because no exposure-time robot state could be matched. Passive
measurement showed about 0.9 s of camera delivery delay, exceeding the former
0.5 s history (and 300-joint-sample buffer at 500 Hz). The transport now retains
2 s of robot/gripper history with room for 2,000 samples. It still requires
state samples within 50 ms of exposure and exact exposure-time tool TF, and
execution separately requires fresh live feedback. Timeout diagnostics name the
missing/mismatched state or TF and include nearest sample offsets. This does not
substitute latest TF/state or authorize motion with missing evidence.

The capture worker verifies the **connected device serial** through
`/camera/get_device_info`. Matching YAML and NPZ names alone is insufficient.
Pose and depth artifacts must agree on serial, observation/depth frames, and the
commissioned depth mode. Legacy UR `base` coordinates mislabeled as `base_link`
retain the established 180-degree conversion.

PoseID observes the color optical frame. Capture computes:

```text
base_T_depth = base_T_color @ inverse(depth_T_color)
```

The legacy camera-link depth-origin relocation is preserved. The current driver
must provide an exact-time `camera_link → camera_depth_optical_frame` transform
that agrees with the recorded calibration within 0.1 mm / 0.01 degrees. The
capture uses the calibrated fixed mount directly; no legacy camera TF publisher
is required and no global mount transform is published by this app.

Depth images and clouds are paired by positive source timestamps within 30 ms,
with bounded buffers, no reuse of an exposure, and a 0.5-second receipt-age
limit. There is no latest-TF fallback. Ten valid fresh pairs are discarded after
each capture request; seven distinct pairs are retained at least 0.2 seconds
apart. Capture, identity lookup and transform waiting share a 60-second deadline.

Native `16UC1`/`mono16` values remain millimetres; `32FC1` values remain metres.
Decoding handles padded rows and byte order. Frame names, 1024×1024 dimensions,
intrinsics, encoding, unbinned mode and uncropped CameraInfo are checked. A
stream change or callback failure stops the worker instead of yielding mixed
capture data.

## Field output

A fixed-field capture contains:

- `raw_pcd/`: seven uncorrected camera-frame XYZ clouds.
- `organized_depth/`: seven original native depth arrays.
- `depth_camera_info.json` and `organized_depth_manifest.json`: reprojection data.
- `calibration_artifacts/`: immutable pose/depth NPZ snapshots with SHA-256 hashes.
- `field_fused_base_link.pcd` and `field_scan.json`: corrected field and provenance.
- `camera_frame.json` and `capture_status.json`: camera metadata and completion.

Field reconstruction retains the legacy policy: temporal median with at least
five supporting/consistent samples within 5 mm, boundary filtering disabled,
distortion-aware backprojection, one axial-depth correction, calibrated transform
to `base_link`, finite field crop, then 1 mm origin-anchored voxel centroids.
An empty filtered or cropped cloud fails the capture. Raw depth and clouds are
never overwritten with their corrected derivatives.

## Settled in-hand views

For a robot and gripper already held stationary by their existing controller:

```bash
python -m stacking_tabletop.acquisition.cli --camera primary --in-hand --output .data/held-view
```

This command only captures. Each exposure requires exact-time `base_link → tool0`
TF plus source-time-matched canonical UR joints and raw Dynamixel position within
50 ms. Bracketing joints are interpolated; sufficiently close single samples are
explicitly recorded as nearest. Local receipt freshness is checked separately.
The burst is rejected if joints move more than 0.25 degrees, gripper position
changes more than five counts, or tool TF moves over 0.5 mm / 0.05 degrees.

A representative raw optical cloud, `frames.csv`, and organized burst manifest
are compatible with `TabletopRawCapture`, Prepare Capture and the reconstruction
workflow. Depth correction is applied once during processing. Snapshot paths are
relative to the capture directory, so captures remain usable after relocation;
missing or changed hashed depth snapshots are rejected. New incomplete captures
are rejected by the field and in-hand loaders; older captures without the new
completion record retain their existing behavior.

For multiple reviewed poses, reuse one `CaptureSession(in_hand=True)` and call
`session.add(camera.capture(in_hand=True, check=...))` after the arm has settled
at each pose. `check` should enforce the coordinator's current arm/gripper health;
`FemtoCamera(cancel_event=...)` can share its cancellation event. No motion is
hidden in acquisition or file writing.

## Validation

```bash
python -m pytest -q tests/test_tabletop_femto_calibration.py \
  tests/test_tabletop_femto_acquisition.py tests/test_tabletop_femto_ros.py \
  tests/test_tabletop_scanning.py tests/test_tabletop_execution_console.py
```

Tests include ported legacy calibration cases, delayed pairing, repeated/stale
frames, source-time state matching, missing exact TF, cancellation, wrong serial,
stream changes, resource cleanup, legacy voxel-grid behavior, incomplete output,
and field/in-hand reader round trips. Scan tests cover native path geometry,
fresh review tokens, exact return, pause-discarded bursts, camera/grip faults,
cancellation, console previews and resource cleanup. Both local camera calibration pairs and
ROS Jazzy bindings were loaded without creating a node. Physical capture and
camera/robot commissioning have not been performed by this migration step.
