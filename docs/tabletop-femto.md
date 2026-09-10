# FEMTO acquisition

The tabletop app now owns FEMTO calibration ingestion, ROS acquisition,
synchronized depth bursts, and capture persistence. It uses the existing
`orbbec_camera` driver and `orbbec_camera_msgs` in `~/ros2_ws`. Calibration
producers remain in PoseID; their installation-specific NPZ files are consumed
and snapshotted, not copied into repository defaults.

The capture CLI supplies fixed-field capture and already-settled in-hand views
without commanding the robot. The Operations Console also composes acquisition
with automatic wrist/pivot motion at the saved execution's held-object review
gate. Live registration and online replanning remain later work.

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
   planning pass. **Continue** starts the scan; **Abort** cancels it.
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
