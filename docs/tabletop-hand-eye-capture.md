# Wrist RGB-D preflight and timestamped capture

The app can now record **one stationary, operator-positioned wrist view** per
directory and verify its raw recording offline. These are observations for
field/scene identification. [Offline PCD replay, multi-view session assembly and
synthetic visualizations](tabletop-hand-eye-replay.md) are also implemented.
Live validation and accuracy evaluation remain pending.

## Do we need hardware now?

Software development and failure-path tests run without hardware. Completing
live verification requires these checks:

| Check | Hardware needed | Robot motion |
| --- | --- | --- |
| Serial, simultaneous RGB/depth profile, units, factory geometry and source timestamps | Connected D435 | None |
| Timestamped camera poses and settling evidence | D435 plus stationary UR5e with live `/joint_states` and TF | None |
| Physical alignment, depth bias and the 1 mm identification target | Cameras, robot and independently measured reference | Later, using existing operator controls |

No hardware was operated during implementation. The current project Python
environment lacks `pyrealsense2`; ROS bindings are present. Tests use injected
device/ROS doubles. They establish software behavior, not physical accuracy.

## Commands

Use a Python environment containing the tabletop acquisition dependencies and
`pyrealsense2` (the app's `hand-eye` extra). For `capture`, also source ROS Jazzy
and the robot's state/TF environment. A camera-only preflight does not need ROS.
The capture host must use the same system-time basis as the robot's ROS stamps;
remote-host synchronization uncertainty must be measured before accuracy tests.

The pinned calibration profile is explained in the
[calibration guide](tabletop-hand-eye-calibration.md). From the repository root:

```bash
# Enumerate advertised modes; this does not verify simultaneous operation.
python -m stacking_tabletop.acquisition.hand_eye_capture_cli profiles

# Start both streams and check ten distinct native frame pairs.
python -m stacking_tabletop.acquisition.hand_eye_capture_cli preflight \
  --depth-size 848 480 --fps 15 --count 10 \
  --output apps/tabletop/data/captures/hand_eye/preflight_001

# Hold the arm stationary through the existing controls, with feedback running.
python -m stacking_tabletop.acquisition.hand_eye_capture_cli capture \
  --depth-size 848 480 --fps 15 --count 10 \
  --output apps/tabletop/data/captures/hand_eye/view_001

# This command needs neither ROS nor the RealSense SDK.
python -m stacking_tabletop.acquisition.hand_eye_capture_cli inspect \
  apps/tabletop/data/captures/hand_eye/view_001
```

To use a portable calibration snapshot, put `--profile PATH/profile.yml` before
the subcommand. Output directories must be new. Repeat `capture` after manually
positioning the arm for each additional view; no capture command moves the arm.

Color stays at the selected calibration's 1280x720 dimensions. The initial depth
candidate is 848x480 at 15 Hz. It is **not yet verified on this device/USB link**.
The requested combination must start exactly; the app does not silently switch
resolutions. If it fails, inspect `profiles`, the USB connection and existing
camera driver ownership. The app does not stop another process or reset a device.

## Recorded data and gates

```text
view_001/
  manifest.json             # Version, completion status, limits, settling evidence, hashes
  device.json               # Serial, USB, firmware/SDK, geometry, units, sensor options
  calibration/              # Exact selected artifacts, audit, portable profile
  frames/0000.npz           # SDK-delivered RGB8 and native Z16 arrays, lossless
  frames/0000.json          # Separate source times, metadata, joints and camera poses
  frames/0001.npz
  frames/0001.json
```

No depth alignment, depth correction, spatial/temporal filtering, point-cloud
fusion or nominal table-height shift is applied. Depth retains its integer
values and the device's metres-per-unit scale. RGB8 retains the SDK-delivered
pixels; this is not an archive of any compressed USB transport packets.

The SDK adapter verifies serial and native color factory intrinsics against the
pinned calibration, and records native depth intrinsics/distortion plus
`color_T_depth`. It checks that stream geometry and depth units remain unchanged.
Factory color geometry is distinct from the fitted color intrinsics used by the
hand-eye solve. Sensor options are saved at startup; supported per-frame metadata
includes actual exposure and gain. Missing metadata is explicitly `null`.

After discarding 30 startup pairs, the recorder drains exposures preceding the
source-time gate. It requires strictly increasing frame numbers/timestamps,
SDK global-time domains, frame ages within 500 ms, future offsets within 20 ms,
color/depth skew within 100 ms, and no host clock step above 20 ms. These are
initial acquisition guards. They do not demonstrate millimetre accuracy or
exposure convergence.

Both sensors' `global_time_enabled` options are temporarily enabled and restored
on close. SDK global time maps the camera clock to the host system clock; it is
not evidence of hardware synchronization. The record preserves SDK timestamps
in milliseconds and converted nanoseconds separately from host receipt times.
Device frame/readout timestamps, sensor exposure-midpoint timestamps and exposure
duration metadata are stored when supported, in SDK units (microseconds for
these metadata times). No unverified conversion from readout to exposure midpoint
is made. See the SDK's [frame timestamp definitions](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/h/rs_frame.h)
and [native depth/extrinsics definitions](https://github.com/realsenseai/librealsense/blob/master/include/librealsense2/h/rs_sensor.h).

The robot reader subscribes only to `/joint_states` and TF. It requires a single
joint-state publisher, system-time ROS, exact requested-time TF and bracketing
joint feedback with gaps no larger than 50 ms. It waits briefly for delayed TF;
there is no latest-pose fallback. Raw joint samples cover the initial 0.5 s
settling interval and the complete accepted burst. Joint excursion over the
whole recording must stay within 0.0005 rad of the initial feedback; per-image
tool transforms must stay within 0.2 mm and 0.0005 rad of the initial transform.
These provisional limits reject observable motion; motion between feedback
samples, clock-mapping error and absolute robot calibration remain unmeasured.

Color and depth get separate robot transforms at their own SDK global timestamps:

```text
base_link_T_depth(t_depth)
  = base_link_T_base @ base_T_tool0(t_depth) @ tool0_T_color @ color_T_depth
```

All transforms use metres and `A_T_B` maps B into A. The existing calibration
loader applies the UR `base` versus `base_link` rotation exactly once. Physical
`tool0` is used. The recorder does not use commanded poses or `grasp_frame`.

Each frame is published before its manifest entry. Cancellation/failure keeps
partial data with `status: incomplete`; offline inspection refuses incomplete
captures. Inspection checks required file hashes, calibration consistency, raw
image shapes/encodings, timing, stationarity and stored transform composition.
It works after relocating the directory and removing original calibration files.

## First live acceptance check

1. Run camera-only preflight. Review the actual USB/profile combination, depth
   units, optical geometry, timestamp domains/ages and metadata support.
2. With stationary real robot feedback, capture and inspect a short view. Check
   joint/TF timing and raw images, including color-depth overlay against a target.
3. Establish warmup/exposure convergence and timestamp-to-exposure uncertainty.
   Pin the actual robot model/calibration provenance before accuracy experiments;
   this recorder currently stores measured TF/joints and camera calibration.
4. Continue with offline projection, FEMTO before/after references and the
   [accuracy experiment](tabletop-hand-eye-experiment.md). Use an independent
   measured reference to assess the 1 mm target.

`complete` means that the recording passed its software gates. Every result
retains `accuracy_1mm_verified: false`.
