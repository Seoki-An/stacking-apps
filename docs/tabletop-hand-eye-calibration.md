# Wrist calibration consistency

The first hand-eye milestone now has a content-pinned D435 calibration profile,
an offline validator, a portable snapshot writer and recorded-data verification.
Camera acquisition and physical accuracy validation are subsequent steps in
[the experiment plan](tabletop-hand-eye-experiment.md).

## Selected baseline, 2026-09-15

`apps/tabletop/config/hand_eye.yml` explicitly selects:

| Input | Selection |
| --- | --- |
| Hand-eye transform | PoseID `handeye_result.npz`, staged Tsai solve |
| Color intrinsics | PoseID `d435_intrinsic_result.npz`, 1280x720, OpenCV k3 |
| Installation binding | PoseID `config/handeye_params.yaml` |
| Configured camera | D435, serial `135522074611` |
| Robot transform source | TF, UR `base` to physical `tool0` |
| Application optical frame | `d435_color_optical_frame` |

The profile pins all three input files by SHA-256. If a calibration tool
overwrites any input, loading fails instead of silently adopting new values.
Relative artifact paths resolve against the profile directory. The local
PoseID originals have not been modified.

The camera serial is established by the installation YAML, not by camera
metadata in these legacy NPZ files. The optical-frame label names the color
coordinate system consumed by PoseID's solver; it is not a claim that a live
ROS publisher uses that name. Present artifact serial/frame metadata is checked
against the binding. Missing legacy metadata remains an explicit provenance
limitation in the audit report.

### Why the refined export is excluded

The current staged result embeds the selected intrinsics exactly:
`fx=903.408325`, `fy=903.035583`. The existing refined export embeds different
intrinsics (`fx=907.389099`, `fy=907.394836`) and different distortion coefficients.
Its extrinsic differs by approximately 0.519 mm and 0.441 degrees.

The archived `refined_extrinsics.npz` reports `solver_success=false` because
the evaluation limit was reached, and `hand_eye_frozen=true`. The refined writer
copies staged diagnostic fields without recomputing them. Consequently, neither
its filename nor copied residuals justify selecting that transform with the
current intrinsic artifact.

The first loader supports the known staged PoseID OpenCV hand-eye methods.
Joint-refined exports are rejected even if their intrinsics match: they need a
separate validated solver-provenance contract before support is added. This is
a limitation of this initial loader, not a claim that joint refinement is
unsuitable.

## Recorded-data verification

The existing PoseID offline solvers were run against copies of their inputs:

- Re-solving the current 64-view refined hand-eye log reproduced **every array**
  in the selected staged hand-eye artifact, including its 59 inliers.
- Re-solving each FEMTO's recorded image pairs using this same hand-eye/color
  pair reproduced the existing FEMTO pose matrices exactly.
- Both replayed FEMTO pose/depth combinations passed `FemtoCalibration`,
  including serial, optical-frame/depth-origin and depth-mode checks. Their
  computed `base_link_T_depth` matrices exactly match the existing profile.

Thus the existing FEMTO extrinsics are consistent with this selected D435
baseline. No change to the operational `femto.yml` was needed.

| Recorded consistency quantity | Mean | Maximum |
| --- | --- | --- |
| Hand-eye inlier board-position spread | 1.100 mm | 2.387 mm |
| Primary FEMTO candidate-position spread | 3.038 mm | 5.914 mm |
| Secondary FEMTO candidate-position spread | 3.316 mm | 8.900 mm |

These are calibration-data consistency residuals. They are not independent
world-pose errors, held-out accuracy measurements or proof of the 1 mm target.

### Reconciliation with the earlier perception thread

Thread `019fff6f-6839-70c0-909f-a8dd4b21c3b8` (August 14-20) establishes
primary `CL83C53005L` as the fixed registration reference and secondary
`CL8956200P5` as the cloud aligned to it. It also identifies the same wrist
D435 and the physical `base -> tool0` calibration chain. These roles are
preserved.

Its August hand-eye result was a 52-pose Park solve with hash
`e89c81004450c4ea87030e4963ca7edf9f021972dd1b3950ecf2222ed16560ec`.
The same source filename now contains the replay-verified 59-inlier Tsai solve
selected here. The old thread's quoted residuals and transform are historical,
not the contents of the current file. The primary depth offset still agrees
with its approximately -25.188 mm value; the current secondary artifact uses
-6.911 mm instead of the thread's -6.704 mm. Both current serial-specific depth
artifacts are preserved in the snapshot and validated by the existing loader.
The consistency audit does not refit depth or restore historical coefficients.

That thread also established post-settle source-time rejection and bounded
joint matching after finding approximately one-second camera backlogs. Those
requirements remain part of the future recorder. Its explicit designated
pick-table-height convention belongs to operational field identification;
the accuracy experiment retains the original calibrated observations so that
normalization cannot conceal a global error. The September migration thread's
later removal of per-stone post-ICP snapping remains in effect.

Local evidence is under:

```text
apps/tabletop/data/calibration/hand_eye_20260915/
  profile.yml                     # Portable, pinned D435 inputs
  hand_eye.npz
  color_intrinsics.npz
  installation.yml
  audit.json
  hand_eye_source_log.npz
  hand_eye_replayed.npz
  hand_eye_replay.log
  excluded_hand_eye_refined.npz
  excluded_refinement_report.npz
  verification.json
  femto_replay/
    replay_inputs.json
    replay_report.json
    femto.yml
    primary.log
    secondary.log
    primary/                      # Copied source captures and replayed outputs
    secondary/
```

Generated calibration/data directories are Git-ignored. Keep the complete
directory with the experiment data. The D435 snapshot independently loads after
relocation. Evidence source paths are descriptive; loading uses relative local
paths and the pinned hashes.

## Audit and snapshot commands

From the repository root in the project Python environment, audit without
opening a device or writing files:

```bash
python -m stacking_tabletop.acquisition.hand_eye_cli
```

Diagnose an alternative and create a new snapshot directory:

```bash
python -m stacking_tabletop.acquisition.hand_eye_cli \
  --compare ~/ros2_ws/src/poseid/calibration_output/handeye_result_refined.npz \
  --output apps/tabletop/data/calibration/hand_eye_review
```

The selected pair must pass before any output directory is created. An
incompatible comparison is reported without selecting it. The CLI exits with
success when the selected baseline passes, even when an explicitly compared
alternative is rejected. `--output` must name a new directory; input artifacts
are copied byte-for-byte and the loadable profile is written last.

Reload the saved snapshot, including after moving its directory:

```bash
python -m stacking_tabletop.acquisition.hand_eye_cli \
  --profile apps/tabletop/data/calibration/hand_eye_20260915/profile.yml
```

After refreshing the editable tabletop installation,
`stacking-tabletop-audit-hand-eye` is the equivalent entry point. The module
command works with the existing editable installation.

### Integration contract

```python
from stacking_tabletop.acquisition.hand_eye import HandEyeCalibration

calibration = HandEyeCalibration.load(profile_path)
base_link_T_color = calibration.base_link_T_color(measured_base_T_tool0)
```

The measured transform must use the base declared in the profile. The loader
applies the established UR `base` to `base_link` conversion exactly once.
`tool_T_color` maps color optical coordinates into physical `tool0`; it is not
a virtual grasp-frame transform.

The [native RGB-D recorder](tabletop-hand-eye-capture.md) calls `validate_color_stream` with the actual serial,
native width/height and factory intrinsic matrix/distortion read from the
device. Factory intrinsics are checked against recorded factory values;
the fitted color matrix remains separate. That check does not establish
native depth geometry, source-time synchronization or physical accuracy.

## Remaining hardware verification

1. Verify live camera identity and native color profile.
2. Record depth intrinsics, distortion, native units and depth/color extrinsics.
3. Establish exposure-clock mapping and measured robot-pose synchronization.
4. Validate the physical calibration against held-out and independent evidence.

This step resolves offline calibration consistency. Its report deliberately
keeps `accuracy_1mm_verified=false`.

## Tests

```bash
python -m pytest -q tests/test_tabletop_hand_eye_calibration.py \
  tests/test_tabletop_femto_calibration.py tests/test_tabletop_femto_acquisition.py \
  tests/test_tabletop_femto_ros.py tests/test_architecture.py
```

The tests exercise incompatible intrinsics/distortion, frame/source/serial
mismatches, inconsistent transform aliases, unsupported refined provenance,
input hash changes, portable snapshots, interrupted writes, native/fitted
intrinsic separation and nontrivial base/tool/color transform composition.
