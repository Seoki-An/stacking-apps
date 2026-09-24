# Offline wrist replay and session assembly

Implemented: native-depth projection, portable multi-view sessions, optional
FEMTO before/after references, synthetic data and offline visualizations.
All processing runs without ROS or a camera SDK. This path produces scene
observations for identification; it does not build new stone models.

For spatial consistency maps of an existing session, use the `diagnose` command
described in the [RGB/FEMTO scene workflow](tabletop-rgb-scene.md).
It compares unchanged table observations in common XY cells and does not
require a known board-to-base pose.

## See a complete synthetic example

From the repository root, using the tabletop app's `hand-eye-replay` extra:

```bash
python -m stacking_tabletop.acquisition.hand_eye_replay_cli demo \
  --output apps/tabletop/data/experiments/hand_eye_synthetic_demo
```

The output directory must be new. This generates four views of four ellipsoidal
stone proxies on a table, with three native RGB/depth pairs per view. A fixed
overview camera produces seven-frame FEMTO-format captures before and after
the wrist sequence through the existing FEMTO writer. These fixtures include
synthetic calibration artifacts; they are not real calibration results or
simulations of measured D435/FEMTO performance.

The wrist fixture uses 320x240 RGB and 160x120 depth, nontrivial hand-eye and
camera poses, a 12 mm color/depth baseline, 0.25 mm depth units, seeded 0.35 mm
axial Gaussian noise and 2.5% dropout. The physical D435 profile is unchanged.
Synthetic timestamps and joint feedback exercise storage and timing contracts;
they do not simulate the UR robot's kinematics or actual clock synchronization.

Open these files in the generated directory:

| File | Contents |
| --- | --- |
| `replay/overview.html` | Standalone interactive 3D clouds and camera positions; drag to rotate, scroll to zoom, click legend entries to toggle views |
| `replay/overview.png` | Static camera/scene overview, with each sensor view colored separately |
| `replay/surfaces.png` | Close 3D and top-down views, colored by height |
| `replay/inputs.png` | Synthetic RGB and native depth for the first exposure of each view |
| `replay/aggregate.pcd` | All wrist observations in `base_link` |
| `replay/aggregate.npz` | Float64 points, source-frame indices and flattened native-pixel indices |
| `replay/replay.json` | Input/output hashes, frame mapping, parameters and completion status |

The HTML embeds Plotly, so viewing it needs no network connection. PNG rendering
uses Matplotlib with its headless backend. Visualization subsamples points only
for display. Full replay exports retain every positive finite depth pixel.
Cloud colors distinguish observations or height; RGB-to-depth color association
is not implemented in this step. Camera frusta are symbolic orientation markers.

## Assemble real recorded views later

First obtain complete view directories using the
[stationary capture command](tabletop-hand-eye-capture.md). List views in source
timestamp order:

```bash
python -m stacking_tabletop.acquisition.hand_eye_replay_cli assemble \
  --views /path/to/view_001 /path/to/view_002 /path/to/view_003 \
  --femto-before /path/to/femto_before \
  --femto-after /path/to/femto_after \
  --reference /path/to/reference.json \
  --output apps/tabletop/data/experiments/session_001

python -m stacking_tabletop.acquisition.hand_eye_replay_cli inspect \
  apps/tabletop/data/experiments/session_001

python -m stacking_tabletop.acquisition.hand_eye_replay_cli replay \
  apps/tabletop/data/experiments/session_001 \
  --output apps/tabletop/data/experiments/replay_001
```

FEMTO captures and reference JSON are optional. Supply both FEMTO captures or
neither. The reference JSON is retained as supplied; it does not establish
independent physical truth simply by being present. Add `--no-visuals` to `replay`
or `demo` for cloud-only output.

```text
session_001/
  session.json              # Status, sensor identity, view intervals, input hashes
  views/000/                # Complete copied native wrist capture
  views/001/
  femto_before/             # Optional complete fixed-field capture
  femto_after/
  reference.json            # Optional supplied reference evidence

replay_001/                  # Separate new directory; source session stays immutable
  frame_0000.pcd
  frame_0000.json            # Source view/exposure, timestamp, transform, pixel mapping
  aggregate.pcd
  aggregate.npz
  femto_before.pcd           # Separate reference cloud, not merged into wrist evidence
  femto_after.pcd
  replay.json
  overview.html
  overview.png
  surfaces.png
```

## Geometry and provenance

Projection uses native depth intrinsics/distortion and recorded metres per unit.
Depth values represent optical Z, not distance along a normalized ray. Invalid
zero/nonfinite pixels are excluded; negative depths and malformed geometry fail.
Each exposure uses its own recorded `base_link_T_depth`, checked against its
robot pose, color/depth extrinsics and pinned hand-eye snapshot.

Supported native models are pinhole and standard five-coefficient Brown-Conrady.
Known RealSense inverse/modified Brown models are accepted only with all-zero
coefficients, where they reduce to pinhole. Nonzero inverse/modified Brown,
fisheye and unknown models fail explicitly. Standard Brown inversion is checked
for convergence and tested against independent forward-projection equations.
See the SDK's [projection/deprojection definitions](https://github.com/realsenseai/librealsense/blob/master/src/rs.cpp).

Wrist exports apply no table-height shift, depth correction, temporal/spatial
filter, ICP refinement, voxel merging or meshing. The aggregate is a concatenation
of observations. PCD coordinates are float32; the companion NPZ retains float64
coordinates and exact frame/pixel provenance. FEMTO reference exports retain
their existing calibration correction, crop and temporal/voxel filtering;
the replay does not register them to the wrist clouds.

Assembly checks capture completeness, calibration hashes, camera identity,
geometry, depth units and source-time ordering. Wrist views must share one
calibration/profile, and must not overlap in time. FEMTO references must be
complete fixed-field `base_link` captures with raw seven-frame evidence, compatible
before/after camera calibration, and timestamps bracketing the wrist sequence.
Synthetic and hardware evidence cannot be mixed.

Inputs are copied with relative paths and hashes. Interrupted assembly or replay
retains an incomplete status. Existing directories are never overwritten.
Inspection checks all session input files; outputs belong in a separate
directory. Each replay records input hashes, source-code hashes and NumPy version.
The session remains usable after relocation without original calibration paths.

## Validation and remaining work

Offline tests cover analytical projection/occlusion, noncommuting transforms,
depth units, distortion, source-pixel provenance, synthetic noise and table height,
deterministic replay, hardware-import isolation, incompatible inputs, timestamp
ordering, corruption and failure status.

```bash
python -m pytest -q tests/test_depth_projection.py \
  tests/test_tabletop_hand_eye_replay.py tests/test_tabletop_hand_eye_capture.py \
  tests/test_tabletop_realsense.py tests/test_architecture.py
```

This synthetic example verifies software behavior. It does not establish a
physical noise model, FEMTO anchoring accuracy, stone identification accuracy
or the 1 mm target. Every output retains `accuracy_1mm_verified: false`.
Next are live profile/capture verification, RGB-depth correspondence checks,
known-stone matching and the [independent accuracy experiment](tabletop-hand-eye-experiment.md).
