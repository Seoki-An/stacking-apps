# Hand-eye capture and accuracy experiment

Current priority: [multi-view RGB/FEMTO scene capture](tabletop-rgb-scene.md),
using wrist RGB MVS for complementary geometry and fixed FEMTO as the operational
reference. Stacked-scene pose solvers remain downstream work. The depth-calibration
capture sequence is paused; existing depth results remain comparison evidence,
not an accepted D435 correction.

The [automated scene capture CLI](tabletop-automated-scene-capture.md) now generates
near and oblique wrist views from one seed and brackets them with FEMTO captures,
using persistent UR bringup. Offline tests pass, and the first live board/table
sequence (`scene_capture_001`) completed with verified replay. Stone-surface ICP
comparison and physical accuracy validation remain pending.

Status: first implementation milestone, 2026-09-15. The
[calibration consistency audit and snapshots](tabletop-hand-eye-calibration.md)
are implemented and verified on the local recorded calibration inputs. The
[native RGB-D recorder and raw offline inspection](tabletop-hand-eye-capture.md)
are implemented with offline tests. [PCD replay, multi-view assembly and
synthetic visualizations](tabletop-hand-eye-replay.md) are implemented too.
Initial live D435 captures and calibrated replay have now been exercised on the
local hardware. Spatial table consistency diagnostics and a
[RGB/FEMTO scene workflow](tabletop-rgb-scene.md) are available. Board-pose
analysis, known-stone replay matching and independent
physical accuracy validation remain pending.

## Objective

Produce a self-contained, replayable dataset from the wrist D435 and fixed
FEMTO, and measure whether those observations support field and placed-scene
stone identification at the user's target of **1 mm or better at stone
surfaces**. The outputs are observations and known-object pose estimates.

Completion means reproducible capture/replay and an evidence-backed accuracy
report. The report may establish that the target is unmet or not yet measurable;
software completion alone does not establish physical accuracy.

## Starting point and findings

- Existing FEMTO capture already records native depth, calibration snapshots,
  corrected clouds and completion status. Reuse its output and behavior; see
  [the FEMTO guide](tabletop-femto.md).
- Existing held-object wrist/pivot scans rotate a stone under a fixed camera.
  This milestone records stationary objects using a moving camera. Each exposure
  therefore needs its own camera-to-world transform.
- Local PoseID configuration identifies a D435 with serial `135522074611` and
  color at 1280x720, 15 Hz. Its USB 2.1 comment concerns an earlier color-only
  setup. Supported simultaneous RGB/depth modes and bandwidth need a live check.
- Local `handeye_result.npz` embeds the same color intrinsics as the current
  `d435_intrinsic_result.npz`. `handeye_result_refined.npz` embeds different
  intrinsics. File naming or modification time is insufficient to choose a set.
- The refined-result writer copies staged diagnostic fields while replacing
  the transform. Those copied residuals do not measure the refined transform.
- These hand-eye/intrinsic artifacts omit an explicit camera serial and color
  optical-frame identity. They need a documented installation binding; the
  loader must not invent that provenance.
- PoseID's current `handeye_node.py::_capture_color` records host ROS time after
  receiving the color image. The new recorder must preserve SDK source
  timestamps and domains separately from receipt time.

The above describes files inspected on this date, not a connected-device test.
Calibration production remains in `~/ros2_ws/src/poseid`; this app loads explicit
accepted artifacts and snapshots them.

## Work packages

### 1. Establish a coherent calibration and stream profile

Implement a read-only calibration audit and a separately invoked device preflight:

1. Validate hand-eye direction, parent/tool/optical frames and rigid transforms.
2. Check the selected color intrinsics against the hand-eye solve's intrinsics,
   image size and distortion convention. Keep factory and fitted values distinct.
3. Verify the actual device serial, supported stream combination, USB connection,
   firmware, SDK version, exposure and emitter state when a device is connected.
4. Record native depth intrinsics, distortion, metres per depth unit and
   color/depth extrinsics. Color calibration does not supply depth calibration.
5. Record the calibrated robot model and pose source. Preserve the difference
   between UR `base`, ROS `base_link`, physical `tool0` and planning world.
6. Validate on held-out target views. Record reprojection/consistency separately
   from independently measured physical error.

RealSense's [D435 tuning guide](https://dev.realsenseai.com/docs/tuning-depth-cameras-for-best-performance/)
provides a depth-mode/exposure baseline. Select a supported profile through
preflight and measurements; the existing color profile is not a depth profile.

### 2. Implement a settled-view recorder

Begin with operator-positioned views through the existing robot controls.
The capture command verifies settling and records; automatic viewpoint motion
can be added after the measurement path is validated.

Each view records a burst of distinct native RGB/depth exposures, with:

- Image dimensions, encoding, intrinsics, distortion and native depth units.
- SDK frame number, source timestamp/domain, receipt timestamps and the mapping
  to the robot clock. Preserve measured synchronization uncertainty.
- Robot joints and measured `base_link_T_tool0` matched to each exposure;
  RGB and depth timestamps are individually retained and checked.
- Color/depth extrinsics and the selected hand-eye calibration snapshot.
- Camera identity, stream settings, exposure/gain/emitter state and acquisition
  errors. Changes invalidate the view or create a new explicitly configured run.
- Stationarity evidence over the complete burst, with limits chosen against
  the experiment's error budget rather than copied from FEMTO defaults.

No latest-TF substitution or reuse of an exposure. A clock reset, unmapped clock,
missing pose, stream change or motion produces an explicit invalid/incomplete
record. Keep original pixels without annotations or additional lossy encoding.
Keep any native compressed source needed to reproduce decoding. Stereo IR is
optional evidence if a later experiment needs it.

Capture FEMTO and D435 sequentially initially, with the scene unchanged and the
arm outside the FEMTO region. Save FEMTO before and after the view sequence to
check for scene changes. The session includes the complete FEMTO captures.

### 3. Implement portable storage and offline replay

Proposed session layout:

```text
apps/tabletop/data/captures/hand_eye/<session>/
  manifest.json              # Version, status, settings, frames and file hashes
  calibration/               # Immutable artifacts and installation binding
  reference.json             # Target dimensions/poses, method and uncertainty
  femto_before/              # Existing complete FEMTO capture format
  views/000/                 # Native image/depth burst and per-exposure metadata
  views/001/
  femto_after/
  results/<processing-run>/  # Derived clouds, poses and report
```

Paths within a session are relative. Use a new directory, incremental atomic
manifests, per-view status and a final session status. Cancellation preserves
completed views and identifies interrupted data. Preserve raw input on replay;
each processing run gets a new output directory with code/dependency versions,
parameters and input hashes.

Replay runs without ROS, a camera SDK or connected hardware. It must:

1. Verify schema, hashes, dimensions, depth units and transform validity.
2. Backproject native depth with its own intrinsics/distortion.
3. Transform each exposure into a single explicit world frame.
4. Export raw and filtered per-view PCDs, retaining view/sensor provenance in
   sidecars; produce an aggregate preview as a derived product.
5. Run existing known-stone fitting and save the resulting pose and diagnostics.

Use `A_T_B` to mean a transform mapping B coordinates into A:

```text
world_T_depth(t) = world_T_base_link
                @ base_link_T_tool0(t)
                @ tool0_T_color
                @ color_T_depth
```

For this experiment, retain calibrated world coordinates. Table fitting is a
measurement; it must not silently move the observations to a nominal height.
Keep calibrated-only results alongside any registration-corrected results, so
ICP cannot hide camera-pose errors. Fit one stone pose across observations;
independent per-view stone fitting is also useful as a consistency diagnostic.

### 4. Collect a small pilot dataset

Use the following as a starting protocol; range/angles must pass actual camera
visibility, robot reach and clearance checks. They are not commissioned paths.

| Test | Suggested acquisition | What it measures |
| --- | --- | --- |
| Stationary target | At near/middle/far intended working distances, retain at least 30 valid frames per condition | Depth variability, bias against a measured reference, valid-pixel coverage |
| Multi-view target | About 8 translated viewpoints: overview and oblique views, spanning two distances | Camera-pose consistency, distortion effects, range/angle dependence |
| Return repeatability | Revisit the same viewpoint near the beginning and end; repeat the sequence three times | Return consistency and temporal drift |
| One known stone | Same view sequence with a stone fixed in place | Pose repeatability, ambiguity, held-out-view fit and visible coverage |
| Placed-scene pilot | Repeat around two touching/partly occluded known stones | Whether usable evidence survives scene occlusion |

Use short native bursts, initially around 10 distinct pairs at each multi-view
stop. Refine burst length after observing temporal correlations. Individual
frames in a burst are not independent trials. Record warmup/exposure convergence
and reference-target stability; do not move the target during a sequence.

Keep a held-out repetition for final evaluation. Any fitted depth correction,
thresholds, camera alignment or algorithm selection uses the development data.
Fix all such choices before evaluating the held-out repetition.

### 5. Evaluate separately defined quantities

| Quantity | Interpretation |
| --- | --- |
| Depth standard deviation and plane-fit residual | Noise/flatness; a best-fit plane can absorb an absolute offset |
| Error against measured target geometry/pose | Absolute dimensional/pose evidence within the reference uncertainty |
| Cross-view and D435/FEMTO disagreement | Consistency; shared calibration errors can remain invisible |
| Repeated stone poses | Repeatability; does not establish the true pose |
| ICP residual and held-out-view residual | Fit to observations; does not independently establish true pose |
| Stone surface displacement against independent true pose | The quantity closest to the user's identification target |

For independently known stone pose `T_ref` and estimated pose `T_est`, report
`norm(T_est @ p - T_ref @ p)` over consistently sampled reference-model surface
points. Also report translation error and rotation error separately. Model
surface uncertainty must be recorded; an inaccurate catalog model limits what
this test can establish.

Report signed bias where meaningful, median, RMS, p95, maximum, coverage,
rejected views/objects and failure rate. Preserve per-condition results. Do not
silently replace the user's 1 mm target with a p95-only or best-view criterion.
An empirical pass applies only to the tested conditions and reference accuracy.

Reference options depend on available equipment:

- A rigid ChArUco board with checked dimensions and a measured step/block can
  support reprojection, flatness and relative-dimension checks.
- A fixture/stage with independently known displacements supports relative-pose
  tests. Full absolute world-pose testing also needs its world transform measured.
- A stone requires an independently established model-to-fixture/world transform
  for absolute pose testing. Calipers alone do not establish that transform.

A board pose computed from the same D435 images, robot command targets, FEMTO
clouds or a pose optimized on the test images cannot serve as independent truth
for the complete camera-to-world measurement chain. When independent reference
data is unavailable, label the report as consistency/repeatability only.

First compare FEMTO-only, one-view D435, multi-view D435 and FEMTO-assisted
multi-view fitting. Keep source weights explicit and preserve separate clouds.
If depth bias limits performance, use the saved RGB views for a pose-guided MVS
comparison; [COLMAP supports supplied camera poses](https://colmap.github.io/faq.html#reconstruct-sparse-dense-model-from-known-camera-poses).
MVS is a follow-up based on evidence, and does not require replacing the recorder.

## Implementation sequence and ownership

1. **Calibration audit + session schema + reader/writer:** tabletop calibration
   ingestion; common immutable numerical values and versioned artifact codecs.
2. **Offline replay + synthetic tests:** common backprojection/frame processing
   and error metrics, with tabletop profile/catalog composition.
3. **D435 recorder + read-only robot-state transport:** device SDK and ROS remain
   under tabletop; no hardware imports are required by replay.
4. **Pilot acquisition + report:** tabletop CLI first, producing portable data,
   per-view/aggregate PCDs, JSON/CSV metrics and a Markdown report.
5. **Decision:** document the supported operating envelope and dominant errors,
   then select calibration improvements, depth fitting or the MVS experiment.

Meaningful tests cover nontrivial color/depth/world transform composition,
distortion and depth-unit mismatch, source/receipt clock differences, missing TF,
motion, duplicate exposures, stream changes, cancellation, relocation, corruption
and exact raw-array round trips. Synthetic known-pose fixtures verify error
metrics and expose a global bias even when plane/ICP fitting residuals are small.
Existing FEMTO and field-identification behavior must retain its regression tests.

## Completion checklist

- A moved copy of a completed capture replays without ROS or camera hardware.
- Raw data and calibration provenance survive cancellation and reprocessing.
- Native and processed observations can be compared in one consistent world.
- One known stone and a small placed-scene case have repeatable recorded trials.
- The report separates fit, consistency, repeatability and independent accuracy.
- The report states whether 1 mm is demonstrated, missed, or not yet measurable,
  and identifies the next change justified by those measurements.
