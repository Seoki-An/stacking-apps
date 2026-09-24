# Wrist RGB and FEMTO scene capture

This is the maintained overview for automated multi-view capture, RGB-derived
scene geometry, FEMTO comparison and offline scene assembly. Detailed reports
stay with their ignored experiment outputs instead of separate repository docs.

## Purpose and sensor roles

The wrist camera gathers additional viewpoints for pick and placement scenes.
The product is replayable scene geometry for later pose solvers, not a canonical
stone reconstruction, watertight mesh or verified ground truth.

| Source | Role |
|---|---|
| Wrist D435 RGB | Dense geometry through known-pose multi-view stereo (MVS) |
| Robot and hand-eye calibration | Camera-pose priors in `base_link` |
| Fixed FEMTO | Broad scene/table reference and before/after bracket |
| Wrist D435 depth | Preserved auxiliary evidence; not accepted for fusion |
| Measured stone PCDs | Offline shape and provisional identity comparison |

FEMTO and RGB remain separate geometry evidence layers. Pick identification uses
a balanced first catalog pose fit using both point clouds. FEMTO supplies
coarse stone regions and the metric frame before RGB reconstruction. It does not install a global RGB-to-FEMTO calibration correction,
and absolute 1-mm accuracy remains unverified.

## Capture

The integrated workflow uses nine wrist views: center, four near translations
and four diagonal oblique views. Every view looks at a target 0.4 m along the
central camera axis and returns to the seed. FEMTO is acquired immediately
before and after the wrist sequence. The workflow derives and validates the
joint route from the current central view; no separate plan command is needed.
The generated audit artifacts follow the format described in
[automated scene capture](tabletop-automated-scene-capture.md).

## Offline processing

The replay CLI exposes these stages. Each command writes a new directory and
records hashes that bind the result to its inputs.

| Command | Result |
|---|---|
| `replay` | Back-project recorded wrist depth for inspection |
| `diagnose` | Compare uncorrected table heights |
| `rgb` | Sparse RGB tracks and triangulation feasibility |
| `rgb-refine` | Diagnostic camera-pose refinement with metric priors |
| `rgb-dense` | Known-pose COLMAP PatchMatch MVS |
| `rgb-filter` | Disjoint-view agreement filtering and coverage classes |
| `scene-assemble` | Portable FEMTO/RGB evidence layers |
| `scene-inspect`, `scene-replay` | Verify and render a portable scene |
| `surface-check` | Shared rigid-fit and held-out surface diagnostics |
| `catalog-check` | Fixed-scale comparison with measured stone PCDs |

The RGB route is:

```text
recorded RGB + robot/hand-eye poses
  → sparse correspondence checks
  → diagnostic pose refinement
  → dense MVS depth maps
  → 1-mm sampling voxels
  → disjoint-view agreement filtering
  → labelled scene evidence
```

The 1-mm voxel size controls sampling density; it is not an accuracy claim. MVS
uses RGB appearance and does not use RealSense or FEMTO depth to estimate depth.
FEMTO defines coarse regions before MVS and contributes geometry to the first
catalog pose fit afterward; it never changes the MVS depth values.

## Integrated pick-scene identification

The pick workflow composes the reviewed robot scan, both FEMTO brackets, FEMTO
stone regions, RGB MVS, a first catalog pose fit using balanced FEMTO/RGB points,
and independent RGB validation. It supports
separated pickup stones; touching and stacked-scene identification require the
later scene solver.

Install the acquisition, perception, operation-window and RGB dependencies:

```bash
python -m pip install -e \
  'apps/tabletop[acquisition,hand-eye,perception,operations,hand-eye-rgb,hand-eye-mvs]'
```

Keep the normal UR bringup running and place the wrist camera at the central
view, approximately 0.4 m from the scene. The integrated command generates the
standard nine-view route from that live pose and executes it:

```bash
python -m stacking_tabletop.acquisition.pick_scene_workflow run \
  --output apps/tabletop/data/experiments/pick_scene_003 \
  --visualize
```

The same workflow is available in the Operations Console under **File → Field
scan / identify… → Capture wrist RGB + FEMTO and identify**. This mode moves the
UR5e up to 0.08 m around its current central view, so clear the capture volume
and use an empty tool. The generated route, URDF, calibration snapshot, preview
and hash remain under `capture_plan/` for replay and audit. The existing
ID-assignment and manual-alignment windows remain the operator review gates.

Reprocess a complete scene capture without moving the robot:

```bash
python -m stacking_tabletop.acquisition.pick_scene_workflow process \
  apps/tabletop/data/experiments/scene_capture_003 \
  --output apps/tabletop/data/experiments/pick_scene_003_retry \
  --visualize
```

If joint pose identification fails after dense MVS, resume from the failed
workflow directory instead of reconstructing RGB again:

```bash
python -m stacking_tabletop.acquisition.pick_scene_workflow resume \
  .data/field_pcd/FAILED_WORKFLOW_DIRECTORY \
  --manual-init-gui --visualize
```

In the Operations Console, select **Resume failed wrist RGB + FEMTO
identification** and choose that failed workflow as the source. Resume reopens
stone-ID review and optional manual alignment, reuses the recorded MVS cloud,
then reruns FEMTO stability, RGB validation and publication. The original failed
stage remains in the workflow log.

FEMTO supplies segmentation, the metric frame and coarse XY regions. Sparse RGB
checks, camera-prior refinement and dense MVS run before catalog identification.
The operator then assigns stone IDs to the clusters; each cluster contains FEMTO
and RGB points for review. The first catalog fit uses equally sampled points
from both sensors, without a FEMTO-only pose or model-distance gate. Each sensor
must independently pass the field coverage and RMSE checks. A final fixed-pose
RGB check and the second FEMTO bracket guard publication. This does not fit or
install a global RGB-to-FEMTO correction. Automatic stone-ID assignment is
currently disabled for the joint workflow; the operator must review IDs.

Every successful joint identification writes
`fused_identification/comparison/overview.html` and `overview.png`. The viewer
has switchable FEMTO observations, wrist-RGB observations and the fitted
catalog model and labeled XYZ pose axes for each selected stone, all in the
planning `world` frame.
The Operations Console opens this pose result when visualization is selected
and provides an **Open identified poses** button afterward. It also keeps the
separate **Open RGB PCD** viewer. The pose viewer is built from the saved PCDs,
so it can be regenerated without capturing or fitting again.

`pose_data.pkl` and `pose_identification/stone_poses.json` appear at the workflow
root only when the FEMTO stability and RGB checks pass. Rejected runs retain all
completed diagnostic artifacts, per-stone reasons, measurements and timing in
`pick_scene_workflow.json`, `femto_stability/summary.json`,
`fused_identification/pose_identification/stone_poses.json` and
`rgb_validation/summary.json`. Passing the gates is an operational consistency
result; absolute 1-mm accuracy remains unverified.

Each workflow stage records `elapsed_s` in `pick_scene_workflow.json`, including
live capture, coarse FEMTO segmentation, sparse RGB, refinement, dense MVS
and joint catalog identification. The
production dense stage uses refined camera poses and five geometric reference
views: center plus the four oblique views (0, 5, 6, 7 and 8).

Before MVS, it crops each undistorted image to the projected padded stone XY
regions over the full recorded Z interval, with another 16 pixels of matching
context. Images keep their original pixel resolution. Each crop has its own
pinhole camera with the principal point shifted by the crop offset. The saved
NPZ pixel indices refer to cropped undistorted images; `crop_xyxy` and
`uncropped_image_shape` preserve their location in the full undistorted image.
This pick-scene optimization intentionally omits distant background; it is not
a full-table scene reconstruction.

Each reference uses up to four explicit sources, selected by shared region
visibility, camera separation and direction diversity. Every visible region
must have at least two overlapping sources with useful parallax. Geometric
references and all their selected sources receive prerequisite photometric maps.
The nine recorded views remain available to the selector; it does not require
all nine depth maps if the chosen dependencies use fewer views. Summary JSON
records crops, source indices, dependency views and per-pass timing.

On `pick_scene_004`, the earlier full-image production schedule took 89.0 s.
Cropped runs took 54.2 s with eight sources, 50.6 s with six, and 36.1 s with four.
The four-source result retained nearby observations for 98.4–99.5% of baseline
stone points within 2 mm. These comparisons measure agreement with the earlier
RGB result, not absolute accuracy. Timings are from one run of each configuration
on the current workstation.

The first joint fit now starts from the combined FEMTO and RGB observation.
A rejected combined or per-sensor fit, or failed fixed-pose RGB validation,
triggers one retry using full images and all sources. Failed fast-run artifacts
remain for review. A second rejection withholds pickup poses. Matching residuals
and repeatability do not establish absolute accuracy.

Saved-capture speed experiments on `pick_scene_004` (four source images per
reference, five geometric references unless noted):

| COLMAP setting | Dense time | Surface and pose observation |
| --- | ---: | --- |
| `window_step=1`, 5 iterations | 36.1 s | Initial cropped reference |
| `window_step=2`, 5 iterations | 22.1 s | 99.67–99.99% of the first cloud's stone points within 2 mm; all pose checks pass |
| `window_step=2`, 3 iterations | 18.0 s | 99.31–99.81% of the 22.1-s cloud's stone points within 2 mm; all pose checks pass |
| `window_step=2`, 2 iterations | 15.6 s | 98.95–99.59% within 2 mm; all pose checks pass |
| `window_step=2`, 1 iteration | 13.1 s | 97.40–99.15% within 2 mm; all pose checks pass, but surface agreement is lower |
| `window_step=2`, 3 iterations, 3 references | 14.8 s | All pose checks pass, but 7–8% fewer stone points than five references |

Three source images per reference took 35.9 s at the original COLMAP settings,
so reducing source count gave no useful speedup. `window_step=3` is invalid in
the installed COLMAP build; the replay function now rejects it before COLMAP
starts. The 18.0-s candidate uses 2.9 s photometric stereo, 5.0 s geometric
stereo, 3.0 s cloud export, 2.2 s preparation and 1.4 s viewer rendering.

`dense_rgb_session` accepts `window_step=2, num_iterations=3` for repeatable
offline trials. Production still uses step 1 and five iterations. Although the
18.0-s result passes all checks, stone 11's fitted orientation differs by 7.7
degrees from the 22.1-s result while their residuals are similar. That is a
pose-ambiguity signal, not evidence that either orientation is correct. The
settings need more scenes and metric accuracy validation before replacing the
production default. All times are single runs on this workstation and exclude
capture, sparse matching, camera refinement and pose fitting. Agreement between
RGB reconstructions is not an absolute accuracy measurement.

Every dense run now writes `rgb_dense/rgb_only/overview.html`, with separate
RGB-reconstructed stone layers that can be toggled in the legend. The complete
RGB voxel cloud remains at `rgb_dense/refined/rgb_dense_voxel_1mm.pcd`; the
RGB-only viewer displays stone regions rather than FEMTO or aligned catalog
models. `rgb_dense/comparison/overview.html` shows the diagnostic RGB/FEMTO
comparison. HTML display decimation does not alter the stored PCD. The joint pose comparison is always written; `--visualize` additionally writes
the independent RGB validation comparison.

In the Operations Console, **Open RGB PCD** opens the dense output actually used
by the accepted pick workflow. Before enabling planning, the dialog checks the
published joint-pose source, the RGB dense-summary hash recorded by the pose
fit, and the copied pose-data file. A saved output from the older FEMTO-only
version must be reprocessed to produce joint FEMTO/RGB pickup poses. For the
`pick_scene_004` capture, the reviewed 9/10/11 assignments are needed: an
unattended all-catalog assignment matched other stone IDs and did not yield a
valid RGB/FEMTO joint fit. The current joint fit keeps RGB points within the coarse FEMTO cluster regions;
it does not filter them using a preliminary catalog pose. The FEMTO frame and
cluster segmentation remain dependencies of the first RGB/FEMTO fit.

## Confidence and coverage labels

For scene 002, a dense RGB voxel is retained when it is within 1.5 mm of both
disjoint-pair controls and has at least six other agreeing voxels within 3 mm.
Coordinates are not moved, smoothed or blended.

Retained stone points are labelled as:

- Reference overlap: within 3 mm of FEMTO.
- Unresolved difference: farther than 3 mm but inside FEMTO's padded XY footprint.
- Possible extension: farther than 3 mm and outside that footprint.

Possible extensions are candidates, not certified newly observed surfaces. The
footprint is a top-view proxy and does not measure 3D surface area or visibility.

## Scene 002 results

Scene 002 contains five wrist views from the original route, three stones and
two FEMTO brackets. Its MVS result has 233,308 scene voxels. Filtering retains
198,250 scene voxels and 21,013 of 24,136 upper stone-region voxels.

| Retained stone class | Points |
|---|---:|
| FEMTO overlap within 3 mm | 11,803 |
| Unresolved difference | 7,457 |
| Possible extension | 1,753 |

Filtered RGB-to-FEMTO nearest-neighbour medians are 2.78, 2.77 and 2.46 mm for
the three regions. These values mix shape, alignment, sampling and visibility.

A shared rigid surface fit reduces median interior height differences from
3.55/5.08/2.73 mm to 1.17/1.30/1.05 mm, but p95 differences remain as high as
5.47 mm. Fits trained on two regions transfer poorly to the third: one reaches a
rotation bound, one does not converge, and one worsens its excluded region. The
correction is diagnostic only.

Fixed-scale comparison against measured stones 9, 10 and 11 provisionally favors
region ordering 0→9, 1→10, 2→11. Spatial held-out point-to-catalog medians after
independent rigid fits are 0.85/0.86/0.86 mm for RGB and 1.14/1.15/1.06 mm for
FEMTO. The sources retain different surfaces, model error is unverified, and
independent fits remove placement errors. These are shape-agreement results,
not absolute accuracy measurements or accepted stone poses.

## Outputs

Detailed summaries, hashes, reproducible commands and viewers live with their
ignored data:

```text
apps/tabletop/data/experiments/scene_capture_002/
  inspection/rgb_check/
  inspection/rgb_refinement/
  inspection/rgb_dense/
  inspection/rgb_filtered/
  inspection/surface_check/
  inspection/catalog_check_001/
  scene_layers/
```

The portable layered scene stores both FEMTO brackets, every original RGB voxel,
selection flags, and CSR view/pixel provenance. PCD files are interchange exports;
NPZ and JSON retain information that an XYZ cloud cannot represent.

## Current decision

Use wrist RGB MVS as the primary source of complementary stone surfaces and
FEMTO as the broad operational scene reference. Preserve RealSense depth and
rejected RGB as auxiliary evidence. Keep disagreement layers separate until a
correction predicts withheld regions or an independent measured target. Evaluate
the nine-view route by accepted new coverage and catalog agreement, not raw count.
