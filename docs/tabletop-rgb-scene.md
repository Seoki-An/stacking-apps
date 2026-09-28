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
FoundationPose on wrist RGB and learned stereo depth, with both point clouds
validating its pose. Explicit multistart ICP remains available. FEMTO supplies
coarse stone regions and the planning frame before RGB reconstruction. It does not install a global RGB-to-FEMTO calibration correction,
and absolute 1-mm accuracy remains unverified.

## Capture

The integrated workflow defaults to thirteen wrist views: center, four near
translations, four diagonal oblique views and four 30° side orbits. At the
0.4 m target distance the side cameras travel up to 0.20 m laterally. Operations
exposes this as **Wrist viewpoints → Wider: 13 views**; choose **Compact: 9**
(or CLI `--viewpoint-pattern coverage9`) for the previous route. Check clearance
for the complete robot/tool paths; these routes are not collision checked. Every view looks at a target 0.4 m along the
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
FEMTO defines coarse regions before MVS and validates catalog poses afterward;
it never changes the MVS depth values.

## Integrated pick-scene identification

The pick workflow composes the reviewed robot scan, both FEMTO brackets, FEMTO
stone regions, RGB reconstruction, FoundationPose estimation, and independent
RGB/FEMTO validation. It supports
separated pickup stones; touching and stacked-scene identification require the
later scene solver.

Install the acquisition, perception, operation-window and RGB dependencies:

```bash
python -m pip install -e \
  'apps/tabletop[acquisition,hand-eye,perception,operations,hand-eye-rgb,hand-eye-mvs]'
```

Keep the normal UR bringup running and place the wrist camera at the central
view, approximately 0.4 m from the scene. The integrated command generates the
selected route (thirteen views by default) from that live pose and executes it:

```bash
python -m stacking_tabletop.acquisition.pick_scene_workflow run \
  --output apps/tabletop/data/experiments/pick_scene_003 \
  --visualize
```

The same workflow is available in the Operations Console under **File → Field
scan / identify… → Capture wrist RGB + FEMTO and identify**. This mode moves the
UR5e through the selected route (up to 0.20 m camera lateral travel by default), so clear the capture volume
and use an empty tool. The generated route, URDF, calibration snapshot, preview
and hash remain under `capture_plan/` for replay and audit. The existing
ID-assignment and result review windows remain available. Manual pre-ICP
alignment applies only when the ICP pose backend is selected.

Reprocess a complete scene capture without moving the robot:

```bash
python -m stacking_tabletop.acquisition.pick_scene_workflow process \
  apps/tabletop/data/experiments/scene_capture_003 \
  --output apps/tabletop/data/experiments/pick_scene_003_retry \
  --visualize
```

After a successful wrist RGB + FEMTO fit, Operations automatically asks you to
**Accept poses** or **Change model IDs**. The comparison viewer opens first when
visualization is enabled. Choosing **Change model IDs** returns to the assignment
window with previous IDs preselected, reuses the saved clouds, and reruns joint
pose fitting and validation. The same prompt appears after each successful refit;
accepting enables **New plan using these poses**. There is no separate workflow
selection for changing IDs.

Each review writes a new sibling output, preserving the previous result even if
review is cancelled or rejected. No robot movement, capture or MVS is repeated.
For a completed result outside the current GUI session, the CLI remains:

```bash
python -m stacking_tabletop.acquisition.pick_scene_workflow reidentify \
  apps/tabletop/data/experiments/pick_scene_004_joint_verified \
  --output apps/tabletop/data/experiments/pick_scene_004_reidentified \
  --visualize
```

This copies only the final clouds and viewers; large MVS working files stay in
the original workflow. The capture directory must still be available for the
FEMTO stability check. Changing IDs does not change the captured region.

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
views for the standard nine-view route: center plus the four oblique views
(0, 5, 6, 7 and 8). A reviewed thirteen-view side route also reconstructs
four 30° side reference views (9, 10, 11 and 12).

Before MVS, it crops each undistorted image to the projected padded stone XY
regions over the full recorded Z interval, with another 16 pixels of matching
context. Images keep their original pixel resolution. Each crop has its own
pinhole camera with the principal point shifted by the crop offset. The saved
NPZ pixel indices refer to cropped undistorted images; `crop_xyxy` and
`uncropped_image_shape` preserve their location in the full undistorted image.
This pick-scene optimization intentionally omits distant background; it is not
a full-table scene reconstruction.

The stereo depth range uses the padded stone boxes over the full recorded Z
interval. Empty corners of the larger FEMTO workspace can lie behind an
oblique camera; those corners do not reject reconstruction of visible stones.

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

### Side-coverage limits observed in simulation

On `scene_side_simulation_gui_002`, the existing XY crop contains all mesh
vertices of stones 9–11. Widening that crop alone cannot reveal their hidden
sides. The thirteen-view route supplies stereo support for approximately
57–72% of side-surface samples, compared with 51–62% for the first nine views.
The pose-fitting stone mask also excludes points within 8 mm of the table;
this intentionally removes lower-surface points along with table contamination.
The full dense cloud retains those points where reconstructed.

Offline trials with 40°, 45°, 50° and 60° side orbits could not produce all four
continuous IK paths from this saved seed. Larger angles require a different
central robot configuration and another offline route/visibility check. Neither
a wider crop nor the thirteen-view option guarantees complete side coverage.


The new `coverage17` route adds 45° and 60° left/right views to the original
thirteen. In the same synthetic scene, side stereo support rose to 70–84%.
See [extended lateral coverage](tabletop-automated-scene-capture.md#extended-lateral-coverage-seventeen-views)
for the new preview, simulated captures and measured comparison. This resolves
the reachable lateral directions individually; it does not require all four
cardinal directions to reach the same larger angle.

## Fast-FoundationStereo default backend

Fast-FoundationStereo is the default dense backend for the integrated RGB-first
pick workflow in the CLI and Operations GUI. Launch the workflow with your normal
workspace Python: the reconstruction stage automatically runs in the installed
`.data/venvs/fast-stereo` environment. This shares the workspace CUDA PyTorch
runtime while keeping model dependencies isolated from ROS and the GUI. The
official source and 68 MB serialized research checkpoint remain under ignored
`.data/vendor`. Install this environment using the setup below on a new workstation.

```bash
python -m stacking_tabletop.acquisition.pick_scene_workflow process \
  apps/tabletop/data/experiments/scene_wide_simulation_001/captures_7views \
  --output apps/tabletop/data/experiments/pick_scene_wide7_fast_001 \
  --visualize
```

This reconstructs RGB **before** opening model selection and running FoundationPose.
RGB and FEMTO independently validate the estimated poses. `run` uses the same
backends for live capture. In Operations, select a wrist RGB + FEMTO workflow;
**RGB reconstruction → Fast-FoundationStereo (default)** and
**Pose estimation → FoundationPose (default)** are preselected.

For the older joint point-cloud fit, select **Multistart ICP** or pass
`--pose-backend icp`. COLMAP reconstruction currently requires this ICP option:
`--dense-backend colmap --pose-backend icp`. Only the explicit ICP path retries
full-image COLMAP after a rejected fit. FoundationPose rejection stops for review;
it never silently substitutes ICP or applies ICP refinement. Runtime/inference
errors also stop and retain the inputs. Each output directory must be new.

Resume and Change model IDs reuse saved clouds and learned stereo pair data.
Older COLMAP-only outputs require explicit ICP selection or new Fast-FoundationStereo
processing. The workflow records `pose_backend`, `dense_backend`, any fallback
reason and `accepted_dense_backend`. Stopping the workflow terminates a running
isolated inference process.

To reconstruct only, reusing the completed sparse/refined cameras:

```bash
.data/venvs/fast-stereo/bin/python -m stacking_tabletop.acquisition.hand_eye_fast_stereo \
  --session apps/tabletop/data/experiments/scene_wide_simulation_001/captures_7views/session \
  --baseline apps/tabletop/data/experiments/pick_scene_wide7_001/rgb_sparse \
  --refinement apps/tabletop/data/experiments/pick_scene_wide7_001/rgb_refinement \
  --output apps/tabletop/data/experiments/pick_scene_wide7_fast_dense_001
```

The adapter selects nearby camera pairs covering every input view, rectifies them
into horizontal stereo images, and predicts forward/reverse disparities. It
rejects invalid sensor borders, occlusions and left/right differences above one
pixel. Metric depth comes from calibrated focal length and camera baseline;
FEMTO supplies spatial regions, not replacement depths. The released checkpoint's
416-pixel disparity capacity is handled by jointly reducing image size and focal
length when necessary. This can reduce detail for wide-baseline pairs.

The seven-view replay uses four pairs and exports their left-reference maps;
the second image in each pair supplies stereo and consistency evidence. Pair
selection is a nearest-view heuristic, not a guarantee of overlap for arbitrary
new capture routes. Pair images, disparities, masks, transforms and timings are
saved under `refined/`. The standard `refined/rgb_dense_voxel_1mm.pcd` contract is
consumed by the existing joint pose solver. `rgb_only/overview.html` displays
reconstructed stones.

### Recorded synthetic replay, 2026-09-25

RTX 5060 Ti 16 GB, PyTorch 2.7.0 CUDA 12.8, eight model iterations, image long
edge at most 960 pixels, four pairs with forward/reverse checks:

| Measurement | Existing COLMAP run | Fast-FoundationStereo |
|---|---:|---:|
| Dense stage, including preparation/export | 61.8 s | 16.8 s |
| Model loading, included above | — | 3.6 s |
| Reconstruction plus RGB viewer | — | 17.8 s |
| Stone points after 1-mm voxel sampling | 21,411 | 21,054 |
| Median distance to synthetic mesh, across three stones | 0.18–0.29 mm | 0.26–0.29 mm |
| 95th-percentile mesh distance | 0.61–1.00 mm | 1.01–1.10 mm |

The recorded dense-stage speedup is about 3.7×. This compares two saved runs,
not repeated controlled timing trials; capture, sparse reconstruction, camera
refinement and operator interaction are excluded. Joint RGB/FEMTO fitting with
previously reviewed IDs 10/9/11 passed all three stones and RGB validation, taking
2.8 s including its viewer. That fit time cannot be compared with earlier GUI
stages containing human interaction. Mesh distance measures surface agreement,
not completeness. These are ideal synthetic images; real-camera millimetre
accuracy remains unverified.

Artifacts: `pick_scene_wide7_001/inspection/fast_stereo_002/`, including
`mesh_comparison.json`, `comparison/overview.html`, `joint_identification/`, and
`rgb_validation/`. The earlier `fast_stereo_001` retains an initial failed run
that exceeded the model's disparity embedding size.

### Recreating the optional environment

Use the workspace Python with working CUDA PyTorch/torchvision, OpenCV, Open3D,
OmegaConf and the editable project packages. Do not replace its CUDA stack.

```bash
git clone https://github.com/NVlabs/Fast-FoundationStereo.git .data/vendor/Fast-FoundationStereo
git -C .data/vendor/Fast-FoundationStereo checkout 476f4249561f7c79ca707326954f9255643412a6
python -m venv --system-site-packages .data/venvs/fast-stereo
python - <<'PY'
import json, pathlib, sysconfig
root = pathlib.Path('.data/venvs/fast-stereo/lib')
site = next(root.glob('python*/site-packages'))
(site/'workspace_runtime.pth').write_text(
    'import site; site.addsitedir(' + repr(sysconfig.get_paths()['purelib']) + ')\n')
PY
.data/venvs/fast-stereo/bin/pip install --no-deps \
  timm==0.9.16 einops==0.8.2 huggingface_hub==0.36.2 safetensors==0.8.0 \
  gdown==6.4.0 beautifulsoup4==4.15.0 soupsieve==2.10 PySocks==1.7.1 \
  filelock==4.0.3 requests==2.34.2 tqdm==4.70.1
.data/venvs/fast-stereo/bin/python - <<'PY'
from pathlib import Path
import gdown
root = Path('.data/vendor/Fast-FoundationStereo/weights/23-36-37')
root.mkdir(parents=True, exist_ok=True)
for ident, name in [('1GDBRYL-ZaLpXEtWfGFRJvkBc_2sywjgj', 'cfg.yaml'),
                    ('1W1V1H64l9bAi97boEQQ2ueNzzGmSMz-E', 'model_best_bp2_serialize.pth')]:
    gdown.download(id=ident, output=str(root/name))
PY
```

Use the official trusted checkpoint: it is a serialized Python model. Its hash
and repository commit are recorded in each run. Source/checkpoint usage follows
the upstream [Fast-FoundationStereo repository and license](https://github.com/NVlabs/Fast-FoundationStereo).
TensorRT conversion, FoundationPose and SAM3D are not part of this backend.

## FoundationPose initializer experiment

The offline `hand_eye_foundation_pose` experiment tests a learned pose initializer
against the existing 24-orientation ICP search. It does not change the production
pose solver or publish execution poses.

Inputs come from saved Fast-FoundationStereo pairs: the rectified left RGB,
calibrated metric depth, camera transform and stereo validity mask. For each
previously assigned stone ID, the experiment selects the pair with the most
valid surface pixels inside its coarse FEMTO region and above the table. It does
not use the previous fitted pose to initialize FoundationPose. Masks, input
hashes and selected pairs are retained for inspection.

```bash
python -m stacking_tabletop.acquisition.hand_eye_foundation_pose prepare \
  --dense apps/tabletop/data/experiments/pick_scene_004_fast_real_001/rgb_dense \
  --reference apps/tabletop/data/experiments/pick_scene_004_rgb_first_verified \
  --output apps/tabletop/data/experiments/foundation_pose_trial/inputs

.data/venvs/foundation-pose/bin/python \
  -m stacking_tabletop.acquisition.hand_eye_foundation_pose run \
  --inputs apps/tabletop/data/experiments/foundation_pose_trial/inputs \
  --output apps/tabletop/data/experiments/foundation_pose_trial/results
```

The optional inference environment is separate from `fast-stereo`. It uses the
[official FoundationPose implementation](https://github.com/NVlabs/FoundationPose),
its released scorer/refiner weights, PyTorch3D, nvdiffrast and the upstream
`mycpp` extension. This workstation's extensions use a local CUDA 12.8 compiler
matching PyTorch 2.7.0+cu128; the system CUDA 13 compiler is left unchanged.
Model source, weights, environments and toolchains remain under ignored `.data/`.

The benchmark records FoundationPose's raw pose, a single multiscale ICP run
initialized by that pose, and a fresh 24-start ICP result on exactly the same
balanced RGB/FEMTO points. It saves per-sensor residuals, acceptance decisions,
timings, transforms and interactive overlays. The prior COLMAP-based poses are
shown only as comparisons. Neither prior fitted poses nor FEMTO are ground truth.
The existing stone meshes are untextured; their geometry also differs slightly
from the measured catalog PCDs. These factors must be considered when interpreting
learned pose quality and orientation ambiguity.

### First real-data FoundationPose result

`foundation_pose_real_001/overview.html` links the input masks, raw initial
poses, refined poses and same-input ICP comparisons. All three raw and refined
estimates pass the existing per-sensor residual gates. However, stone 10's
refined orientation differs by 179.6° from the 24-start ICP result despite
similar combined residuals (3.04 versus 3.06 mm). The corresponding disagreements
for stones 9 and 11 are 18.9° and 16.9°. These are disagreements, not measured
pose errors; neither estimate is ground truth.

FoundationPose evaluated 252 hypotheses per stone. Inference took 7.10 s total
for three stones, plus 3.21 s model/renderer load and 0.36 s object setup. Its
single-seed ICP refinement took 0.07 s total. A fresh 24-start ICP search on the
same observations took 0.66 s total, excluding visualization. These single-run
measurements show no pose-stage speed advantage for FoundationPose here. The
benchmark alone did not establish the correct real-data orientation; a lower
residual alone is insufficient to choose between ambiguous orientations.

### Synthetic ground-truth comparison

The saved seven-view simulation was tested with the same FoundationPose adapter
and measured-catalog ICP. Ground truth was withheld from inference and used only
by the separate evaluator:

```bash
python -m stacking_tabletop.acquisition.hand_eye_foundation_pose evaluate-truth \
  --inputs apps/tabletop/data/experiments/foundation_pose_sim_001/inputs \
  --results apps/tabletop/data/experiments/foundation_pose_sim_001/results \
  --simulation apps/tabletop/data/experiments/scene_wide_simulation_001 \
  --output apps/tabletop/data/experiments/foundation_pose_sim_001/ground_truth_new
```

The evaluator verifies simulation/capture provenance and mesh hashes, converts
predictions from the corrected planning frame to the original simulator UR-base
frame, and reports translation, rotation, corresponding-vertex displacement
(ADD) and nearest-vertex displacement (ADD-S). It does not fit an alignment to
ground truth or assume rotational symmetries. A low ADD-S can conceal incorrect
orientation, so rotation and ADD remain separate metrics.

Recorded results (`foundation_pose_sim_001/ground_truth/overview.html`):

| Stone | FoundationPose initial: mm / degrees | FoundationPose + ICP: mm / degrees | 24-start ICP: mm / degrees |
|---|---:|---:|---:|
| 9 | 0.46 / 4.52 | 1.27 / 5.63 | 2.61 / 14.68 |
| 10 | 0.50 / 1.95 | 1.41 / 3.56 | 3.00 / 5.30 |
| 11 | 0.26 / 2.40 | 1.50 / 2.55 | 2.71 / 10.78 |

FoundationPose's raw estimate is better on all three stones in this scene.
Current joint ICP refinement worsens its translation and rotation accuracy,
despite passing every residual gate. Initial ADD is 1.56 / 0.82 / 0.83 mm for
stones 9 / 10 / 11; submillimetre translation therefore does not establish
submillimetre surface accuracy. FoundationPose inference took 6.57 s total,
versus 0.66 s for the fresh 24-start ICP search; this is an accuracy benefit in
this test, not a speed benefit.

This simulation renders the same mesh supplied to FoundationPose. ICP uses
measured catalog PCDs, which differ slightly from those meshes. That model
mismatch and the synthetic image/depth conditions limit generalization. The
result supports further validation of the learned initializer and reconsidering
unconditional ICP refinement. It does not settle stone 10's orientation in the
separate real capture or establish real-world 1-mm accuracy.


## FoundationPose default pickup pose estimation

The wrist RGB + FEMTO workflow now defaults to FoundationPose in `run`, `process`,
`resume`, `reidentify`, and the Operations dialog. FEMTO-only workflows retain
their existing ICP solver. The default pipeline is:

1. FEMTO segmentation supplies coarse stone regions and the planning table frame.
2. Fast-FoundationStereo reconstructs metric geometry from wrist RGB views.
3. The operator assigns catalog IDs to the combined RGB/FEMTO clusters.
4. For each assigned stone, select the stereo pair with the most valid masked
   pixels and estimate its mesh pose with FoundationPose.
5. Convert the result into the planning frame. Independently validate it against
   RGB, FEMTO, and balanced observations; check FEMTO bracket stability.
6. Publish the accepted learned poses and open the observed-cloud/model overlay.

The learned pose is used directly: **no ICP refinement or automatic ICP fallback**.
FoundationPose's own learned refinement remains enabled (five iterations).
The report records `pose_source: foundationpose_rgb_femto_validated`, input and
mesh hashes, camera and planning transforms, and per-sensor validation metrics.
Input masks and raw estimates remain under the identification directory in
`foundation_pose_inputs` and `foundation_pose_results`.

The workflow uses `.data/venvs/foundation-pose/bin/python` and the installed
`.data/vendor/FoundationPose` weights, keeping GUI/ROS dependencies separate.
Each selected model needs `model_<id>_mesh.ply` beside its measured PCD, in metres
and the **same catalog coordinate frame**. Missing meshes or runtime files stop
with a clear error; a custom PCD directory must supply matching meshes.
Manual pre-ICP alignment is enabled only with explicit **Multistart ICP**.

No extra backend flag is needed:

```bash
python -m stacking_tabletop.acquisition.pick_scene_workflow run \
  --output apps/tabletop/data/experiments/pick_scene_foundation_001 \
  --visualize
```

Restart Operations to load the updated default. Use **Capture wrist RGB + FEMTO
and identify**, or its existing-capture counterpart. The result viewer and
workbench handoff accept the learned pose source while retaining provenance and
sensor checks. This change favors the observed synthetic pose accuracy; it does
not claim a speed gain or verified 1-mm real-world accuracy.
