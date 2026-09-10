# Excavator desktop perception

The desktop pipeline imports recorded frames, constructs purpose-specific
transfer crops, runs pose identification and writes reviewable proposals.
It uses the measured legacy PCD catalog for registration and the commissioned
stone_grab geometry for jaw evidence, bounded per-view alignment and gripper removal. It does not command
the robot or update a planning run.

## Try it without a NUC

Use new output directories, from the repository root:

```bash
python -m stacking_excavator.acquisition.emulation \
  --repository . --output /tmp/excavator-perception-demo

python -m stacking_excavator.perception.cli identify \
  /tmp/excavator-perception-demo/field/desktop \
  /tmp/excavator-perception-demo/field/priors.json \
  /tmp/excavator-field-result --repository .

python -m stacking_excavator.perception.cli identify \
  /tmp/excavator-perception-demo/inhand/desktop \
  /tmp/excavator-perception-demo/inhand/priors.json \
  /tmp/excavator-inhand-result --repository .
```

Each result directory contains `proposal.json` and, when nonempty,
`processed.ply`, whose hash is bound into the CLI proposal. The latter uses world coordinates for field/scene, and
stone_grab geometry coordinates for in-hand data. A proposal's `valid` means
that the configured numeric/evidence gates passed. Its scope is
`desktop_replay`; this is not authorization to execute a trajectory.

The new `stacking-excavator-perception` entry point accepts the same arguments
after refreshing the editable package installation. Open3D is in the excavator
`perception` extra. No ROS installation is required.

## Perception in the Operations Console

The **Perception** tab answers: “Where is this expected stone, based on this
recorded scan?” Identification creates a proposal for review. **Accept proposal**
is the separate action that updates the desktop session's accepted scene.
Opening a scan, running identification, or changing a display layer does not
accept a pose or start motion.

### What each purpose means

The crop manifest selects the purpose automatically; there is no separate
purpose selector in the tab.

| Purpose | What it estimates | Pose reference frame |
| --- | --- | --- |
| `field` | The pose of one expected stone before pickup. Ground is retained by default to reproduce legacy recovery. | `world` |
| `inhand` | The pose of one held stone relative to stone_grab, using jaw evidence and per-view gripper alignment before merging. | `stone_grab_geometry` |
| `scene` | Updated poses for the expected stones around the target structure. Points are assigned to at most one expected stone. | `world` |

The IDs come from the crop request and initial poses. Scene identification
refines those supplied identities; it does not discover arbitrary new stone
IDs. The in-hand frame is the gripper geometry frame, so an in-hand pose must
not be entered as a world pose.

### Inputs and buttons

| Control | What to supply or expect |
| --- | --- |
| **Open session folder…** | Select a directory for durable accepted poses, proposals and execution history. Session files are initialized if needed. Reopen the same directory to continue reviewing that session. |
| **Verified crop product** | The received crop directory containing `manifest.json` and its frame `.npz` files, such as the emulation's `field/desktop` directory. A standalone `.pcd` or `.ply` file must first go through import and crop preparation. |
| **Initial poses JSON** | A JSON file with `scene_revision`, `ground_height` in metres, and `poses` keyed by object ID. Each pose is a 4×4 object-to-reference-frame transform; see the example below in “Crops and identification requests.” |
| **Identify in background** | Validates the product and runs registration in a separate process. A completed result is staged for review, including results that fail quality gates. |
| **Stop identification** | Stops that worker. It does not accept a partial result or undo a previously accepted observation. |
| **Open proposal.json…** | Stages an existing result without rerunning registration. Keep its matching `processed.ply` alongside it to display the observed cloud. |
| **Proposal dropdown** | Selects a pending, accepted or rejected proposal from the session. Selecting an older result displays its recorded poses; it does not roll back the accepted scene. |

The crop, initial poses and session must agree on the scene revision. The
initial-pose IDs must exactly match the IDs requested by the crop. A new
session starts at revision **0**; the supplied emulation examples use **1**.

For a first walkthrough, run the field example in **Try it without a NUC**
above. Then stage its result into a **new** session initialized at revision 1:

```bash
python -m stacking_excavator.perception.cli stage \
  /tmp/excavator-perception-review/review.db \
  /tmp/excavator-field-result/proposal.json --initial-revision 1

python -m stacking_excavator.gui.operations.application . \
  --session /tmp/excavator-perception-review
```

Select **Perception**, then **Open proposal.json…** and choose
`/tmp/excavator-field-result/proposal.json`. This also associates the matching
processed cloud with the session's display. To run identification from the tab,
set **Verified crop product** to `/tmp/excavator-perception-demo/field/desktop`
and **Initial poses JSON** to `/tmp/excavator-perception-demo/field/priors.json`.
Use **Identify in background** while the session is still at revision 1.
Initializing a review store does not overwrite the revision of an existing one.

### Reading the 3D view

| Layer | Meaning |
| --- | --- |
| **Observed cloud** — gray | The merged, processed scan used for inspection. In-hand points have undergone alignment and jaw removal; field/scene processing follows the corresponding ground policy. This is not the untouched sensor capture. |
| **Initial poses** — blue transparent surfaces | The supplied starting estimates. Compare these with the fitted poses to see how far registration moved each stone. |
| **Proposed poses** — green or red transparent surfaces | Green means that object's gates passed; red means it failed a gate. Neither color indicates an operator decision. No proposed surface can be drawn when insufficient evidence leaves the pose unknown. |
| **Crop boundary** — yellow wire box | The requested transfer crop including its margins, in the displayed reference frame. Desktop filtering can leave a smaller cloud inside that box. |

Toggle layers to inspect overlap without the other surfaces obscuring it.
Use **Reset camera** or `R` to fit the scene. The toolbar's **DSF surface / PLY
mesh** setting changes the displayed stone surfaces only. Registration still
uses the measured PCD catalog, so switching the display does not recompute or
improve a pose. Display surfaces and measured reference points can differ by
centimetres.

The gray cloud is shown only if its file hash matches the proposal. If only
poses appear, check that the matching `processed.ply` is still beside the
proposal and has not changed. Older proposals without a cloud hash can still
show their poses. The session retains a reference to an externally opened
result directory, so keep that directory available when reopening the session.

### Understanding the quality summary

**Passes gates** means that the configured numerical checks passed. It is not
a probability that the pose is correct and does not establish physical accuracy.
The full proposal, hashes and per-view details are available in **History**.

| Value | Interpretation |
| --- | --- |
| **Target coverage** | Fraction of the object's observed target points within 3 cm of the fitted reference cloud. Low coverage means much of that observed region is unexplained by the model. The default stone-pose minimum is 35%. |
| **Source coverage** | Fraction of reference-model points within 3 cm of the observation. Occlusion can reduce this fraction. The default minimum is 20%. |
| **RMSE** | Root mean squared nearest-point distance over inliers from both directions, shown in millimetres. A small value alone is insufficient: a small matching patch can have low error and poor coverage. |
| **Opening view N** | Estimated jaw opening and separately recorded measured opening, in radians. `None` means no measurement was recorded, not that the gripper is open or empty. |
| **Per-view correction** | Translation and rotation used to align an in-hand view to the gripper. These are frame-alignment corrections, not the stone's pose correction. Rejected corrections are not applied. |

For example, the restored stone-17 field replay matches the saved legacy pose
numerically but has **28.9% target coverage**, below **35%**. Its surface is red
and acceptance is disabled. Matching legacy behavior does not bypass a quality
gate.

### Accepting, rejecting and troubleshooting

**Accept proposal** is enabled for a valid, pending result at the current
revision. Acceptance advances the revision once. Field/scene acceptance updates
the proposed world poses while preserving other objects. In-hand acceptance
stores the stone-to-gripper attachment geometry separately. If simulated
execution is active, acceptance stops that attempt and sets `needs_replan`;
it does not rewrite the trajectory or automatically invoke planning.

**Reject proposal** records the decision without changing accepted poses or
advancing the revision. Either decision consumes that proposal's review token.
An older accepted or rejected proposal remains inspectable but cannot be decided
again. A revision change also makes other pending results at the old revision
stale.

| Symptom | What to check next |
| --- | --- |
| **Stale scene revision** | Use a capture, crop and priors associated with the current session revision. For independent inspection of an old recording, use a separate replay session initialized to that recording's revision. Recropping preserves the raw capture's revision; editing only the priors cannot rebase it. |
| **Insufficient evidence / unknown pose** | Check the requested ID, initial position and coordinate frame. Inspect whether the transfer crop actually contains the stone; derive a wider crop from retained raw data when appropriate. An absent fit does not prove the stone has disappeared. |
| **Low coverage / ambiguous pose** | Compare the cloud with the initial and fitted surfaces. Check occlusion, unrelated points and competing orientations; obtain better evidence or improve the initial estimate before retrying. |
| **Unvalidated opening / gripper alignment** | Inspect both jaws, opening diagnostics and individual view corrections. Missing jaw evidence or an excessive alignment correction blocks attachment acceptance. |
| **Accept remains disabled** | Check validity, pending/consumed decision status and the current scene revision. Green geometry alone does not mean the result is still eligible for acceptance. |

This tab reviews both imported products and proposals produced automatically
by [closed-loop emulation](excavator-closed-loop-emulation.md). In that workflow,
the Execution tab requests scans of the changing virtual world and holds motion
for review. Physical scan acquisition, WiFi control and hardware execution
remain separate commissioning work.

## Recorded inputs

Explicit sidecars can import NPY, NPZ, PLY and PCD files:

```bash
python -m stacking_excavator.perception.cli import recording.json /tmp/imported-capture
```

The sidecar has `schema_version: 1`, `capture_id`, `kind`, `scene_revision`, and
a nonempty `frames` list. Each entry contains the fields returned by
`CaptureFrame.metadata()`, plus `source` (relative to the sidecar, or absolute)
and `source_sha256`. Transforms are 4×4 matrices, joints are the six canonical
radian coordinates, and timestamps are integer nanoseconds. NPZ inputs contain
`xyz` and optional `attr_NAME` arrays. PCD/PLY numeric channels are preserved
using Open3D's tensor reader; malformed or nonfinite values are rejected.

Legacy logs have a dedicated importer:

```bash
python -m stacking_excavator.perception.cli import-legacy \
  ../stacking-planner/logs/260707/plan_260707_7/exec_232318_1_nuc/step_44/field_scan \
  /tmp/legacy-field-capture \
  --kind field --capture-id field-step-44 --revision 0 --repository .
```

Supported layouts:

| Purpose | Files |
| --- | --- |
| Field/scene | `KIND_raw_TOPIC_swingN.npy`, matching `KIND_lidar_frame_TOPIC_swingN.npy` and `KIND_scan_q_actual_swingN.npy`. |
| In-hand | `pcd_N.ply` and matching rows of `q_list.npy`. These PLY files are already in the gripper geometry frame. |

Legacy records lack verified sensor/joint timestamp pairing. The importer
records `timestamp_source: recorded_unknown` and null timestamps; it never
fills them from file modification times. Unrecorded measured jaw opening is
null and is kept separate from estimated opening. For field/scene it uses the
logged sensor transform. For in-hand it preserves the existing geometry-frame
cloud without applying another sensor transform. Original in-hand sensor data
discarded by the legacy logger cannot be recovered.

Imported manifests record source-file and model hashes. Their calibration hash
identifies a legacy transform snapshot, not a commissioned calibration result.
The stationary acquisition service rejects these unverified recordings as live
sensor evidence.

## Crops and identification requests

`perception.roi.enclosing_crop` encloses expected/observed stone surfaces plus
uncertainty margins. Scene crops require target corners and include ground
context and stone overhang. In-hand crops require a jaw envelope covering the
opening uncertainty and include the held stone. Only spatial membership is
applied before transfer; fine filtering stays on the desktop.

Serialize `CropRequest.values()` to `crop.json`, then derive a product:

```bash
python -m stacking_excavator.perception.cli crop \
  /tmp/legacy-field-capture crop.json /tmp/field-crop-v1 \
  --product-id field-crop-v1
```

A wider crop uses the same retained raw source, a new output directory and a
new product ID. The identification priors JSON has:

```json
{
  "scene_revision": 0,
  "ground_height": 0.0,
  "poses": {
    "17": [[1,0,0,-3], [0,1,0,-5.8], [0,0,1,0.25], [0,0,0,1]]
  }
}
```

These numbers are illustrative, not a measured pose. Object IDs must exactly
match the crop request. Poses map object coordinates to world coordinates for
field/scene, and to stone_grab geometry coordinates for in-hand. Supply the
appropriate measured support height; the current ground filter is a height
band, not a terrain-plane estimator.

## Behavior and differences from legacy

- Field identification retains the legacy 64 absolute XYZ orientation seeds
  and multiscale schedule. In-hand identification retains four left-multiplied
  roll seeds, the width-to-opening polynomial and 6 cm gripper-removal distance.
- Registration uses measured catalog XYZ copied without resampling into
  `assets/perception`. PLY meshes remain ray-casting/display assets; they differ
  from measured PCD surfaces by centimetres. The reference export is reproducible
  with `tests/export_excavator_perception_models.py` and records both hashes.
  The 49 compressed reference clouds occupy approximately 134 MiB.
- Point-to-point ICP is ranked by final-scale fitness minus RMSE. Additional
  coverage, residual, correction, correspondence and ambiguity gates reject
  unsupported solutions. Multiple seeds converging to the same pose are not
  ambiguity. Distinct, similarly scored poses are.
- Opening validity is separate from measured-opening availability. It requires
  an in-range estimate and bilateral jaw support, plus consistency with measured
  opening when available. Empty scans do not imply an empty gripper.
- Scene identification assigns each point to at most one expected object,
  then refines those priors. Missing or sparse evidence produces a rejected
  proposal; objects are not silently removed. This is ICP reconciliation, not
  the legacy native SceneID solver or automatic unknown-stone discovery.
- Capture-time transforms establish the input frame. In-hand views then receive
  a bounded, reported ICP correction against the gripper before merging.

Defaults are in `apps/excavator/config/perception.yml` and require field
commissioning. Before the restoration below, the recorded field-log comparison for stone 17 produced a
proposal about **0.136 m and 31.4°** from the saved legacy result. Controlled
replay identified ground preprocessing as the cause: the legacy crop retains
ground points, whereas the earlier migrated pipeline removed points at or below
`ground_height + 0.08 m` before downsampling and registration.

| Controlled replay | Difference from the saved legacy pose |
| --- | --- |
| New ICP backend, unchanged legacy target | Numerical precision (less than 1e-12 m and 1e-10 degrees). |
| Legacy crop, only remove the ground band | 0.13593 m / 31.43845 degrees: reproduces the new pose. |
| New crop, retain ground | 0.01133 m / 2.37832 degrees. |
| New crop and ground removal | 0.13593 m / 31.43845 degrees. |

These runs use exactly the same measured reference XYZ and all 64 orientation
seeds. Quality gates select the same winner as ungated ranking in each case;
they did not cause this pose change. Running the original legacy ICP function
on the winning legacy seed also reproduces the saved pose to numerical precision.
At the legacy final matching distance, 1,350 of 7,206 matches at the legacy pose
are against the ground band. Removing those possible matches changes the
optimization and its winning orientation seed. This explains the disagreement;
it does not establish which pose is physically correct.

The recorded in-hand stone-1 log also produced a proposal with valid opening
evidence, but has no independent ground-truth acceptance. Field/scene terrain
handling, gripper calibration refinement and quality thresholds still need
that acceptance. The investigation itself made no setting changes. The subsequent
implementation restored legacy field behavior as described below.

### Restored field default and expanded recorded checks

`field_ground_removal: false` now retains ground for field recovery. The legacy
box (initial position minus `[0.8, 0.8, 0.5]`, plus `[0.8, 0.8, 1.0]` metres) is
applied before voxelization, and transfer ROIs contain that box. Scene
identification still removes the configured ground band. Setting
`field_ground_removal: true` is an explicit experiment; it reproduces the
previous 13.6 cm / 31.4 degree disagreement on the recorded fixture.

Both the portable saved-crop regression and full imported field replay now
reproduce the legacy pose to numerical precision. The restored field result
has target coverage 0.2892 against the existing minimum 0.35 and remains
invalid for acceptance. The gate was not relaxed to make the recorded case pass.

In-hand processing now aligns each cropped view against the stone_grab surface
before jaw removal and merging. Rejected corrections are not applied and make
the proposal invalid. Reports retain each frame's transform, coverage, RMSE and
correction size. Six recorded views from two sessions pass the bounded alignment
gates, with corrections below 4 cm and 5 degrees. Two recorded scene regions
verify rejection of deliberately unobserved expected objects; controlled model
scenes verify positive multi-object reconciliation. These tests do not supply
independent physical pose ground truth.

`tests/export_excavator_perception_regressions.py` reproduces the portable
fixtures from the legacy repository and records source hashes. Tests run without
loading legacy code or requiring that repository.

## Review a proposal durably

The replay review store records immutable proposals and one-use decisions in
SQLite. Accepting requires the current scene revision. Concurrent or stale
decisions cannot overwrite newer poses; accepting a partial scene observation
preserves all other objects. In-hand acceptance stores attachment geometry
separately from world poses.

```bash
python -m stacking_excavator.perception.cli stage \
  /tmp/excavator-review.db /tmp/excavator-field-result/proposal.json \
  --initial-revision 1

# Use the token printed by stage:
python -m stacking_excavator.perception.cli decide \
  /tmp/excavator-review.db TOKEN accept --revision 1
```

Use `reject` to record rejection. A restarted store retains poses and decisions.
The store is explicitly a replay session, with no hardware authority.

## Acquisition and delivery boundary

`StationaryCaptureService` accepts a source that yields `CaptureFrame` values.
It owns identified capture jobs and cannot command joints. `CaptureSpool`
reserves capacity before acquisition, bounds frame count and payload, retains
raw data across outages, and refuses automatic recapture after an interrupted
job. Duplicate successful requests return the existing manifest. Reservations
are conservatively retained; administrative reconciliation/retention expiry is
not yet automated. A concrete ROS source must enforce its own acquisition
deadline and supply commissioned timestamp/calibration metadata.

`delivery.pull_product` pulls only a requested crop directory through rsync/SSH,
using compression, partial staging, an explicit bandwidth limit, I/O and total
attempt time bounds, and hash verification before publication. It terminates
bulk transfer when the supplied control-freshness check fails. A caller must
serialize transfers and reserve staging capacity. The adapter is tested with
injected subprocesses; no real SSH host or WiFi link has been commissioned here.

These boundaries support the next acquisition/device integration work. The
simulated execution owner and desktop operations views are now implemented;
see [the integrated workflow](excavator-simulation.md). Concrete ROS adapters and
moving scan choreography remain deployment work in the
[overall plan](excavator-implementation-plan.md).

## Validation

Before the steps 1–4 extension, 98 focused tests passed across perception, acquisition/delivery,
capture persistence, emulation, common point-cloud/architecture and tabletop
acquisition regressions. Run them with:

```bash
python -m pytest \
  tests/test_excavator_perception.py \
  tests/test_excavator_acquisition_service.py \
  tests/test_excavator_capture.py \
  tests/test_excavator_pcd_emulation.py \
  tests/test_pointcloud.py tests/test_architecture.py \
  tests/test_tabletop_femto_acquisition.py -q
```

The CLI capture → crop → identify → stage → replay-accept flow was also run.
Real recorded field, scene and in-hand imports completed without ROS. Compact
real frame fixtures retain source hashes and compare field/scene transforms
against independently saved world clouds. The tests do not establish live
ROS, controller, WiFi or physical perception acceptance.

The subsequent steps 1–4 implementation passed 170 focused checks across the
expanded perception suite, simulated execution, operations and tabletop
regressions. See [integrated validation and commands](excavator-simulation.md).
