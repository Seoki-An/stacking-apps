# Automated wrist and FEMTO scene capture

For normal pick-scene capture and identification, use the integrated
`stacking_tabletop.acquisition.pick_scene_workflow run` command. It materializes
the standard route internally, so it does not require `--plan` or `--approve`.
The separate plan/run commands below remain the low-level interface for route
development, offline previews and capture-only experiments.

The scene capture CLI generates nine overlapping wrist views from one seed view,
records fixed FEMTO before/after observations and assembles an offline replay
session. **Keep the existing UR bringup and External Control program running.**
The application sends references to the existing admittance controller; it does
not restart bringup or rewrite its startup/home settings for each viewpoint.

Status: implemented and tested offline with simulated device lifecycle and native
kinematics. The first live board/table sequence, `scene_capture_001`, completed
all five wrist views (50 RGB-D frames), both seven-exposure FEMTO references and
verified offline replay. This exercises the sequence on hardware; capture
completion does not certify collision freedom, accuracy, scene immobility or
successful ICP fusion.

The second live sequence, `scene_capture_002`, contains three stones and also
completed all 50 wrist frames and both FEMTO references. Its offline rigid GICP
check improved held-out stone median agreement in only 2/15 view/stone tests;
10/15 candidate transforms exceeded the provisional 3-degree / 20-mm bounds.
Ordinary point-to-point ICP improved none of the 15 tests. These results do not
support automatic ICP fusion. They also do not isolate depth bias from visibility,
calibration or weak registration geometry. The local report and interactive
overlay are under
`apps/tabletop/data/experiments/scene_capture_002/inspection/icp_check/`.
The consolidated [RGB/FEMTO scene workflow](tabletop-rgb-scene.md) describes the
sparse checks, camera-prior refinement, dense MVS, confidence filtering, scene
assembly and catalog comparison performed on these captures. Calibration accuracy
remains a separate investigation; the 1-mm accuracy target remains unverified.

The route was expanded after the scene-002 RGB MVS experiment showed useful
shape agreement with the measured stone catalog but incomplete side coverage.
Existing scene-001/002 data and reviewed five-view plans retain their original
meaning; plan schema 1 remains valid. New plans use the nine-view schema 2 route.

## One seed, one preview, one automatic sequence

Use an empty tool and an existing stationary camera view aimed at the region.
The seed should leave useful FEMTO visibility; the first version returns there
for the final reference rather than planning a separate arm-clear parking pose.
You do not specify nine joint poses. Provide an approximate distance from the
seed camera to the region centre, a near translated offset and a larger oblique
offset.

The generated views are center; left, right, up and down at the near radius; and
four diagonal oblique views at the larger radius. All views aim at the same
point. Each offset view follows a nearby IK branch and returns along the reverse
path to the seed. Defaults use a 40-mm near radius, 80-mm oblique radius and 5-mm
Cartesian planning increments. At a 0.4-m target distance, these correspond to
view angles of 5.71° and 11.31°. Wrist roll alone is not the scan motion.

The near views preserve connected image overlap. The oblique views add translated
and angular diversity for side surfaces. They are still modest angles and cannot
guarantee visibility of hidden lower sides. Their benefit must be evaluated from
the resulting MVS coverage rather than assumed from camera count.

### Wider RGB coverage

Use `--viewpoint-pattern coverage13` to add four orbital views at 30° from
the seed optical axis, one from each side. At a 0.4-m target distance their
camera centers are about 0.20 m sideways from the seed and about 0.054 m
closer to the target. This gathers more side-surface evidence and expands the
set of camera positions. All thirteen views still aim at one target; a
different target or a tiled route is needed to inspect a separate area of the
table. Dense RGB processing uses the center, four oblique, and four side
images as reference depth maps for this route.

Generate the route and inspect `preview.html` before moving the robot:

```bash
python -m stacking_tabletop.acquisition.scene_capture_cli plan \
  --distance 0.4 --offset 0.04 --oblique-offset 0.08 \
  --viewpoint-pattern coverage13 \
  --output apps/tabletop/data/experiments/scene_plan_side_001
```

The preview verifies IK, joint limits, and the camera paths. It does not check
collisions or guarantee RGB overlap. Inspect clearance around the table,
stones, camera, cables and wrist for all four lower, farther-reaching legs.
The integrated `pick_scene_workflow run` now defaults to this thirteen-view
route. Operations exposes **Wrist viewpoints → Wider: 13 views** and **Compact:
9 views**. CLI `--viewpoint-pattern coverage9` selects the compact route. The
standalone plan CLI retains its nine-view default so existing plan commands
keep their behavior. No new route is executed by an offline simulation.

### Extended lateral coverage: seventeen views

Use `coverage17` when the 30° side route does not expose enough stone surface.
It preserves the original thirteen poses and adds left/right pairs at **45° and
60° relative to the central camera axis**. The two extra angles give stereo
partners for newly visible sides. Up/down remain at 30° because steeper arcs
were unreachable from the recorded `view_005` seed. At 0.4 m target distance,
maximum lateral camera travel increases from 0.20 m to approximately **0.35 m**.
The region is still centered on the same target.

A generated offline plan is at
`apps/tabletop/data/experiments/scene_wide_preview_001/preview.html`.
Its simulation, RGB gallery and comparison are at
`apps/tabletop/data/experiments/scene_wide_simulation_001/overview.html`.
Select that simulation's **`captures`** directory for identification in Operations.
This is a new route; `scene_side_preview_001` remains the original thirteen views.

Reproduce with fresh output directory names:

```bash
python -m stacking_tabletop.acquisition.scene_capture_cli plan \
  --distance 0.4 --offset 0.04 --oblique-offset 0.08 \
  --viewpoint-pattern coverage17 \
  --seed-view apps/tabletop/data/captures/hand_eye/view_005 \
  --urdf apps/tabletop/data/experiments/scene_side_preview_001/robot.urdf \
  --output apps/tabletop/data/experiments/scene_wide_preview_002

python -m stacking_tabletop.acquisition.scene_capture_simulation \
  --plan apps/tabletop/data/experiments/scene_wide_preview_002 \
  --output apps/tabletop/data/experiments/scene_wide_simulation_002
```

On the same synthetic stone placements, side-surface samples visible from at
least two camera positions improved as follows:

| Stone | 13 views | 17 views |
|---|---:|---:|
| 9 | 63.3% | 78.9% |
| 10 | 71.9% | 83.8% |
| 11 | 57.1% | 69.9% |

These are geometric visibility measurements, including modeled occlusions and
parallax; actual MVS accuracy and hardware clearance remain unvalidated. A tested
alternative seed with four 45° sides provided less support than this route.

Operations exposes **Wrist viewpoints → Extended: 17 views**. The integrated CLI
accepts `--viewpoint-pattern coverage17`; its default remains thirteen views.
Dense processing uses center, oblique and all eight side images as reference
maps (thirteen references instead of nine), so capture and reconstruction take
more time. Complete IK paths are checked, but collisions are not; selecting this
route for live capture requires clearance for its larger robot/tool envelope.

### Offline simulation in the Operations cell

Use a saved thirteen-view plan to test camera coverage without ROS or hardware:

```bash
python -m stacking_tabletop.acquisition.scene_capture_simulation \
  --plan apps/tabletop/data/experiments/scene_side_preview_001 \
  --output apps/tabletop/data/experiments/scene_side_simulation_002
```

The experiment uses the Operations tables, FEMTO obstruction, UR5e and gripper
meshes, the saved plan's robot kinematics and hand-eye calibration, and catalog
stone meshes 9, 10 and 11. Native posegen synthesizes separated pick placements
on the configured pick table. Its convergence and residuals are recorded per
stone; a nonconverged candidate is not a verified stable placement. Rendered
meshes are seated vertically on the table after the DSF solve; both original
and adjusted poses and the height corrections are retained.

Open `overview.html` for the robot/view slider, generated RGB images, and the
nine-versus-thirteen-view coverage comparison. Each view also saves ideal axial
depth and `ideal_cloud.pcd` in `base_link`. These are raycast geometry, not MVS
output. The images use procedural texture and ideal pinhole cameras. Coverage
uses area samples and first-hit occlusion, including the modeled robot. The
reported two-view support requires at least one degree of parallax. It measures
an opportunity for stereo, not a successful match or an accuracy guarantee.

The simulator also creates a complete `captures/` directory containing
`capture.json`, timestamped `wrist/` frames, `femto_before/`, `femto_after/`,
and a hash-verified `session/`. In Operations, open **Scene capture and stone
identification**, select **Identify existing wrist RGB + FEMTO capture**, and
choose this `captures/` directory as the source. Choose a fresh output folder
and keep the pickup-table height at 0.100 m. The normal RGB reconstruction and
joint pose identification run, including the stone-ID assignment window.
All generated observations and workflow outputs retain synthetic provenance.
The modeled FEMTO view is positioned at the top of its Operations obstruction
box and aimed at the stones; it is not a measured fixed-camera calibration.

For these synthetic placements, posegen explicitly uses the table top as
`ground_height` (0.100 m in this cell). The simulation overrides contact-force
absolute and relative tolerances to `1e-7`; the ordinary tabletop solver profile
is unchanged. The earlier `1e-3`/`1e-4` force tolerances caused stones 10 and 11
to reach the pose optimizer's 75-iteration limit. Moving the equivalent ground
plane from local zero to world 0.100 m alone did not change those results.
Solver flags, iterations, gradient, trust radius and residuals are retained.

The experiment does not authorize motion or perform collision validation.
Camera housing, cables, and unmodeled objects are absent. It does not establish
real-world lighting robustness or real-camera millimetre accuracy.

### 1. Prepare without moving

Source ROS Jazzy and the existing ROS workspace in the same environment used
for wrist capture. Leave UR bringup running, with current joint/tool feedback
and `/robot_description`. Stop other motion clients such as a running saved-plan
execution; the capture command checks for competing reference publishers.

From the repository root, for a region approximately 0.4 m ahead of the camera:

```bash
python -m stacking_tabletop.acquisition.scene_capture_cli plan \
  --distance 0.4 --offset 0.04 --oblique-offset 0.08 \
  --output apps/tabletop/data/experiments/scene_plan_003
```

This reads the current joints and running robot description, snapshots its
kinematics and the hand-eye calibration, solves the entire route, and writes:

- `plan.json`: all joint paths, camera poses and asset hashes.
- `robot.urdf`: kinematics from the running robot, without mesh dependencies.
- `calibration/`: pinned hand-eye/color artifacts.
- `preview.html`: camera paths, viewing axes and robot link skeletons.

Open `preview.html`. The command prints `approval_sha256`. Preparing a plan
sends no arm or gripper commands. It does advertise the existing transport's
publishers while connected, so close other motion clients before live use.

**Review the complete path and return legs once.** Generated IK and speed/cable
limits do not check collisions. The preview uses link skeletons, not full
camera/gripper volumes, so include their clearance when reviewing the real setup.
The explicit approval hash binds the run to that reviewed route and calibration;
there is no approval prompt at every viewpoint.

### 2. Execute and capture

Keep the robot at the seed. Replace `PLAN_SHA256` with the value printed above:

```bash
python -m stacking_tabletop.acquisition.scene_capture_cli run \
  --plan apps/tabletop/data/experiments/scene_plan_001 \
  --approve PLAN_SHA256 \
  --fps 15 --count 10 \
  --output apps/tabletop/data/experiments/scene_capture_001
```

**This command moves the UR5e.** Approval confirms the empty tool and review of
the full path. It does not open or close the gripper. The first release supports
one fixed FEMTO reference camera, primary by default; `--camera secondary` selects
the other calibrated camera. Existing `--femto-config` selects the installation
profile. Its current field crop ends at 0.20 m: use appropriate bounds for taller
placed scenes. Full native depth evidence is retained for later recropping.

Automatic sequence:

```text
verify plan/assets/current robot and seed
  → stationary FEMTO before
  → open wrist recorder once
  → center and four near captures, returning to seed after each
  → four diagonal oblique captures, returning to seed after each
  → close wrist recorder
  → stationary FEMTO after
  → verify and assemble the replay session
```

The known working paired profile defaults to RGB 1280×720, depth 848×480 at
10 fps. The RealSense connection and robot timestamp recorder stay open across
all nine views. Actual measured joint settling precedes each burst, and the
existing wrist recorder enforces its tighter stationarity/source-time guards.
Motion references use the existing adapter with caps of 0.15 rad/s joints,
0.03 m/s tool translation and 0.30 rad/s tool rotation.

The workflow owns FEMTO driver startup/cleanup at the before/after brackets.
It stops that owned driver during wrist acquisition to avoid simultaneous active
depth emitters. These camera starts are automatic; **UR bringup stays alive**.
A separately running FEMTO driver is rejected; stop it once before using this
workflow. The app never kills an externally owned launch process. Continuous
FEMTO operation with validated stream/emitter switching is a later optimization.

Repeated scans reuse the saved plan and persistent bringup when the robot is
back at its seed and the route remains appropriate for the physical scene. Use
a new output directory each time. A changed robot description, calibration,
joint branch or path requires a new plan/review.

## Results and viewing

```text
scene_capture_001/
  capture.json              # running, complete, failed or cancelled
  executed_plan.json
  reviewed_plan/            # exact approved plan and calibration snapshot
  femto_before/
  wrist/view_000_center/    # nine independently inspectable wrist views
  wrist/view_001_left/
  ...
  wrist/view_008_oblique_left_down/
  femto_after/
  session/                  # existing portable hand-eye replay session
```

Use the existing replay CLI to create the PCD viewer after recording:

```bash
python -m stacking_tabletop.acquisition.hand_eye_replay_cli replay \
  apps/tabletop/data/experiments/scene_capture_001/session \
  --output apps/tabletop/data/experiments/scene_capture_001/replay
```

This replay shows calibrated observations. The D435-to-FEMTO ICP comparison is
currently an offline experiment with post-fit transform bounds, described above;
Sparse RGB triangulation is available through `rgb`, camera-prior refinement
through `rgb-refine`, and diagnostic dense MVS through `rgb-dense`.
The offline `rgb-filter` step uses disjoint-pair agreement to reject unsupported
dense samples and separates FEMTO overlap, unresolved differences and possible
coverage extensions. `scene-assemble` packages those RGB layers with both FEMTO
scans; `scene-inspect` and `scene-replay` verify and view the portable result.
See the [RGB/FEMTO scene workflow](tabletop-rgb-scene.md) for all offline stages.
No automatic correction or
millimetre-accuracy claim is attached to the capture.

## Failure and cancellation

Ctrl-C/SIGTERM, stale controller feedback, competing command publishers, camera
failure or capture validation failure stop further capture/motion and request
hold at the measured pose. **There is no automatic return on failure.** Partial
data and the failed/cancelled manifest remain available; no session is labelled
complete unless all views, both references and assembly succeed.

A hold request is not an acknowledgement of a hardware emergency stop. The
capture process never releases a held object, and an existing gripper hold blocks
startup. After interruption, reconcile the robot and scene before a new run;
there is no automatic resume from an assumed waypoint. Bringup need not restart.

## Offline development

An existing wrist recording can supply the seed joints for preview development:

```bash
python -m stacking_tabletop.acquisition.scene_capture_cli plan \
  --seed-view apps/tabletop/data/captures/hand_eye/view_004 \
  --urdf apps/tabletop/assets/ur5e/ur5e_ik.urdf \
  --distance 0.4 --offset 0.04 --oblique-offset 0.08 \
  --output apps/tabletop/data/experiments/scene_preview_offline
```

The repository URDF is nominal, so this example is an offline illustration.
A live run requires an exact match to the current driver's kinematic description
and checks model/controller FK at the seed. Prefer a live `plan` for execution.
The new tests exercise near/oblique view geometry, legacy-plan compatibility,
branch rejection, asset/review
binding, exclusive ownership, settling, bracket order, cleanup and abort without
return. They operate without ROS or cameras.
