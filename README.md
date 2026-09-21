# stacking-apps

Application workflows for the stacking project.

This repository is organized around one platform-neutral Python package and
thin platform applications. The tabletop migration validated the shared
contracts first. The excavator app now provides offline placement search and
native motion planning using the commissioned VDK23/stone-grab models.

```text
packages/stacking-common   reusable planning, search, processing, and rendering
apps/tabletop              UR5e/SR-gripper, Femto, ROS, and execution integration
apps/excavator             VDK23/stone-grab offline MCTS and motion planning
```

`stacking-core` remains the native source of truth for bodies, scenes,
simulation, pose generation, grasping, and motion planning. Python bindings
are consumed through common adapters; the common package does not recreate the
mutable legacy `diffsimpy.Context` interface.

## Development

Run the structural tests without installing the packages:

```bash
pytest
```

For editable package development:

```bash
python -m pip install -e packages/stacking-common -e apps/tabletop
```

The root Makefile provides one-line editable installation commands with their
local dependencies installed in order:

```bash
make install-core       # stacking-core Python bindings
make install-common     # stacking-core + stacking-common
make install-tabletop   # core + common + tabletop perception dependencies
make install-excavator  # core + common + offline excavator planning
make install-excavator-operations # excavator planning and replay console
make install-workbench  # object-reconstruction workbench dependencies
make install-operations # stacking operations console dependencies
make install            # default tabletop perception environment
```

Run these commands from the `stacking-apps` repository root. `PYTHON` can be
set when a specific interpreter should own the editable installations:

```bash
make install PYTHON=/path/to/python
```

When using the native simulation backend, install all three sibling packages:

```bash
python -m pip install \
  -e ../stacking-core/python \
  -e packages/stacking-common \
  -e apps/tabletop
```

See [docs/architecture.md](docs/architecture.md),
[docs/migration.md](docs/migration.md), and the
[visualization migration baseline](docs/visualization.md).

For setup, planning, hardware startup, scanning, and reconstruction commands,
see the [tabletop command reference](docs/tabletop-commands.md).

For recorded-scene MCTS, native direct/regrasp planning, `.stack` runs, and
checkpoint resume, see [the excavator planning guide](docs/excavator-planning.md).
Native search-only timings are recorded in the
[excavator PMBS benchmark](docs/excavator-mcts-pmbs-benchmark.md).

Open its offline Operations Console with `stacking-excavator-operations .`
or `stacking-excavator-operations . --run /path/to/run.stack`.

## FEMTO capture

The FEMTO acquisition layer captures fixed fields and settled in-hand views using
serial-checked calibration, synchronized raw depth bursts and exact-time TF.
Output feeds the existing field fusion and reconstruction workflows. See the
[FEMTO guide](docs/tabletop-femto.md) for driver startup and capture commands.

For the proposed wrist-D435 field and scene identification milestone, see the
[hand-eye capture and accuracy experiment](docs/tabletop-hand-eye-experiment.md).
Its [calibration consistency audit](docs/tabletop-hand-eye-calibration.md) now
pins compatible PoseID inputs and creates portable offline snapshots.
The [wrist capture guide](docs/tabletop-hand-eye-capture.md) covers live RGB-D
preflight, timestamped stationary views and offline raw-record verification.
The [offline replay guide](docs/tabletop-hand-eye-replay.md) adds portable session
assembly, native-depth PCD exports and a synthetic demo with interactive visuals.

```bash
python -m stacking_tabletop.acquisition.cli --camera primary --output .data/field-primary
```

Start the matching Orbbec driver first. Acquisition sends no robot/gripper
commands. The Operations Console offers automatic wrist/pivot scans at the
held-object placement review gate, with a motion preview and explicit approval.

## Tabletop MCTS smoke run

The MCTS runner reads tree, CEM, validation, and debug inputs from one profile:

```bash
python apps/tabletop/scripts/debug/mcts.py \
  --config apps/tabletop/config/mcts.yml \
  --save-run /tmp/tabletop-mcts.stack
```

Edit the profile's `debug` section to select object IDs, placement count,
native or pass-through pose solving, and kinematic or stacking-core transitions.
Command-line values such as `--iterations 8` remain available as temporary
overrides.
Ordinary depth-two-and-deeper search uses PMBS-style global PUCT selection
with temporary virtual visits and distinct action reservations. See
[MCTS batching](docs/mcts-batching.md) for the algorithm, serial baseline,
and differences from the PMBS paper.
Add `--plan-motion` to pass every selected placement through the configured
native grasp and motion planner. For a short end-to-end smoke run:

```bash
python apps/tabletop/scripts/debug/mcts.py \
  --config apps/tabletop/config/mcts.yml \
  --object-ids 1 --steps 1 --iterations 1 \
  --max-search-time 60 \
  --transition-backend kinematic --pose-solver passthrough \
  --plan-motion --save-run /tmp/tabletop-planning.stack
```

For legacy-equivalent pickup geometry, pass the trusted registered pose file:

```bash
python apps/tabletop/scripts/debug/mcts.py \
  --config apps/tabletop/config/mcts.yml \
  --pick-pose-data /path/to/pose_data.pkl \
  --object-ids 1 --steps 1 --plan-motion \
  --save-run /tmp/tabletop-planning.stack
```

The loader preserves each object's measured world-frame position and
orientation, uses the remaining objects as pickup-scene obstacles, and records
the source path, hash, and poses in the run. Without `--pick-pose-data`, the
runner uses `debug.planning_pick_position` and `initial_yaw_degrees` as an
explicitly synthetic smoke-test scene. Production grasp and motion settings can
take a few minutes per attempted placement.
The optional `.stack` output is one transactional run document and can be
opened directly by `stacking-tabletop-operations`. Each completed search step
is committed without rewriting earlier steps.

Production composition can use `TabletopApplication.build_planning_sequence`
to feed motion-planning failures back into MCTS, retry another root candidate,
advance successful states, and append every attempt to the same run document.

## Tabletop object reconstruction

The canonical command runs prepare, cross-capture merge, and reconstruction
under one resumable manifest. Mass is recorded with the build and used to emit
commissioned density and inertia metadata:

```bash
stacking-tabletop-object-build . \
  --object-id 1 \
  --capture-directory apps/tabletop/data/captures/object_1
```

The tabletop density is loaded from `apps/tabletop/config/inventory.yml`.
Supply `--mass-kg` only when a measured mass should override that estimate.
The profile supports object-specific material values while retaining one
platform default:

```yaml
objects:
  density: 1000.0
  density_overrides:
    7: 1250.0
```

Reconstruction uses the closed triangle mesh for volume, center of mass, and
inertia when its topology is consistently oriented and watertight. Otherwise,
it falls back to convex-hull properties and records `volume_method` in the
metadata.

To rerun one stage, pass the workflow directory printed by the first command:

```bash
stacking-tabletop-object-build . \
  --resume apps/tabletop/data/object-build/1/SESSION_ID \
  --stage reconstruct
```

The optional reconstruction workbench reads and updates the same `workflow.json`:

```bash
stacking-tabletop-reconstruction-workbench . --object-id 1
```

`stacking-tabletop-workbench` remains a compatibility alias for the
reconstruction workbench.

Prepare crop/removal, cross-capture alignment, optional reconstruction cleanup,
manual DSF partition seeds, and final mesh/DSF review all run in the embedded
viewport. Enable manual cleanup or manual partitioning before starting the
reconstruct stage. Stage computation runs in a worker thread so the viewport
remains responsive while the common pipeline executes.

The original prepare, merge, and reconstruct commands remain available as
focused compatibility entry points.

## Tabletop stacking operations

The Stacking Operations Console is the shared tabletop application surface for
plan visualization, Search Inspector diagnostics, trajectory replay, and live
execution integration. Run `stacking-tabletop-operations .` and choose
**File → New planning run…** (`Ctrl+N`, also on the toolbar) to select a profile,
output file, optional pickup poses/object IDs, step count, and run time limit.
Full grasp/motion planning is enabled unless **Placement search only** is checked.
**Pose from scan override** selects **On**, **Off**, or **Use configuration**
for placement orientations without editing YAML. Off uses model principal axes;
recorded pickup poses remain available for grasp planning in either mode.
The dialog provides live logs, Stop, and Open resulting run, without commanding
hardware. Blank pickup/ID fields retain the profile values; if no pickup file is
configured, the planner uses the synthetic pickup scene.

Open an existing versioned stacking run with:

```bash
stacking-tabletop-operations . --run path/to/tabletop-mcts.stack
```

The console implements `.stack` loading, search-step and candidate navigation,
explicit pose/render modes, trajectory replay, diagnostic contact and force
layers, camera preservation, and lit tabletop mesh rendering. Its environment
panel independently toggles the translucent target, ground, support tables, and
planning obstacles. Tabletop reference bounds and layer defaults live in
`apps/tabletop/config/visualization.yml`; support tables and obstacles start
hidden. The Plan section replays every native joint-space segment using UR5e
and SR-gripper forward kinematics, including sample-aligned gripper opening.
Playback defaults to **Whole step**, with one slider spanning all motion
segments; **Selected segment** limits playback and scrubbing to one segment.
Playback interpolates joints and gripper opening between recorded waypoints
at a nominal 60 Hz. Recorded segment timing is respected; untimed paths use
a display-only 0.6 rad/s joint speed with a minimum 0.1 s per waypoint interval.
The saved motion is unchanged. Robot actors are reused between frames.
Stationary segment boundaries animate gripper-opening changes; discontinuous
joint paths are not bridged with invented motion.
Robot rendering uses smooth URDF visual meshes, UR5e vertex colors, a black
SR gripper, and soft specular highlights.
The Plan panel also offers **Collision meshes** and **Visual + collision
wireframe**. Collision hulls use outward cell normals and flat shading.
The Execution section connects saved direct plans to the existing UR5e and
Dynamixel ROS controllers. It provides explicit pickup/close/place/release gates,
Hold/Resume, Abort, bounded joint corrections, live measured robot rendering and
estimated object poses. Loading a run sends no hardware commands. Execution
checks recorded models and context, journals every attempt, and requires physical
reconciliation after failure or interruption. See [the hardware and execution
guide](docs/tabletop-hardware.md). New plans retain native grasp events and
attachment modes; older plans missing these semantics must be regenerated.

Common execution records, planning checkpoints, context snapshots, and derived
CSV/JSON exports support the integration; see [the execution
handoff](docs/common-execution.md). Automatic wrist/pivot scanning is integrated with saved execution. Physical
commissioning and live in-hand registration remain pending. When debug `--pick-pose-data` omits some
requested objects, the runner warns and uses the available requested IDs. With
no matching IDs it warns and exits without starting planning.

Tabletop floor-first search ranks orientation-dependent footprints by support,
open/fill area, frontier contact, and target boundary proximity. The same
active-floor geometry policy filters solved proposals before simulation and
checks the final fixed-scene placement (90% fill threshold, 20 mm bottom
tolerance). Placement-only rejections retire a pose, not the whole object;
only motion/achieved-support failures count toward the two-failure object ban.
Seeds are distributed across open-floor regions and eight yaw angles. Lookahead
starts at the first placement (`initial_depth1_steps: 0`) so packing choices can
account for future placements; early searches therefore do more work.
Tabletop simulation groups contact/solver settings under `physics.settling`
(single-point) and `physics.structural` (limit-surface, shared by short and long
validation). Both use absolute/relative solver tolerances `1e-5`/`1e-6` for the
current native solver; settling no longer uses the looser legacy fast tolerances.
`phases` holds settling, short structural, and long structural timing/stability
checks; `final_extra_steps` gives the final stone its longer quiet check.
Integration, ground, scene, and post-simulation validation settings are shared.
Each structural rollout starts at the settled poses with zero velocities.
Old flat profiles remain readable, including the structural-physics fallback
when no settling physics is specified; mixing old and new layouts is rejected.
This aligns configuration and phase ownership, not the internal numerical
implementation of diffsim and stacking-core or a guarantee of physical settling.

`integration.damping` applies implicit
mass-proportional viscous damping (`D = d M`, with d in 1/s) in both phases.
The velocity solve uses `(1 + dt * d) * M`, with unchanged physical mass/inertia
and incoming momentum. Contacts, pose integration, and snapshots use the solved
endpoint velocity. Unforced linear and angular velocities have the same decay
factor `1 / (1 + dt * d)` per step. The configured rate still needs validation
for this model. The midpoint-contact experiment was reverted after it increased
residual motion in saved stone scenes and exceeded the existing stability limit.
Set it to `0` to restore undamped dynamics; omitted values also default to zero.
This requires the updated stacking-core binding. It is independent of
`solver.constraint_scale_reference`, which only sets numerical constraint scaling
and must be positive. The old `solver.damping` key is rejected with a rename
instruction; physical `integration.damping` is unchanged.
The trial coefficient is not calibrated: damping can suppress genuine instability as
well as jitter, so passing a damped rollout is not evidence of hardware safety.

CEM removes execution-excluded poses before its final per-object and keep caps.
The sequence runner tries the returned ranked batch before applying accumulated
neighborhood/object exclusions to the next search, matching the legacy retry
order. Progress logs and run statistics report skipped candidates and MCTS
execution exclusions separately from CEM eligibility filtering. Native motion
failure messages include the worst collision's entity IDs, sample index, gap,
and required clearance, or the violating joint's position and limits.

CEM recovers empty parent pools by consuming untried fresh seeds, bounded by
its round limit and the search deadline. Deadline checks are cooperative between
native batches, not cancellation of an in-flight solve. Planning-mode logs and
each search-attempt record in `.stack` retain CEM rounds, rejection reasons,
retry numbers, fresh-stream exhaustion, and timeout status, including zero-root
searches.

### Planning checkpoints, resume, and interruption

New `--plan-motion --save-run` runs record resolved planning contexts and an
atomic checkpoint after each selected successful placement. Resume branches
into a **new** file; it does not overwrite the source or reuse an unfinished
attempt. For example:

```bash
python apps/tabletop/scripts/debug/mcts.py \
  --config apps/tabletop/config/mcts.yml \
  --resume-run /tmp/tabletop-planning.stack \
  --save-run /tmp/tabletop-planning-resumed.stack \
  --steps 5 --max-run-time 600 \
  --export-summary /tmp/tabletop-planning-resumed.json
```

`--steps` counts additional placements on resume. The stored pickup scene and
search settings are reused; profiles, runtime settings, native binary and assets
must still match the recorded context. Restore changed inputs before resuming.
Old files without a planning checkpoint cannot be resumed. Runs with execution
commands require physical reconciliation instead of this offline restart.

Ctrl+C stops the isolated planning worker even during native computation.
`--max-run-time` optionally sets a whole-run wall limit (seconds); it is distinct
from cooperative per-search budgets. Completed records survive interruption;
the unfinished attempt is retried from the last checkpoint. This does not stop
hardware. A run interrupted before its first checkpoint must start afresh.

The Operations Console's **File** menu provides **Resume planning into new run…**
(with logs and Stop) and **Export run summary…**. `.json` and `.csv` summaries are
derived from the `.stack` file, including after interruption. Choose unused
output paths: neither CLI run destinations nor summary exports overwrite files.
