# Offline excavator planning

`stacking_excavator` composes common MCTS/CEM, native pose generation and
simulation, and native direct/regrasp motion planning. It uses the commissioned
VDK23 and stone-grab models from `stacking-planner`. This delivery implements
planning, persistence, and the offline Operations Console. Excavator action
sampling follows the legacy ablation score policy through common array kernels;
tabletop retains its existing sampler and filters.

## Run

From the `stacking-apps` checkout, with the sibling `stacking-core` checkout:

```bash
make install-excavator
stacking-excavator-plan \
  --repository "$PWD" \
  --pick-pose-data /path/to/pick.json \
  --save-run /tmp/excavator-plan.stack \
  --steps 1 --seed 0 --max-run-time 600
```

The native rebuild includes `configure_excavator_ik`; an older installed core
without this binding cannot construct the excavator planner. The editable
installation reads assets and profiles from the repository specified by
`--repository`. Source-checkout invocation is also available:

```bash
PYTHONPATH=packages/stacking-common/src:apps/excavator/src \
  python -m stacking_excavator.planning.cli --help
```

Excavator YAML angle settings use explicit `_degrees` fields, including
`robot.home_positions_degrees`, gripper opening offsets/limits,
`spin_step_degrees`, `target_rot_tol_degrees`, and
`max_handoff_orientation_distance_degrees`. The loaders convert these to radians
once; runtime objects and saved-run contexts retain their canonical SI values.
Legacy unsuffixed radian fields remain supported, but specifying both forms of
the same field is an error. The simulation device uses the same planning-profile
loader as the planner.

Transforms such as `world_T_stack` remain `[x, y, z, qx, qy, qz, qw]` quaternion
poses. Direction vectors, optimizer radii, weights and mixed pose-error tolerances
are not degree-valued angles. Perception's opening limits and measured-angle
difference use degrees; its calibrated polynomial coefficients remain unchanged
and produce radians.

Pickup JSON maps catalog IDs to world-frame SI poses using XYZW quaternions:

```json
{"55": [2.0, -4.0, 0.4, 0.0, 0.0, 0.0, 1.0]}
```

Values may alternatively be `[position_xyz, quaternion_xyzw]`. Existing
trusted local `pose_data.pkl` mappings are supported explicitly. No legacy
planner classes or repositories are imported at runtime. `--object-ids`
selects a subset of the recorded scene. Requested IDs missing from the scene
produce a warning and are skipped. If none match, the command exits normally
without creating a run; omit `--object-ids` to use all recorded objects.

`--search-only` saves placement search without invoking robot motion planning.
With the default `pose_from_scan: false`, pickup data is optional: the run uses
all objects from `inventory.yml`, or the catalog subset passed with
`--object-ids`. Initial orientations come from the assets' principal axes.
The saved context labels its identity inventory poses as synthetic search
initialization, not measured pickup poses. Motion planning and
`pose_from_scan: true` still require `--pick-pose-data` for new runs.

Override `orientation.pose_from_scan` from `action_generation.yml` explicitly:

- `--pose-from-scan`: initialize MCTS orientations from the recorded stone
  orientations and enable the scan-orientation filter.
- `--no-pose-from-scan`: use principal-axis orientations and configured rolls.
- Omit both flags to use the YAML value. The startup log prints the effective
  value and whether it came from the profile, CLI override, or saved run.

Both modes use recorded pickup poses for motion planning. This switch changes
placement orientation generation/filtering; it does not activate live scanning
or disable the pickup input. In **New planning run**, **Pose from scan override**
offers **Use configuration (no override)** (the default), **true**, and **false**.
Use configuration follows the YAML value without passing an override flag;
true/false passes your selection as a CLI override without changing the
configuration file. Resume displays the recorded value and preserves it;
start a new run to choose a different value.

For example, append `--pose-from-scan` to a planning command with
`--pick-pose-data /path/to/pose_data.pkl` and a new `--save-run` destination.

The grasp solver's `trust_region.radius_init` sets the initial optimization-step
radius, while `radius_max` caps its growth. The step combines translation,
rotation and gripper opening, so these are not a workspace radius or a pure
distance in metres. The solver shrinks the radius when its local approximation
predicts improvement poorly and may grow it when prediction is good. Require
`0 < radius_init <= radius_max`; the excavator profile currently uses `0.1`
for both. An inconsistent explicit pair is rejected before search begins.

```bash
stacking-excavator-plan --repository . \
  --config apps/excavator/config/mcts.yml \
  --search-only --steps 10 --save-run /tmp/excavator-search3.stack
```

`--iterations`, `--max-search-time`, and `--config` (MCTS YAML) bound or replace
search settings. Small time limits can expire during initial CEM generation.
`--max-run-time` limits the entire worker, including native grasp/motion work.
Ctrl+C and wall expiry terminate its dedicated process group and preserve
committed `.stack` records, marking the run cancelled.

The CLI uses the same terminal reporter as the tabletop MCTS debug runner.
It prints the resolved workload, `[Progress]` phase messages and 10-second
heartbeats during long calls, `[INFO]` CEM/tree/validation diagnostics, and
`[Timing]` phase and total durations. Search-only runs finish with the saved
run path, discounted return, target IoU, and completed placement count.
Motion runs also report candidate attempts and native planning phases.
Search-only `.stack` records contain the same detailed search statistics as
tabletop. This output also streams into the Operations Console planning log.

Resume a search-only or motion-planning run from its last successfully committed placement:

```bash
stacking-excavator-plan --repository "$PWD" \
  --resume-run /tmp/excavator-plan.stack \
  --save-run /tmp/excavator-continued.stack --steps 3 --max-run-time 1800
```

Supply the same `--config` if the original run used a custom MCTS profile.
Recorded seed and search budgets are reused. Resume keeps search inputs
immutable: omit `--seed`, `--iterations`, and `--max-search-time`, or repeat
their recorded effective values. Changed values fail before creating the new
run. `--steps` and `--max-run-time` may change. Start a new run to use different
search inputs; the retained plans remain bound to their original context.
Already-unit pickup quaternions are preserved when copying/loading checkpoints.
For older excavator runs, resume restores the exact context poses after checking
that checkpoint differences are limited to quaternion normalization roundoff
(eight machine epsilons). Position, scan provenance, and larger orientation
changes still fail validation; the original artifact is not modified.

Resume validates resolved profiles, input poses, native binary, model hashes,
frames, and joint order. It creates a new document containing the committed
prefix; the source is unchanged. Resume restores the original search-only or
motion-planning mode automatically. Do not supply new pickup poses or object
IDs. `--steps` specifies **additional** steps: after nine committed steps,
`--steps 21` requests continuation from step 10 through step 30. The run-time
limit is a total wall-clock limit for this new invocation, not a per-step limit.

Search-only runs now commit the full selected search state and its checkpoint
in the same transaction as each completed step. A timeout during the next search
can therefore resume from that checkpoint. The GUI preserves search-only mode
in its disabled mode checkbox when resuming. A failed first placement has no
selected state to resume. Restart rebuilds the search tree and observation;
it does not restore worker processes or promise an identical random search path.

**Older search-only files:** versions before this fix saved candidate and
visualization records without checkpoints. Those files can show completed
steps but report `missing planning_checkpoint/N` when attempting to resume.
The console now explains this limitation directly. They remain viewable, but
require starting a new run; the missing full state is not reconstructed from
viewer poses. A saved checkpoint records simulated planning state, not
physical execution.

## Operations Console

In **Plan → Planning attempt → Step**, select the placement whose robot motion
you want to replay. Each entry states whether motion was recorded. Select an
attempt, then use **Play** or scrub the timeline. The step selection is shared
with Search Inspector; changing steps stops playback and resets its timeline.

Both operations consoles offer the same **Objects: DSF surface / PLY mesh**
toolbar selector, available in Plan and Search Inspector. DSF is the default:
the renderer samples each component's curved support surface using its stored
nodes and sharpness, rather than displaying the OBJ's node-hull triangles.
PLY mode shows the corresponding reconstructed stone mesh. Both modes are
cached and preserve the camera when switching, including during replay.
All 49 excavator stones have matched PLY assets with source hashes.

Plan's robot options match tabletop: **Visual meshes**, **Collision meshes**,
and **Visual + collision wireframe**. DSF collision assets use the same curved
surface tessellation. For excavator, the available collision geometry is the
gripper geometry recorded in the commissioned model. Candidate visibility,
pose selection, contacts, forces, environment layers, and video controls match
the tabletop Search Inspector. The object surface selector also applies to
trajectory video exports.

Candidate lighting and transparency use the same policy in both consoles:
rough diffuse material (ambient 0.42, diffuse 0.58, specular 0), a fixed key
light at elevation 45°/azimuth -35° with intensity 0.8, and smooth outward
surface normals. Translucent stones cull their rear faces so inward-facing
surfaces do not show through as reversed lighting. Target/environment shells
retain their own two-sided rendering.

Scenes with transparent candidates, target shells, or environment layers use
depth peeling with multisampling disabled and no fixed peel-layer limit.
SSAO is enabled only for fully opaque scenes: its VTK delegate pass bypasses
the standard depth-peeling path. The platform-sized values remain distinct:
SSAO radius is 0.03 m for tabletop and 0.3 m for excavator; force-arrow length
is 0.035 m and 0.35 m respectively. These are display dimensions, not changes
to the collision geometry or solver configuration.

```bash
make install-excavator-operations
stacking-excavator-operations . --run /path/to/excavator-plan.stack
```

Omit `--run` to open the console before planning. From an uninstalled checkout:

```bash
PYTHONPATH=packages/stacking-common/src:apps/excavator/src \
  python -m stacking_excavator.gui.operations.application .
```

The window contains **Plan** and **Search Inspector** tabs. The toolbar opens
recorded runs, starts new planning, resumes a checkpoint into a new document,
reloads incremental results, and exports JSON/CSV summaries. The planning
dialog accepts recorded pickup poses, an MCTS YAML profile, additional step
count, run time limit, and an optional search-only mode. It streams subprocess
output without blocking the window. Stop terminates the CLI's isolated worker
through its supervisor; the resulting cancelled document remains inspectable.

Plan shows every recorded attempt, including failed and partially completed
steps. Play/pause and the timeline replay a whole step or a selected segment.
The viewport projects the commissioned VDK23 and stone-grab URDFs, recorded
joint coordinates, signed jaw openings, and carried-object target tracks.
Pick, handoff, and place phases use their recorded support planes and scenes.
The target volume uses the saved stack-to-world transform. Visual meshes,
gripper collisions, and a combined view are available. Failed attempts without
trajectories show their input scene and the robot at the recorded profile's
home coordinates.

Replay restores the run's planning profile and verifies mesh/URDF content
hashes. Current solver tuning does not replace saved values. Missing object
tracks remain unknown; faint pickup/placement ghosts provide reference poses.
Untimed paths use display-only playback timing. Recorded opening samples are
used when present; otherwise the profile opening is the display fallback.
The source robot URDF has no body collision meshes, so collision mode renders
the gripper. Search Inspector retains candidate selection, pose modes,
simulation timelines, contact/force layers, and trajectory MP4 export.

This is an offline console. Hardware execution and operator control of the
excavator remain outside this application.

## Ownership and behavior

Generic profile schemas, OBJ decoding, height-map rendering, search composition,
native result conversion, context hashing, and process supervision live in
`stacking-common`. Tabletop keeps its existing import names through small
aliases and supplies its original geometry policies. Physical YAML values
remain independently owned by each application.

The excavator app owns:

- The six-joint VDK23 chain and an owned native analytic IK seed callback.
  The calibrated bucket/wrist zero orientations are accepted alongside the
  original native reference geometry; generic IK refines the seed.
- The stone-grab floating root, independent jaw coordinates, and eight declared
  pad/tooth contact geometries. Both opening coefficients are positive because
  the URDF joint axes oppose one another.
- Separate pickup, stack-placement, and handoff scenes with explicit ground
  planes and configured fixed boxes. Remaining pickup objects are pickup
  obstacles; committed stack objects are placement/handoff obstacles.
- Stack-to-world conversion and the 0.20 m production placement lift. Motion
  targets use the lift once; successful commits retain the settled local pose.
- The nine ordered legacy handoff locations. Only retryable failures that can
  depend on handoff location advance to the next location. Fixed pickup or
  placement grasp failures return to candidate selection.
- Per-candidate fixed-scene resettling, common MCTS rejection feedback, attempt
  journaling, and atomic selected-step/checkpoint persistence.

The source IK URDF contains robot visuals but **no robot collision geometry**.
The migrated motion model preserves that scope: gripper and carried-object
collisions are represented. Full excavator-body collision validation is not
established by this delivery.

## Source-to-profile mapping

| Source | Migrated boundary |
| --- | --- |
| `agent/configs/config.yml`: production Q network disabled | Heuristic prior/value bootstrap; no Torch/Ray/model loader |
| `agent/configs/ablation.yml`: MCTS depth 3 after 10 placements, 200 iterations, 240 seconds; CEM 96/24/3/12 | `config/mcts.yml`; shared tree/search/sequence implementation |
| Principal axes, rolls 0/180°, yaws 0/90/180/270°, scan mode false | Common principal-axis provider and platform orientation selection |
| Production catalog/density 1000; ablation target 2 × 3.5 × 2.5 m, 5° taper, origin (0, 0, 0) | `inventory.yml`; asset mass and inertia multiplied by density once |
| Observation, rewards, fast settling/full structural phases | `environment.yml`, `solvers/simulation.yml` |
| Native posegen setup, diffsim defaults, and `PoseSolver` cosine schedule | `solvers/posegen.yml`, including platform-owned `schedule.objective_weights` |
| `planning/planning.py` grasp and motion setup | `solvers/grasp.yml`, `solvers/motion.yml`; all supported native fields materialized |
| `scripts/desktop/generate_sequence.py` frame/home/lift constants | `planning.yml`; world stack offset (-6, 3), production lift 0.20 m |
| Recorded July 1 plan | Fixture preserves its different 0.10 m lift and support heights; never substitutes current defaults |

The excavator MCTS profile follows the local `stacking-planner`
`agent/configs/ablation.yml`: depth 1 for the first 10 placements, then depth 3;
200 iterations, 240 seconds, 16 children, 8 validation candidates, and CEM
population/elite/iterations/keep of 96/24/3/12 with 15-degree yaw mutation.
Final selection uses `max_descendant` with weight 1.0; descendant validation
is disabled and its configured depth is 0. Simulation depth resolves to the
legacy Python default of 5. The local native worker allocation remains 12.
Action sampling and build-time filtering also follow ablation, as detailed
below. Reward weights, settling budgets, and supported grasp/motion settings
remain independently resolved in their respective profiles.
The target geometry also follows ablation: width 2 m, length 3.5 m, height
2.5 m, 5-degree taper on all four sides, and fixed stack-local origin
(0, 0, 0). The stack-to-world transform remains in `planning.yml`.
Ablation's synthetic inventory and other environment settings remain separate
from this MCTS, action-policy, and target alignment. Existing `.stack` files retain their
recorded target geometry; the updated target applies to newly planned runs.
The placement count is still controlled by `--steps` (default 1), independently
of MCTS depth. For example, `--steps 20` allows the run to reach depth 3 after
the initial 10 placements, if valid placements remain available.

### Action sampling and filtering

`action_generation.yml` owns the excavator's legacy score-mode settings:
orientation-aware hull support kernels, max-downsampled and smoothed height
maps, connected floor proposals, taper-aware anchor bounds, seven score terms,
and representative/elite/Boltzmann selection. Temperature is 0.3, elite fraction
is 0.25, excavator-distance weight is 2.0, and support contact-gap tolerance is
0.10 m. The distance origin defaults to stack-local (0, 0), as in ablation;
`scored_sampling.score.excavator_xy` can supply the reference position that the
legacy sequence entrypoint set from its target offset.

Fresh populations use private seeded Python and NumPy RNGs, balanced per-object
allocation, and the legacy first-placement shuffle/later-floor priority. They
do not perturb tabletop's random state. CEM yaw mutations retain object-specific
score lookup. End-to-end search trees need not be identical: the native solver,
parallel scheduling, inventory, and simulation profiles also influence results.

Initial height comes from `model_<id>_mesh.ply`, with hull fallback when absent.
Both footprints receive 0.12 m padding and the pose receives 0.15 m clearance.
PLY assets used by planning are included in saved-context hashes. The build gate
checks XY target overlap, gap residual <= 0.005, and equilibrium residual < 1.0;
native convergence flags are diagnostic rather than an additional rejection.
Volumetric containment remains part of simulation/reward evaluation. Placed
stones farther than 0.5 m from the target's XY rectangle become posegen boundary
bodies. No catalog stones are banned by default.

The active ablation score path bypasses the optional large-stone/pre-pose height
filters and disables mesh-face fitting, U-shape rejection, and hard support or
stacked-floor rejection. This migration preserves those active choices. It does
not enable those inactive legacy modes. Tabletop keeps its deterministic sampler,
mesh-height initialization, convergence check, and pre-simulation containment
filter. Regression fixtures capture legacy scores and seeded proposals on both
empty and occupied scenes; separate tests cover filter boundaries and native
posegen boundary-body selection.
CLI flags override YAML: the earlier `--iterations 10 --max-search-time 120`
example still requests 10 iterations and 120 seconds. Omit those flags to use
the profile defaults; `--config` replaces only the MCTS profile.

The pose solver now uses the excavator schedule instead of the inherited
tabletop weights. In `schedule.objective_weights`, each entry is a sum of its
coefficients times the named bases: `constant = 1`,
`ramp = (1 - cos(pi*t))/2`, `anti_ramp = 1 - ramp`, `sine = sin(pi*t)`, and
`gap_feedback = t * previous_c_gap`. The wrench coefficient multiplies
`diag(1, 1, 1, 10, 10, 10)`. Scheduled weights override the corresponding
`objective` values at each solve stage. The excavator target weight decreases
from 5 to 0; complementarity increases from 0 to 10. Tabletop retains its
existing schedule in its own profile. Profiles without explicit schedule
weights retain the previous shared schedule for compatibility.

Posegen force-solver tolerances are 0.001 absolute and 0.0001 relative, matching
diffsim. Grasp seed retreat is 0.75 m, resolved from the legacy 1.5 m gripper
height times its 0.5 surface-grasp offset scale. Motion approach distance
remains 1.5 m; it is a separate setting.

Some controls have no direct legacy equivalent: MCTS evaluation batching,
validation time reserve/state-drift limit, yaw deduplication, and native
trajectory candidate limits retain the migration's values. Legacy grasp
score/orientation rejection thresholds are not all exposed by the current
shared/native adapters. Matching supported configuration values does not establish
complete planner behavior parity.

The modern core deliberately removed diffsim's mass-normalized artificial
drag. Legacy solver `damping` is not physical damping and is not copied into
that field. The app uses physical damping 0 and the current native numerical
constraint-scale reference 0.001. Therefore matching input poses and profiles
does not imply identical old simulation trajectories or candidate rankings.

The active ablation sampling path now has legacy score-map and seeded-proposal
fixtures, plus a successful two-placement native CLI run. Longer multi-seed
searches, including the depth-3 phase after 10 placements, still need acceptance
coverage. Reward and simulation behavior also need comparison independently of
sampling: for example, the current place-stability weight is 0.5 while ablation
specifies 0.2. CNN scoring, training, perception/reconstruction, alternate
attachments, inactive legacy action modes, and ROS/hardware execution remain
outside this offline delivery. This is not a full legacy numerical-parity claim.

### Motion recovery

The excavator motion profile enables `joint_grasp_recovery: true` and
`scene_approach: true` for both `direct` and `regrasp`. Core defaults remain
false, preserving the tabletop policy.

- Failed endpoint IK triggers grasp-and-joint refinement. It first attempts
  the fixed pickup/placement endpoint, then jointly checks the target-relative
  grasp at both endpoints. Recovered joints and grasp openings feed motion
  planning. Subgoal IK uses the legacy home/goal midpoint and retains the goal
  branch for joints spanning a full rotation.
- Grasp sampling starts at `retreat_distance: 0.75` metres. An undersized pool
  is supplemented at `fallback_retreat_distances: [0.975, 1.2, 1.8]`, corresponding
  to the legacy stone_grab height factors. Direct planning fills a pool of
  `fallback_min_candidates: 8` reachable grasps in bounded batches. Independent
  pickup/placement grasp generation also uses these fallback offsets.
- Approach direction follows scene plane normals and separation gradients
  from nearby objects. `minimum_approach_up_component: 0.5` maintains at least
  30 degrees of elevation. Approach/retreat distance retries use factors
  `[1, 0.75, 0.5, 0.25]`.
- After grasp and standoff retries, a handoff-dependent failure may advance to
  the next configured handoff location. Fixed pickup/placement IK or motion
  failures, fixed placement grasp-generation failures, and invalid inputs
  stop that XY retry loop. Handoff target-boundary mismatches are retryable and
  report achieved position/orientation errors. Total timing sums all handoff
  attempts; each attempt retains its own diagnostics.

Collision checks and the 2 cm / 10 degree target-boundary gate still apply.
Legacy sometimes accepts an approximate subgoal IK result; this implementation
retains the core's strict waypoint IK checks. Recovery increases the amount of
work per placement and does not guarantee a feasible trajectory. See
[the recorded-case comparison](excavator-motion-legacy-comparison.md).

## Validation and replay data

The tabletop/common gate passed 155 focused tests before the excavator package
was introduced. The implementation then passed the complete Python suite;
the excavator tests additionally exercise native search on recorded objects,
frame/lift semantics, phase scenes, contact mappings, failure feedback,
context-based resume, and an actual seeded direct-motion solve. Native core
coverage checks binding ownership/refinement and the original C++ IK oracle.
Verification after console integration: 558 application tests (plus 9 subtests), 15 native Python
binding tests, and 3 focused native C++ tests passed. The application suite ran
with `QT_QPA_PLATFORM=offscreen` and pytest available through
`PYTHONPATH=/tmp/stacking-apps-test-deps`.

A bounded native search selected a validated placement from recorded objects
56, 58, and 8. A bounded CLI motion run reached grasp generation and preserved
two infeasible attempts before exhausting candidates. The seeded direct-motion
test isolates trajectory solving; it is not a grasp-quality test. Successful
fresh grasp generation and end-to-end regrasp on the recorded production scene
are not yet demonstrated by these smoke runs.

Common motion segments now retain optional sample-aligned target poses and an
explicit pick/handoff/place/transfer phase, alongside joints, openings,
attachment mode, and grasp events. Recorded direct and regrasp trajectories
round-trip through these contracts. Missing target samples stay unknown;
timestamps are preserved only when supplied. Old schema-v1 records remain
readable and keep their old encoding when the new fields are absent.
The excavator console consumes these records using shared headless selection
and URDF projection. Qt widgets and platform appearance remain owned by the
applications. Direct/regrasp fixture tests cover target interpolation, jaw
mapping, partial-step inspection, geometry checks, and actual planning-worker
cancellation; a virtual-display smoke test verifies 3D rendering.
