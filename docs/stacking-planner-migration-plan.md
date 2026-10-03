# Stacking-planner migration implementation plan

This document preserves the original offline migration plan. The
[current overall implementation plan](excavator-implementation-plan.md)
supersedes its remaining-work ordering: perception → execution → operations,
with further planning work deferred and stone_grab as the only gripper.

Status: reference/tabletop gate completed; shared extraction and excavator
MCTS/native planning CLI and offline Operations Console implemented. Recorded
direct/regrasp replay and active ablation sampling fixtures are covered.
Full behavior parity and fresh production-scene grasp success remain open; see
[implementation and validation details](excavator-planning.md) and the
[overall code review and full-project backlog](excavator-code-audit.md).
Scope agreed on 2026-09-07: **offline planning and replay first**.

## First milestone

Deliver an excavator application that loads commissioned objects and recorded
pickup poses, searches for placements, generates native direct or regrasp
motions, writes a resumable `.stack` run, and replays its robot, gripper, and
object motion without loading ROS or either legacy application.

Use only the commissioned stone_grab gripper for the current migration. The
legacy production selection is `assets/excavator_3` plus `assets/stone_grab_3`.
MT15 and other grippers are excluded from the current scope, including the
perception, execution and operations stages. A direct-only solver is an intermediate delivery,
not completion of this milestone.

Both legacy repositories remain read-only behavioral references. Preserve the
tabletop application throughout the migration. Do not create `apps/excavator`
until the applicable shared contracts have passed the tabletop validation gate
below, as required by [AGENTS.md](../AGENTS.md).

## Findings that determine the design

| Finding | Implementation consequence |
| --- | --- |
| Common domain, environment, actions, MCTS, sequence coordination, and `.stack` persistence already exist. | Reuse them; migrate demonstrated gaps rather than copying the old environment or integrated planner. |
| Legacy `IntegratedPlanner` eagerly imports Ray, Torch, ROS, and device wrappers. Legacy replay subclasses an execution worker. | Build thin offline entrypoints using common collaborators. Importing or replaying a run must work without execution dependencies. |
| Native grasp/motion schemas, result conversion, OBJ decoding, and NumPy height-map rendering still live under tabletop. | Extract the mechanisms required by the second platform with tabletop characterization tests. Keep platform policy and resolved values in the apps. |
| C++ has `ExcavatorIkInitializer`, but the Python `MotionRobot` binding initializes its hook to empty and does not expose it. | Complete a small, separately tested stacking-core binding change before claiming native excavator planning support. |
| Legacy production planning permits regrasp; tabletop currently disables it. | Validate direct and regrasp scenes, analytic IK, retries, grasp events, and replay explicitly. |
| Current normalized motion segments omit native target-pose samples. Tabletop replay selects scenes using a `pick_` name prefix and shows static target overlays. | Add explicit motion-phase/scene semantics and carried-object tracks before sharing full replay behavior. |
| Legacy physical values are spread across YAML, Python constants, and native defaults. Saved sessions can differ from current defaults. | Capture effective inputs and their sources; create independent SI profiles. Use saved values when importing a saved run. |
| `tests/test_architecture.py` explicitly forbids an excavator directory. | Record tabletop contract validation first, then replace that temporary assertion with actual cross-platform boundary tests. |

Source entrypoints and semantics are documented in the legacy
[planning guide](../../stacking-planner/docs/llm/planning.md). In particular,
`scripts/desktop/execute_offline.py` controls hardware despite its name and is
outside this milestone.

## Ownership and proposed layout

The dependency direction remains:

```text
stacking_tabletop   stacking_excavator
        \             /
         stacking_common
                |
         stacking_core bindings
```

After the validation gate, add:

```text
apps/excavator/
  pyproject.toml
  assets/                    required commissioned model and mesh closure
  config/
    inventory.yml            catalog, density, target
    environment.yml          observation, rewards, simulation validation
    action_generation.yml
    mcts.yml
    planning.yml             frames, models, collision roles, supports, handoffs
    visualization.yml        platform overlays and camera defaults
    solvers/{simulation,posegen,grasp,motion}.yml
  src/stacking_excavator/
    composition.py
    config/
    environment/             platform catalog and rendering policy
    planning/                native models, scenes, attempt policy, CLI
    visualization/           excavator model projection and overlays
    gui/operations/          offline application window
```

Do not scaffold empty ROS, execution, acquisition, or training packages. Keep
the first implementation specific to the demonstrated stone-grab platform;
do not introduce an attachment plugin framework or a global scale setting.

Common changes are limited to:

- Solver schemas and mapping from resolved values into native configuration.
  Platform code continues to load YAML and supply every physical value.
- Native result normalization, including failures, motion segments, timing
  when available, target trajectories, and grasp events.
- Reusable commissioned OBJ decoding and rasterization. Catalog discovery,
  selected IDs, density, target bounds, and height-map policy remain platform
  inputs. Tabletop pickup-table inference remains tabletop code.
- Artifact conversion and canonical replay projection. Rendering of canonical
  geometry can use an optional common backend; Qt windows, operator interaction,
  robot appearance, and environment overlays remain platform-owned.
- Small planning-attempt or restart helpers only if both apps demonstrate the
  same behavior. Do not relocate the whole tabletop coordinator or console.

## Phase 0 — establish reference inputs and the tabletop baseline

This phase changes tests and documentation before adding a platform package.

1. Make the documented project environment capable of running the existing
   tests; record the Python and native binding versions. Run the focused
   tabletop/shared baseline and report failures or skips explicitly.
2. Create `tests/fixtures/legacy_planner/` with compact, versioned records:
   source paths and hashes, effective configuration, model hashes, object
   ID/index mapping, pickup and placed scenes, selected/rejected actions,
   direct and regrasp motion samples, and expected phase/event outcomes.
3. Use local recorded sessions as extraction sources, not runtime or CI
   dependencies. Candidates already present are
   `sessions/260701/plan_260701_5` and `sessions/260730/plan_260730_1` in the
   legacy planner. The former's motion log contains successes, regrasp stages,
   and failed candidate retries; select cases from decoded motion records,
   since a log can retain a failed direct-stage diagnostic after fallback
   ultimately succeeds.
4. Export trusted legacy records in an isolated compatibility process to
   plain JSON/NPZ values. Avoid importing legacy runtime classes in common
   code or making tests depend on the full ignored session tree.
5. Build a source-to-contract inventory. Mark each behavior as already covered,
   missing, intentionally changed, or deferred. Inspect the effective policy
   and model checkpoint usage; an algorithm named `bc_mcts` alone does not
   establish a dependency on a learned model. If the chosen baseline actually
   uses one, preserve inference or explicitly document a different baseline.

The tabletop validation gate has two checkpoints: the existing baseline in
Phase 0 and the extraction/extended-contract checks in Phase 1. Both must pass
before Phase 2 creates the app. They cover these contracts:

| Contract | Required evidence |
| --- | --- |
| Domain and scenes | Stable sparse IDs versus indices, quaternion/frame roundtrips including rotation, separate pickup and placed objects, stable native entity mapping. |
| Search and simulation | Existing legacy policy comparisons, independent evaluator state, settling/short/long phase behavior, and rejection propagation. |
| Native planning | Tabletop profiles, model/contact construction, remaining pickup obstacles, failure normalization, and its existing known-motion smoke test. |
| Sequence and persistence | Failed attempts do not advance state; successful selected state/checkpoint commit consistently; interrupted attempts preserve the committed prefix; context mismatches reject resume. |
| Visualization | Phase 0 records existing scene/candidate/trajectory and robot replay results. Phase 1 keeps them passing and adds direct/regrasp roundtrip and replay tests for the new motion-track fields. |

The current known-motion native test supplies a grasp candidate and minimal
motion. It does not prove end-to-end grasp generation or regrasp. Those need
separate representative native acceptance cases in the later phases.

Record both checkpoint results in `docs/migration.md`. Only after both pass, replace
`test_no_excavator_package_is_scaffolded` with checks that common cannot import
either platform, neither platform imports the other, and offline paths cannot
import ROS, device controllers, or legacy `agent`/`diffsimpy` packages.

Physical tabletop commissioning and broader reconstruction parity remain
separate gates for future execution/perception reuse. Passing this offline
gate does not claim those workflows are commissioned.

## Phase 1 — complete shared mechanisms and native accessibility

Run these two tracks in parallel after the reference inventory is available.

**Shared Python track:** extract grasp/motion schemas from tabletop
`planning/profile.py`, native result conversion from `planning/adapter.py`,
and the demonstrated common portions of `environment/assets.py` and
`environment/heightmap.py`. Keep the tabletop public entrypoints working
through thin adapters while callers move. Re-run their existing behavioral
tests after each extraction.

Extend common motion values only where the fixtures require it: sample-aligned
target poses and explicit pickup/handoff/placement scene identity, alongside
the existing joint positions, scalar openings, attachment mode, and grasp
events. Preserve native timestamps when provided. Keep absent timing explicit.
Add versioned artifact decoding for old records; do not invent attachment
motion for old files that lack enough information.

**Native track:** expose the existing C++ excavator initializer through Python
with an explicit joint-chain/tool-frame description and a supported hook on
the native robot. Link the required native target and test lifetime ownership,
invalid model rejection, FK/IK agreement, and use by motion planning. This is a
stacking-core change governed by that repository's instructions; native solver
implementation stays there.

Reuse `stacking-core/test/planner/assets/stacking_planner_plan_state.txt` and
the existing excavator kinematics tests where applicable. The native URDF
loader already supports floating roots and `dsf_vert`; verify the real asset
semantics before proposing loader changes. Never silently substitute generic
IK seeds for the baseline analytic initializer.

**Exit:** the tabletop gate remains green through extraction, extended records
roundtrip, and Python can construct/use the validated excavator initializer.

## Phase 2 — add independent excavator profiles and model composition

Depends on the recorded tabletop gate and the relevant Phase 1 contracts.

Create the app/package, installation entry, and explicit profiles. Copy only
the selected object catalog and the URDF/mesh dependency closure required for
the milestone. Record provenance and preserve link origins, mesh scales,
joint limits, tool transforms, component friction/sharpness, contact flags,
teeth geometry, and opposing jaw motion. URDF mesh scales deliberately mix
`0.001` and `1`; relocating files must not rescale their geometry.

Build a profile mapping table from:

- `agent/configs/config.yml`, `diffsim.yml`, and `diffsim_fast.yml`;
- `agent/env/components/contexts.py` and `agent/env/simulator.py`;
- `planning/planning.py` and `scripts/desktop/generate_sequence.py`;
- saved configuration and planning parameters for each reference session.

Examples demonstrating why tabletop profiles cannot be reused:

| Input | Current legacy reference to resolve explicitly |
| --- | --- |
| Production target dimensions | 3.0 × 4.0 × 2.5 m, with separate target-origin and world/stack transforms. |
| Object mass/inertia | Unit-density OBJ metadata multiplied by the resolved density exactly once; legacy default density is 1000 kg/m³. |
| Robot/gripper | Six-joint stone-grab chain, analytic IK, signed opening interval of -80° to 0°, converted to radians. |
| Handoff | Base XY `[-2, -5]` m and the existing ordered candidate locations. |
| Placement lift | Current source default 0.20 m; the July 1 reference log uses 0.10 m. Imported runs must retain their saved value. |
| Floor policy | Planner production disables strict floor-first rejection; current tabletop enables it. |
| Solver values | Resolve implicit posegen tolerances, phase physics, support checks, grasp margins, and motion constraints from their actual assignments. |

Do not claim a one-to-one mapping between diffsim and stacking-core numerical
parameters without checking their meaning. Record unmapped inputs and native
behavior differences. Do not inherit tabletop damping or stability tuning as
excavator defaults.

**Exit:** loading the platform profiles and catalog needs no legacy imports;
model properties, FK, joint ordering, collision roles, and visual transforms
match the captured references. Sparse object IDs and full inventory size do
not collide with fixed/robot/gripper entity IDs.

## Phase 3 — reproduce placement search on recorded scenes

Compose common inventory, observation, action generation, MCTS, and simulation
with the excavator profiles. Reuse the existing `stacking_core` backend.

Add planner-specific characterization alongside the current tabletop MCTS
comparisons. Cover canonical/scan-relative orientations, directional taper,
support and floor decisions, root validation, preserved-tree revalidation,
candidate retry order, and rejection-memory behavior needed by the chosen
production baseline. Preserve object subset and already-placed prefix handling.

Separate exact checks of deterministic Python policy and serialization from
native numerical comparisons. For native runs, record seeds where supported,
thread counts, inputs, tolerances, rejection classifications, final poses, and
elapsed time. Set physical tolerances from the reference cases before judging
results; stochastic solvers need not emit identical joint samples.

**Exit:** bounded recorded cases produce valid proposals and expected policy
decisions; a search-only `.stack` artifact reloads with matching scenes and
diagnostics. Kinematic/pass-through runs are wiring checks, not native parity.

## Phase 4 — integrate direct/regrasp motion and sequence persistence

Build excavator robot/gripper roles and independent pickup, placement, and
handoff scenes around the common placement problem. Validate separate ground
heights and preserve the remaining pickup obstacles. Reuse native direct and
regrasp solvers rather than porting their algorithms into Python.

Keep three pose concepts explicit: the committed settled stack pose, the
lifted world-frame motion target, and the achieved/released motion pose. Apply
placement lift after fixed-scene re-settling. Future stack scenes must use the
unlifted committed pose, so the lift never accumulates. Tabletop's released-pose
support check must not be copied with assumptions that reject the excavator's
deliberate release clearance; characterize the excavator release policy.

Preserve handoff retry eligibility. In particular, a failure such as
`regrasp_b_grasp_generation`, which is independent of handoff XY, must not
trigger the entire location retry list. Retain original failure details even
when fallback later succeeds.

Connect to common `MCTSPlanningSequenceCoordinator` behavior: exhaust the returned
ranked candidate batch as required by the baseline, record every attempt,
apply exclusions at the correct point, and advance only successful placements.
Keep canonical diagnostics for zero-candidate, exhausted, infeasible,
cancelled, and native-failure outcomes.

Persist resolved profiles, exact scene geometry, joint ordering, attachment
identity, model/asset hashes, native/runtime identity, target tracks, events,
and source provenance. Use existing transactional `.stack` checkpoints and
process isolation for native interruption. Resume into a new destination from
the last committed checkpoint and reject changed planning contexts.

Non-destructive branching is an intentional departure from legacy in-place
truncation. An imported legacy run is eligible for planning resume only when
its context and state can be established; replay-only conversion must not
manufacture a resumable checkpoint.

**Exit:** representative native direct and regrasp cases include actual grasp
generation and nontrivial motion; a short multi-placement run verifies failure
retry, no lift accumulation, interruption, checkpoint integrity, and branching.

Proposed public commands follow the existing tabletop run conventions:

```bash
stacking-excavator-plan --config apps/excavator/config/mcts_cpu.yml \
  --pick-pose-data /path/to/pose_data.pkl --save-run /tmp/excavator.stack
stacking-excavator-plan --resume-run /tmp/excavator.stack \
  --save-run /tmp/excavator-resumed.stack --steps 2
stacking-excavator-operations --run /tmp/excavator.stack
```

These are planned entrypoints, not commands available in the current tree.
Run destinations must be unused paths.

## Phase 5 — deliver standalone offline replay

Build an excavator offline Operations Console surface with run selection,
search/candidate inspection, plan-attempt selection, segment and whole-step
playback, scrubbing, and visual/collision mesh display. Reuse canonical scene,
timeline, and projection behavior; keep platform meshes and overlays in the
excavator app. Do not reuse an execution worker to obtain a viewer.

Extract a common headless replay controller and canonical renderer helpers
where tabletop already supplies reusable behavior. If an optional PyVista
backend is shared, isolate it from backend-neutral contracts and lazy imports;
keep the Qt application/window outside common and update architecture tests
to enforce that separation.

Replay joints, signed gripper openings, and the target pose together through
pickup, carry, release, handoff, reacquire, placement, and retreat. At a
handoff release the object remains in the handoff scene while the free robot
moves. Avoid duplicate static/carried targets. Use explicit scene semantics,
not segment-name prefix inference. Preserve discontinuities rather than
inventing motion between disconnected paths.

Use saved timing when available. An untimed recording may use a documented
display rate; it must not be presented as measured execution timing. Preserve
camera state across playback, layer, step, and candidate changes, and update
actor transforms without rebuilding unchanged geometry every frame.

Provide a bounded, explicit importer for the characterized legacy session
format as part of this milestone. Keep schema conversion in artifacts
and legacy file discovery/model resolution in the platform adapter. Reject
unsupported or incomplete semantics clearly; do not run an old planner merely
to replay its output.

**Exit:** fresh native runs and characterized imported runs replay direct and
regrasp motion with correct frames, events, object tracks, and openings.
Replay starts in a process without ROS, the old repositories on `PYTHONPATH`,
or a planner solve. Missing assets/context produce a useful error rather than
silently using current tabletop or excavator defaults.

## Phase 6 — acceptance and handoff

The milestone is complete only when:

1. Shared and tabletop regressions pass, including tests affected by schema
   and renderer extraction; native acceptance tests actually run rather than
   being counted as passes when skipped.
2. The new application loads independently installed assets and profiles and
   performs native search, direct planning, and regrasp planning on the agreed
   recorded cases.
3. A bounded multi-placement run survives interruption, branches from its last
   committed checkpoint, and retains failed attempts and provenance.
4. A fresh process can open that run and replay robot, gripper, carried object,
   and handoff behavior without ROS, legacy imports, or replanning.
5. Documentation gives installation, planning, resume, replay, and validation
   commands; records numerical differences and measured runtime; and identifies
   all unsupported legacy workflows. No source implementation is removed merely
   because an initial smoke run passes.

Suggested review units follow Phases 0–6. Within Phase 1, the native binding
and common Python extraction can proceed independently. After motion records
are established, the offline viewer can be developed from fixtures alongside
search/sequence integration. Avoid combining solver behavior changes, profile
retuning, schema migration, and GUI work into one unreviewable change.

## Later milestones

The [excavator online architecture](excavator-online-architecture.md) now gives
the concrete ROS/NUC ownership, interfaces, and sequential perception →
execution → operations implementation plan. Further planning work is deferred.

| Milestone | Prerequisite and scope |
| --- | --- |
| Recorded perception and reconstruction | Extend tabletop parity beyond hull/DSF to ICP, pose graphs, meshes, and CoACD; implement an independent excavator object-build adapter with recorded LiDAR/joint inputs. |
| Acquisition and scene identification | Excavator ROS messages, LiDAR frames, scan calibration, onboard NUC coordination, and SceneID belong to this platform; common algorithms continue to consume arrays/transforms. |
| Field execution and recovery | Validate platform commands/transports, cancellation, holds, reconciliation, and in-hand replanning against shared execution contracts, followed by separate physical commissioning. |
| Research | Training, datasets, learned-policy integration, and evaluation tooling as separately scoped work. Other gripper models are excluded from the current migration. |

## Validation status for this planning session

Read the applicable AGENTS files, migration/architecture notes, legacy planning
documentation and selected source paths, native binding surfaces, and existing
regression tests. Confirmed the reference session files exist; they have not
yet been decoded into fixtures or validated against native results.

Attempted:

```bash
direnv exec . python -m pytest -q \
  tests/test_architecture.py tests/test_mcts_legacy_regression.py \
  tests/test_planning_sequence.py tests/test_planning_contracts.py \
  tests/test_execution_artifacts.py tests/test_visualization_contracts.py
```

The configured Python 3.12.3 environment reported `No module named pytest`.
No tests ran, and the tabletop gate remains pending. Installing test
dependencies and running the baseline is the first implementation action.

Primary implementation references:

- [Current migration state](migration.md) and [architecture](architecture.md).
- [Tabletop native planning](../apps/tabletop/src/stacking_tabletop/planning/native.py),
  [result adapter](../apps/tabletop/src/stacking_tabletop/planning/adapter.py),
  [replay controller](../apps/tabletop/src/stacking_tabletop/gui/operations/controller.py).
- [Common motion contracts](../packages/stacking-common/src/stacking_common/planning/contracts.py)
  and [sequence coordinator](../packages/stacking-common/src/stacking_common/workflows/planning_sequence.py).
- [Legacy planner setup](../../stacking-planner/planning/planning.py)
  and [sequence generation](../../stacking-planner/scripts/desktop/generate_sequence.py).
- [Native Python motion bindings](../../stacking-core/python/bindings_planner_motion.cpp)
  and [excavator initializer](../../stacking-core/include/stacking_core/planner/excavator/closed_form_initializer.hpp).
