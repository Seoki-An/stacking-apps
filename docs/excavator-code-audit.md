# Excavator migration: overall code review

Reviewed on 2026-09-09 against the local `stacking-planner` source and the
shared/tabletop implementation in `stacking-apps`.

Follow-up on 2026-09-09: **all five defects below are fixed**. The findings and
original test result are retained as review history. Committed scenes now
rebuild observations/rewards; optional PyVista adapters have a separate common
backend boundary; target taper survives storage and replay; excavator scan
orientations match legacy fixtures; and resume explicitly requires unchanged
search inputs while allowing a new step count and run-time limit.

The combined follow-up validation passed **222 tests**, including tabletop,
excavator, architecture, action generation, committed sequences, DSF rendering,
tapered target vertices, and resume controls. The next areas are specified in
the [perception, execution, and operations design](excavator-online-architecture.md).
The [current implementation plan](excavator-implementation-plan.md) supersedes
the historical recommended ordering below and includes the desktop/NUC,
purpose-specific crop, WiFi and stone_grab-only constraints.

The excavator application implements placement search, native motion-planning
integration, saved runs, and an operations console for planning and replay.
It does not yet replace the complete legacy project. This review includes the
implemented paths and an inventory of the remaining legacy capabilities;
it is not an exhaustive verification of every source line or physical hardware.
Runtime code was not changed during the original review; the follow-up fixes
are described above.

## Defects in implemented paths

### 1. Committing a motion-approved pose leaves the search observation stale

Priority: high. Applies to the shared motion-planning sequence, including its
excavator caller; the search-only loop does not use this commit path.

[`_commit_planned_pose`](../packages/stacking-common/src/stacking_common/workflows/planning_sequence.py)
updates the selected node's state, action, and action-history pose, restoring
previously placed objects to their parent-state poses. It does not rebuild the
node's observation. The subsequent `session.promote(selected,
preserve_subtree=True)` retains that observation. Search passes it into action
generation, and the excavator scorer consumes its `scene_height_map`.

Reproduction using actual excavator assets and the array observation builder:
commit object 55 from x=0 to x=0.2 m. The promoted state reports x=0.2 while
`observation.object_poses` still reports x=0.0; rebuilding the observation
changes 403 height-map cells. This can make subsequent proposals inconsistent
with the committed scene whenever fixed-scene settling changes a pose.

Preserved children are scheduled for revalidation, so this is not a claim that
subtrees are never revalidated. That mechanism does not refresh the root's
observation. Rebuild the committed observation before promotion and assess
pose-dependent reward/diagnostics for the saved selected candidate as well.
Add a two-step test in which planning deliberately adjusts the accepted pose.

### 2. Common visualization currently fails its dependency-boundary test

Priority: medium; a reproducible regression gate failure.

[`visualization/dsf.py`](../packages/stacking-common/src/stacking_common/visualization/dsf.py)
imports PyVista in `read_dsf_surface`. The architecture test forbids GUI backend
imports throughout the common visualization-contract tree, including lazy
imports. `test_common_visualization_contracts_do_not_depend_on_gui_backends`
fails with this file as its only reported violation.

The import is lazy: this does not establish a headless import crash. Keep the
numeric DSF support calculation reusable, and put the optional mesh-rendering
adapter behind an explicit backend boundary with corresponding tests. Preserve
both applications' rendering behavior instead of globally relaxing the gate.

### 3. The displayed target omits the configured taper

Priority: medium; the operations view misrepresents the planning target.

[`TargetVolume`](../packages/stacking-common/src/stacking_common/visualization/contracts.py)
contains dimensions, origin, and offset, but no taper. The
[excavator CLI](../apps/excavator/src/stacking_excavator/planning/cli.py) creates
that contract from dimensions alone, and the
[scene viewer](../apps/excavator/src/stacking_excavator/gui/operations/scene_view.py)
builds a rectangular shell in `_add_target` / `_open_target_shell`.

The configured 2 m wide, 2.5 m tall target with 5-degree taper on both width
sides should be approximately 1.563 m wide at the top; the shell stays 2 m wide.
The full target profile remains recorded in the run context, and planning uses
the tapered target. The missing geometry is in the visualization contract/path.
Carry taper through serialization and projection, retaining zero taper for old
artifacts and tabletop profiles, and verify displayed vertices.

### 4. Scan-based orientation initialization differs from legacy excavator

Priority: medium; affects `pose_from_scan: true`, not the default false branch.

[`ExcavatorApplication.build_orientation_provider`](../apps/excavator/src/stacking_excavator/composition.py)
chooses the shared `ScanAlignedOrientationProvider`, which first corrects the
measured longest-axis heading. Legacy
[`_orientation_quaternion_pairs`](../../stacking-planner/agent/env/components/action/planar.py)
uses configured yaw multiplied directly by measured scan rotation in this mode.

For object 55 with measured XYZ Euler angles [20, -10, 37] degrees, each of the
four generated rotations differs from its nearest legacy rotation by
27.304441 degrees. The default principal-axis branch has separate parity
coverage; that coverage does not validate scan mode. Align the excavator scan
branch, including its support/sampling rotation semantics, without changing the
tabletop provider. Add a non-axis-aligned measured-pose fixture.

### 5. Resume override documentation conflicts with context validation

Priority: medium; unchanged-context resume works, changed search controls fail.

The CLI accepts seed/iteration/search-time overrides on resume, and the
[guide](excavator-planning.md) says recorded overrides are reused unless supplied
again. Those values participate in the context hash, while
[`validate_planning_resume`](../packages/stacking-common/src/stacking_common/artifacts/planning_context.py)
requires exact context equality. A seed-only runtime change on a valid test
checkpoint reproduces `resume context mismatch: profiles=['runtime'], assets=[]`.
Actual CLI changes can also change the MCTS profile.

Define the supported behavior explicitly: either reject changed values early
and document immutable resume inputs, or support a new context for subsequent
steps while retaining the original context on the committed prefix. Do not
simply remove the provenance check; branching currently binds retained plans
to their original context too.

## Full-project migration coverage

| Area | Current implementation | Remaining work |
| --- | --- | --- |
| Shared contracts and ownership | Domain/state, observations, scenes, profiles, planning results, artifacts, and search composition are reused by both apps. Other architecture checks passed. | Resolve the rendering boundary failure and state/observation commit inconsistency. Keep ROS and platform policy out of common code. |
| Excavator MCTS and actions | Scored floor sampling, support kernels, CEM, feasibility/overlap gates, principal-axis orientations, rejection feedback, and native validation exist. | Fix scan mode; compare longer multi-seed runs, including the depth change after ten placements, and inactive legacy modes if required. |
| Rewards and simulation | Common reward evaluation and current native simulation are composed with excavator profiles. | Compare accepted/rejected scenes, reward components, and ranking against legacy. Modern numerical damping semantics differ deliberately; copying YAML cannot establish numerical parity. |
| Grasp and motion planning | VDK23/stone-grab models, analytic IK, contact mapping, direct/regrasp adapter, handoff retries, phase scenes, and target tracks exist. | Demonstrate fresh grasp generation and successful full direct/regrasp runs on representative scenes; review legacy scene-aware approach and grasp rejection policies missing from current adapters. |
| Collision modeling | Gripper, carried stone, other stones, support planes, and configured obstacles are modeled. | The commissioned robot IK model has no robot-body collision geometry. Full robot/environment and applicable self-collision coverage require explicit commissioning. This preserves a legacy boundary rather than proving execution clearance. |
| Run storage and restart | Attempts, committed checkpoints, resolved contexts, asset hashes, branch validation, and partial-run inspection exist. | Resolve override semantics; cover interruption and continuation across longer real planning sequences. Search-only continuation is not implemented. |
| Operations console | Launch/cancel planning subprocesses, inspect MCTS, replay direct/regrasp tracks, and select DSF/PLY display modes. | Fix target projection; add live telemetry, execution controls, measured/planned reconciliation, and operator recovery as part of execution migration. Process cancellation is not a hardware stop. |
| Field acquisition and NUC coordination | Recorded pick scenes can be imported. Shared array point-cloud workflows and tabletop acquisition examples exist. | Port excavator field/gripper/in-hand scan choreography, transforms, sensor adapters, synchronization, and durable acquisition sessions. No excavator acquisition runtime is present. |
| Perception and scene reconciliation | Shared registration/reconstruction building blocks are available. | Integrate excavator PoseID/SceneID, in-hand object pose updates, scene merge, field recovery, and confidence/failure handling; validate recorded datasets before live use. |
| Robot execution and ROS | Common execution/artifact contracts and tabletop execution provide references. | Implement excavator joint/phase/grasp-status interfaces, motion dispatch, feedback, manual intervention, recovery, and restart reconciliation. These remain absent from `apps/excavator`. |
| Simulation of online workflows | Native placement/motion simulation supports offline planning. | Migrate the legacy excavator, field-scan, and in-hand-scan simulator nodes to exercise the execution/perception loop without hardware. |
| Learning and datasets | Default excavator search uses analytic scored sampling. | Legacy sampling/data-generation scripts, learned value/height-map models, training, dataset merge/checks, and optional inference integration have not been migrated. They are a separate capability from the active ablation path. |
| Ablation and evaluation tooling | Individual parity fixtures and search/planning tests exist. | Port the legacy batch ablation runner, comparison/state summaries, reproducible run matrices, and aggregate metrics. |
| Alternative grippers | The commissioned VDK23 stone_grab path is implemented. | Excluded by the user's scope clarification: no MT15/other-gripper work or generalization is required for the current migration. |
| Packaging and deployment | Both excavator console scripts are declared; GUI dependencies are optional. | Verify clean installation and native-binding compatibility on target hosts; package/deploy NUC and ROS services when those applications exist. Repository-based tests do not validate deployment. |

Legacy capability inventory was checked against `planning/execution*.py`,
`agent/integrated_planner.py`, `ros2/`, `scripts/nuc/`, `scripts/desktop/`,
`scripts/simulation/`, `scripts/evaluation/`, and learning/sampling modules.
In particular, legacy `scripts/desktop/execute_offline.py` controls hardware;
it is not equivalent to the current offline planning CLI.

## Original recommended implementation order (superseded)

Follow the [current implementation plan](excavator-implementation-plan.md)
instead. The list below records the recommendation made during the initial
audit, before the user deferred planning work.

1. Fix the five concrete defects above, with tabletop regression coverage for
   each common-code change.
2. Establish longer planning and fresh grasp/direct/regrasp acceptance cases;
   add batch evaluation so parity and failures are measurable.
3. Integrate recorded excavator perception and scene reconciliation using
   common numeric workflows and platform-owned acquisition contracts.
4. Implement the excavator execution state machine and operator recovery
   against simulated devices, with persisted command/feedback state.
5. Integrate NUC/sensor/ROS devices and commission collision models, calibrated
   transforms, joint control, grasp feedback, and interruption/recovery on the
   physical platform.
6. Migrate learning/data pipelines according to demonstrated product needs.
   They need not block the active analytic planner. All current excavator work
   is limited to stone_grab; other grippers are excluded.

## Review validation

The focused review run completed with **141 passed, 1 failed in 67.15 seconds**:

```bash
QT_QPA_PLATFORM=offscreen PYTHONPATH=/tmp/stacking-apps-test-deps python -m pytest -q \
  tests/test_architecture.py tests/test_excavator_cli.py \
  tests/test_excavator_mcts.py tests/test_excavator_planning.py \
  tests/test_excavator_operations.py tests/test_excavator_action_parity.py \
  tests/test_planning_sequence.py tests/test_planning_contracts.py \
  tests/test_planning_scene.py tests/test_mcts_session.py \
  tests/test_execution_artifacts.py tests/test_visualization_contracts.py \
  tests/test_tabletop_planning.py tests/test_tabletop_planning_restart.py \
  tests/test_tabletop_mcts.py
```

The original single failure was finding 2. Additional small reproductions established
the observation mismatch, scan-rotation discrepancy, and strict resume hash
behavior. These are not yet committed regression tests. No hardware, fresh
production grasp acceptance run, full-suite run, or clean-install deployment
test was performed in this review. Earlier validation results in other guides
are historical. The follow-up regression run described at the top supersedes
this original failure.
