# Shared planning and execution handoff

The common package supplies execution commands, status events, result records,
planning checkpoints, context snapshots, and `.stack` storage. The platform
implements `MotionExecutor` and owns robot/gripper control, interpolation,
controller limits, sensor acquisition, operator gates, and physical recovery.

Tabletop supplies UR5e and Dynamixel adapters, a ROS transport, and an
operator-gated saved-sequence `MotionExecutor` integrated with the console and
journal; see [the hardware and execution guide](tabletop-hardware.md).

## Record the inputs used for planning

Build a `PlanningRunContext` from the resolved inputs used by the solver. Its
`frame_id`, ordered `joint_names`, profiles, fixed collision geometry, and asset
hashes are independent of current YAML files. A box stores full XYZ lengths in
metres; a plane uses its pose's local +Z normal; a mesh references hashed bytes.
Include robot/gripper and object assets, source poses, and calibration assets as
appropriate. `verify_context_assets` checks supplied files or embedded bytes
against every recorded asset role. The platform decides which inputs constitute
its complete context; common code cannot inspect native solver internals.

Before persisting a newly generated plan:

```python
from dataclasses import replace

context_hash = writer.append_run_context(context)
record = replace(record, context_sha256=context_hash)
writer.append_placement_plan(record)
```

Associate the context actually used to generate that plan. Existing plans with
no recorded context remain readable by the viewer but cannot become execution
commands through this API. Do not retrofit today's profile onto an old plan.
The tabletop debug CLI now captures this context for new saved motion-planning
runs and binds every attempt to it. Older runs remain view-only unless they
already contain the required context and checkpoint; provenance is not invented.

Optional `MotionPlanSegment.mode` and `PlacementPlanningResult.grasp_events`
retain attachment state and explicit acquire/release sample locations, poses and
model opening angles. Commands preserve these fields. The v1 codecs omit them
when absent, retaining historical plan digests; individual platforms may require
them for execution. Signed model opening angles are validated by the platform.

## Execute a selected plan

`StackRunStore.load_committed_placement_plans(path)` resolves each selected step
to exactly one successful motion plan. Search-only runs cannot supply such a
sequence. Command creation copies the path, validates joint columns and supplied
timing, and binds the command to hashes of its plan and context.

```python
from stacking_common.artifacts import (
    StackRunStore, build_execution_command, verify_context_assets,
)

plans = StackRunStore.load_committed_placement_plans(path)
context = StackRunStore.load_run_context(path, plans[0].context_sha256)
verify_context_assets(context, resolved_assets)
command = build_execution_command(plans[0], context)

with StackRunStore.open_append(path) as writer:
    writer.append_execution_command(command)
    result = executor.execute(command, progress=writer.append_execution_event)
```

The platform calls this only after its normal execution checks. It must use
the recorded frame/joint ordering and construct or verify its controller
configuration from the context. Segment times may be absent in a planning
path; platform time parameterization remains necessary before sending it to a
controller. The common protocol does not prescribe a platform's segment names.

The first event is `running`, or a terminal `failed`/`cancelled` rejection before
motion. Subsequent events report running, held, or terminal status. Events have
contiguous sequence numbers per command; timestamps use Unix nanoseconds.
Sequence numbers, not wall-clock timestamps, define ordering. A terminal event
is the persisted result. `load_execution_results` projects those final events
into `ExecutionResult` values.

`hold`, `resume`, and `cancel` request actions from the live executor. An API
return does not itself confirm that the robot stopped or resumed; subsequent
events report the observed state. Once terminal, a command accepts no further
events. A new attempt requires a new command ID and incremented attempt number.

Create/use a writer on the thread consuming its callbacks. SQLite connections
are not shared with arbitrary callback threads. Each journal append commits
independently; transaction locking prevents competing writers from queuing the
same step concurrently. Reopening a run preserves its existing records and its
planning completion status. Execution outcomes have their own records.

## Recovery and planning restart

`load_execution_checkpoint` reports completed steps, pending steps, and
unconfirmed command IDs. A queued command with no telemetry, a running command
left by a crash, a held command, and a failed/cancelled command all require
physical reconciliation. The last sample index never implies permission to
restart a trajectory. Planning success also does not imply robot execution.

After stopping/checking the hardware and restoring or replanning the scene,
the platform records a failed/cancelled terminal event if none exists, then
appends `ExecutionReconciliation` with its reason and evidence. Only then may
it queue a new attempt. Common code records that confirmation; the tabletop
state machine performs the actual checks.

Planning restart uses a separate `PlanningCheckpoint`. It retains the full
domain state (including histories, simulation trajectories, contacts, and
pose-identified IDs), full pickup scene/provenance, stack-to-world transform,
and context hash. Supply `checkpoint=` to `append_mcts_step` to commit both in
one transaction. A failed checkpoint validation rolls back the step as well.

`MCTSPlanningSequenceCoordinator` accepts an optional
`checkpoint_builder(step, state)` callback that returns this value after a
successful placement plan. The platform supplies its original pickup snapshot,
frame transform, and context. When restarting:

1. Load `StackRunStore.load_planning_checkpoint(path)` and its context.
2. Verify assets and rebuild the environment/observation from the saved state.
3. Create a fresh evaluated MCTS root/session.
4. Run the coordinator with `first_step=checkpoint.step + 1` and an append writer.

Native workers, trees, and RNG internals are not serialized. The new search can
make different choices. A run without a checkpoint for its latest committed
step fails explicitly rather than silently restarting from older state. A
partially written planning attempt also requires platform handling.

The tabletop CLI's `--resume-run SOURCE --save-run NEW` performs that handling
by branching the committed prefix into a new document. It verifies the current
resolved profiles, runtime settings, native binary and asset hashes against the
saved context, restores pickup poses and the planned state, and starts a fresh
tree at the next step. Failed/uncommitted attempts stay in the original file;
their IDs do not collide with new attempts. Runs containing execution commands
are refused: offline planning restart is not physical reconciliation. Existing
output files are never overwritten. `--steps` is the number of additional
placements; omitted search settings are inherited from the source request.

A source still marked `running` is refused to avoid racing its writer. Normal
Stop/Ctrl+C finalizes it automatically. After an external crash, first confirm
the offline worker is gone, then explicitly finalize the journal with
`StackRunStore.open_append(path)` / `writer.finish("cancelled")` before branching.

The Operations Console offers **File → Resume planning into new run…**, with
a nonblocking log dialog and Stop button. Profile/asset changes cause an explicit
refusal; restore the recorded inputs or start a new planning run. A native rebuild
also changes the context. Restarts do not serialize the old search tree or RNG.

The CLI supervises its own isolated process group. Ctrl+C or `--max-run-time`
stops the group, escalates to SIGKILL after a two-second grace period if needed,
reaps the worker, and marks the run cancelled. Atomic SQLite commits/checkpoints
survive; an in-flight attempt is not a committed placement. The whole-run limit
includes initialization, search, planning, and cleanup. Per-search deadlines
remain cooperative between native batches. This is an offline computation stop,
**not** an emergency stop or a guarantee that any physical robot has stopped.

## Derived reports and perception

`summarize_run(path)` returns step/search statistics, compact planning attempts,
execution results, and recovery status. `export_run_summary(path, destination)`
creates `.json` or `.csv`. CSV uses one row per step and preserves variable
statistics/attempts in JSON columns. Exports preserve the original `.stack` and
refuse to overwrite existing output files. Use the CLI's `--export-summary` or
the Operations Console's **File → Export run summary…** for JSON/CSV output.
The CLI exports after completion or interruption, if a run document was created.

`pointcloud.IncrementalRegistration` now owns accumulated-reference registration,
quality gates, accepted-view merging, rejection, and undo. Inputs are common
`PointCloudData` already transformed into one nominal frame. The tabletop
`IncrementalInHandAligner` delegates to it while retaining capture metadata,
operator inspection, and progress behavior. FEMTO depth decoding, predefined
UR5e scan trajectories, and acquisition synchronization stay in the tabletop app.

## Verification

Use the same interpreter for native dependencies and tests:

```bash
python -m pip install -e '../stacking-core/python'
python -m pip install -e 'packages/stacking-common[perception,test]'
python -m pip install -e 'apps/tabletop[operations]'
QT_QPA_PLATFORM=offscreen make test
```

The focused tests exercise a mock executor through the real journal, crash and
retry handling, context/path mismatch, legacy record reading, atomic planning
checkpoints, summary exports, and registration rejection/undo. Hardware
commissioning and recorded multi-stone legacy parity remain separate integration
validation; these tests do not claim physical execution equivalence.
