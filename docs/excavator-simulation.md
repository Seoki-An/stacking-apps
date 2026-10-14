# Excavator desktop execution and operations simulation

The desktop provides saved-PCD identification, durable proposal review,
a local VDK23/stone_grab joint-motion simulator, and the Operations Console. It
requires no NUC or ROS installation. Planning algorithm work remains deferred.

The [closed-loop emulation](excavator-closed-loop-emulation.md) now connects
execution to scans of a changing virtual world, desktop identification and
reviewed continuation. Use that workflow for pickup/carry/placement acceptance.
The smaller joint demos below remain useful for isolated controller tests.
Physical hardware commissioning remains deferred.

## Run without hardware

From the repository root, use separate new session directories:

```bash
python -m stacking_excavator.execution.cli \
  --repository . --session /tmp/excavator-direct-session --demo direct

python -m stacking_excavator.execution.cli \
  --repository . --session /tmp/excavator-regrasp-session --demo regrasp

python -m stacking_excavator.gui.operations.application . \
  --session /tmp/excavator-console-session
```

The two demos exercise four/eight segments and acquire/release ordering with
small joint excursions. They are workflow examples, not computed placement
solutions. The CLI uses accelerated virtual time; the console uses wall time.
After refreshing the editable installation, the corresponding executables are
`stacking-excavator-simulate` and `stacking-excavator-operations`.

A saved motion plan can also supply a single simulation attempt:

```bash
python -m stacking_excavator.execution.cli \
  --repository . --session /tmp/excavator-saved-session \
  --run /tmp/excavator-motion.stack --step 1
```

Only a committed selected successful placement with a recorded excavator
context and matching geometry is eligible. Search-only runs and uncommitted
attempts cannot supply motion. A placement must contain explicit acquire and
release events, canonical joints, valid joint/opening limits, and an initial
joint state matching the simulator. Old artifacts missing these values remain
viewable but are not executed by guessing missing events. This command does
not execute a whole multi-placement sequence.

## Console workflow

The existing Plan and Search Inspector tabs remain available. For input examples,
overlay colors, quality metrics and review decisions, see
[Perception in the Operations Console](excavator-perception.md#perception-in-the-operations-console).
Three tabs add the session workflow:

1. **Perception:** select **Open session folder…**, choose a verified crop
   product and its initial-poses JSON, then select **Identify in background**.
   The worker runs in a separate process and can be stopped. You can also open
   a saved `proposal.json`. Inspect the processed cloud, initial/proposed
   surfaces, reference frame, crop boundary, opening evidence and alignment
   diagnostics. Layer checkboxes and the shared DSF/PLY selector control the
   display. Processed clouds are shown only when their hash matches the proposal.
2. **Review:** accept/reject the selected pending proposal. Invalid proposals
   cannot be accepted. Stale revisions and repeated decisions are rejected;
   unobserved objects are never inferred to have disappeared. Opening estimates
   and recorded measurements remain distinct. Accepted in-hand geometry is
   stored separately from world poses.
3. **Execution:** load a committed placement or a direct/regrasp demo, then
   **Run simulation**. Inspect measured/planned joint values, opening, progress
   and `needs_replan`. Hold, resume and cancel address the current attempt.
   Disable operator heartbeat or inject stalled-device, missing-feedback or
   unknown-gripper faults to exercise failure handling.
4. **History:** inspect durable execution events and perception decisions.
   Reopening a session restores them. An interrupted attempt fails on restart;
   it never resumes from a saved sample. Explicitly confirm that the simulated
   gripper is empty, reconcile, and load a fresh command before retrying.

If the session already contains accepted observations, the input JSON and crop
must carry its current scene revision. A derived crop preserves the raw
capture's revision, so recropping or changing only the priors JSON cannot
rebase an old recording. Use capture evidence associated with the current
revision, or a separate replay session for inspecting an older recording.

## Execution behavior

`execution/runtime.py` owns the session independently of Qt. A process lock
allows one owner per session. Commands are persisted before dispatch; common
`ExecutionCommand` and `ExecutionEvent` codecs retain plan/context binding and
ordered per-command events. `review.db` stores accepted observations and
one-use decisions; `execution.db` stores commands, events and recovery state.
These files are separate from the source planning `.stack` document.

The device checks command sequence numbers and its lease before accepting a
setpoint. Acknowledgement does not advance a sample: two fresh feedback samples
must meet joint/opening tolerances and the segment's minimum dwell. Simulation
motion is bounded by URDF joint velocities and a 0.2 rad/s jaw speed. The owner
uses a one-second operator lease, a half-second feedback age limit, and bounded
sample completion deadlines. The device watchdog stops before integrating a
motion interval that spans an expired lease. These are simulation parameters,
not commissioned hydraulic controller settings.

Hold stops motion and resume reissues the current incomplete sample. Cancel,
feedback/lease loss, tracking failure and unknown gripper state stop the device
and require reconciliation. Accepting perception during an active attempt
stops it before updating the scene and sets `needs_replan`. An independently
accepted revision is detected on the next owner tick. Perception changes do
not silently patch an existing trajectory.

## Validation and limits

Portable tests exercise recorded direct/regrasp joint paths, all grasp-event
ordering, acknowledgement versus completion, holds, cancellation, injected
faults, duplicate/out-of-order commands, single ownership, crash recovery and
perception invalidation. Qt tests exercise background identification, review,
simulation, reopening, and actual PCD/ROI/material rendering. Recorded jaw
feedback is not available in the motion fixture; jaw events in that test are
explicit simulation fixtures.

This milestone provides a desktop simulator, not a deployed distributed device
service. The simulated device does not model hydraulic dynamics, contact,
load/slip, or physical grasp success. Its numeric gripper coordinate is not a
mapping to the legacy open/close/stay actuator interface. The console displays
accepted world poses as reference geometry; it does not claim to measure a
moving stone from commanded joints. Concrete ROS capture/control adapters,
independent NUC watchdog commissioning, moving scan choreography, real WiFi
measurements, and physical perception acceptance remain follow-up work under
the [online architecture](excavator-online-architecture.md).

Validation on 2026-09-09: **170 focused tests passed**, split into 121
perception/execution/acquisition/architecture checks and 49 GUI/rendering checks,
including tabletop regressions. Direct and regrasp CLI demos also completed,
and the full imported field capture reproduced the saved legacy pose. The
real VTK views were inspected using a local Xvfb display with software OpenGL;
the sandbox alone could not initialize a reliable graphics context.

For the GUI checks on a headless machine with Xvfb installed:

```bash
xvfb-run -a env QT_QPA_PLATFORM=xcb LIBGL_ALWAYS_SOFTWARE=1 \
  python -m pytest tests/test_excavator_operations.py \
  tests/test_excavator_session_console.py \
  tests/test_tabletop_operations_console.py tests/test_tabletop_execution_console.py \
  tests/test_dsf_rendering.py tests/test_plan_playback_rendering.py -q
```


## Placement robustness reward

Both platforms configure `reward.place_stability` in `config/environment.yml`.
After a placement passes nominal physical checks **and settles**, each trial
starts from the final structural snapshot. All poses and static/dynamic roles
are preserved. Residual velocities are reset; only the newly placed stone gets
an initial Gaussian linear velocity in world axes and angular velocity in body
axes. Supporting stones that were dynamic in structural validation remain dynamic.
Pose-identified and structurally frozen stones remain fixed. Trials do not alter
the committed scene, action, or nominal trajectories.

| Field under `reward.place_stability` | Excavator | Tabletop |
| --- | --- | --- |
| `enabled` | `true` | `true` |
| `n_noise` | 4 | 8 |
| `linear_velocity_std` (world XYZ, m/s) | `[0.05, 0.05, 0.05]` | `[0.02, 0.02, 0.02]` |
| `angular_velocity_std_deg` (each body axis, deg/s) | 5 | 5 |
| `rotation_weight` (metres/radian) | 0.1 | 0.01 |
| `reject_failed_trials` | `true` | `true` |
| `fail_displacement` (metres of combined excursion) | 0.15 | 0.015 |
| `seed` | 0 | 0 |

Velocity magnitudes are provisional platform settings, not conversions from
previous pose-noise magnitudes. Calibrate them against measured release motion.
All axes use signed, independent, zero-mean Gaussian draws; downward velocity
is allowed without changing the initial geometry. Seeds depend on placement
step, inventory index, object ID, and configured seed.

The score is the largest **peak** excursion across trials and dynamic bodies:
translation from that body's nominal settled position plus `rotation_weight`
times its shortest rotation angle in radians. Every integration step is checked;
a stone cannot hide a large excursion by returning to its initial pose.
`reward.weights.place_stability` multiplies the negative score. Reward clipping
via `max_displacement` does not affect rejection checks.

Trials use structural physics, twice the structural timestep, and its time and
quiet-window bounds. Energy must remain below threshold for the quiet window
for **all dynamic bodies**. The cumulative speed-integral cutoff is disabled for
intentional disturbances. CUDA uses the existing multi-body phase stopping rule;
its older single-body `CudaRobustnessStopping` is not used. Per-step histories
are downloaded in batches sized to an estimated 512 MiB sample-storage budget,
then discarded. This is not a total simulator memory limit.

If energy settling is not reached, each body's terminal-window motion can still
qualify as `bounded_rocking` when configured tolerances are satisfied (excavator:
2 mm and 0.5 degrees; tabletop: disabled). Unresolved motion is `not_settled` and
follows `reject_failed_trials`. Nonfinite/invalid simulations, missing history,
and excursion above `fail_displacement` always reject. Excursion failures use
`excursion_limit`; other invalid CUDA statuses use `simulation_cuda_<status>`.
The candidate rejection is `place_robustness`; nominal physical failures retain
precedence and skip disturbance trials.

Diagnostics include `place_robustness_model: velocity_disturbance`,
`place_robustness_evaluated`, per-trial excursions/outcomes/failure reasons, raw
score, and wall time. An enabled but unevaluated trial set means nominal
validation did not provide an eligible settled scene. Ordinary structural
validation still uses its own motion-integral limits.

Pose perturbation and clearance/lift correction have been removed. Old
`position_std` / `rotation_std_deg` settings fail explicitly instead of silently
changing their units. Saved runs retain their historical diagnostics; use the
current configuration for a fresh run. Velocity disturbances test dynamic
stability under release motion, not placement-pose uncertainty or robot/gripper
insertion feasibility; the latter remains a motion-planning responsibility.

Replay saved candidates with `tests/benchmark_place_velocity_disturbance.py
--stack PATH --steps 1 3 4 --repeats 3 --output /tmp/velocity.json`. The comparison
with nominal validation alone measures candidate overhead, not full-MCTS speed.

### Saved first-floor replay — 2026-10-14

The [recorded replay](benchmarks/excavator-velocity-disturbance-20261014.json)
used `excavator_plan5.stack`, current long-validation physics, and three timed
repeats after warmup. Median GPU batch times:

| Step | Candidates | Nominal only | With velocity trials | Accepted |
| --- | ---: | ---: | ---: | ---: |
| 1 | 32 | 0.163 s | 0.421 s | 32 |
| 3 | 57 | 0.317 s | 0.929 s | 57 |
| 4 | 67 | 0.382 s | 1.199 s | 67 |

All 664 trials settled, including those for 52 candidates originally rejected
for initial clearance. Maximum combined excursion was 0.0156 m. Added trial
cost was approximately 8–12 ms per candidate in these batches. These are not
whole-MCTS timings or a controlled speed comparison with the former pose-noise
implementation. All candidates passing at these provisional disturbance scales
does not establish real release robustness; calibration remains necessary.
