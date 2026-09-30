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

Both platform `config/environment.yml` files now expose `reward.place_stability`.
This restores the noisy-placement trials from `stacking-planner` and
`stacking-tabletop`; it is distinct from the ordinary structural motion limit.
After initial settling, each trial perturbs that settled pose, creates a fresh
simulation with previously placed stones fixed at their pre-action poses, and
moves only the new stone. The metric is the maximum across trials of
translation distance plus `rotation_weight` times shortest rotation angle in
radians, measured from the noisy start to the final pose. Trials never replace
the nominal scene or action pose. Seeds depend on configured seed, placement
step, inventory index and object ID, as in the legacy implementations.

| Field under `reward.place_stability` | Excavator | Tabletop |
| --- | --- | --- |
| `enabled` | `true` | `true` |
| `n_noise` | 4 | 8 |
| `position_std` (XYZ, metres) | `[0.02, 0.02, 0.05]` | `[0.005, 0.005, 0.005]` |
| `rotation_std_deg` | 5 | 5 |
| `rotation_weight` (metres/radian) | 0.1 | 0.01 |
| `reject_failed_trials` | `true` | `false` |
| `fail_displacement` | `null` | `null` |
| `seed` | 0 | 0 |

Noise values follow each legacy project's `agent/configs/config.yml`. The excavator
legacy ablation profile disabled robustness and used smaller XY noise;
set `enabled: false` to skip the trials. Missing sections in older profiles
also leave trials disabled. More trials increase candidate evaluation cost.
Trials use the selected short/long structural phase's stopping thresholds,
twice its time step, and the corresponding structural physics in fresh
simulators. Existing placements remain at rest throughout each trial.

`reward.weights.place_stability` multiplies the negative worst displacement.
The existing `reward.max_displacement` caps only the reward value;
`reward.nonfinite_displacement` supplies its nonfinite fallback.
`reject_failed_trials: true` rejects a candidate if **any** noisy trial exceeds
its structural `velocity_integral_threshold`, produces nonfinite simulation
values, or stops without settling. Settling requires energy below
`energy_threshold` for the phase's extra steps after `min_time`; reaching
`max_time` without that rest period fails. The legacy near-motion-limit stopping
shortcut is retained for scoring but does not count as genuine rest. The option
is enabled for excavator and defaults to false when omitted; tabletop exposes
it as false so its existing acceptance behavior is preserved. Both short search
and long validation apply it. Failures use rejection reason `place_robustness`,
so another reward term cannot compensate for a failed trial.

The independent `fail_displacement` option rejects a result whose worst trial
displacement exceeds its numeric limit (including nonfinite results).
`reject_failed_trials: false` together with `fail_displacement: null` leaves
robustness as a soft reward signal. Unlike the old pre-rejection clipping, the hard check uses
the raw displacement so the reward cap cannot hide failure.

Action diagnostics retain per-trial and raw displacement, enabled/failed flags,
and `place_robustness_trial_failure_reasons` (one entry per trial: `null` for a
pass, otherwise `target_motion_limit`, `not_settled`, or `nonfinite_simulation`),
plus `simulation_robustness_wall_seconds`. The measured value propagates through
`PlacementResult`, action history, and saved action artifacts into the reward.
Ordinary structural acceptance still uses `velocity_integral_threshold` in
`config/solvers/simulation.yml`; it is not a substitute for these noisy trials.
