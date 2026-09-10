# Tabletop plan1 motion-planning diagnosis

Investigated `.stacks/tabletop_plan1.stack` on 2026-09-10. No hardware was
commanded and the saved run was not modified.

## Finding

The run contains an invalid synthetic pickup scene: objects 1–20 all have the
same position `[0, -0.7, 0.2]` metres and 90-degree yaw. Its recorded scene
source is `synthetic_debug_pose`, not a measured pickup-pose file.

When no pickup file is supplied, the
[tabletop launcher](../apps/tabletop/scripts/debug/mcts.py) assigns the same
pose to every requested object (lines 270–293). The
[native adapter](../apps/tabletop/src/stacking_tabletop/planning/native.py)
includes the target and every other pickup object in collision checking;
non-target objects are static (lines 133–164). Each grasp is therefore evaluated
against 19 objects overlapping its target.

All 23 recorded attempts failed at step 1:

| Failure | Attempts |
| --- | ---: |
| `direct_grasp_generation` | 14 |
| `simulation_missing_both_side_contact` | 7 |
| `simulation_unsettled` | 2 |

No attempt produced a refined grasp or a motion candidate. The recorded run
failed before trajectory optimization could produce a path.

## Controlled native replay

The installed native binary matched the recorded binary hash, and current
planning and native solver profiles matched the recorded context. Replayed
saved placement problems using the same models, target poses, placement poses,
robot, gripper, fixed obstacles, and solver settings. In the diagnostic variants,
only the other 19 pickup objects were removed.

| Saved attempt | Pickup scene | Replay outcome | Wall time |
| --- | --- | --- | ---: |
| 4, object 18 | All 20 recorded objects | Missing two-sided contact; no motion candidates | 42.86 s |
| 4, object 18 | Target only | Success; 10 motion candidates | 8.91 s |
| 1, object 9 | Target only | Success; 20 motion candidates | 31.48 s |

Attempt 1 originally failed during direct grasp generation. These replays
demonstrate that the overlapping pickup objects prevent otherwise feasible
plans for at least these two saved placements. They do not establish that all
23 placements would succeed in a realistic measured scene. Removing real
obstacles is not a production remedy.

## Comparison with legacy tabletop

The relevant legacy workflow is `stacking-tabletop/scripts/generate_sequence.py`:
it loads `pose_data.pkl`, infers the pickup support height from the posed object
point clouds (lines 3211–3226), and adds each object at its own loaded pose
(lines 3572–3585). This avoids the migrated launcher's uniform synthetic fallback
when a valid pickup file is used.

The inspected UR5e and gripper URDF differences are mesh path prefixes. Object
9 and 18 meshes are byte-identical to the legacy meshes.

Separate grasp-planner parity differences remain:

- Legacy `planning/planning.py` configures fallback retreats of 5, 10, and
  20 mm. The migrated grasp profile does not configure those fallbacks.
- Legacy `diffsim/src/planner/planning/direct.cpp` derives the nominal grasp
  alignment from the robot home configuration. Current
  `stacking-core/src/planner/grasp/sampling.cpp` derives its default frame from
  the pick/place poses when no preferred parallel axis is configured.

These are code-comparison findings, not demonstrated causes of this run's
failures. The legacy solver was not executed for this investigation.

## Correct next step

Create a fresh planning run with a trusted measured `pose_data.pkl` selected
in the planning GUI's pickup-pose field, or passed with `--pick-pose-data`.
The **Pose from scan override** changes placement orientation proposals; it
does not acquire a scan or load pickup poses. Resuming this failed run would
reuse its recorded overlapping scene.

For synthetic motion smoke tests, use a single object, as shown in
[tabletop commands](tabletop-commands.md). The launcher and GUI should reject
multi-object synthetic motion runs early with an actionable pickup-pose error.
After validating measured-scene inputs, compare the remaining grasp differences
using identical saved problems before changing solver parameters.
