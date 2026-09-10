# Excavator motion comparison with legacy planning

Inspected on 2026-09-10 using `test4.stack`, the current excavator app,
`stacking-core`, `stacking-planner`, and `diffsim`. This is a diagnosis, not a
claim that motion migration has reached behavioral parity.

## Recorded failure

The corrected grasp radii (`radius_init = radius_max = 0.1`) are present in
`test4.stack`. Recorded pickup poses are used; `pose_from_scan` is false.

| Attempt | Object | Recorded result |
| --- | --- | --- |
| 1:1 | 12 | Success; 58 grasp candidates, 17 motion candidates; 34.32 s. |
| 2:1 | 2 | `regrasp_pick_approach_motion`: waypoint IK failed; 207 grasp candidates; 85.21 s. |
| 2:2 | 2 | Handoff motion failures at `[-2,-5]`, `[-1,-5]`, `[-3,-5]`; last failure is a target-boundary mismatch. |

Attempt 2:2 records three handoff solves taking 68.63, 77.46 and 71.32 seconds.
Its top-level `elapsed_seconds` reports only the final solve, not their sum
(217.40 seconds). The file is marked cancelled at the 600-second overall
deadline. It contains one committed successful placement. This differs from
`test2.stack`/`test3.stack`, where all 22 attempts produced zero grasps.

## Differences identified before recovery implementation

The descriptions below record the original diagnosis. The implementation
status is summarized at the end of this document.

1. **Approach direction.** Legacy `Context::graspSubgoalOffsetDir` normally
   calls `sceneApproachDir`, combining plane normals with separation directions
   from nearby bodies. The stacking-planner profile sets the minimum upward
   component to `sin(pi/6)`; `gripper_y_approach_enabled` defaults to false.
   The migrated direct/regrasp code offsets along gripper-local `[0,-1,0]`.
   This changes approach/retreat targets and is directly relevant to the
   recorded approach-IK failures. Both implementations shorten standoff with
   factors `1, .75, .5, .25`; changing the distance alone does not align them.
   Sources: `diffsim/src/planner/planner.cpp:334`,
   `stacking-planner/planning/planning.py:516`,
   `stacking-core/src/planner/regrasp.cpp:291`.

2. **IK recovery and branch choice.** Legacy dual-scene IK can refine the
   grasp together with robot joints when initial IK fails. Approach subgoals
   use a home/goal midpoint seed for bounded joints, retaining the goal branch
   for broad-range joints and normalizing the result. The current grasp IK
   path solves a fixed grasp pose and rejects failed IK; motion waypoint IK
   uses the current state and optional excavator initializer. These are not
   equivalent recovery paths.
   Sources: `diffsim/src/planner/planning/regrasp.cpp:543`,
   `diffsim/src/planner/planning/regrasp_common.hpp:331`,
   `stacking-core/src/planner/manipulation.cpp:233`,
   `stacking-core/src/planner/motion.cpp:315`.

3. **Additional grasp seeds.** Legacy direct planning fills an undersized
   candidate pool with strategy-specific fallback seeds. For stone_grab those
   include offsets at `.65`, `.8`, and `1.2` times gripper height. The migrated
   antipodal sampler uses the single configured retreat distance, currently
   `.75 m`. Its independent pick/place fallback does not reproduce that seed
   schedule. This can change which grasps reach motion planning.
   Sources: `diffsim/src/planner/planning/direct.cpp:637`,
   `diffsim/src/planner/planning/grasp_initialization.cpp:311`,
   `stacking-core/src/planner/grasp/sampling.cpp:299`,
   `stacking-core/src/planner/pick_place.cpp:41`.

4. **Handoff failure and retries.** The migrated planner checks the achieved
   handoff target against the designed pose with a 2 cm / 10 degree gate.
   When motion reports success but this check fails, `stage_failure` inherits
   `retryable=false` from the successful motion result. The app then stops
   handoff retries. This occurred at `[-3,-5]` in attempt 2:2. Legacy also
   checks carry continuity, but that check compares adjoining trajectory
   boundaries; it is not this designed-pose gate. Legacy XY retry decisions
   use failure-stage text rather than the native retryable bit.
   Sources: `stacking-core/src/planner/regrasp.cpp:130,238,365`,
   `apps/excavator/src/stacking_excavator/planning/native.py:16`,
   `diffsim/src/planner/planning/regrasp_common.hpp:202`,
   `stacking-planner/planning/planning.py:117`.

5. **Trust-region semantics.** Legacy actually runs with `delta_init=1.0`
   and `delta_max=.1`: it accepts that pair and applies the cap on expansion.
   Core rejects an initial radius larger than the cap. The previous `.1/.1`
   correction makes core's input valid but does not reproduce legacy's initial
   optimizer step. Equivalent parameter names did not imply equivalent
   validation or numerical behavior.
   Sources: `stacking-planner/planning/planning.py:465`,
   `diffsim/src/planner/graspgen/graspgen.cpp:800,867`,
   `stacking-core/src/planner/grasp/evaluation.cpp:600`.

## Geometry and experiment controls

The robot and stone_grab URDFs have identical semantic XML after normalizing
mesh filenames; all referenced geometry files have matching SHA-256 hashes.
The comparison uses the recorded stone OBJ assets, world-frame poses,
surrounding pick/place stones, home joints, zero-height ground, handoff
`[-2,-5]`, and 15/10 move/grasp samples.

The hardware-free legacy replay script is
`/tmp/excavator-motion-comparison/legacy_attempt.py`, with stdout in
`/tmp/excavator-motion-comparison/legacy.log`. It reconstructs attempt 2:1
from `test4.stack` and invokes legacy `get_planner` / `regrasp_planning`.
It uses legacy solver defaults, including the original trust-region behavior.
Random grasp sampling and parallel scheduling prevent treating this as a
deterministic one-variable experiment.

The replay reached the explicit 150-second experiment deadline (exit 124).
Legacy direct planning found no feasible motion; fallback reported 66 feasible
place-side grasps and subsequently found pick-to-handoff grasps. It did not
return a complete trajectory before the deadline. Thus this experiment does
not establish that legacy succeeds on this placement, or identify a single
difference as the sole cause. The behavioral differences above are confirmed
from source; their individual effect still requires controlled regression runs.

## Recovery implementation

The excavator profile now selects core joint-grasp recovery, scene-based
approaches, and the additional stone_grab seed offsets. Fixed-endpoint joint
refinement seeds the two-endpoint solve, which checks a consistent
target-relative grasp in both scenes. Subgoal seeds retain the legacy
home/goal midpoint and broad-joint branch rules. Direct recovery fills eight
reachable candidates in bounded batches. Tabletop leaves these policies off.

Handoff target-boundary mismatches now permit another handoff location and
report numerical endpoint errors. Fixed pickup/placement failures still stop
the XY loop. Timing diagnostics sum all handoff attempts.

Core regression tests cover a fixed IK failure repaired through grasp
optimization, the two-endpoint recovery path, fallback sampling after a failed
nominal offset, and scene approaches around mirrored obstacles and a tilted
plane. App tests verify profile isolation, retry classification, and timing.

This does not establish full numerical parity. Core still enforces strict
subgoal IK feasibility and the designed-pose handoff gate; legacy can accept
approximate subgoal IK and checks adjoining motion boundaries. The trust-region
semantics described above are also unchanged. Recovery replays of attempt 2:1,
both before and after adding bounded candidate recovery, hit the 300-second
test deadline without returning a plan. The latter log is
`/tmp/excavator-motion-comparison/recovered-final.log`. No successful trajectory
has been established for that recorded placement.

Validation: all 34 core test targets and 74 application tests passed, including
tabletop planning regressions and the installed excavator native adapter.

Do not treat enabling `pose_from_scan` or loosening collision tolerances as a
substitute for these implementation comparisons. Scan orientation can change
the requested placement, but the recorded motion problems already contain
pickup poses and now produce many grasp candidates.
