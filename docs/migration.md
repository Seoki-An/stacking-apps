# Tabletop-first migration

The existing `stacking-tabletop` repository remains the behavioral reference
while code is moved incrementally.

## Implemented sequence

1. Stacking-core Python bindings and typed scene/problem conversion live under
   `stacking_common.planning`.
2. Domain, environment, inventory, target, and artifact values are common.
3. Registration, segmentation, cross-capture fusion, pose graphs, cleanup,
   meshing, CoACD, DSF fitting, and reconstruction are split into common
   numerical services and tabletop interaction adapters.
4. A single versioned object-build manifest connects prepare, merge, and
   reconstruct, including measured-mass metadata and stale-output handling.
5. The existing stage commands remain compatibility entry points. The
   canonical `stacking-tabletop-object-build` command and optional
   `stacking-tabletop-reconstruction-workbench` use the unified workflow.
6. Mesh volume integration rejects self-intersections, attempts conservative
   topology repair, and records the validation result before falling back to
   convex-hull inertia.
7. A versioned fixture characterizes the shared legacy hull routine and the
   commissioned DSF asset using a migrated recorded tabletop model. The hull
   source was byte-identical in both legacy projects.
8. Prepare crop/removal, cross-capture alignment, optional reconstruction
   cleanup/partitioning, and final review now run in the workbench viewport.
9. Backend-neutral visualization contracts and native `.stack` runs feed the
   tabletop Stacking Operations Console and its Search Inspector.
10. Reusable sequence orchestration feeds failed motion plans back into MCTS,
    preserves every attempt, and advances the tree only after planning succeeds.
11. The Plan viewport replays native UR5e/SR-gripper joint trajectories and
    sample-aligned gripper openings from the run document.
12. Trusted legacy `pose_data.pkl` pickup scenes cross a typed snapshot boundary;
    motion planning keeps remaining pickup objects separate from placed objects
    and stores pose provenance in the native run.
13. Common execution commands, status/events, results, and a `MotionExecutor`
    protocol define the hardware adapter boundary. The `.stack` journal records
    attempts and reconciliation without conflating planning and execution.
14. Content-addressed planning contexts retain resolved profiles, exact fixed
    geometry, joint ordering, and asset hashes. New execution commands require a
    context-bound plan; the tabletop adapter must supply those resolved inputs.
15. Planning checkpoints can commit atomically with selected steps and restart
    a fresh search tree. The coordinator supports continued step numbering;
    interrupted physical execution requires a separate reconciliation.
16. Common JSON/CSV summary exports derive reports from `.stack` records.
17. Incremental in-hand registration and accepted-reference undo now live in
    common; tabletop retains capture metadata and operator interaction.
18. Tabletop UR5e and Dynamixel device adapters preserve bounded physical-tool
    reference sampling, gripper mapping/contact/hold logic, and the existing ROS
    topics. Fake-transport tests cover device faults, cancellation, stale data,
    and ROS resource ownership.
19. Native grasp events and attachment modes survive saved-plan serialization.
    A direct-sequence executor validates recorded context/assets, enforces
    operator gates and acknowledged holds, and journals physical attempts.
20. Operations Console execution controls provide Hold/Resume/Abort, bounded
    manual corrections, measured robot/estimated object rendering, and explicit
    physical reconciliation before whole-step retry.

21. FEMTO acquisition validates both predefined serials and calibration frames,
    preserves the color-to-depth origin correction, captures synchronized depth
    bursts, and writes field/in-hand artifacts with calibration snapshots.
    A read-only ROS transport enforces exact-time transforms and source-time
    robot-state matching; field fusion and reconstruction consume the output.
22. Automatic wrist/pivot scans run in the saved-execution worker at the held
    placement gate. The console provides motion preview/approval, view progress,
    Hold/Resume/Abort and a fresh placement gate after exact return. Scan output
    retains calibration and model provenance for reconstruction.

See [the shared execution handoff](common-execution.md) for APIs, integration
requirements, and recovery semantics. The debug CLI captures contexts and
checkpoints for saved motion-planning runs; the Operations Console provides
offline planning resume, summary export, and saved-sequence execution controls.
The integrated hardware workflow still requires physical commissioning.

## Next migration work

For excavator, follow the [current overall implementation plan](excavator-implementation-plan.md):
perception → execution → operations for stone_grab, then further planning work.
It includes desktop processing, NUC cropping and resilient field WiFi transfer.

The [stacking-planner implementation plan](stacking-planner-migration-plan.md)
starts with offline planning and replay. The tabletop gate has passed and
`apps/excavator` now composes common MCTS with native VDK23/stone-grab planning.
Its CLI saves attempts, resolved contexts, target tracks, and checkpoints.
See [the implementation guide](excavator-planning.md) for commands, validation,
and remaining numerical-parity work. The excavator Operations Console now
supports planning launch/resume, search inspection, and recorded direct/regrasp
replay with its own platform renderer and shared headless projection.

The offline contract gate passed on 2026-09-08: 155 focused tabletop/shared
tests passed before and after extracting profile parsers, native result
normalization, object loading, rasterization, and search composition. Four
additional recorded-planner/native tests cover direct/regrasp target-track
roundtrips, optional sample ownership, and commissioned excavator analytic IK.
The first baseline exposed stale assertions against operator-tuned YAML;
the YAML was preserved and those assertions were corrected. Tests run with
`QT_QPA_PLATFORM=offscreen`; pytest is installed in `/tmp/stacking-apps-test-deps`.
This validates offline shared contracts, not physical commissioning.

1. Extend recorded-data parity beyond hull/DSF commissioning to ICP edges,
   pose-graph transforms, mesh statistics, and CoACD partitions.
2. Commission the integrated UR5e/Dynamixel sequence on the physical tabletop,
   including calibrated FK, contact phases, holds, and operator recovery.
3. Commission the integrated [FEMTO wrist/pivot scans](tabletop-femto.md), then
   add console live in-hand registration checks. See the
   [hardware and execution guide](tabletop-hardware.md).
4. Implement excavator perception, execution and operations using the current
   plan above. Defer recorded-scene grasp success coverage, longer search runs
   and reward/simulation comparisons to the later planning phase.
5. Retire the old standalone reconstruction-session compatibility module after
   downstream callers have moved to `workflow.json`.

## Explicitly excluded

The tabletop migration does not carry forward:

- Excavator online execution and field recovery.
- Excavator joint/phase/grasp-status ROS wrappers.
- NUC-side scan choreography.
- LiDAR SceneID runtime.
- Deprecated `poseinit` usage.

These components do not belong in the tabletop application. Excavator execution,
ROS wrappers, NUC coordination and LiDAR SceneID are deferred to a later,
separate platform migration after tabletop validates the shared contracts.
