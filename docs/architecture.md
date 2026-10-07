# Architecture

## Dependency direction

```text
stacking_tabletop   stacking_excavator
        \             /
         stacking_common
      |
      v
stacking_core Python bindings
      |
      v
stacking-core C++ targets
```

Dependencies never point from common code to a platform application. Native
`stacking-core` also remains independent of Python application concerns.

## Common package

`stacking_common` owns reusable application-level behavior:

- `domain`: stack objects, actions, state snapshots, and object trajectories.
- `environment`: episode orchestration and observation/reward contracts.
- `search`: MCTS and CEM.
- `profiles`: shared schemas for independently resolved platform YAML values.
- `assets`: commissioned convex-component OBJ decoding and model construction.
- `planning`: conversion to typed stacking-core scenes, problems, and results.
- `execution`: platform-neutral commands, progress/results, and driver contracts.
- `pointcloud`: immutable point-cloud values, registration and quality gates,
  support-surface segmentation, incremental reference registration, and cross-capture fusion.
- `reconstruction`: cleanup, pose graphs, meshing, CoACD decomposition, DSF
  fitting, inertial estimation, and a headless reconstruction pipeline.
- `workflows`: resumable use-case orchestration independent of a GUI or CLI.
- `visualization`: scene, plan, trajectory, grasp, and debug rendering.
- `learning`: observation conversion; model/training migration is a later research task.
- `artifacts`: versioned plan, session, and debug serialization.

Common modules operate on arrays, transforms, canonical entities, and typed
results. They do not know ROS topic names, message types, camera drivers,
controller services, or operator controls.

`artifacts` is the persistence boundary. It converts external records into
validated domain values and converts domain values into independently owned
records for storage or transport. Schema versions, legacy field aliases, and
migration logic belong there; runtime domain classes stay independent of file
formats and historical schemas.

Execution events and planning checkpoints use distinct records. A committed
plan advances simulated planning state; only a confirmed terminal execution
event marks a physical step complete. Commands bind to a recorded plan and
resolved planning context by content hash. Common persistence provides recovery
assessment; platform code verifies the physical scene and controls retries.
See [common-execution.md](common-execution.md) for the tabletop integration API.

`planning.SceneObjectInstance` resolves an object's pose and mobility for one
specific solve. `planning.CoreSceneBuilder` is the single conversion point from
those application roles to native `BodyInstance` and `SceneSnapshot` values.
The stable mapping reserves entity `0` for fixed platform geometry and maps an
inventory `object_index` to entity `object_index + 1`.

`domain.ConvexGeometry` stores validated NumPy mesh and half-space data needed
by application features; it does not replace stacking-core collision geometry.
Each immutable `StackObject` keeps the corresponding native `BodyModel`, while
`ObjectInventory` owns ordering and initial poses and `State` owns per-scene
poses. `TargetGeometry` owns only the target shape: explicit dimensions,
directional taper, and support overlap. `environment.VoxelOverlapEvaluator`
owns the platform-tuned approximation resolution and precomputed evaluation
grid.

`environment.InventoryObservationAdapter` precomputes padded mesh, surface,
and physical-feature arrays around a domain inventory. It receives a platform
height-map renderer, which owns the fixed bounds and target map. This keeps
learning layouts and rendering dependencies out of `ObjectInventory`.

The runtime `environment.Observation` is only a typed value. Full record
encoding and decoding live in `artifacts`; compact height-map model inputs live
in `learning`. Neither storage schemas nor model-specific dtypes are methods on
the runtime observation.

Filesystem asset discovery, mesh conversion, Open3D rendering, and height-map
ray casting are adapters around these types rather than responsibilities of the
domain models.

`environment.CoreSimulationProfile` is the typed boundary for resolved native
simulation values. It validates the platform mapping, constructs a fresh
stacking-core `SimulationConfig`, and supplies `CorePlacementBackend` through
its profile factory. Common code therefore does not read platform YAML or
select physical values.

`search.mcts` separates tree mechanics from application policy. `MCTSNode`
owns only tree topology, evaluated transition values, search statistics, and
explicit rejection state. `MCTSConfig`, the tree policy, and discounted backup
implement root limits, progressive widening, PUCT selection, and subtree reuse
without importing OmegaConf, Torch, Ray, Gym, or a simulator. Action
generation, prior scoring, transition evaluation, authoritative final
validation, and diagnostics remain separate collaborators. Training-sample
conversion is intentionally outside the runtime coordinator.

Those collaborators exchange `ActionGenerationResult` and
`TransitionEvaluation` values through the `ActionGenerator`, `PriorScorer`,
`TransitionEvaluator`, and `ValueEstimator` protocols. Generation reports its
attempts, exhaustion, and rejection counts independently from accepted
candidates. Transition results retain the requested `none`, `short`, or `long`
evaluation fidelity and own a branch-independent state snapshot.

`search.mcts.MonteCarloTreeSearch` coordinates selection, retry-bounded action
generation, sibling-batched evaluation, depth-dependent fidelity, wall-clock
budgets, value estimation, discounted backup, final long validation, and
preserved-subtree revalidation. It neither constructs nor reaches inside an
environment. `StackingEnvironmentEvaluator` restores a parent state and steps
the configured environment for each fidelity. `ParallelTransitionEvaluator`
uses independent evaluator instances while all tree mutation remains on the
coordinator thread. `MaskedActionGeneratorAdapter` remains only as a bounded
compatibility seam.

The tabletop policy admits root actions through authoritative long validation
before descendant search and seeds each surviving root result once. Optional
root lookahead spends the first search iterations giving every admitted,
non-terminal root one descendant attempt before ordinary PUCT selection. Final
selection can rank root means, maximum known descendant returns, or a weighted
mixture; promoted descendant paths may be long-validated to a bounded depth.
If final validation rejects every root, failed slots are retired and one or
more bounded CEM retries use a changed mutation seed while excluding earlier
root poses. The first configured number of placements use the legacy depth-1
schedule, after which the full tree depth becomes active without changing the
canonical local stacking frame.

Migration regressions include captured legacy policy outcomes and, when the
sibling legacy checkout is present, execute its actual MCTS/node sources in an
isolated subprocess. This keeps depth scheduling, PUCT selection, and bootstrap
backup behavior checked against the implementation being replaced rather than
only against a rewritten approximation.

The root CEM generator draws fresh proposals after the initial population,
recomputes height-map scores at mutated poses, and replaces an exhausted fresh
quota with parent mutations. MCTS reserves a configured portion of its total
wall budget for final long validation and stops scheduling validation batches
at the absolute deadline.

Long `CorePlacementBackend` evaluation settles once, runs the configured short
gate, and continues the same native snapshot, trajectories, velocity
integrals, elapsed time, and quiet-step count through the long extension. A
failed short gate skips the extension. Final native contacts and cumulative
motion provide explicit target-motion, scene-motion, ground/scene penetration,
and missing-support rejection reasons; unsettled and outside-target outcomes
are completed by `StackingEnvironmentEvaluator`.

`pointcloud.PointCloudData` is the immutable NumPy boundary between perception
adapters and common workflows. Processing backends consume this value and
return validated typed results. Open3D, SciPy, CoACD, and PyVista are optional,
lazy implementation dependencies: importing the common or tabletop packages
does not load them. ROS and camera SDKs never enter the common package.

`artifacts.ObjectBuildSession` is the single persisted source of truth for an
object build. Each prepare, merge, or reconstruct attempt records parents,
status, platform profile values, metrics, operator decisions, coordinate-frame
explicit artifacts, and failures. Rerunning an upstream stage marks accepted
downstream attempts stale rather than silently reusing incompatible output.
`workflows.ObjectBuildRunner` owns these transitions and delegates only the
platform edge to `ObjectBuildPlatformAdapter`.

## Tabletop platform

`stacking_tabletop` owns:

- UR5e and SR-gripper construction and collision roles.
- Femto/D435 acquisition, synchronization, and calibration ingestion.
- ROS 2 bring-up and controller/gripper integration.
- Tabletop execution state machines and safety/operator gates.
- Live execution UI and tabletop-specific visual overlays.
- Fully resolved tabletop solver profiles in SI units.

Inherited excavator online execution, NUC coordination, LiDAR SceneID, and
excavator ROS wrappers are intentionally outside this platform.

`TabletopApplication.build_environment` is the tabletop composition root:

```text
ObjectInventory ---> TabletopHeightMapRenderer ---> observation builder
        |                                                |
        +-----------> voxel overlap evaluator ---> reward evaluator

PlacementBackend ---> SimulationTransition

observation builder + reward evaluator + transition ---> StackingEnvironment
```

Both platform compositions use `ScoredPlanarPoseSampler` and
`ScoredPlacementActionGenerator` from `stacking-common`, with their own
physical profiles and orientation providers. The shared sampler ranks grid
anchors and draws fresh populations using score weights and spatial diversity.
The former tabletop legacy-builder adapter and deterministic height-map
sampler were removed after tabletop adopted this strategy. Composition uses
scheduled stacking-core pose generation,
equilibrium/containment priors, root-only CEM, horizon bootstrap values, and
independent short/long simulation environments. Neither common search nor the
platform composition imports the legacy `agent` package.

The placement backend remains injectable for tests and alternative execution
paths. `TabletopApplication.build_core_backend` is the production factory that
combines inventory body models with the resolved tabletop simulation profile;
solver state is not hidden in the application object. The renderer
implementation lives under `stacking_tabletop.environment`; ROS and sensor
code are not involved.

`TabletopObjectAssetLoader` is the commissioned-asset adapter. It reads the
component metadata from each OBJ, retains convex-component boundaries and
per-component friction and sharpness, applies the configured density to mass
and inertia, and constructs both domain geometry and the native multi-geometry
body model. The configured location is `apps/tabletop/assets/objects`; the
application has no runtime dependency on a legacy repository.
The inventory profile explicitly enumerates the app-owned object IDs. Every
catalog entry has an OBJ collision model, a PCD perception model, and a PLY
visualization mesh under the same `model_<id>` stem.

`TabletopPcdCatalog` is the Open3D file-format adapter for the PCD member of
each catalog entry. It loads model points into the common immutable array type.
`Open3dRegistrationBackend` implements point-to-point multiscale ICP behind the
common registration contract, and `TabletopObjectRegistrar` composes catalog
lookup with registration. Open3D is imported lazily, so configuration,
planning, and simulation do not acquire a perception runtime dependency.

The online tabletop perception composition is:

```text
ROS PointCloud2 -> RosPointCloudAdapter -> CapturedPointCloud
                                             |
                                             v
                                     TabletopCaptureBuffer

recorded/captured XYZ -> depth correction -> field segmentation
                                               |
                                               v
PCD catalog -> orientation-bank ICP -> quality gates -> cluster assignment

two corrected base-frame clouds -> bounded GICP -> dual-Femto fused PCD
```

Plane fitting peels orientation-incompatible dominant planes before a
total-least-squares refit. Field clustering, the 24-orientation initializer,
one-to-one assignment, and field/in-hand acceptance thresholds are configured
independently. ROS owns only message conversion and subscriptions; numerical
processing remains callable with recorded NumPy point data.

The offline object-build composition is:

```text
raw Femto captures
        |
        v
tabletop prepare adapter -> shared pose graph + consensus merge
        |
        v
tabletop alignment UI -> shared cross-capture GICP/fusion
        |
        v
tabletop review UI -> shared cleanup/mesh/CoACD/DSF pipeline
        |
        v
PCD + mesh + DSF + density/mass metadata
```

The command-line workflow and optional Qt/PyVista Object Reconstruction
Workbench use the same `workflow.json`; the GUI does not define a second
reconstruction pipeline. It is separate from the Stacking Operations Console,
which owns plan viewing, Search Inspector diagnostics, replay, and future live
execution interaction.
The tabletop layer owns capture discovery, Open3D/PyVista operator interaction,
visualization, and artifact locations. Common owns the numerical algorithms.
Tabletop reconstruction resolves an object-specific density override first and
the inventory default second. A measured `--mass-kg` remains an explicit
override. Unit-density volume, center of mass, and inertia come from the
reconstructed mesh only when every edge has opposite two-face incidence, the
surface has no self-intersections, and the integrated properties are valid. A
conservative repair pass removes duplicate/degenerate topology, removes
non-manifold edges, and reorients triangles; it does not invent hole patches.
Meshes that remain invalid use the convex-hull fallback. Exported metadata
records `mass_source`, `volume_method`, and the mesh validation/repair result.

For a future planner/excavator-scale integration, implement
`ObjectBuildPlatformAdapter` with an independent, complete SI-unit profile.
Do not import tabletop profiles and multiply them by a global scale. Its sensor
acquisition, ROS topics, execution, and operator UI stay in that platform app.
No excavator application is scaffolded until this contract has characterization
coverage against the legacy workflow.

## Configuration

The common package owns configuration contracts. A platform owns every
resolved physical value. Profiles are separated by solver so changes remain
reviewable:

```text
apps/tabletop/config/solvers/
  simulation.yml
  posegen.yml
  grasp.yml
  motion.yml
apps/tabletop/config/
  environment.yml
  inventory.yml
  perception.yml
  visualization.yml
```

No `scale` setting generates these files. Length, angular, time, force, and
dimensionless objective parameters must be migrated and reviewed explicitly.
Environment observation sizes, reward thresholds, and reward weights are also
resolved by the platform rather than supplied as common defaults.
`stacking_common.environment.EnvironmentProfile` defines their typed contract,
while the tabletop package owns YAML loading. Height-map margin, surface sample
count, and voxel-overlap resolution are independent explicit values.
Field-scan, in-hand, and dual-camera registration have separate metric voxel
pyramids, correspondence distances, iteration counts, and quality limits in
`perception.yml`; none is generated by scaling another profile.
Platform reference geometry for the Operations Console is separately resolved
in `visualization.yml`. Ground, support tables, and planning obstacles remain
tabletop-owned overlays and do not enter common visualization code as Qt or
PyVista objects.

## Height-map renderer

The dependency-free NumPy renderer has deterministic golden coverage and is
checked against the former Open3D ray-casting implementation. Their outputs
agree within micrometre-scale numerical tolerance for the commissioned assets,
but Open3D is currently faster, especially for multiple objects. Open3D is
therefore kept as an optional benchmark dependency while backend selection
remains an explicit tabletop decision. See
[heightmap-benchmark.md](heightmap-benchmark.md) for the measurements.
