# Augmented excavator stone catalog

`apps/excavator/assets/objects_augmented` contains 100 simulation stones:
49 real commissioned stones, copied without changing their original IDs or
OBJ/PLY bytes, and 51 synthetic stones with IDs 59–109. Historical gaps in the
real catalog's numbering are preserved. The original `objects` directory and
default `inventory.yml` remain the real 49-stone catalog.

The opt-in profile is `apps/excavator/config/inventory_augmented.yml`. It uses
the same SI units, density, target registry, and material properties as the real
catalog. Synthetic stones have no measured pickup poses or scan references;
use model-derived orientations (`pose_from_scan=False`) for simulation studies.

## Generation

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    python apps/excavator/scripts/build_augmented_objects.py --workers 6
```

The default seed is 20261112. Each real stone is selected as a parent once,
with two additional parents drawn from a second seeded permutation. New IDs
receive separate reproducible random seeds. The shared implementation lives in
`stacking_common.reconstruction.augmentation`; fitting values are owned by the
excavator script.

The generator applies anisotropic scaling and four smooth Gaussian displacement
fields in the source mesh's principal axes. The normalized displacement
Jacobian norm is bounded by 0.30. It preserves the triangle connectivity and
then independently rejects meshes with self intersections or invalid topology.
It integrates the resulting surface's volume, center of mass, and inertia,
recenters both surface and collision geometry at that center of mass, and
writes unit-density inertial values for the normal asset loader.

Every synthetic surface receives a fresh CoACD decomposition followed by the
existing optimized DSF fitter. The decomposition allows up to four components;
the initial DSF budget is 48 nodes per stone and 300 evaluations per component.
Support errors are checked after commissioning integer sharpness. Difficult
fits retry with 96 nodes and 600 evaluations, then linear loss if necessary.
The 95th-percentile directional error for each component must be at most 5% of
the stone's largest extent. CoACD's requested threshold is 0.02; its four-part
cap can prevent that threshold being reached. Independent surface/DSF volume
checks are therefore reported separately rather than equating the threshold
with achieved accuracy. Optimizer iteration limits are retained in provenance;
acceptance uses the measured support error, not a convergence claim.

Existing completed objects are hash-verified and reused on restart. Failed
objects remain incomplete and are retried without replacing accepted models.
Changing sources, seed, count, or fitting settings requires a different output
directory. `--total` can request a larger catalog.

## Sequential fit viewer

```bash
python apps/excavator/scripts/view_augmented_objects.py
```

The viewer starts at the first synthetic stone (59) and covers all 100 catalog
IDs. Three linked panes show the surface mesh, the actual curved DSF components
in different colours, and a transparent overlay. Mouse rotation and zoom apply
to all panes. Changing stones preserves the view angle and fits the new model
to the viewport. The displayed DSFs evaluate their fitted support functions;
they are not the OBJ node hulls.

- **Right**, **Space**, or **N**: next stone; **Left** or **P**: previous.
- **Home** / **End**: first / last stone. Navigation wraps at the ends.
- **R**: reset the view; **S**: save a screenshot; **Q** / **Esc**: close.

Use `--start 93` to begin at a particular object. Screenshots saved with **S**
go to `.analysis/objects-augmented/viewer/stone_ID_fit.png`. For a headless
preview without opening a window:

```bash
python apps/excavator/scripts/view_augmented_objects.py --start 93 \
    --screenshot .analysis/objects-augmented/viewer/stone_093_fit.png
```

## Files and review

- `model_ID.obj`: fitted DSF components, inherited friction, unit-density mass
  and inertia, in the existing runtime format.
- `model_ID_mesh.ply`: matching original or deformed triangle surface.
- `model_ID.json`: synthetic parent, source hashes, deformation, fitting values,
  support errors, and output hashes.
- `manifest.json`: completion state, all 100 IDs, source hashes, seeded parent
  schedule, settings, software versions, and implementation hashes.
- `validation.json`: independent native-loading, geometry, inertial, and
  surface/DSF overlap audit.

Reproduce that audit with:

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
    python apps/excavator/scripts/validate_augmented_objects.py
```

All 100 models load through the native asset loader. For the 51 synthetic
stones, the independent voxel audit gives mean surface/DSF IoU 0.910 and minimum
IoU 0.862; the minimum original-surface volume covered by the fitted DSF union
is 0.935. The audit samples a grid with 45 cells along each stone's largest
bounding-box extent. It compares the original surface against sampled smooth
DSF surfaces rather than using the fitter's support residual as an IoU proxy.
This catalog has 343 DSF components; a full scene needs 101 bodies including
ground and 344 geometries including the target.

Render the complete inventory with the existing invisible 4×4 grid renderer:

```bash
python tests/inventory_paper_figures.py \
    --inventory apps/excavator/config/inventory_augmented.yml \
    --output .analysis/objects-augmented/inventory
```

For a Python search experiment, load this profile and replace the application's
`inventory_profile` using `dataclasses.replace`, with `pose_from_scan=False`.
The asset catalog does not change planner profiles. A 100-stone scene exceeds
the existing 64-body CUDA capacity; size body and geometry capacities for the
experiment using the counts recorded in `validation.json` before starting a
long run.
