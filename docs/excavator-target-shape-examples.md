# Full-inventory target-shape examples

The batch runner uses the same `stacking_excavator.planning.cli` worker as
**New planning run → Placement search only** in `stacking-excavator-operations`.
It plans placements with native pose optimization, settling, and stability
validation. Robot grasping and motion planning are omitted. Every example
starts with an empty stack and the complete 49-stone excavator inventory.

Prepare a batch without starting a long search:

```bash
python -m stacking_excavator.planning.shape_batch \
  --output .analysis/target-shape-batch --prepare-only
```

The shared [target registry](../apps/excavator/config/target_shapes.yml)
supplies the batch runner, CLI `--target-shape`, and the operations GUI's
**New planning run → Target shape** dropdown. `inventory.yml` now contains only
object settings. The default shape named `default` preserves the wall from
`excavator_plan17.stack`; `tapered_tower` remains selectable, and `two_walls`
demonstrates one composite goal with two disconnected DSFs.
`l_shaped_wall` instead overlaps two perpendicular 2 × 4 × 3 m wall DSFs:
their ground overlap is 2 × 2 m, and the L-shaped footprint spans [-2, 2] m
on both axes, excluding the upper-right 2 × 2 m notch. Both components taper
5° on all four side faces. They form one connected
sampling region with shared floor progress; union coverage counts overlap once.
Native containment still requires each stone to fit inside at least one DSF,
so placements that only fit by spanning the component seam remain restricted.

Select **l_shaped_wall** in the operations GUI's target dropdown, or start a
bounded placement-only CUDA test (the default target is unchanged):

```bash
python -m stacking_excavator.planning.cli \
  --config apps/excavator/config/mcts_cuda.yml \
  --target-shape l_shaped_wall --search-only \
  --steps 5 --iterations 200 --max-search-time 60 --max-run-time 600 \
  --save-run /tmp/excavator-l-shaped.stack
```

The four original example shapes below are also registered. Dimensions are at ground level;
the targets include the platform's normal support overlap below ground.

Composite proposals share one `planar.grid_shape` grid over the complete target's
XY bounds (32×32 by default). Support maps are computed once per stone and
orientation; region masks exclude gaps and preserve balanced sampling. Each
region still uses its own `planar.floor.grid_size` coverage grid and advances its
layers independently. If a stone has no eligible proposals in a region on the
shared grid, sampling retries the original region-local grid at the configured
resolution. This also protects narrow or widely separated components that the
shared grid misses. Per-region score maps expose `grid_source` as `shared` or
`region_fallback` for diagnostics.

The shared grid changes proposal discretization, not the native target geometry,
continuous pose optimization, MCTS policy, simulation, or rewards. Coarser seeds
can still change the resulting sequence; local fallback prevents empty regional
pools but does not guarantee identical proposal coverage. Existing `.stack`
documents remain readable.

| Shape | Base width × length | Height | Top width × length | Sloped faces |
| --- | --- | --- | --- | --- |
| One-sided wall | 1.6 × 4.5 m | 3.0 m | 1.0 × 4.5 m | +X only |
| Two-sided wall | 2.0 × 4.0 m | 3.0 m | 1.48 × 4.0 m | +X and −X, 5° each |
| Pyramid | 4.0 × 4.0 m | 2.8 m | 0.7 × 0.7 m | All four sides |
| Box | 2.8 × 3.2 m | 2.0 m | 2.8 × 3.2 m | None |

The pyramid is truncated: a finite top accommodates finite stones and avoids
the target model's prohibited zero-width apex. The box is solid, rather than
a hollow enclosure. Target volumes are approximately 18–21.4 m³, compared with
11.53 m³ of summed stone convex-component volumes; the extra capacity allows
voids. These are starting target designs, not a guarantee that the planner can
place all 49 stones or perfectly fill each target. The batch never inserts
hand-positioned stones or relaxes physical validation to reach the count.

Preparation copies the current excavator profiles into each shape's private
configuration snapshot and validates all profiles, target geometry, 49 native
object models, and grasp/motion schemas. Original application YAML files are
not edited. Assets are referenced through a symlink to the checkout; recorded
native/model hashes still govern checkpoint resume. Source profiles, target
definitions, IDs, seed, and profile hashes are retained in `manifest.json`.
The default MCTS profile is `mcts_cuda.yml`. Select `--config
apps/excavator/config/mcts_cpu.yml` when preparing a separate CPU batch.

Run the four original examples sequentially, using one GPU at a time
(omit `--shapes` to include every registered shape):

```bash
python -m stacking_excavator.planning.shape_batch \
  --output .analysis/target-shape-batch \
  --shapes one_sided_wall two_sided_wall pyramid box
```

The default limit is **six hours per shape per invocation**, with at most
three advancing attempts within that time. Use `--max-run-time 14400` for a
four-hour limit. A shape's time budget includes its retries. A failure without
any new committed placements stops retries for that shape and advances to the
next shape. Nonzero worker exits and logs are retained. The runner emits a
heartbeat every 30 seconds; each worker log includes the normal search timing
and progress output. For an unattended terminal run:

```bash
nohup python -u -m stacking_excavator.planning.shape_batch \
  --output .analysis/target-shape-batch \
  > .analysis/target-shape-batch/batch.log 2>&1 &
```

Check progress without starting another planner:

```bash
python -m stacking_excavator.planning.shape_batch \
  --output .analysis/target-shape-batch --status
```

Rerun the normal command to resume incomplete shapes. It branches from each
shape's checkpoint with the largest number of committed placements, asks for
the remaining stones, and skips shapes already containing all 49. Attempts
are saved to new `runs/attempt-NNN.stack` files; parent files are preserved.
Ctrl+C/SIGTERM terminates the active worker group and retains its committed
checkpoint. A batch lock, inherited by the active worker, prevents two runners
from competing for the same output even if the parent process dies.

Resume preserves the frozen search settings and seed. Changes to copied
profiles are rejected; prepare a new output directory to change the target,
MCTS configuration, or seed. `--shapes box pyramid` selects a subset, and
`--shapes-file /path/to/targets.yml` supplies another target collection when
preparing a new batch. The standard per-run native/profile/asset checks also
apply. A clean planner exit alone is insufficient for batch completion:
the final checkpoint must contain every catalog ID exactly once. Exit code
0 means all requested shapes reached 49; 1 means at least one is incomplete;
2 signals a batch input/orchestration error; 130 means interrupted.

`--shapes-file` accepts either the registry schema (`units`, `default`, `shapes`)
or the old standalone mapping retained in `config/examples/target_shapes.yml`.
That old file is a compatibility input, not the active GUI registry. Composite
batch volumes use the shared voxel-union evaluator, so overlap is not counted
twice. Named selections and resolved targets are frozen in each batch snapshot;
existing prepared batches retain their original targets.

Each shape's `summary.json` contains `best_run.path`, placement count, attempt
statuses, and the latest worker log. Open the path printed by `--status`:

```bash
stacking-excavator-operations . \
  --run .analysis/target-shape-batch/box/runs/attempt-001.stack
```

The console restores the run's recorded target, rather than the current
`inventory.yml` target. It can display the full placement sequence and export
images using the normal scene viewer. Partial runs are also viewable; their
summary remains explicitly incomplete.
