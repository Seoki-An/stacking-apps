# Visualization migration

The visualization migration shares canonical scene composition, timelines,
MCTS diagnostics, and rendering without sharing platform execution logic.
Tabletop ROS, operator decisions, robot controls, sensor acquisition, and
platform overlays remain under `apps/tabletop`.

## Legacy characterization baseline

`tests/fixtures/legacy_visualization/mcts_candidate_viewer.json` is the small,
human-readable golden case for the first migration phase. It represents the
behavior that the new common projection and tabletop viewer must preserve; it
is not the new artifact schema.

The baseline covers:

- successful, rejected, unevaluated, selected, and zero-candidate outcomes;
- explicit selected-candidate identity;
- all, opaque-all, selected, and state-only candidate rendering;
- initial, solved, designed action, simulation-start, and final poses;
- one score map per search state rather than per selected candidate;
- separate simulation contacts and pose-solve forces;
- failed motion of an existing scene object;
- synchronized multi-object tracks with hold-last-pose replay; and
- camera preservation across candidate, layer, pose, mesh, and refresh changes.

The legacy `settled_pose` name is ambiguous. It is a compatibility alias for
the designed action pose in the legacy debug record, not a guarantee that it is
the first pose of structural simulation. New contracts should use explicit
`action_pose`, `simulation_start_pose`, and `final_pose` names.

## Standalone pose-optimization paper figures

The offline experiment in `tests/posegen_paper_figures.py` exports paired,
matched-camera figures from a recorded excavator planning step:

```bash
MPLCONFIGDIR=/tmp/stacking-paper-mpl \
  python tests/posegen_paper_figures.py \
  --run .stacks/excavator_plan21.stack --step 20 \
  --output .analysis/posegen-paper
```

Use an unused output directory. `--candidate` optionally selects a recorded
candidate ID; the selected candidate is the default. This requires a run with
a planning checkpoint at the chosen step and its recorded object models.
Rendering is offscreen; on X-only VTK builds, prefix Python with `xvfb-run -a`.
Add `--stone-mesh ply` to display reconstructed `model_<id>_mesh.ply` stone
meshes; `--stone-mesh dsf` (the default) displays the solver's DSF surfaces.
This option changes display geometry only; optimization and force estimates
still use DSF models. The chosen mesh mode, paths and hashes are saved in the
numerical JSON record. Missing requested surfaces raise an error, and surface
hashes are checked against the run when recorded.
The script checks participating model hashes and restores the run's posegen,
target and density profiles. It uses CPU posegen even for a CUDA-recorded run;
the replayed optimum can differ from the recorded CUDA optimum.

The trajectory panel follows the reference paper style: opaque gray
scene stones, a yellow initial candidate, a green final candidate, and faint
gray ghost poses on a solid brown ground plane. Ghosts are sampled **accepted
native trust-region updates**, with original rotations. Native profiling identifies
accepted trials; deterministic prefix replays recover their poses while
preserving all preceding continuation stages. It does not interpolate poses
or restart the optimizer after each iteration. Spatially spaced ghost meshes
avoid obscuring the final pose with nearly identical samples. Already placed
stones retain their input poses. Original display meshes are rendered with
smooth shading and no edge outlines or mesh simplification. Colors and
transparency are preserved, and mesh cell counts are recorded.

Both Operations Consoles also use the paper figures' brown ground color
(`#5a5046`), including the excavator's world-frame planning replay ground.

The force panel uses that same replay's final pose and whole-scene posegen force
output, restored to newtons by the standard adapter's mean-mass scaling. Arrows
point in the direction of force **on each stone**; stone/stone contacts retain
both incidences. Stones are transparent gray, and arrows are red with fixed
shaft and arrowhead widths. To keep forces of different orders of magnitude
visible together, their lengths use the compressed mapping
`L = 0.16 + 0.39 * sqrt(F / F_max)` in meters, rather than linear scaling.
Thus positive-force arrow lengths range from 0.16 to 0.55 m, while zero forces
are omitted. Small white markers locate arrow origins. The mapping and glyph
dimensions are recorded in the rendering JSON for the
paper caption; labels, center lines and color bars are omitted to match the references.
Only native returned contacts are drawn, subject to the recorded profile's
contact tolerance and boundary roles. These are posegen force estimates at the
optimized configuration, before physical simulation.

Outputs are two 2400×1800 PNGs at 300 dpi, corresponding PDFs containing raster
renderings, and JSON files retaining poses, stage/iteration indices, forces,
solver convergence, recorded profiles, source/model/native hashes, camera and
rendering settings. Large titles and panel letters are left to the paper.
No search or native-core implementation changes are required.

## Standalone height and placement-score paper figures

`tests/height_score_paper_figures.py` exports a six-panel overview and individual
PNG/PDF maps for scene height, total placement score, support score, filling
score, target boundary score and height score:

```bash
MPLCONFIGDIR=/tmp/stacking-paper-mpl \
  python tests/height_score_paper_figures.py \
  --run .stacks/excavator_plan21.stack --step 20 \
  --output .analysis/height-score-paper
```

Use an unused output directory. `--candidate` optionally selects a recorded
candidate ID; otherwise the selected candidate is used. The scene is the
pre-search scene, before adding that stone. The script loads the recorded
profiles and checks participating model hashes using the pose-figure loader,
then calls `initial_pose_score_view`, the same diagnostic used by the excavator
Operations Console. It does not solve poses or run simulation.

Scores depend on the stone's footprint and orientation, not just the scene.
These maps hold the recorded initial orientation fixed, apply normal floor
gates and normalize height over that orientation's eligible cells. They do
not reconstruct multi-orientation normalization or retry relaxation during the
original search. Unsupported historical scoring settings are rejected.

The four score panels display signed **weighted contributions**, so higher is
better, including for the negatively weighted height term. The total includes
every recorded term, including planar bias if its weight is nonzero; it is
checked against the weighted sum before export. Each panel has its own color
scale. Score colors show computed values under the original eligibility mask.
Unscored or excluded regions inside the physical target base are neutral gray
(`#808080`); outside the target is black. Scores are clipped to the exact
target rectangle, and no extrapolation is rendered.

Individual `*_masked.png/.pdf` files retain black masked regions for comparison.
The numerical `raw_*` and `weighted_*` arrays and eligibility mask are unchanged.
Older exports may also contain `display_*` and `boundary_display_*` diagnostic
continuation arrays; these are retained in their archives but are not rendered.

Boundary proximity is measured relative to the candidate's allowed anchor
rectangle, not directly to the yellow target-base outline. An asymmetric
rotated footprint shifts that rectangle; scene-dependent support heights and
taper change its size at each cell. Sampling admission uses base-height anchor
bounds, while the boundary score uses height-dependent bounds and returns zero
outside them. Consequently an admitted cell can have a zero boundary score
near a tapered wall. Longer footprints along Y also require wider top/bottom
anchor margins than left/right margins. Recorded target profiles, rather than
the current YAML, determine these figures.
The physical height map remains in meters and includes the whole target area.
Figures follow the Operations Console's **XY score maps** conventions with a
white background and dark text: X is horizontal and Y is vertical, every panel
uses the full scene's bounds. The overview retains titles and horizontal color
bars with three tick labels; individual PNG/PDF maps omit titles and color bars
for composition in the paper, while retaining coordinate axes and overlays.
Solid yellow outlines show the target base, and a magenta point marks the
candidate's initial XY position. Its original reconstructed PLY mesh is
overlaid as a 50%-opaque grayscale bottom-height map, with lower points white
and higher points black. The overlay uses the scene height-map resolution and
the candidate's full initial pose. It samples the minimum triangle-surface Z
at each XY location, preserving concavities and masking empty space; no convex
component outlines or whole-stone convex hull are drawn. This display overlay
is independent of the scorer's hull-based support kernels. Viridis
colors match the console. Heatmaps show actual cells without smoothing.

`height_score_maps.npz` preserves raw and weighted terms, all grid coordinates,
the eligibility mask, scene heights and candidate bottom heights in meters
(world Z, with NaN outside the mesh). The JSON sidecar
retains source/context/model hashes, recorded settings, poses, weights and
coordinate conventions, and the PLY source hash and bottom-map color limits.
PNGs use 300 dpi; PDF labels and target outlines remain vector.

The separate `candidate_bottom_height.png/.pdf` shows the candidate's contact-side
height map, cropped to its sampled footprint on a white background. It uses the
same original PLY lower surface and initial orientation as the translucent
overlay, with opaque viridis colors. XY coordinates are relative to the candidate's
initial XY origin and keep the scene's axis directions, without mirroring.
Heights are in meters above the lowest sampled underside point, following the
separate object elevation-map convention in [From Rocks to Walls, Section 3.1
and Figure 2(c)](https://vislab.isr.tecnico.ulisboa.pt/wp-content/uploads/paper_pdfs/2021_amenezes_from_ocks.pdf).
The clean figure has no title or color bar; `candidate_bottom_height_annotated`
adds both for inspecting the height scale. NPZ exports retain the cropped
`candidate_bottom_x`, `candidate_bottom_y`, `candidate_bottom_relative_height`,
and absolute `candidate_bottom_minimum_z`, while the JSON records these conventions.

## Source behavior

The characterization comes from these read-only legacy references:

- `stacking-tabletop/scripts/debug/candidate_viewer.py`
- `stacking-planner/scripts/debug/candidate_viewer.py`
- `stacking-planner/docs/llm/debugging.md`
- `stacking-planner/scripts/debug/candidate_simulation_tracks.py`
- `stacking-tabletop/scripts/execute_offline.py`

The execution viewer establishes one additional requirement for the later live
GUI: static named geometry is replaced only when the scene layout changes,
while frequent robot and carried-object motion is delivered as latest-only
transform updates. Rendering updates must be applied on the GUI thread.

## Deliberate gaps

The current common MCTS keeps complete admitted tree candidates but retains
generation failures only as aggregate rejection counts. A later opt-in trace is
required before rejected proposal geometry can be shown. The characterization
does not require retaining that extra data during normal search.

## Common visualization contracts

`stacking_common.visualization` now defines immutable, backend-neutral values
for camera views, target volumes, scene snapshots, object pose tracks,
synchronized timelines, score fields, simulation contacts, and pose-solve
forces. A timeline holds the final sample of a shorter object track during
replay. The common package does not import PySide, PyVista, VTK, Open3D, ROS,
or platform modules.

The MCTS projection turns an evaluated root and its admitted children into one
`MCTSDebugStep`. It reads existing domain state, action stages, trajectories,
contacts, tree statistics, and diagnostics without mutating the search tree.
Generation failures remain aggregate rejection counts; the projection does not
invent geometry that normal search did not retain.

Candidate pose modes are explicitly named `initial`, `solved`, `action`,
`simulation_start`, and `final`. Candidate identity and outcome are explicit,
and render-mode ordering preserves the legacy selected-last behavior for the
transparent all-candidate view.

## Native `.stack` run document

In both Operations Consoles, select a step in **Search Inspector** and read
the **Search outcome** panel. It shows the recorded step status, per-search
timeout and evaluation counts, retained-tree rejection counts when available,
CEM rejection counts, root candidate failures, and root generation rejections.
Selecting a candidate shows its individual rejection reason in the selectable
**Failure reason** field below the candidate selector and in the dropdown's
tooltips, including
`missing_validated_descendant` when required descendant validation did not
produce a usable path. A zero-candidate step can still show CEM failures.
Each count is labelled by its scope; overlapping categories should not be
added together. Older runs without detailed records are identified explicitly.
Use **Refresh run** to read newly committed search steps.
The inspector scrolls vertically to keep the controls and failure details
accessible in smaller windows.

`stacking_common.artifacts.StackRunStore` owns the schema-versioned native run
format. One `.stack` file contains compact JSON records, checksummed compressed
typed arrays, run/configuration provenance, MCTS visualization steps, and
per-step search accounting. SQLite supplies transactional incremental writes
and indexed reads but is an internal implementation detail; `.stack` is the
public format and extension.

The MCTS runner creates the document before search, commits each completed
step independently, and records a terminal `complete` or `failed` status. It
does not rewrite prior steps. Numeric payloads large enough to benefit from
binary storage are compressed and content-addressed; small vectors remain in
their owning JSON record to avoid row overhead.

`MCTSDebugArtifactStore` remains the JSON boundary for small fixtures and
explicit interchange tests. The Operations Console accepts those JSON values
read-only, while new planning/search runs are written as `.stack`. The legacy
characterization fixture has a dedicated decoder and is not a production
schema. Object meshes currently remain in the tabletop asset catalog and are
resolved by stable object IDs. The format already provides checksummed embedded
attachments for replay assets and derived MP4 exports. The Operations Console
can explicitly export the active candidate's synchronized timeline to one MP4
using the current camera and environment-layer visibility. Embedding that
derived export back into the source `.stack` remains later integration work.

## Application naming

The overall planning, search-diagnostics, replay, and execution application is
the **Stacking Operations Console**. Its MCTS-focused section is the **Search
Inspector**. “Candidate viewer” is retained only when identifying the legacy
programs that supplied migration behavior.

The existing perception application remains the **Object Reconstruction
Workbench**. Its canonical command is
`stacking-tabletop-reconstruction-workbench`; the shorter historical command
remains a compatibility alias. The operations console uses the distinct
`stacking-tabletop-operations` command and `gui/operations` package.

The first operations-console shell now loads `.stack` runs and provides search
step/candidate navigation, render and pose modes, trajectory replay, diagnostic
layers, camera preservation, and tabletop mesh resolution. Live search events
and planning/execution operator controls are intentionally not part of the
common artifact contract and remain later platform integrations.

The tabletop viewport keeps the legacy Open3D candidate viewer's fully rough
look: one fixed directional key light provides diffuse illumination, while
screen-space ambient occlusion supplies soft ground and contact shadows. VTK's
shadow-map pass is intentionally not used because it drops translucent target
and candidate actors. Zero specular response keeps stones non-reflective, while
smoothed vertex normals hide individual reconstruction triangles as in the
legacy Open3D viewer. A stronger material ambient response softens directional
contrast. The target is a translucent white volume. A shared dark stylesheet
keeps the Operations Console and Object Reconstruction Workbench visually
consistent. The tabletop ground and layer defaults come from
`config/visualization.yml`. Support tables and exclusion volumes use the saved
planning context, falling back to the current planning profile for runs without
recorded inputs. Search Inspector (including trajectory export) transforms these
world-frame boxes using the inverse recorded `world_T_stack`, including rotation;
its reference ground stays at stack `z=0`. Plan and Execution retain world-frame
geometry. The operator can toggle each semantic layer without altering the
planning result.

The viewport preserves its camera pose across step, candidate, render-mode,
pose-mode, diagnostic-layer, environment-layer, planning-attempt, and tab
changes. Camera reset occurs only when a new run is loaded or the operator
explicitly selects **Reset camera**.

Both operations consoles provide **Refresh run** beside the open-file controls
and in the Planning toolbar / File menu (`Ctrl+R`). This rereads committed
updates from the currently open `.stack` without reopening the file picker,
including when a planning process is still writing the run. Existing step,
candidate, planning-attempt and replay selections are retained where available;
new steps and attempts appear in their selectors. Refresh pauses playback and
preserves the current tab, camera and rendering toggles. It is unavailable
before a `.stack` is opened, and requests during video export or active tabletop
execution are blocked.

Starting planning in either console automatically opens its output `.stack`
as soon as the committed header is readable (and, for excavator, the recorded
planning context is available). The viewer may initially have no completed
steps; use **Refresh run** during planning to inspect newly committed results.
Completion automatically loads the final result, including the published path
when replacing an existing run. Planning progress is a non-modal window with
its own minimize/maximize controls: minimizing it leaves the viewer usable and
planning running. Selecting **New planning run** again restores an active
minimized progress window instead of launching another planner.

Planning log output is appended independently of the text selection, so
selecting/copying logs with `Ctrl+A` / `Ctrl+C` does not replace or erase them
when new output arrives. Auto-scrolling pauses while selecting text or reading
earlier output. The existing 3,000-line display limit still applies.

Save-path browsers use the Qt widget file chooser in both consoles. Right-click
a file or directory for **Rename** / **Delete**, or create a **New Folder**.
Deletion requires confirmation and is permanent, not a move to Ubuntu Trash;
avoid deleting a run that is still being written. Cancelling the save chooser
does not undo filesystem changes already confirmed in it.

Scene replacement is render-suppressed until all actors and the preserved
camera are ready, preventing a blank intermediate frame. Replay slider changes
are coalesced over a short interval so rapid scrubbing does not enqueue a full
render for every raw slider event. Video capture also suppresses widget paints
and restores the original replay and pose-mode state after export.
Simulation replay and trajectory videos show moving stones in opaque gray.
Tracked poses replace static copies of the same stones; untracked scene stones
remain visible at their original poses.
Export displays frame progress followed by a final MP4 finishing stage. The
operator can cancel between frames; cancellation discards the temporary video
and preserves an existing destination file.

Simulation replay reuses mesh actors and updates their transforms instead of
rebuilding the ground, target, lights, and stone geometry for each frame.
New simulation trajectories record elapsed seconds from the start of structural
simulation (after fixed-scene settling), including across short/long phases.
These timestamps are preserved in `.stack`. **Video FPS** resamples timed tracks
using linear translation and shortest-arc quaternion interpolation, so changing
FPS changes smoothness rather than simulation speed. Export includes a final
endpoint frame (up to one frame interval of endpoint padding for aligned times,
and less than two for unaligned times).
Older artifacts without timestamps retain one-sample-per-frame export; their
playback speed still depends on FPS. Interactive simulation scrubbing remains
sample-index based; the Video FPS control applies only to MP4 export.

Candidate scores use the legacy blue–cyan–yellow–red gradient, with the active
candidate shown in opaque amber and no wireframe selection outline. The target
uses the same rough diffuse lighting as the other scene geometry while
remaining translucent white. Its rendered shell deliberately omits the bottom
face, avoiding depth and ambient-occlusion artifacts where it meets the ground.
The ambient-occlusion pass uses a higher-sample blurred kernel to avoid visible
sampling stripes at shallow stone–ground contacts.

The migrated legacy table and `third_view_camera` bounds are reference
overlays. Future motion-planning runs should persist the exact collision boxes
or mesh asset references used by that run in `.stack`, rather than assuming the
current profile still matches an older result.
