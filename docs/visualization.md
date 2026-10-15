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
