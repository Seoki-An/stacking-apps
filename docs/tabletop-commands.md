# Tabletop command reference

Run commands from the repository root using the project Python 3.12 environment.
Replace `/path/to/...`, `YOUR_UR5E_IP`, and `SESSION_ID` with local values. CLI run
paths, summary exports, and capture output directories must be new; the planning
GUI also supports explicit overwrite as described below.

## Setup

```bash
cd ~/workspace/stacking/stacking-apps

# Install core, common, and tabletop perception dependencies.
make install-tabletop

# Add both consoles and the hardware/camera adapters.
python -m pip install -e 'apps/tabletop[operations,workbench,execution,acquisition]'
```

For hardware and camera work, source ROS in **each terminal**:

```bash
source /opt/ros/jazzy/setup.bash
source ~/ros2_ws/install/setup.bash
```

## Offline planning and replay

To create a plan from the GUI, open the console without a run:

```bash
stacking-tabletop-operations .
```

Choose **File → New planning run…** (`Ctrl+N`) or the toolbar button. Select
the MCTS profile and a new output `.stack` file, optionally supply trusted pickup
poses and object IDs, and set the step count and whole-run time limit. Leave
**Placement search only** unchecked to include grasp and motion planning.
Click **Start planning**, then **Open resulting run** to inspect/replay the
saved results. **Stop planning** preserves committed records. Blank pickup/ID
fields use profile values; an unset profile pickup path uses a synthetic scene.
This workflow computes plans offline and sends no hardware commands.

To reuse an existing `.stack` filename, check **Overwrite existing run on
successful completion**. The old file stays intact while planning runs and is
replaced only after successful completion. Failed/stopped results remain in a
separate `*-pending-*.stack` file shown in the log and available through
**Open resulting run**. A resumed run cannot overwrite its source. When selecting
an existing destination for tabletop resume, review this option and click Start.

**Pose from scan override** has three choices:

- **Use configuration (no override)**: use `orientation.pose_from_scan` in
  `apps/tabletop/config/action_generation.yml` (default: `true`).
- **On**: use scan-aligned pickup orientations and the scan-orientation filter.
- **Off**: use model principal-axis orientations with 0°/180° roll variants,
  without the scan-orientation filter.

This setting controls placement orientation proposals. Recorded pickup poses
still supply the grasp-planning pickup geometry in either mode; without a pickup
file, the existing synthetic-scene behavior remains. The override does not edit
YAML. Resume preserves the recorded setting. Equivalent CLI flags are
`--pose-from-scan` and `--no-pose-from-scan`; omit both to use configuration.

Run MCTS with the current profile:

```bash
python apps/tabletop/scripts/debug/mcts.py \
  --config apps/tabletop/config/mcts.yml \
  --save-run /tmp/tabletop-mcts.stack
```

Short synthetic smoke run, including native grasp/motion planning:

```bash
python apps/tabletop/scripts/debug/mcts.py \
  --config apps/tabletop/config/mcts.yml \
  --object-ids 1 --steps 1 --iterations 1 --max-search-time 60 \
  --transition-backend kinematic --pose-solver passthrough \
  --plan-motion --save-run /tmp/tabletop-smoke.stack
```

Plan from a trusted, measured pickup scene:

```bash
python apps/tabletop/scripts/debug/mcts.py \
  --config apps/tabletop/config/mcts.yml \
  --pick-pose-data /path/to/pose_data.pkl \
  --object-ids 1 --steps 1 --plan-motion \
  --save-run /tmp/tabletop-planning.stack
```

Open the Operations Console and replay the run:

```bash
stacking-tabletop-operations . --run /tmp/tabletop-planning.stack
```

In **Plan → Step**, select a step marked **motion recorded**, select a successful
attempt, and click **Play**, or scrub the
whole-step slider. The solid stone follows the gripper during attached motion
and stays at its recorded pose after release. Faint pickup and placement stones
remain as reference markers. This requires a trajectory with recorded target
poses; failed attempts have no motion to replay.
The Step selection is shared with Search Inspector. Changing steps stops
playback and resets the motion slider; steps without saved motions are labeled
**no motion recorded** and have playback disabled.

The **Rendering** toolbar offers the same mesh choices as excavator:

- **Objects: DSF surface / PLY mesh** selects the stone surface representation.
- **Robot / gripper: Visual meshes / Collision meshes / Visual + collision
  wireframe** selects robot and gripper geometry. The combined mode overlays
  orange collision wireframes on the visual meshes.

The robot choice is synchronized with the Plan tab and also applies to measured
Execution poses and cyan scan previews. Changing it preserves playback position.
Failed planning attempts show the robot at the recorded profile's home pose
for geometry inspection; this is not a planned motion or measured robot state.
Motion segment labels include **free** or **attached** when recorded.

Pickup and placement tables are visible by default. Plan and Execution use
the saved world-frame planning bounds, including the measured pickup height;
Search Inspector uses its search-frame reference geometry. The **Support**
layer checkbox in Search Inspector controls table visibility in Plan and Search.

Opening a run does not command hardware. The synthetic smoke scene is for
software checks; physical execution needs a plan matching the actual scene.

Resume offline planning into a new run and export its summary:

```bash
python apps/tabletop/scripts/debug/mcts.py \
  --config apps/tabletop/config/mcts.yml \
  --resume-run /tmp/tabletop-planning.stack \
  --save-run /tmp/tabletop-planning-resumed.stack \
  --steps 5 --max-run-time 600 \
  --export-summary /tmp/tabletop-planning-resumed.json
```

On resume, `--steps` means additional placements. Recorded inputs must still
match. Physical execution recovery uses the console's reconciliation workflow,
not this offline resume command. The File menu also exports existing runs to
JSON or CSV.

## UR5e and Dynamixel startup

These commands start hardware controllers from `~/ros2_ws`. Use the installation's
commissioned calibration and settings; physical commissioning remains pending
for the migrated application. Start each controller once, in separate terminals.

```bash
# Terminal 1: UR driver + admittance controller; gripper is separate.
ros2 launch ur5e_admittance bringup.launch.py \
  robot_ip:=YOUR_UR5E_IP \
  kinematics_params_file:=/path/to/ur5e_factory_calibration.yaml
```

```bash
# Terminal 2: single owner of the Dynamixel serial device.
ros2 launch ur5e_admittance gripper_bringup.launch.py device:=/dev/ttyUSB0
```

Use the actual serial device path. The arm launch defaults to
`headless_mode:=false`, requiring the robot's External Control program.
In the Operations Console, select **Execution → Start supervised execution**
and follow the pickup, close, placement, and release gates. See the
[hardware guide](tabletop-hardware.md) for model checks and recovery behavior.

## FEMTO capture and automatic scanning

For fixed-field capture → filtering → stone identification, choose
**File → Field scan / identify…** in the Operations Console. After success,
**New plan using these poses** fills the planning inputs. Equivalent commands
after installing `apps/tabletop[acquisition,perception,operations]`:

```bash
stacking-tabletop-scan-field --visualize
stacking-tabletop-scan-field --both-cameras --visualize
```

These own their camera drivers; stop existing drivers first, or use
`--use-running-camera` for a single camera. See the
[field workflow guide](tabletop-femto.md#field-capture-and-identification-in-the-operations-console)
for ID assignment, manual alignment, capture retries, and output paths.

Start the primary camera driver in its own ROS-sourced terminal:

```bash
python /opt/ros/jazzy/bin/ros2 launch apps/tabletop/launch/femto.launch.py camera:=primary
```

Capture a fixed field from another ROS-sourced terminal:

```bash
stacking-tabletop-capture-femto --camera primary --output .data/field-primary
```

Capture one view of an object already held stationary by the robot:

```bash
stacking-tabletop-capture-femto \
  --camera primary --in-hand --output .data/held-view-primary
```

The capture CLI does not move the robot. For **automatic wrist/pivot scanning**,
use **Prepare held-object scan…** at the console's placement review gate. Select
the camera and output parent directory, review the motion preview, then Continue.
The scan returns to a fresh placement review gate after saving its views.

For the secondary camera, stop the primary driver and substitute `secondary`
in both launch and capture commands. Primary is `CL83C53005L`; secondary is
`CL8956200P5`. They share `/camera` and run sequentially. Calibration paths and
field crop are in `apps/tabletop/config/femto.yml`. See the
[FEMTO guide](tabletop-femto.md) for synchronization and output details.

## Object reconstruction

Prepare, merge, and reconstruct a completed in-hand scan:

```bash
stacking-tabletop-object-build . \
  --object-id 1 --capture-directory /path/to/completed-in-hand-scan
```

Optionally add `--mass-kg 0.25` with the object's measured mass. Otherwise,
density comes from `apps/tabletop/config/inventory.yml`.

Rerun reconstruction in the workflow directory printed by the build command:

```bash
stacking-tabletop-object-build . \
  --resume apps/tabletop/data/object-build/1/SESSION_ID --stage reconstruct
```

Open an existing workflow for interactive review, or start a new one:

```bash
stacking-tabletop-reconstruction-workbench . \
  --resume apps/tabletop/data/object-build/1/SESSION_ID

# Alternative: create a new object-build workflow.
stacking-tabletop-reconstruction-workbench . --object-id 1
```

## Offline hardware-integration tests

With pytest installed in the project environment:

```bash
QT_QPA_PLATFORM=offscreen python -m pytest -q \
  tests/test_tabletop_hardware_adapters.py tests/test_tabletop_ros_hardware.py \
  tests/test_tabletop_saved_execution.py tests/test_tabletop_execution_console.py \
  tests/test_tabletop_femto_calibration.py tests/test_tabletop_femto_acquisition.py \
  tests/test_tabletop_femto_ros.py tests/test_tabletop_scanning.py
```

These tests use simulated devices and do not command the hardware. Append
`--help` to any CLI above to inspect all available options.
