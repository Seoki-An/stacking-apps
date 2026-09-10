# Excavator PCD tests on the desktop

The desktop harness runs without a NUC, ROS installation, sensor connection or
robot commands. It follows `stacking-planner`'s
`perception/utils/simulate_lidar.py`, used by its field and in-hand simulation
nodes: 120° × 70° field of view, 0.15° × 0.36° angular sampling, a ground plane,
nearest mesh intersections and points returned in the sensor frame. It uses
the excavator's PLY stone assets and VDK23/stone_grab visual meshes and kinematics.
Both sensor placements and capture timestamps are explicitly emulated.

## Run all three purposes

From the repository root, with the project environment active:

```bash
python -m stacking_excavator.acquisition.emulation \
  --repository . \
  --output /tmp/excavator-pcd-test
```

The output directory must be new. The harness runs field identification,
in-hand identification and placed-scene capture examples. Each uses two sensor
views, keeps a raw capture, makes a purpose-specific crop, interrupts a local
transfer after 1,024 bytes, resumes it, verifies the hashes and exports merged
world-frame clouds. This is an acquisition/processing test; it does not run
complete PoseID or SceneID.

Run only the in-hand example:

```bash
python -m stacking_excavator.acquisition.emulation \
  --repository . \
  --purpose inhand \
  --output /tmp/excavator-inhand-pcd-test
```

The new `stacking-excavator-emulate-pcd` entry point accepts the same arguments
after refreshing the editable installation. If Open3D or the entry point is
missing, install the extra in the project environment:

```bash
python -m pip install -e 'apps/excavator[perception]'
```

The module command works without refreshing the entry point when the package
is already installed in editable mode and Open3D is available.

## Artifacts and results

`report.json` records counts, frame payload sizes, byte fractions and verified
product hashes. For each of `field`, `inhand` and `scene`, inspect:

- `raw/`: immutable original sensor XYZ, channels and capture-time metadata.
- `crop-v1/`: selected sensor rows and a versioned crop manifest.
- `desktop/`: resumed, verified product and an acknowledgment receipt.
- `raw-world.ply` and `crop-world.ply`: transformed and merged visualization data.

XYZ and attributes retain their values and dtypes in the NPZ frame products.
The PLY exports contain XYZ only. Neither crop nor transfer deletes the raw
capture. `prepare_crop` can derive a wider product in a new directory from the
same raw source; it rejects attempts to recrop an earlier crop.

The initial run retained 21,539 / 213,613 points for field, 19,447 / 185,806 for
in-hand, and 46,796 / 214,023 for scene. Frame payload sizes were approximately
10.1%, 10.5% and 21.9% of raw sizes, respectively. These ratios describe these
synthetic scenes and uncompressed NPZ payloads; they exclude manifest overhead
and are not predictions of WiFi throughput.

## Automated checks

```bash
python -m pytest \
  tests/test_excavator_capture.py \
  tests/test_excavator_pcd_emulation.py \
  tests/test_pointcloud.py \
  tests/test_architecture.py \
  tests/test_tabletop_femto_acquisition.py -q
```

Tests check legacy ray-casting fixtures, two sensor views, rotated and offset
gripper geometry frames, jaw-opening occlusion, purpose crops, exact channel
preservation, empty crops, wider recropping, calibration/timestamp/revision
rejection, interrupted/resumed transfers, lost acknowledgments, corruption and
missing frames. A registration test uses the existing common multiscale ICP
backend to recover a known displacement of an emulated cloud. Tabletop and
architecture checks cover the shared boundary.

Validation on 2026-09-09: the command above passed **70 tests** in the project
environment. The full three-purpose CLI also completed and wrote its report
and PLY artifacts. This is focused desktop validation, not a full-suite or
physical-sensor acceptance result.

Legacy parity fixtures include field, in-hand and scene rays at a smaller
30° × 24° field of view to keep the stored reference compact. The full CLI test
uses 120° × 70°. Tests load JSON/NPZ references without importing the legacy
application. To explicitly regenerate from a trusted local legacy checkout:

```bash
python tests/export_legacy_lidar_fixture.py ../stacking-planner
```

The exporter runs the original ray-casting function with supplied mesh/sensor
adapters, keeping its ray and intersection math unchanged. Metadata records
the legacy source hash, asset hashes, sensor transforms and Open3D version.
This verifies the ray caster, not the legacy ROS node or its URDF adapter.

## Implementation boundary

The [desktop perception pipeline](excavator-perception.md) now consumes these
products for field/in-hand identification and scene reconciliation proposals.
New field/in-hand emulation outputs include ready-to-use `priors.json` files.

This completes the desktop emulation test harness and provides capture/crop
contracts for the perception phase. The local receiver is a fault-injection
test utility, not a network service; callers serialize writes to one receiver.
Physical calibration, real capture timing, stationary ROS acquisition, WiFi
transport commissioning, raw-spool retention expiry and physical
field/in-hand/scene acceptance remain in the
[implementation plan](excavator-implementation-plan.md). Execution and online
operations follow perception. No tabletop configuration or planning behavior
changes are part of this harness.
