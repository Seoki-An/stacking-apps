# Legacy planner reference

`scored_sampling.npz` captures the legacy `floor_fill.py` and `support_map.py`
score kernels and 32 seeded proposals for stone 8, with inventory [55, 50, 8],
on an empty scene and a scene containing stone 55. `scored_sampling.json`
records the source hashes, seed, input pose and grid bounds. The source kernels
were executed from the sibling planner, using its ablation configuration;
only imports were redirected to load the pure modules without the agent stack.
Tests consume these numeric fixtures without importing the sibling repository.

`scan_sampling.npz` captures scan mode separately, using stone 55 with a measured
XYZ rotation of [20, -10, 37] degrees. It records four support kernels, score maps,
and 32 seeded final proposals for empty and occupied scenes. Orientation pairs
come from the legacy `_orientation_quaternion_pairs` method extracted with AST;
the legacy floor/support modules produce the maps and selected candidates. Final
poses compose each selected yaw with the measured scan once, as in legacy
`ActionBuilder._build_sample_with_solver`. `scan_sampling.json` records the
source hashes, scene inputs, seed and zero planar variance. This fixture checks
measured heading preservation and prevents a second scan composition during
height initialization or action construction.

`reference.json` and `motion.npz` were exported from the trusted local
`stacking-planner/sessions/260701/plan_260701_5` session with
`tests/export_legacy_planner_fixture.py`. The JSON records source paths/hashes,
effective saved parameters, the placed prefix, pickup poses, and the hash of
the NumPy archive. Runtime tests do not unpickle or import the old planner.

Step 39 contains four recorded direct-motion sections; step 40 contains eight
regrasp sections. The old action's `motion_mode` reflects an entrypoint setting,
so section count identifies the actual recorded case. Joint coordinates and
target transforms are recorded observations. Test-injected opening samples
are synthetic and are not a legacy opening-trajectory oracle.

`excavator_kinematics.urdf` is the visual-free commissioned chain from
`stacking-core/test/planner/assets/stacking_planner_excavator_kinematics.urdf`.
It verifies the native binding independently of the application's mesh assets.

`posegen_schedule.json` records all four stages of the excavator and tabletop
`PoseSolver._set_obj_weights` and `_set_trust_region` methods, with an incoming
gap of 0.002 and tabletop target guidance enabled. Each entry includes its
source path and SHA-256. The methods were extracted with Python AST and run
with NumPy and a configuration stub, without importing either legacy package.
Tests compare the current native solver configurations against these recorded
outputs, including the intermediate stages and gap feedback.
# LiDAR emulation reference

`lidar.npz` / `lidar.json` contain sensor-frame ray hits produced by the legacy
`perception/utils/simulate_lidar.py` for field, in-hand and scene examples, using
real excavator PLY/STL assets and two explicitly emulated sensor poses each.
The metadata includes the source and asset hashes and Open3D version.
`tests/export_legacy_lidar_fixture.py` regenerates them from a trusted checkout;
it supplies projected geometry and sensor poses to the unmodified legacy
function without importing its ROS/URDF wrappers. See
`docs/excavator-pcd-emulation.md` for scope and test commands.

`captures.npz` / `captures.json` contain compact subsets of actual field, scene
and in-hand logs, with original file hashes. Field/scene include independently
saved world-frame clouds to check sensor transforms. In-hand inputs were
already saved in gripper geometry coordinates. Timestamps are explicitly
unknown. Regenerate with `tests/export_legacy_capture_fixture.py`; these are
frame-regression fixtures, not ground-truth pose or physical calibration data.
