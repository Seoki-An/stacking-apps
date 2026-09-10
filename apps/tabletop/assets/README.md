# Tabletop assets

These assets belong to the tabletop application because their dimensions,
physical calibration, perception density, and collision decomposition are
specific to this platform.

`objects/` contains the curated measured tabletop object catalog. Each object
ID listed in `config/inventory.yml` owns one file stem:

- `model_<id>.obj`: convex DSF components and inertial/contact metadata.
- `model_<id>.pcd`: dense model used for perception and registration.
- `model_<id>_mesh.ply`: reconstructed mesh used for visualization.

The common package owns the corresponding domain types, not these files.
`ur5e/` and `sr_gripper/` contain only the kinematic URDFs and compact DSF
collision assets required by native motion planning. Large visualization meshes
remain outside the reusable planning path.

`TabletopPcdCatalog` validates the configured IDs and loads only the PCD member
through the optional tabletop perception dependency. The resulting point data
crosses into common code as an immutable NumPy value rather than an Open3D
object.
