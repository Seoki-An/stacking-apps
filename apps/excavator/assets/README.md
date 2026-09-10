# Commissioned offline model assets

This directory contains the 49 commissioned convex-component stone OBJs,
their matching reconstructed PLY surfaces, and
the VDK23/stone-grab URDF dependency closure copied from `stacking-planner`.
`provenance.json` records original paths and SHA-256 values. URDF asset filenames
were relocated to sibling files; joint geometry, inertial values, contact flags,
visual origins, and mesh scales were preserved. Runtime contexts hash the
relocated files actually used by a run.

The robot IK URDF has visuals but no robot collision bodies. Gripper collision
meshes and their pad/tooth declarations are present. The stone PLY files are
display assets copied from the same source catalog; their OBJ counterparts
were byte-matched before copying. The viewer can switch between reconstructed
PLY meshes and curved surfaces evaluated from the DSF nodes and sharpness.
Measured catalog XYZ used by legacy PoseID is now included separately under
`perception/`, compressed losslessly as NPZ without resampling. Its own
`provenance.json` records the original PCD hashes and converted-file hashes for
all 49 stones. Normals are not migrated: the current point-to-point registration
uses XYZ. Runtime perception verifies each reference against these hashes.
Recorded field captures are not part of the asset catalog. Search continues to
use the commissioned DSF collision models.
