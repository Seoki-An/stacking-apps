# UR5e visual meshes

These seven PLY files were copied unchanged from the migrated reference
`stacking-tabletop/assets/ur5e/visual`. They are the existing conversions of the
official `Universal_Robots_ROS2_Description/meshes/ur5e/visual` Collada meshes,
with diffuse material colors retained as per-vertex RGBA and authored normals.
The associated upstream license is retained here as `LICENSE`.

`ur5e_ik.urdf` supplies each visual's link-relative origin. The robot base is a
visual-only component. Collision DSF assets remain in the separate `collision`
directory; the viewer's collision mode does not replace or modify those files.
