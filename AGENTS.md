# AGENTS.md — stacking-apps

## Architecture

- Keep reusable Python workflows in `packages/stacking-common`.
- Keep ROS, hardware control, sensor acquisition, and operator interaction in
  `apps/<platform>`.
- Common point-cloud code may process arrays and transforms, but must not
  import ROS messages, camera drivers, or platform packages.
- Common visualization renders canonical scenes, plans, meshes, and debug
  data. Live execution windows and platform overlays stay in the platform app.
- Solver schemas are common; resolved physical values are owned by each
  platform profile. Do not derive a platform profile from one global scale.

## Migration

- `stacking-tabletop` is the first platform migrated into this repository.
- Do not migrate its inherited excavator online execution, NUC workflows, or
  excavator ROS wrappers.
- Preserve behavior with tests before removing the original implementation.
- Do not add an excavator application until the shared contracts have been
  validated by the tabletop migration.

## Code Style

- Be as simple as possible.
- Prefer easily interpretable code.
- Do not add configuration options without a demonstrated platform need.
