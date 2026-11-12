# Augmented simulation stones

100 distinct stones: the 49 real commissioned OBJ/PLY pairs copied byte for
byte, plus 51 deformed and freshly decomposed/fitted variants (IDs 59–109).
Original IDs are preserved, including gaps in their numbering.

Use `apps/excavator/config/inventory_augmented.yml` to load this catalog.
Synthetic variants inherit their parent's friction and density. Their masses
and inertias are recomputed from the deformed watertight surface, with both
PLY and DSF coordinates centered at the new center of mass.

`manifest.json` records completion, identities, source hashes, seeds, and fitting
settings. Synthetic `model_ID.json` files retain each deformation and fit.
`validation.json` contains independent surface/DSF overlap, topology, inertial,
and native-loader checks. Display meshes are the PLY surfaces; collision models
are freshly fitted DSFs, rather than transformed copies of the parent DSF.

Reproduce using `apps/excavator/scripts/build_augmented_objects.py`.
Audit with `apps/excavator/scripts/validate_augmented_objects.py`. All 100 models
passed native loading; synthetic surface/DSF voxel IoU is 0.910 on average,
with minimum 0.862. The original 49 model pairs remain byte-identical.
See [the augmentation guide](../../../../docs/excavator-object-augmentation.md)
for details and the opt-in simulation workflow.
