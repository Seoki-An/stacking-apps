"""Export original inventory meshes on invisible 4x4 grids, viewed from above.

    python tests/inventory_paper_figures.py --output .analysis/inventory-paper

Like stacking-planner/scripts/test/compare_stone_sizes.py, sort by original
bounding-box volume and align principal axes, without normalizing stone sizes.
The default is the excavator inventory, split into 16-stone pages. An optional
--object-ids list selects a subset; --show-labels adds stone IDs. Requires
PyVista and Pillow; use xvfb-run -a on X-only VTK installations.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "packages/stacking-common/src"))

from stacking_common.profiles.inventory import load_inventory_profile
from stacking_common.visualization.dsf import object_surface_path
from stacking_common.visualization.inventory import inventory_grid, render_inventory_pages


def main():
    import pyvista as pv

    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--inventory", type=Path, default=ROOT / "apps/excavator/config/inventory.yml")
    parser.add_argument("--object-ids", type=int, nargs="+", help="subset of IDs from the inventory")
    parser.add_argument("--show-labels", action="store_true", help="add stone IDs below each mesh")
    parser.add_argument("--output", type=Path, required=True, help="new directory for PNG/PDF pages and JSON")
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    profile = load_inventory_profile(args.inventory)
    ids = profile.object_ids if args.object_ids is None else tuple(args.object_ids)
    if len(set(ids)) != len(ids) or not set(ids).issubset(profile.object_ids):
        raise ValueError("object IDs must be unique members of the selected inventory")
    meshes, sources = {}, {}
    for object_id in ids:
        path = object_surface_path(ROOT / profile.model_directory / f"model_{object_id}.obj", "ply")
        meshes[object_id] = pv.read(path)
        sources[str(object_id)] = dict(path=str(path.resolve()), sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    records, cell = inventory_grid(meshes)
    args.output.mkdir(parents=True, exist_ok=False)
    metadata = dict(inventory=str(args.inventory.resolve()),
        inventory_sha256=hashlib.sha256(args.inventory.read_bytes()).hexdigest(),
        meshes=sources, grid=[4, 4], cell_spacing_m=cell, ordering="ascending original AABB volume, top-left row first",
        orientation="principal axes: largest variance X, second Y, smallest Z; bottom at Z=0",
        camera="orthographic +Z; +Y up; parallel scale = 2.05 * cell spacing",
        background="white", stone_color=[.58, .58, .58], opacity=1.,
        grid_visible=False, labels=args.show_labels, rescaling=False,
        window_size=[2400, 2400], dpi=300, pdf_content="raster",
        stones=[{**entry, "transform": entry["transform"].tolist()} for entry in records])
    (args.output / "inventory_layout.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(f"Rendering {len(ids)} stones on {records[-1]['page']} pages; spacing {cell:.3f} m", flush=True)
    for path in render_inventory_pages(args.output, meshes, records, cell, show_labels=args.show_labels):
        print(path, flush=True)


if __name__ == "__main__":
    main()
