"""Sequential keyboard viewer for the 100-stone augmented surface/DSF catalog.

    python apps/excavator/scripts/view_augmented_objects.py

Right/Space/N: next; Left/P: previous; Home/End: first/last; R: reset;
S: screenshot; Q/Esc: close. Mouse orbit/zoom is shared by all three views.
"""

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "packages/stacking-common/src"))

from stacking_common.visualization.asset_fit_viewer import AssetFitViewer


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--catalog", type=Path, default=ROOT / "apps/excavator/assets/objects_augmented")
    parser.add_argument("--start", type=int, help="initial stone ID; defaults to the first synthetic stone")
    parser.add_argument("--screenshot", type=Path, help="export one headless view instead of opening a window")
    args = parser.parse_args()
    viewer = AssetFitViewer(args.catalog, start_id=args.start, off_screen=args.screenshot is not None,
                           screenshot_directory=ROOT / ".analysis/objects-augmented/viewer")
    if args.screenshot:
        args.screenshot.parent.mkdir(parents=True, exist_ok=True)
        viewer.plot.screenshot(args.screenshot)
        viewer.close()
        print(args.screenshot)
    else:
        print(f"Reviewing {len(viewer.ids)} stones, starting at {viewer.object_id}.", flush=True)
        print("Right/Space/N: next; Left/P: previous; Home/End: first/last; R: reset; S: save; Q: close.", flush=True)
        viewer.show()


if __name__ == "__main__":
    main()
