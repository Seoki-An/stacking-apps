"""Run recorded excavator perception on the desktop without ROS or a NUC."""

import argparse
import json
from pathlib import Path
import numpy as np

from stacking_excavator.acquisition.capture import CropRequest, digest
from stacking_excavator.acquisition.recorded import import_recording
from stacking_excavator.acquisition.legacy import import_legacy_log
from stacking_excavator.acquisition.storage import atomic_json, prepare_crop, file_hash
from .pipeline import DesktopPerception
from .review import ReplayReviewStore


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="recorded sensor files plus explicit metadata sidecar")
    importer.add_argument("sidecar", type=Path)
    importer.add_argument("output", type=Path)
    legacy = commands.add_parser("import-legacy", help="import field/scene raw logs or in-hand geometry-frame logs")
    legacy.add_argument("source", type=Path)
    legacy.add_argument("output", type=Path)
    legacy.add_argument("--kind", choices=("field", "inhand", "scene"), required=True)
    legacy.add_argument("--capture-id", required=True)
    legacy.add_argument("--revision", type=int, required=True)
    legacy.add_argument("--repository", type=Path, default=Path.cwd())
    crop = commands.add_parser("crop", help="derive a new crop from retained raw files")
    crop.add_argument("raw", type=Path)
    crop.add_argument("request", type=Path, help="JSON CropRequest fields")
    crop.add_argument("output", type=Path)
    crop.add_argument("--product-id", required=True)
    identify = commands.add_parser("identify", help="write a proposal; does not accept scene changes")
    identify.add_argument("product", type=Path)
    identify.add_argument("priors", type=Path, help="JSON with scene_revision, ground_height, poses by object ID")
    identify.add_argument("output", type=Path, help="new proposal directory")
    identify.add_argument("--repository", type=Path, default=Path.cwd())
    stage = commands.add_parser("stage", help="stage a proposal for replay-session review")
    stage.add_argument("database", type=Path)
    stage.add_argument("proposal", type=Path)
    stage.add_argument("--initial-revision", type=int, default=0)
    decide = commands.add_parser("decide", help="accept/reject a staged replay proposal")
    decide.add_argument("database", type=Path)
    decide.add_argument("token")
    decide.add_argument("decision", choices=("accept", "reject"))
    decide.add_argument("--revision", type=int, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "import":
            result = import_recording(args.sidecar, args.output)
        elif args.command == "import-legacy":
            result = import_legacy_log(args.source, args.output, repository=args.repository,
                kind=args.kind, capture_id=args.capture_id, scene_revision=args.revision)
        elif args.command == "crop":
            request = CropRequest(**json.loads(args.request.read_text()))
            result = prepare_crop(args.raw, args.output, args.product_id, request,
                                  maximum_timestamp_skew_ns=50_000_000)
        elif args.command == "stage":
            token = ReplayReviewStore(args.database, initial_revision=args.initial_revision).stage(
                json.loads(args.proposal.read_text()))
            result = {"sha256": token, "next_action": "decide"}
        elif args.command == "decide":
            result = ReplayReviewStore(args.database).decide(args.token,
                accept=args.decision == "accept", expected_revision=args.revision)
            print(json.dumps(result))
            return 0
        else:
            import open3d as o3d
            request = json.loads(args.priors.read_text())
            args.output.mkdir(parents=True, exist_ok=False)
            result, cloud = DesktopPerception(args.repository).identify(args.product,
                {int(k): v for k, v in request["poses"].items()},
                expected_revision=request["scene_revision"], ground_height=request["ground_height"])
            if len(cloud):
                pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.array(cloud.points, copy=True)))
                if not o3d.io.write_point_cloud(str(args.output / "processed.ply"), pcd):
                    raise OSError("could not export processed point cloud")
                result.pop("sha256")
                result["processed_cloud_sha256"] = file_hash(args.output / "processed.ply")
                result["sha256"] = digest(result)
            atomic_json(args.output / "proposal.json", result)
        print(json.dumps({k: result[k] for k in ("sha256", "valid", "next_action") if k in result}))
    except (ValueError, OSError, RuntimeError, KeyError, TypeError, ImportError) as error:
        parser.exit(1, f"perception error: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
