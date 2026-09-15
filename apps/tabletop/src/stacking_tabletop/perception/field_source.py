"""Resolve operator-selected field workflow and result folders to raw captures."""

import json
from pathlib import Path


def resolve_field_capture_directory(directory: str | Path) -> Path:
    """Accept a capture, workflow root, or identification result directory."""
    visited = set()

    def resolve(selected):
        root = Path(selected).expanduser().resolve()
        if root in visited:
            raise ValueError(f'cyclic field capture reference: {root}')
        visited.add(root)
        if not root.is_dir():
            raise ValueError(f'field capture directory does not exist: {root}')
        if (root / 'field_scan.json').is_file():
            return root
        if root.name == 'pose_identification' and (root.parent / 'field_scan.json').is_file():
            return root.parent

        manifest = root / 'field_workflow.json'
        if manifest.is_file():
            data = read_report(manifest)
            if data.get('pose_input_dir'):
                source = Path(data['pose_input_dir']).expanduser()
                # Workflow captures are immediate children. Prefer the local
                # copy when a complete workflow has been moved or copied.
                local = root / source.name
                return resolve(local if (local / 'field_scan.json').is_file()
                               else source if source.is_absolute() else root / source)
            if len(data.get('cameras', [])) > 1:
                if (root / 'merged' / 'field_scan.json').is_file():
                    return root / 'merged'
                raise ValueError('two-camera workflow has no merged capture; '
                                 'finish fusion or explicitly select a single camera capture')

        for report in (root / 'stone_poses.json', root / 'pose_identification' / 'stone_poses.json'):
            if report.is_file():
                data = read_report(report)
                if data.get('source_directory'):
                    source = Path(data['source_directory']).expanduser()
                    return resolve(source if source.is_absolute() else report.parent / source)

        if (root / 'merged' / 'field_scan.json').is_file():
            return root / 'merged'
        cameras = [root / name for name in ('primary', 'secondary')
                   if (root / name / 'field_scan.json').is_file()]
        if len(cameras) == 1:
            return cameras[0]
        if len(cameras) > 1:
            raise ValueError('multiple camera captures found without a merged capture; '
                             'select primary or secondary explicitly')
        raise ValueError(f'no field capture found in {root}; select a capture containing '
                         'field_scan.json, its workflow root, or its pose_identification folder')

    return resolve(directory)


def read_report(path):
    try:
        data = json.loads(path.read_text())
        if not isinstance(data, dict):
            raise ValueError('expected a JSON object')
        for key in ('pose_input_dir', 'source_directory'):
            if key in data and data[key] is not None and not isinstance(data[key], str):
                raise ValueError(f'{key} must be a path string')
        return data
    except (OSError, ValueError) as error:
        raise ValueError(f'cannot read field capture reference {path}: {error}') from error
