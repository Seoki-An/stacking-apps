"""Locations of one platform's explicitly resolved solver profiles."""

from dataclasses import dataclass
from pathlib import Path


_SOLVERS = ("simulation", "posegen", "grasp", "motion")


@dataclass(frozen=True)
class SolverProfileFiles:
    """A complete platform profile without implicit geometric scaling."""

    simulation: Path
    posegen: Path
    grasp: Path
    motion: Path

    @classmethod
    def from_directory(cls, directory: str | Path) -> "SolverProfileFiles":
        root = Path(directory)
        paths = {name: root / f"{name}.yml" for name in _SOLVERS}
        missing = [str(path) for path in paths.values() if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "missing solver profile files: " + ", ".join(missing)
            )
        return cls(**paths)
