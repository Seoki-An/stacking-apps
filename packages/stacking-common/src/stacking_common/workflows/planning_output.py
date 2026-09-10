"""Publish replacement planning runs only after a successful offline process."""

from pathlib import Path
from uuid import uuid4

from stacking_common.artifacts import StackRunStore


class PlanningRunOutput:
    def __init__(self, destination, *, overwrite=False, source=None):
        self.destination = Path(destination).resolve()
        if source is not None:
            source = Path(source).resolve()
            if source == self.destination or (self.destination.exists() and source.samefile(self.destination)):
                raise ValueError("A resumed run must not overwrite its source")
        if self.destination.exists() and not self.destination.is_file():
            raise ValueError("Choose a .stack file path, not a directory")
        if self.destination.exists() and not overwrite:
            raise FileExistsError("Choose a new output path or enable Overwrite existing run")
        self.path = (self.destination.with_name(f"{self.destination.stem}-pending-{uuid4().hex}.stack")
                     if self.destination.exists() else self.destination)

    def finish(self, success):
        """Return the available result; failed/partial runs retain their own path."""
        if (self.path != self.destination and success and self.path.is_file()
                and StackRunStore.info(self.path).status == "complete"):
            self.path.replace(self.destination)
            self.path = self.destination
        return self.path
