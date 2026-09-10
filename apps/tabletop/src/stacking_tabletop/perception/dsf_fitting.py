"""Tabletop compatibility names for shared DSF fitting."""

from stacking_common.reconstruction import (
    DsfFitResult,
    DsfFitter,
    distribute_vertex_budget,
)


TabletopDsfFitter = DsfFitter

__all__ = [
    "DsfFitResult",
    "TabletopDsfFitter",
    "distribute_vertex_budget",
]
