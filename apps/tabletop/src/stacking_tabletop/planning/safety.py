"""Tabletop physical limits for shared active-floor placement checks."""

from stacking_common.actions.floor import ActiveFloorPolicy


class TabletopPlacementSafety(ActiveFloorPolicy):
    """Use the resolved tabletop floor profile for search and execution."""

    support_tolerance = 0.02
