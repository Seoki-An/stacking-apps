"""Tabletop floor geometry and released-pose support tolerance."""

from stacking_common.actions.floor import ActiveFloorPolicy


class TabletopPlacementSafety(ActiveFloorPolicy):
    """Track sampling layers and expose the physical release tolerance."""

    support_tolerance = 0.02
