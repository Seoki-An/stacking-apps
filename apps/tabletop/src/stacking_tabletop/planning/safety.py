"""Tabletop physical limits for shared active-floor placement checks."""

from stacking_common.actions.floor import ActiveFloorPolicy
from stacking_common.domain import ObjectInventory


class TabletopPlacementSafety(ActiveFloorPolicy):
    """Keep the legacy tabletop floor thresholds for search and execution."""

    support_tolerance = 0.02

    def __init__(self, inventory: ObjectInventory) -> None:
        super().__init__(
            inventory, bottom_tolerance=0.02, fill_ratio=0.90,
            grid_size=32, max_layers=8, minimum_contact_cells=1,
        )
