"""Tabletop physical limits for shared active-floor placement checks."""

import numpy as np

from stacking_common.actions.floor import ActiveFloorPolicy


class TabletopPlacementSafety(ActiveFloorPolicy):
    """Use the resolved tabletop floor profile for search and execution."""

    support_tolerance = 0.02
    resettle_floor_translation_tolerance = 0.001
    resettle_floor_rotation_tolerance = np.deg2rad(0.5)
