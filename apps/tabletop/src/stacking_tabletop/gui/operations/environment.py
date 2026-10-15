"""Cell geometry in world coordinates or the search's stack frame."""

from dataclasses import dataclass

from stacking_common.planning.frames import inverse_pose

from stacking_tabletop.config import TabletopSceneBox


@dataclass(frozen=True)
class TransformedSceneBox(TabletopSceneBox):
    """A box whose native bounds are transformed into the displayed frame."""

    pose: tuple[float, ...]


def search_scene_boxes(planning, visualization):
    # The search ground is a reference plane at stack z=0, not the world floor.
    ground = tuple(box for box in visualization.boxes if box.layer == "ground")
    stack_T_world = tuple(inverse_pose(planning.world_T_stack.parent_T_child))
    return ground + tuple(
        TransformedSceneBox(
            box.component_id, box.layer, box.minimum, box.maximum,
            box.color, box.opacity, stack_T_world,
        )
        for box in planning_scene_boxes(planning)
    )


def planning_scene_boxes(planning):
    boxes = []
    for obstacle in planning.obstacles:
        table = obstacle.obstacle_id in {"pick_table", "place_table"}
        boxes.append(TabletopSceneBox(
            obstacle.obstacle_id,
            "support" if table else "obstacle",
            obstacle.minimum,
            obstacle.maximum,
            (.48, .34, .22) if table else (1., 1., 1.),
            1.0 if table else .2,
        ))
    return tuple(boxes)
