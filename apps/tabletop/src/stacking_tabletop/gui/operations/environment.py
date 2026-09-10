"""World-frame cell geometry shared by plan replay and live execution."""

from stacking_tabletop.config import TabletopSceneBox


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
