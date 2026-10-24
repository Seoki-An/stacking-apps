"""Shared rendering policy for opaque and translucent operations scenes."""


def stone_opacity(opacity, transparent_stones=False):
    """Cap all stones' opacity without making existing ghost overlays more opaque."""
    return min(opacity, 0.35) if transparent_stones else opacity


def scene_is_translucent(frame, environment_boxes, visibility, *, transparent_stones=False):
    """Account for every translucent layer drawn by the operations viewers."""
    if frame.target is not None and visibility.get("target", True):
        return True  # The target shell has opacity 0.12.
    if any(stone_opacity(value.opacity, transparent_stones) < 1.0 for value in frame.objects):
        return True
    if any(part.part_id.startswith(("scan_preview/", "motion_preview/"))
           for part in frame.robot_parts):
        return True
    return any(box.opacity < 1.0 and visibility.get(box.layer, False)
               for box in environment_boxes)


def configure_scene_passes(view, *, translucent, occlusion_radius, occlusion_bias):
    """Keep transparent surfaces depth-sorted, with outward-facing lighting.

    MSAA prevents reliable depth peeling on the supported VTK OpenGL backend.
    The SSAO delegate pass also bypasses the renderer's peeling path. Restore
    the standard pass for translucent frames; opaque frames retain SSAO.
    """
    view.render_window.SetMultiSamples(0)
    view.enable_depth_peeling(number_of_peels=0, occlusion_ratio=0.0)
    view.disable_shadows()
    if translucent:
        view.disable_ssao()
    else:
        view.enable_ssao(radius=occlusion_radius, bias=occlusion_bias,
                         kernel_size=128, blur=True)


def object_surface_culling(opacity):
    """Do not blend a closed stone's rear faces into its translucent exterior."""
    return "back" if opacity < 1.0 else False
