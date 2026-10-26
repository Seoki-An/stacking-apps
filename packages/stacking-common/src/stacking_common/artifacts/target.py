"""Versioned target display geometry and resolved-profile replay boundary."""

from dataclasses import asdict

from stacking_common.domain import ConvexGeometry, DirectionalTaper, TargetComponent
from stacking_common.visualization.contracts import CompositeTargetVolume, TargetVolume, target_display


def encode_target(value):
    if isinstance(value, CompositeTargetVolume):
        return {
            "kind": "composite", "schema_version": 1,
            "structure_offset": value.structure_offset.tolist(),
            "components": [{"id": item.component_id, "pose": item.pose.tolist(),
                            "points": item.geometry.points.tolist(), "faces": item.geometry.faces.tolist(),
                            "sharpness": item.geometry.sharpness} for item in value.components],
        }
    # Keep the historical layout byte-for-byte compatible with context hashes.
    return {"origin": value.origin.tolist(), "width": value.width,
            "length": value.length, "height": value.height,
            "structure_offset": value.structure_offset.tolist(), "taper": asdict(value.taper)}


def decode_target(values):
    kind = values.get("kind", "single")
    offset = values.get("structure_offset", (0., 0., 0.))
    if kind == "composite":
        if values.get("schema_version") != 1:
            raise ValueError("unsupported composite target schema version")
        return CompositeTargetVolume(tuple(TargetComponent(
            item["id"], ConvexGeometry(item["points"], item["faces"], sharpness=item["sharpness"]),
            item["pose"]) for item in values["components"]), offset)
    if kind != "single":
        raise ValueError(f"unsupported target kind: {kind}")
    return TargetVolume(values["origin"], values["width"], values["length"], values["height"],
                        offset, DirectionalTaper(**values.get("taper", {})))


def target_from_context(context, artifact_target=None):
    """Use recorded geometry, never the operator's current inventory YAML."""
    from dataclasses import replace
    from stacking_common.profiles.inventory import target_profile_from_record

    if "target_geometry" in context.profiles:
        result = decode_target(context.profiles["target_geometry"])
    else:
        profile = target_profile_from_record(context.profiles["inventory_profile"]["target"])
        result = target_display(profile.geometry())
    return (result if artifact_target is None else
            replace(result, structure_offset=artifact_target.structure_offset))
