"""Utilities for composing MuJoCo task scenes from reusable model instances."""

from __future__ import annotations

import copy
import math
import os
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def resolve_project_path(path_text: str | Path, *, base_dir: Path | None = None) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = (base_dir or PROJECT_ROOT) / path
    return path.resolve()


def rel_path(path: Path, output_dir: Path) -> str:
    return os.path.relpath(path, output_dir).replace(os.sep, "/")


def format_floats(values: list[float]) -> str:
    return " ".join(f"{float(value):.12g}" for value in values)


def yaw_quat(yaw: float) -> list[float]:
    half = 0.5 * yaw
    return [math.cos(half), 0.0, 0.0, math.sin(half)]


def quat_mul(a: list[float], b: list[float]) -> list[float]:
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def rotate_z(vec: list[float], yaw: float) -> list[float]:
    c = math.cos(yaw)
    s = math.sin(yaw)
    x, y, z = vec
    return [c * x - s * y, s * x + c * y, z]


def find_child(parent: ET.Element, tag: str) -> ET.Element:
    child = parent.find(tag)
    if child is None:
        child = ET.SubElement(parent, tag)
    return child


def append_asset_model(asset: ET.Element, *, name: str, file_path: Path, output_dir: Path) -> None:
    model = ET.SubElement(asset, "model")
    model.set("name", name)
    model.set("file", rel_path(file_path, output_dir))
    model.set("content_type", "text/xml")


def resolve_nested_body(parent: ET.Element, path: list[str]) -> ET.Element:
    current = parent
    for name in path:
        found = None
        for child in current.findall("body"):
            if child.get("name") == name:
                found = child
                break
        if found is None:
            raise ValueError(f"Could not find body path component {name!r} under {current.get('name', current.tag)!r}")
        current = found
    return current


def set_base_pose(root: ET.Element, pose: dict[str, Any]) -> None:
    body_path = pose.get("body_path", [])
    if not body_path:
        return
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("Base model has no worldbody")
    body = resolve_nested_body(worldbody, [str(value) for value in body_path])
    if "pos" in pose:
        body.set("pos", format_floats([float(value) for value in pose["pos"]]))
    if "quat" in pose:
        body.set("quat", format_floats([float(value) for value in pose["quat"]]))


def set_viewer_options(root: ET.Element, viewer: dict[str, Any]) -> None:
    statistic = viewer.get("statistic", {})
    if statistic:
        elem = find_child(root, "statistic")
        if "center" in statistic:
            elem.set("center", format_floats([float(value) for value in statistic["center"]]))
        for key in ("extent", "meansize"):
            if key in statistic:
                elem.set(key, f"{float(statistic[key]):.12g}")

    global_options = viewer.get("global", {})
    if global_options:
        visual = find_child(root, "visual")
        elem = find_child(visual, "global")
        for key in ("azimuth", "elevation"):
            if key in global_options:
                elem.set(key, f"{float(global_options[key]):.12g}")


def make_empty_scene_root(name: str) -> ET.Element:
    root = ET.Element("mujoco", {"model": name})
    ET.SubElement(root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        root,
        "option",
        {"integrator": "implicitfast", "cone": "elliptic", "impratio": "10", "timestep": "0.005"},
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(
        visual,
        "headlight",
        {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"},
    )
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(root, "asset")
    ET.SubElement(root, "worldbody")
    return root


def rewrite_asset_file_paths(root: ET.Element, *, source_dir: Path, output_dir: Path) -> None:
    asset = root.find("asset")
    if asset is None:
        return
    for elem in asset:
        filename = elem.get("file")
        if not filename:
            continue
        file_path = Path(filename).expanduser()
        if not file_path.is_absolute():
            file_path = (source_dir / file_path).resolve()
        elem.set("file", rel_path(file_path, output_dir))


def add_free_object(
    worldbody: ET.Element,
    *,
    object_id: str,
    model_name: str,
    body_name: str,
    prefix: str,
    pos: list[float],
    quat: list[float],
    freejoint: bool,
) -> ET.Element:
    wrapper = ET.SubElement(worldbody, "body")
    wrapper.set("name", object_id)
    wrapper.set("pos", format_floats(pos))
    wrapper.set("quat", format_floats(quat))
    if freejoint:
        ET.SubElement(wrapper, "freejoint", {"name": f"{object_id}_free"})
    ET.SubElement(wrapper, "attach", {"model": model_name, "body": body_name, "prefix": prefix})
    return wrapper


def add_static_model(
    worldbody: ET.Element,
    *,
    object_id: str,
    model_name: str,
    body_name: str,
    prefix: str,
    pos: list[float],
    quat: list[float],
) -> ET.Element:
    frame = ET.SubElement(worldbody, "frame")
    frame.set("name", object_id)
    frame.set("pos", format_floats(pos))
    frame.set("quat", format_floats(quat))
    ET.SubElement(frame, "attach", {"model": model_name, "body": body_name, "prefix": prefix})
    return frame


def iter_scene_items(config: dict[str, Any], section: str) -> list[dict[str, Any]]:
    items = config.get(section, [])
    if not isinstance(items, list):
        raise ValueError(f"{section!r} must be a list")
    return items


def find_scene_item(config: dict[str, Any], *, section: str, item_id: str) -> dict[str, Any]:
    for item in iter_scene_items(config, section):
        if str(item.get("id")) == item_id:
            return item
    raise ValueError(f"layout group references missing item {section}.{item_id}")


def apply_layout_groups(config: dict[str, Any]) -> None:
    """Apply deterministic group transforms while preserving member-relative poses."""

    for group in config.get("layout_groups", []):
        source_origin = [
            float(value)
            for value in group.get("source_origin", [0.0, 0.0, 0.0])
        ]
        target_origin = [float(value) for value in group.get("target_origin", source_origin)]
        yaw = float(group.get("yaw", 0.0))
        q_yaw = yaw_quat(yaw)
        for member in group.get("members", []):
            section = str(member["section"])
            item_id = str(member["id"])
            item = find_scene_item(config, section=section, item_id=item_id)
            init = item.setdefault("init", {})
            pos = [float(value) for value in init.get("pos", [0.0, 0.0, 0.0])]
            quat = [float(value) for value in init.get("quat", [1.0, 0.0, 0.0, 0.0])]
            rel = [pos[i] - source_origin[i] for i in range(3)]
            rotated = rotate_z(rel, yaw)
            init["pos"] = [target_origin[i] + rotated[i] for i in range(3)]
            init["quat"] = quat_mul(q_yaw, quat)


def build_task_scene(config: dict[str, Any], *, config_dir: Path | None = None) -> Path:
    output_model = resolve_project_path(config["output_model"], base_dir=config_dir)
    output_dir = output_model.parent
    scene_name = str(config.get("name", "task_scene"))
    config = copy.deepcopy(config)
    apply_layout_groups(config)

    base_model_value = config.get("base_model")
    if base_model_value:
        base_model = resolve_project_path(base_model_value, base_dir=config_dir)
        tree = ET.parse(base_model)
        root = tree.getroot()
        root.set("model", str(config.get("name", root.get("model", "task_scene"))))
        set_base_pose(root, config.get("base_pose", {}))
        rewrite_asset_file_paths(root, source_dir=base_model.parent, output_dir=output_dir)
    else:
        root = make_empty_scene_root(scene_name)
        tree = ET.ElementTree(root)

    set_viewer_options(root, config.get("viewer", {}))

    asset = find_child(root, "asset")
    worldbody = find_child(root, "worldbody")

    for item in config.get("model_instances", []):
        object_id = str(item["id"])
        model_name = str(item.get("model_name", object_id))
        source = resolve_project_path(item["source"], base_dir=config_dir)
        append_asset_model(asset, name=model_name, file_path=source, output_dir=output_dir)
        init = item.get("init", {})
        add_static_model(
            worldbody,
            object_id=object_id,
            model_name=model_name,
            body_name=str(item["body"]),
            prefix=str(item.get("prefix", f"{object_id}/")),
            pos=[float(value) for value in init.get("pos", [0.0, 0.0, 0.0])],
            quat=[float(value) for value in init.get("quat", [1.0, 0.0, 0.0, 0.0])],
        )

    for item in config.get("static_models", []):
        object_id = str(item["id"])
        model_name = str(item.get("model_name", object_id))
        source = resolve_project_path(item["source"], base_dir=config_dir)
        append_asset_model(asset, name=model_name, file_path=source, output_dir=output_dir)
        init = item.get("init", {})
        add_static_model(
            worldbody,
            object_id=object_id,
            model_name=model_name,
            body_name=str(item["body"]),
            prefix=str(item.get("prefix", f"{object_id}/")),
            pos=[float(value) for value in init.get("pos", [0.0, 0.0, 0.0])],
            quat=[float(value) for value in init.get("quat", [1.0, 0.0, 0.0, 0.0])],
        )

    for item in config.get("objects", []):
        object_id = str(item["id"])
        model_name = str(item.get("model_name", object_id))
        source = resolve_project_path(item["source"], base_dir=config_dir)
        append_asset_model(asset, name=model_name, file_path=source, output_dir=output_dir)
        init = item.get("init", {})
        add_free_object(
            worldbody,
            object_id=object_id,
            model_name=model_name,
            body_name=str(item["body"]),
            prefix=str(item.get("prefix", f"{object_id}/")),
            pos=[float(value) for value in init.get("pos", [0.0, 0.0, 0.0])],
            quat=[float(value) for value in init.get("quat", [1.0, 0.0, 0.0, 0.0])],
            freejoint=bool(item.get("freejoint", False)),
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(output_model, encoding="utf-8", xml_declaration=True)
    return output_model


def load_yaml_config(path: Path) -> dict[str, Any]:
    import yaml

    with path.open("r", encoding="utf-8") as f:
        config = yaml.safe_load(f) or {}
    if not isinstance(config, dict):
        raise ValueError(f"Scene config must contain a YAML mapping: {path}")
    return copy.deepcopy(config)
