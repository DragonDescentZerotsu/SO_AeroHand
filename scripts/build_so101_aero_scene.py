import copy
import os
import xml.etree.ElementTree as ET
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SO101_XML = PROJECT_ROOT / "third_party/SO-ARM100/Simulation/SO101/so101_new_calib.xml"
AERO_XML = PROJECT_ROOT / "mujoco_menagerie/tetheria_aero_hand_open/right_hand.xml"
OUTPUT_DIR = PROJECT_ROOT / "models/so101_aero_hand"
OUTPUT_XML = OUTPUT_DIR / "scene.xml"

# SO101's stock gripperframe is near the removed jaw tip, about 9.8 cm past the
# visible wrist-roll hardware. Use a closer fixed point so the mountless Aero
# palm visually sits on the arm instead of floating beyond the stripped gripper.
SO101_AERO_ATTACH_POS = [-0.0079, -0.000218121, -0.035]

AERO_CLASS_RENAMES = {
    "tetheria_rh": "aero_tetheria_rh",
    "visual": "aero_visual",
    "tip": "aero_tip",
    "thumb_tip": "aero_thumb_tip",
    "rot": "aero_rot",
    "pip": "aero_pip",
    "dip": "aero_dip",
    "thumb_cmc": "aero_thumb_cmc",
    "thumb_axl": "aero_thumb_axl",
    "thumb_mcp": "aero_thumb_mcp",
    "thumb_ipl": "aero_thumb_ipl",
    "cmc_spring": "aero_cmc_spring",
    "distal_spring": "aero_distal_spring",
    "mcp_spring": "aero_mcp_spring",
    "mcp_tendon": "aero_mcp_tendon",
    "flex_tendon": "aero_flex_tendon",
}


def rel_path(path: Path) -> str:
    return os.path.relpath(path, OUTPUT_DIR).replace(os.sep, "/")


def set_mesh_paths(asset: ET.Element, asset_dir: Path) -> None:
    for mesh in asset.findall("mesh"):
        filename = mesh.get("file")
        if filename:
            mesh.set("file", rel_path(asset_dir / filename))


def rename_aero_classes(elem: ET.Element) -> None:
    class_name = elem.get("class")
    if class_name in AERO_CLASS_RENAMES:
        elem.set("class", AERO_CLASS_RENAMES[class_name])
    childclass_name = elem.get("childclass")
    if childclass_name in AERO_CLASS_RENAMES:
        elem.set("childclass", AERO_CLASS_RENAMES[childclass_name])
    for child in list(elem):
        rename_aero_classes(child)


def remove_body_by_name(parent: ET.Element, body_name: str) -> bool:
    for child in list(parent):
        if child.tag == "body" and child.get("name") == body_name:
            parent.remove(child)
            return True
        if remove_body_by_name(child, body_name):
            return True
    return False


def find_body(parent: ET.Element, body_name: str) -> ET.Element | None:
    if parent.tag == "body" and parent.get("name") == body_name:
        return parent
    for child in list(parent):
        found = find_body(child, body_name)
        if found is not None:
            return found
    return None


def remove_actuator_by_name(actuator: ET.Element, actuator_name: str) -> None:
    for child in list(actuator):
        if child.get("name") == actuator_name:
            actuator.remove(child)


def remove_asset_by_name(asset: ET.Element, tag: str, name: str) -> None:
    for child in list(asset):
        if child.tag == tag and child.get("name") == name:
            asset.remove(child)


def remove_contact_excludes_with_body(contact: ET.Element | None, body_name: str) -> None:
    """Drop contact excludes that reference a body removed from the combined model."""
    if contact is None:
        return
    for child in list(contact):
        if child.tag == "exclude" and (child.get("body1") == body_name or child.get("body2") == body_name):
            contact.remove(child)


def parse_floats(text: str | None, default: str) -> list[float]:
    return [float(value) for value in (text or default).split()]


def format_floats(values: list[float]) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def quat_to_mat(q: list[float]) -> list[list[float]]:
    """Convert MuJoCo wxyz quaternion to a 3x3 rotation matrix."""
    w, x, y, z = q
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def mat_vec_mul(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(mat[row][col] * vec[col] for col in range(3)) for row in range(3)]


def mat_transpose_vec_mul(mat: list[list[float]], vec: list[float]) -> list[float]:
    return [sum(mat[row][col] * vec[row] for row in range(3)) for col in range(3)]


def quat_mul(a: list[float], b: list[float]) -> list[float]:
    """Multiply MuJoCo wxyz quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return [
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ]


def quat_inv(q: list[float]) -> list[float]:
    return [q[0], -q[1], -q[2], -q[3]]


def normalize_quat(q: list[float]) -> list[float]:
    norm = sum(value * value for value in q) ** 0.5
    return [value / norm for value in q]


def ensure_site(parent: ET.Element, name: str, pos: list[float], group: str = "4") -> ET.Element:
    site = parent.find(f"site[@name='{name}']")
    if site is None:
        site = ET.SubElement(parent, "site", {"name": name})
    site.set("pos", format_floats(pos))
    site.set("group", group)
    site.set("size", "0.003")
    site.set("rgba", "0.2 1 0.2 1")
    return site


def ensure_aero_landmark_sites(palm: ET.Element) -> None:
    """Add the 21 Quest/Aero retargeting landmarks to an upstream Aero Hand XML."""
    body_sites = {
        "palm": [
            ("aero_wrist_site", [0.002, -0.0015, 0.022], "1 0.2 0.2 1"),
            ("aero_wrist_lm", [0.002, -0.0015, 0.022], "0.2 0.8 1 1"),
        ],
        "right_index_proximal_link": [
            ("aero_index_proximal_site", [0, -0.003, 0.019], "1 0.2 0.2 1"),
            ("aero_index_proximal_lm", [0, -0.003, 0.019], "0.2 0.8 1 1"),
        ],
        "right_index_middle_link": [
            ("aero_index_intermediate_site", [0, -0.001, 0.006], "1 0.2 0.2 1"),
            ("aero_index_intermediate_lm", [0, -0.001, 0.006], "0.2 0.8 1 1"),
        ],
        "right_index_distal_link": [
            ("aero_index_distal_site", [0, -0.003, 0.014], "1 0.2 0.2 1"),
            ("aero_index_distal_lm", [0, -0.003, 0.014], "0.2 0.8 1 1"),
            ("aero_index_tip_site", [0, -0.003, 0.023], "1 0.2 0.2 1"),
            ("aero_index_tip_lm", [0, -0.003, 0.023], "0.2 0.8 1 1"),
        ],
        "right_middle_proximal_link": [
            ("aero_middle_proximal_site", [0, -0.003, 0.019], "1 0.2 0.2 1"),
            ("aero_middle_proximal_lm", [0, -0.003, 0.019], "0.2 0.8 1 1"),
        ],
        "right_middle_middle_link": [
            ("aero_middle_intermediate_site", [0, -0.001, 0.006], "1 0.2 0.2 1"),
            ("aero_middle_intermediate_lm", [0, -0.001, 0.006], "0.2 0.8 1 1"),
        ],
        "right_middle_distal_link": [
            ("aero_middle_distal_site", [0, -0.003, 0.014], "1 0.2 0.2 1"),
            ("aero_middle_distal_lm", [0, -0.003, 0.014], "0.2 0.8 1 1"),
            ("aero_middle_tip_site", [0, -0.003, 0.023], "1 0.2 0.2 1"),
            ("aero_middle_tip_lm", [0, -0.003, 0.023], "0.2 0.8 1 1"),
        ],
        "right_ring_proximal_link": [
            ("aero_ring_proximal_site", [0, -0.003, 0.019], "1 0.2 0.2 1"),
            ("aero_ring_proximal_lm", [0, -0.003, 0.019], "0.2 0.8 1 1"),
        ],
        "right_ring_middle_link": [
            ("aero_ring_intermediate_site", [0, -0.001, 0.006], "1 0.2 0.2 1"),
            ("aero_ring_intermediate_lm", [0, -0.001, 0.006], "0.2 0.8 1 1"),
        ],
        "right_ring_distal_link": [
            ("aero_ring_distal_site", [0, -0.003, 0.014], "1 0.2 0.2 1"),
            ("aero_ring_distal_lm", [0, -0.003, 0.014], "0.2 0.8 1 1"),
            ("aero_ring_tip_site", [0, -0.003, 0.023], "1 0.2 0.2 1"),
            ("aero_ring_tip_lm", [0, -0.003, 0.023], "0.2 0.8 1 1"),
        ],
        "right_pinky_proximal_link": [
            ("aero_little_proximal_site", [0, -0.003, 0.019], "1 0.2 0.2 1"),
            ("aero_little_proximal_lm", [0, -0.003, 0.019], "0.2 0.8 1 1"),
        ],
        "right_pinky_middle_link": [
            ("aero_little_intermediate_site", [0, -0.001, 0.006], "1 0.2 0.2 1"),
            ("aero_little_intermediate_lm", [0, -0.001, 0.006], "0.2 0.8 1 1"),
        ],
        "right_pinky_distal_link": [
            ("aero_little_distal_site", [0, -0.003, 0.014], "1 0.2 0.2 1"),
            ("aero_little_distal_lm", [0, -0.003, 0.014], "0.2 0.8 1 1"),
            ("aero_little_tip_site", [0, -0.003, 0.023], "1 0.2 0.2 1"),
            ("aero_little_tip_lm", [0, -0.003, 0.023], "0.2 0.8 1 1"),
        ],
        "right_t_link": [
            ("aero_thumb_metacarpal_site", [-0.01, 0.015, 0.005], "1 0.2 0.2 1"),
            ("aero_thumb_metacarpal_lm", [-0.01, 0.015, 0.005], "0.2 0.8 1 1"),
        ],
        "right_thumb_mcp_link": [
            ("aero_thumb_proximal_site", [0, 0, 0.014], "1 0.2 0.2 1"),
            ("aero_thumb_proximal_lm", [0, 0, 0.014], "0.2 0.8 1 1"),
        ],
        "right_thumb_proximal_link": [
            ("aero_thumb_distal_site", [0, -0.002, 0.012], "1 0.2 0.2 1"),
            ("aero_thumb_distal_lm", [0, -0.002, 0.012], "0.2 0.8 1 1"),
        ],
        "right_thumb_distal_link": [
            ("aero_thumb_tip_site", [0, -0.005, 0.027], "1 0.2 0.2 1"),
            ("aero_thumb_tip_lm", [0, -0.005, 0.027], "0.2 0.8 1 1"),
        ],
    }

    for body_name, sites in body_sites.items():
        body = palm if body_name == "palm" else find_body(palm, body_name)
        if body is None:
            raise RuntimeError(f"Could not find Aero Hand body named {body_name!r}")
        for name, pos, rgba in sites:
            site = ensure_site(body, name, pos)
            site.set("rgba", rgba)


def strip_so101_gripper_geometry(gripper_body: ET.Element) -> None:
    """Remove the stock SO101 gripper geometry before attaching the Aero Hand.

    Keep the wrist_roll joint, inertial, and gripperframe site as the attachment
    reference. Remove direct gripper geoms, camera mount, and moving-jaw child.
    """
    for child in list(gripper_body):
        if child.tag == "geom":
            gripper_body.remove(child)
        elif child.tag == "body" and child.get("name") in {"moving_jaw_so101_v1", "camera_mount"}:
            gripper_body.remove(child)


def append_children(dst: ET.Element, src: ET.Element | None) -> None:
    if src is None:
        return
    for child in list(src):
        dst.append(copy.deepcopy(child))


def build_scene() -> ET.ElementTree:
    so_tree = ET.parse(SO101_XML)
    aero_tree = ET.parse(AERO_XML)
    so_root = so_tree.getroot()
    aero_root = aero_tree.getroot()

    output_root = ET.Element("mujoco", {"model": "so101_aero_hand"})
    ET.SubElement(output_root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        output_root,
        "option",
        {
            "integrator": "implicitfast",
            "timestep": "0.005",
            "cone": "elliptic",
            "iterations": "10",
            "ls_iterations": "20",
            "impratio": "10",
        },
    )
    statistic = ET.SubElement(output_root, "statistic")
    statistic.set("center", "0.1 0 0.15")
    statistic.set("extent", "0.8")
    statistic.set("meansize", "0.01")

    visual = ET.SubElement(output_root, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "160", "elevation": "-20"})

    for default in so_root.findall("default"):
        output_root.append(copy.deepcopy(default))
    for default in aero_root.findall("default"):
        default_copy = copy.deepcopy(default)
        rename_aero_classes(default_copy)
        output_root.append(default_copy)

    asset = ET.SubElement(output_root, "asset")
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "skybox",
            "builtin": "gradient",
            "rgb1": "0.3 0.5 0.7",
            "rgb2": "0 0 0",
            "width": "512",
            "height": "3072",
        },
    )
    ET.SubElement(
        asset,
        "texture",
        {
            "type": "2d",
            "name": "groundplane",
            "builtin": "checker",
            "mark": "edge",
            "rgb1": "0.2 0.3 0.4",
            "rgb2": "0.1 0.2 0.3",
            "markrgb": "0.8 0.8 0.8",
            "width": "300",
            "height": "300",
        },
    )
    ET.SubElement(
        asset,
        "material",
        {
            "name": "groundplane",
            "texture": "groundplane",
            "texuniform": "true",
            "texrepeat": "5 5",
            "reflectance": "0.2",
        },
    )
    so_asset = copy.deepcopy(so_root.find("asset"))
    aero_asset = copy.deepcopy(aero_root.find("asset"))
    set_mesh_paths(so_asset, SO101_XML.parent / "assets")
    set_mesh_paths(aero_asset, AERO_XML.parent / "assets")
    remove_asset_by_name(aero_asset, "mesh", "tetheria_mount")
    append_children(asset, so_asset)
    append_children(asset, aero_asset)

    contact = copy.deepcopy(aero_root.find("contact"))
    remove_contact_excludes_with_body(contact, "tetheria_mount")
    if contact is not None:
        output_root.append(contact)

    worldbody = ET.SubElement(output_root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 3.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "size": "0 0 0.05", "pos": "0 0 0", "type": "plane", "material": "groundplane"})
    ET.SubElement(worldbody, "camera", {"name": "side", "pos": "-0.35 0.65 0.45", "xyaxes": "-0.88 -0.48 0 0.24 -0.44 0.86"})

    so_world = copy.deepcopy(so_root.find("worldbody"))
    aero_world = copy.deepcopy(aero_root.find("worldbody"))
    remove_body_by_name(so_world, "moving_jaw_so101_v1")
    so_gripper = find_body(so_world, "gripper")
    if so_gripper is None:
        raise RuntimeError("Could not find SO101 body named 'gripper'")
    strip_so101_gripper_geometry(so_gripper)
    ensure_site(so_gripper, "so101_aero_attach_site", SO101_AERO_ATTACH_POS)
    aero_mount = find_body(aero_world, "tetheria_mount")
    if aero_mount is None:
        raise RuntimeError("Could not find Aero Hand body named 'tetheria_mount'")
    aero_mount = copy.deepcopy(aero_mount)
    rename_aero_classes(aero_mount)
    palm = find_body(aero_mount, "palm")
    if palm is None:
        raise RuntimeError("Could not find Aero Hand body named 'palm'")
    direct_palm = copy.deepcopy(palm)
    direct_palm.set("childclass", "aero_tetheria_rh")
    ensure_aero_landmark_sites(direct_palm)

    # Remove the Tetheria mounting plate and use the exposed hand wrist/bottom
    # site as the new fixed connection point to the SO101 wrist roll frame.
    mount_quat = [0.5, -0.5, 0.5, 0.5]
    attach_pos = SO101_AERO_ATTACH_POS
    palm_quat = parse_floats(direct_palm.get("quat"), "1 0 0 0")
    direct_quat = normalize_quat(quat_mul(mount_quat, palm_quat))
    wrist_site = direct_palm.find("site[@name='aero_wrist_site']")
    if wrist_site is None:
        raise RuntimeError("Could not find Aero Hand site named 'aero_wrist_site'")
    wrist_local = parse_floats(wrist_site.get("pos"), "0 0 0")
    wrist_offset = mat_vec_mul(quat_to_mat(direct_quat), wrist_local)
    direct_pos = [attach_pos[i] - wrist_offset[i] for i in range(3)]
    direct_palm.set("pos", format_floats(direct_pos))
    direct_palm.set("quat", format_floats(direct_quat))

    grasp_site = aero_mount.find("site[@name='grasp_site']")
    if grasp_site is not None:
        direct_grasp = copy.deepcopy(grasp_site)
        grasp_mount = parse_floats(direct_grasp.get("pos"), "0 0 0")
        palm_mount = parse_floats(palm.get("pos"), "0 0 0")
        grasp_from_palm = [grasp_mount[i] - palm_mount[i] for i in range(3)]
        grasp_local = mat_transpose_vec_mul(quat_to_mat(palm_quat), grasp_from_palm)
        direct_grasp.set("pos", format_floats(grasp_local))
        direct_grasp.set("quat", format_floats(quat_inv(palm_quat)))
        direct_palm.append(direct_grasp)

    so_gripper.append(direct_palm)
    append_children(worldbody, so_world)

    tendon = copy.deepcopy(aero_root.find("tendon"))
    if tendon is not None:
        rename_aero_classes(tendon)
        output_root.append(tendon)

    actuator = ET.SubElement(output_root, "actuator")
    so_actuator = copy.deepcopy(so_root.find("actuator"))
    remove_actuator_by_name(so_actuator, "gripper")
    append_children(actuator, so_actuator)
    aero_actuator = copy.deepcopy(aero_root.find("actuator"))
    rename_aero_classes(aero_actuator)
    append_children(actuator, aero_actuator)

    sensor = copy.deepcopy(aero_root.find("sensor"))
    if sensor is not None:
        output_root.append(sensor)

    equality = copy.deepcopy(aero_root.find("equality"))
    if equality is not None:
        output_root.append(equality)

    return ET.ElementTree(output_root)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tree = build_scene()
    ET.indent(tree, space="  ")
    tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {OUTPUT_XML}")


if __name__ == "__main__":
    main()
