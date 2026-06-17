import copy
import os
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

PIPER_XML = PROJECT_ROOT / "mujoco_menagerie/agilex_piper/piper.xml"
AERO_XML = PROJECT_ROOT / "mujoco_menagerie/tetheria_aero_hand_open/right_hand.xml"
OUTPUT_DIR = PROJECT_ROOT / "models/piper_aero_hand"
OUTPUT_XML = OUTPUT_DIR / "Piper_aerohand.xml"

# Attach the Aero wrist close to the Piper wrist-roll axis, shifted slightly
# outward along link6 +Z so the palm base does not intersect the wrist shell.
# The stock parallel fingers link7/link8 were mounted farther out at local
# z=0.13503, which leaves a large empty span after those finger bases are
# removed.
PIPER_AERO_ATTACH_POS = [0.0, 0.0, 0.03]
PIPER_AERO_PALM_VISUAL_OFFSET = [0.0, 0.0, 0.0]

# Map the Aero wrist-to-fingertip direction (+Z_palm) onto Piper link6 +Z.
# The aero_wrist_site is still placed exactly at the Piper wrist-roll mounting
# point; this makes the wrist sit at the arm connection and the fingers point
# outward from the arm. The 180-degree roll around that axis is intentionally
# undone relative to the previous right-hand-looking orientation.
PIPER_AERO_PALM_ATTACH_QUAT = [0.0, 0.0, 0.0, 1.0]

from scripts.scenes.build_so101_aero_scene import (
    append_children,
    ensure_aero_landmark_sites,
    ensure_site,
    find_body,
    format_floats,
    mat_transpose_vec_mul,
    mat_vec_mul,
    parse_floats,
    quat_inv,
    quat_to_mat,
    remove_actuator_by_name,
    remove_asset_by_name,
    remove_body_by_name,
    remove_contact_excludes_with_body,
    rename_aero_classes,
)


def rel_path(path: Path) -> str:
    return os.path.relpath(path, OUTPUT_DIR).replace(os.sep, "/")


def set_mesh_paths(asset: ET.Element, asset_dir: Path) -> None:
    for mesh in asset.findall("mesh"):
        filename = mesh.get("file")
        if filename:
            mesh.set("file", rel_path(asset_dir / filename))


def remove_mesh_asset_by_filename(asset: ET.Element, filename: str) -> None:
    for child in list(asset):
        if child.tag == "mesh" and Path(child.get("file", "")).name == filename:
            asset.remove(child)


def append_nonlight_world_bodies(dst: ET.Element, src_worldbody: ET.Element) -> None:
    for child in list(src_worldbody):
        if child.tag == "body":
            dst.append(copy.deepcopy(child))


def remove_gripper_children(link6: ET.Element) -> None:
    for child in list(link6):
        if child.tag == "body" and child.get("name") in {"link7", "link8"}:
            link6.remove(child)


def remove_link6_gripper_base_geometry(link6: ET.Element) -> None:
    """Remove Piper's stock gripper base geometry from link6.

    Keep link6 itself, its inertial, and joint6. The long transverse link6 mesh
    visually belongs to the original parallel gripper mount, so it should not
    remain between the Piper wrist roll and the Aero palm.
    """
    for child in list(link6):
        if child.tag == "geom":
            link6.remove(child)


def remove_gripper_equality(equality: ET.Element | None) -> ET.Element | None:
    if equality is None:
        return None
    equality = copy.deepcopy(equality)
    for child in list(equality):
        if child.tag == "joint" and {child.get("joint1"), child.get("joint2")} & {"joint7", "joint8"}:
            equality.remove(child)
    return equality if list(equality) else None


def build_scene() -> ET.ElementTree:
    piper_tree = ET.parse(PIPER_XML)
    aero_tree = ET.parse(AERO_XML)
    piper_root = piper_tree.getroot()
    aero_root = aero_tree.getroot()

    output_root = ET.Element("mujoco", {"model": "piper_aero_hand"})
    ET.SubElement(output_root, "compiler", {"angle": "radian", "autolimits": "true"})
    ET.SubElement(
        output_root,
        "option",
        {
            "integrator": "implicitfast",
            "cone": "elliptic",
            "impratio": "10",
            "timestep": "0.005",
        },
    )
    ET.SubElement(output_root, "statistic", {"center": "0.25 0 0.25", "extent": "0.7", "meansize": "0.01"})
    visual = ET.SubElement(output_root, "visual")
    ET.SubElement(visual, "headlight", {"diffuse": "0.6 0.6 0.6", "ambient": "0.3 0.3 0.3", "specular": "0 0 0"})
    ET.SubElement(visual, "rgba", {"haze": "0.15 0.25 0.35 1"})
    ET.SubElement(visual, "global", {"azimuth": "120", "elevation": "-20"})

    for default in piper_root.findall("default"):
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
    piper_asset = copy.deepcopy(piper_root.find("asset"))
    aero_asset = copy.deepcopy(aero_root.find("asset"))
    set_mesh_paths(piper_asset, PIPER_XML.parent / "assets")
    set_mesh_paths(aero_asset, AERO_XML.parent / "assets")
    for mesh_name in ("link6", "link7", "link8"):
        remove_asset_by_name(piper_asset, "mesh", mesh_name)
    for mesh_filename in ("link6.stl", "link7.stl", "link8.stl"):
        remove_mesh_asset_by_filename(piper_asset, mesh_filename)
    remove_asset_by_name(aero_asset, "mesh", "tetheria_mount")
    append_children(asset, piper_asset)
    append_children(asset, aero_asset)

    piper_contact = copy.deepcopy(piper_root.find("contact"))
    if piper_contact is not None:
        output_root.append(piper_contact)
    aero_contact = copy.deepcopy(aero_root.find("contact"))
    remove_contact_excludes_with_body(aero_contact, "tetheria_mount")
    if aero_contact is not None:
        output_root.append(aero_contact)

    worldbody = ET.SubElement(output_root, "worldbody")
    ET.SubElement(worldbody, "light", {"pos": "0 0 1.5", "dir": "0 0 -1", "directional": "true"})
    ET.SubElement(worldbody, "geom", {"name": "floor", "size": "0 0 0.05", "type": "plane", "material": "groundplane"})
    ET.SubElement(worldbody, "camera", {"name": "side", "pos": "-0.45 0.65 0.45", "xyaxes": "-0.85 -0.53 0 0.26 -0.42 0.87"})

    piper_world = copy.deepcopy(piper_root.find("worldbody"))
    link6 = find_body(piper_world, "link6")
    if link6 is None:
        raise RuntimeError("Could not find Piper body named 'link6'")
    remove_gripper_children(link6)
    remove_link6_gripper_base_geometry(link6)

    aero_world = copy.deepcopy(aero_root.find("worldbody"))
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

    attach_pos = PIPER_AERO_ATTACH_POS
    palm_quat = parse_floats(direct_palm.get("quat"), "1 0 0 0")
    direct_quat = PIPER_AERO_PALM_ATTACH_QUAT
    wrist_site = direct_palm.find("site[@name='aero_wrist_site']")
    if wrist_site is None:
        raise RuntimeError("Could not find Aero Hand site named 'aero_wrist_site'")
    wrist_local = parse_floats(wrist_site.get("pos"), "0 0 0")
    wrist_offset = mat_vec_mul(quat_to_mat(direct_quat), wrist_local)
    direct_pos = [attach_pos[i] - wrist_offset[i] for i in range(3)]
    direct_pos = [direct_pos[i] + PIPER_AERO_PALM_VISUAL_OFFSET[i] for i in range(3)]
    direct_palm.set("pos", format_floats(direct_pos))
    direct_palm.set("quat", format_floats(direct_quat))
    ensure_site(link6, "piper_aero_attach_site", attach_pos)

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

    link6.append(direct_palm)
    append_nonlight_world_bodies(worldbody, piper_world)

    piper_equality = remove_gripper_equality(piper_root.find("equality"))
    if piper_equality is not None:
        output_root.append(piper_equality)

    tendon = copy.deepcopy(aero_root.find("tendon"))
    if tendon is not None:
        rename_aero_classes(tendon)
        output_root.append(tendon)

    actuator = ET.SubElement(output_root, "actuator")
    piper_actuator = copy.deepcopy(piper_root.find("actuator"))
    remove_actuator_by_name(piper_actuator, "gripper")
    append_children(actuator, piper_actuator)
    aero_actuator = copy.deepcopy(aero_root.find("actuator"))
    rename_aero_classes(aero_actuator)
    append_children(actuator, aero_actuator)

    sensor = copy.deepcopy(aero_root.find("sensor"))
    if sensor is not None:
        output_root.append(sensor)

    aero_equality = copy.deepcopy(aero_root.find("equality"))
    if aero_equality is not None:
        output_root.append(aero_equality)

    return ET.ElementTree(output_root)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tree = build_scene()
    ET.indent(tree, space="  ")
    tree.write(OUTPUT_XML, encoding="utf-8", xml_declaration=True)
    print(f"Wrote {OUTPUT_XML}")


if __name__ == "__main__":
    main()
