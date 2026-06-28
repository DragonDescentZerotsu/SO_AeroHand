#!/usr/bin/env python3
"""Render Blender liquid demos with the real AutoBio pipette tip and centrifuge tube."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
import trimesh


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.blender_render import BlenderRenderConfig, prepare_blender_render  # noqa: E402
from aero_tasks.lerobot_export import RenderCameraSpec  # noqa: E402
from aero_tasks.liquid import ContainerState, PipetteLiquidController, PipetteTipState, PlungerModel  # noqa: E402
from aero_tasks.liquid_meshplane import MeshPlaneGeometry  # noqa: E402
from scripts.debug.demo_meshplane_pipette_centrifuge_liquid import (  # noqa: E402
    AUTOBIO_ROOT,
    DEFAULT_BASE_MODEL,
    GRAVITY_WORLD,
    LIQUID_COLOR,
    TIP_SITE_LOCAL_Z,
    TUBE_MESH,
    TUBE_POS,
    DemoFrame,
    exchange_frames,
    interp,
    quat_to_matrix,
    tube_quat,
)


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/debug_rollouts/blender_real_pipette_centrifuge_liquid"
TIP_MESH = AUTOBIO_ROOT / "assets/container/tip_200ul_vis/visual.obj"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--render", action="store_true", help="Run Blender after preparing the trajectory and manifests.")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--initial-tube-volume-ul", type=float, default=900.0)
    parser.add_argument("--tip-capacity-ul", type=float, default=200.0)
    parser.add_argument("--stroke-volume-ul", type=float, default=200.0)
    parser.add_argument("--hold-frames", type=int, default=18)
    parser.add_argument("--motion-frames", type=int, default=72)
    parser.add_argument("--move-frames", type=int, default=36)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def find_direct_body(worldbody: ET.Element, name: str) -> ET.Element:
    for child in list(worldbody):
        if child.tag == "body" and child.get("name") == name:
            return child
    raise ValueError(f"Missing direct worldbody body {name!r}")


def prepare_model_xml(base_model: Path, out_dir: Path) -> Path:
    tree = ET.parse(base_model)
    root = tree.getroot()
    asset = root.find("asset")
    worldbody = root.find("worldbody")
    if asset is None or worldbody is None:
        raise ValueError(f"Expected <asset> and <worldbody> in {base_model}")

    for body_name in ("source_tube", "target_well"):
        try:
            worldbody.remove(find_direct_body(worldbody, body_name))
        except ValueError:
            pass

    pipette = find_direct_body(worldbody, "pipette")
    original_pos = pipette.get("pos", "0 0 0")
    original_quat = pipette.get("quat", "1 0 0 0")
    pipette.set("pos", "0 0 0")
    pipette.set("quat", "1 0 0 0")
    worldbody.remove(pipette)

    pipette_wrapper = ET.Element("body", {"name": "pipette_free_body", "pos": original_pos, "quat": original_quat})
    pipette_wrapper.append(ET.Element("freejoint", {"name": "pipette_free"}))
    pipette_wrapper.append(pipette)
    worldbody.append(pipette_wrapper)

    asset.append(ET.Element("mesh", {"name": "centrifuge_tube_visual", "file": str(TUBE_MESH), "scale": "0.001 0.001 0.001"}))
    asset.append(ET.Element("material", {"name": "centrifuge_tube_clear", "rgba": "0.58 0.86 1.0 0.32", "specular": "0.65", "shininess": "0.65"}))

    tube = ET.Element("body", {"name": "centrifuge_tube", "pos": "0 0 0", "quat": "1 0 0 0"})
    tube.append(ET.Element("freejoint", {"name": "centrifuge_tube_free"}))
    tube.append(
        ET.Element(
            "geom",
            {
                "name": "centrifuge_tube_visual",
                "type": "mesh",
                "mesh": "centrifuge_tube_visual",
                "material": "centrifuge_tube_clear",
                "contype": "0",
                "conaffinity": "0",
                "group": "2",
            },
        )
    )
    worldbody.append(tube)

    ET.indent(tree, space="  ")
    out_path = out_dir / "generated_real_pipette_centrifuge_blender.xml"
    tree.write(out_path, encoding="unicode")
    return out_path


def build_meshplane_geometry() -> MeshPlaneGeometry:
    mesh = trimesh.load(TUBE_MESH)
    mesh.apply_scale(0.001)
    return MeshPlaneGeometry.from_trimesh(
        mesh,
        autobio_root=AUTOBIO_ROOT,
        split_top=True,
        split_bottom=False,
        opening="top",
    )


def make_container(volume_ul: float, geometry: MeshPlaneGeometry) -> ContainerState:
    return ContainerState(
        name="centrifuge_tube",
        geometry=geometry,
        volume_ul=volume_ul,
        capacity_ul=1500.0,
        sample_id="tube_sample",
        liquid_color=LIQUID_COLOR,
    )


def make_controller(args: argparse.Namespace) -> PipetteLiquidController:
    tip = PipetteTipState(capacity_ul=args.tip_capacity_ul, liquid_color=LIQUID_COLOR)
    plunger = PlungerModel(qpos_rest_m=0.0, qpos_pressed_m=-0.008, stroke_volume_ul=args.stroke_volume_ul)
    return PipetteLiquidController.from_initial_qpos(tip=tip, plunger=plunger, qpos_m=-0.008)


def estimate_tip_liquid_geometry(tip_mesh: Path = TIP_MESH) -> dict[str, float | int]:
    """Estimate a safe liquid column from the real 200 uL tip visual mesh."""

    mesh = trimesh.load(tip_mesh, force="mesh")
    mesh.apply_scale(0.001)
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    local_z = vertices[:, 2]
    radius = np.linalg.norm(vertices[:, :2], axis=1)
    ring_stats: list[tuple[float, float]] = []
    for z_value in np.unique(np.round(local_z, decimals=7)):
        mask = np.isclose(local_z, z_value, atol=5e-8)
        if mask.any():
            ring_stats.append((float(z_value), float(np.min(radius[mask]))))
    if len(ring_stats) < 2:
        return {"height_m": 0.0325, "bottom_radius_m": 0.00012, "top_radius_m": 0.00098}

    lower_z = min(z for z, _ in ring_stats)
    upper_candidates = [(z, r) for z, r in ring_stats if z > lower_z + 1e-4 and r > 0.001]
    upper_z, upper_inner_radius = min(upper_candidates, key=lambda item: item[0]) if upper_candidates else max(ring_stats, key=lambda item: item[0])
    height_m = max(0.020, float(upper_z - lower_z - 0.0015))
    top_radius_m = min(0.00105, max(0.00075, 0.50 * float(upper_inner_radius)))
    return {
        "height_m": height_m,
        "bottom_radius_m": 0.00012,
        "top_radius_m": top_radius_m,
    }


def joint_qpos_adr(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"Missing joint {name!r}")
    return int(model.jnt_qposadr[joint_id])


def site_id(model: mujoco.MjModel, name: str) -> int:
    item = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if item < 0:
        raise RuntimeError(f"Missing site {name!r}")
    return int(item)


def set_freejoint(qpos: np.ndarray, adr: int, pos: np.ndarray, quat: np.ndarray) -> None:
    qpos[adr : adr + 3] = np.asarray(pos, dtype=np.float64).reshape(3)
    qpos[adr + 3 : adr + 7] = np.asarray(quat, dtype=np.float64).reshape(4)


def surface_dict(geometry: MeshPlaneGeometry, container: ContainerState, tube_R: np.ndarray, acceleration_world: np.ndarray) -> dict[str, object]:
    surface = geometry.surface(
        container.volume_ul,
        gravity_world=GRAVITY_WORLD,
        acceleration_world=acceleration_world,
        container_pos_world=TUBE_POS,
        container_rot_world=tube_R,
    )
    return surface.as_json()


def liquid_bulk_mesh_dict(geometry: MeshPlaneGeometry, surface: dict[str, object], tube_R: np.ndarray) -> dict[str, object] | None:
    distance = surface.get("distance_m")
    if distance is None:
        return None
    liquid_mesh = geometry.meshplane.calculate_mesh(float(distance))
    vertices = np.asarray(liquid_mesh.vertices, dtype=np.float64)
    faces = np.asarray(liquid_mesh.faces, dtype=np.int64)
    if vertices.size == 0 or faces.size == 0:
        return None
    vertices_world = TUBE_POS + vertices @ tube_R.T
    face_list: list[list[int]] = faces.astype(int).tolist()
    boundary = np.asarray(getattr(liquid_mesh, "boundary", []), dtype=np.int64)
    if boundary.size >= 3:
        face_list.append(boundary.astype(int).tolist())
    return {
        "vertices": np.round(vertices_world, decimals=7).tolist(),
        "faces": face_list,
    }


def qpos_for_frame(
    base_qpos: np.ndarray,
    model: mujoco.MjModel,
    ids: dict[str, int],
    frame: DemoFrame,
    surface: dict[str, object],
) -> tuple[np.ndarray, np.ndarray, bool, bool, float, float]:
    data = mujoco.MjData(model)
    qpos = base_qpos.copy()
    tube_q = tube_quat(frame.tube_roll_rad, frame.tube_pitch_rad)
    tube_R = quat_to_matrix(tube_q)
    normal = np.asarray(surface["normal_world"], dtype=np.float64)
    center = np.asarray(surface["center_world"], dtype=np.float64)
    tip_target = center - normal * float(frame.tip_depth_m)
    pipette_pos = tip_target - np.array([0.0, 0.0, TIP_SITE_LOCAL_Z], dtype=np.float64)
    set_freejoint(qpos, ids["pipette_free_qpos_adr"], pipette_pos, np.array([1.0, 0.0, 0.0, 0.0]))
    set_freejoint(qpos, ids["tube_free_qpos_adr"], TUBE_POS, tube_q)
    qpos[ids["button_qpos_adr"]] = float(frame.qpos_m)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    tip_pos = data.site_xpos[ids["tip_site"]].copy()
    local_tip = tube_R.T @ (tip_pos - TUBE_POS)
    signed_depth = float(np.dot(center - tip_pos, normal))
    radial = float(np.linalg.norm(local_tip[:2]))
    in_tube = bool(radial < 0.0047 and -0.001 <= local_tip[2] <= 0.043)
    submerged = bool(signed_depth > 0.0 and in_tube)
    return qpos, tip_pos, in_tube, submerged, signed_depth, radial


def build_frames(args: argparse.Namespace) -> list[DemoFrame]:
    frames: list[DemoFrame] = []
    frames += exchange_frames(args, camera_mode="full")
    for qpos_m in interp(-0.008, -0.008, 8):
        frames.append(DemoFrame("Holding real tip after exchange", qpos_m, -0.018, camera_mode="full"))
    return frames


def build_trajectory(model_path: Path, args: argparse.Namespace, out_dir: Path) -> tuple[Path, Path, dict[str, object]]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    base_qpos = data.qpos.copy()
    ids = {
        "pipette_free_qpos_adr": joint_qpos_adr(model, "pipette_free"),
        "tube_free_qpos_adr": joint_qpos_adr(model, "centrifuge_tube_free"),
        "button_qpos_adr": joint_qpos_adr(model, "pipette_button"),
        "tip_site": site_id(model, "tip_site"),
    }
    geometry = build_meshplane_geometry()
    tip_liquid_geometry = estimate_tip_liquid_geometry()
    container = make_container(args.initial_tube_volume_ul, geometry)
    controller = make_controller(args)
    frames = build_frames(args)
    qpos_rows: list[np.ndarray] = []
    records: list[dict[str, object]] = []
    for index, frame in enumerate(frames):
        tube_R = quat_to_matrix(tube_quat(frame.tube_roll_rad, frame.tube_pitch_rad))
        pre_surface = surface_dict(geometry, container, tube_R, frame.acceleration_world)
        _, _, pre_in_tube, pre_submerged, _, _ = qpos_for_frame(base_qpos, model, ids, frame, pre_surface)
        events = controller.update(
            frame.qpos_m,
            source=container if frame.source_name == container.name else None,
            target=container if frame.target_name == container.name else None,
            tip_in_liquid=bool(pre_submerged),
            tip_in_target=bool(pre_in_tube),
        )
        surface = surface_dict(geometry, container, tube_R, frame.acceleration_world)
        if container.volume_ul > 1e-6:
            surface["bulk_mesh_world"] = liquid_bulk_mesh_dict(geometry, surface, tube_R)
        qpos, tip_pos, in_tube, submerged, signed_depth, radial = qpos_for_frame(base_qpos, model, ids, frame, surface)
        qpos_rows.append(qpos)
        records.append(
            {
                "frame_index": index,
                "stage": frame.stage,
                "qpos_m": frame.qpos_m,
                "tip_site_world": tip_pos.tolist(),
                "tip_axis_world": [0.0, 0.0, 1.0],
                "tip_liquid_geometry": tip_liquid_geometry,
                "tip_signed_depth_m": signed_depth,
                "tip_radial_m": radial,
                "tip_in_tube": in_tube,
                "tip_submerged": submerged,
                "acceleration_world": frame.acceleration_world.tolist(),
                "tube": container.as_json(),
                "source": container.as_json(),
                "tip": controller.tip.as_json(),
                "source_hit": {
                    "container_name": container.name,
                    "tip_in_container": in_tube,
                    "tip_in_liquid": submerged,
                    "surface": surface,
                },
                "events": [event.as_json() for event in events],
            }
        )

    trajectory_path = out_dir / "real_pipette_centrifuge_liquid.npz"
    wet_state_path = out_dir / "wet_state.jsonl"
    np.savez_compressed(trajectory_path, qpos=np.asarray(qpos_rows, dtype=np.float64), model=str(model_path))
    with wet_state_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    summary = {
        "frames": len(records),
        "trajectory": str(trajectory_path),
        "wet_state": str(wet_state_path),
        "final_tube_volume_ul": records[-1]["tube"]["volume_ul"],
        "final_tip_volume_ul": records[-1]["tip"]["volume_ul"],
        "initial_tube_volume_ul": args.initial_tube_volume_ul,
        "tip_capacity_ul": args.tip_capacity_ul,
        "stroke_volume_ul": args.stroke_volume_ul,
        "tip_liquid_geometry": tip_liquid_geometry,
    }
    return trajectory_path, wet_state_path, summary


def eye_from_mujoco_free_camera(lookat: tuple[float, float, float], distance: float, azimuth: float, elevation: float) -> tuple[float, float, float]:
    target = np.asarray(lookat, dtype=np.float64)
    az = np.deg2rad(float(azimuth))
    el = np.deg2rad(float(elevation))
    horizontal = float(distance) * np.cos(el)
    delta = np.array(
        [
            np.sin(az) * horizontal,
            -np.cos(az) * horizontal,
            -np.sin(el) * float(distance),
        ],
        dtype=np.float64,
    )
    return tuple(float(v) for v in (target + delta))


def render_camera_specs() -> tuple[RenderCameraSpec, ...]:
    tube_lookat = (float(TUBE_POS[0]), float(TUBE_POS[1]), float(TUBE_POS[2] + 0.020))
    return (
        RenderCameraSpec(
            name="real_liquid_full",
            mode="world",
            lookat=(0.0, 0.0, 0.120),
            eye_offset_world=eye_from_mujoco_free_camera((0.0, 0.0, 0.120), 0.95, 142.0, -10.0),
        ),
        RenderCameraSpec(
            name="real_tip_close",
            mode="world",
            lookat=(0.0, 0.0, 0.030),
            eye_offset_world=eye_from_mujoco_free_camera((0.0, 0.0, 0.030), 0.13, 142.0, -14.0),
        ),
        RenderCameraSpec(
            name="real_tube_liquid_close",
            mode="world",
            lookat=tube_lookat,
            eye_offset_world=eye_from_mujoco_free_camera(tube_lookat, 0.12, 112.0, -18.0),
        ),
    )


def prepare_or_render(
    *,
    args: argparse.Namespace,
    trajectory_path: Path,
    wet_state_path: Path,
    model_path: Path,
    out_dir: Path,
    camera: str,
    output_name: str,
) -> Path:
    config = BlenderRenderConfig(
        trajectory=trajectory_path,
        model=model_path,
        wet_state=wet_state_path,
        out_dir=out_dir,
        output_name=output_name,
        camera=camera,
        fps=args.fps,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        stride=1,
        blender=args.blender,
        engine="AUTO",
        samples=64,
        visible_groups=(0, 1, 2),
    )
    manifest, command = prepare_blender_render(config, camera_specs=render_camera_specs())
    if args.render:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return Path(json.loads(manifest.read_text(encoding="utf-8"))["output_video"])


def main() -> None:
    args = parse_args()
    args.base_model = resolve_path(args.base_model)
    args.out_dir = resolve_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = prepare_model_xml(args.base_model, args.out_dir)
    trajectory_path, wet_state_path, summary = build_trajectory(model_path, args, args.out_dir)
    videos = [
        prepare_or_render(
            args=args,
            trajectory_path=trajectory_path,
            wet_state_path=wet_state_path,
            model_path=model_path,
            out_dir=args.out_dir,
            camera="real_liquid_full",
            output_name="01_blender_real_pipette_centrifuge_full.mp4",
        ),
        prepare_or_render(
            args=args,
            trajectory_path=trajectory_path,
            wet_state_path=wet_state_path,
            model_path=model_path,
            out_dir=args.out_dir,
            camera="real_tip_close",
            output_name="02_blender_real_tip_close.mp4",
        ),
        prepare_or_render(
            args=args,
            trajectory_path=trajectory_path,
            wet_state_path=wet_state_path,
            model_path=model_path,
            out_dir=args.out_dir,
            camera="real_tube_liquid_close",
            output_name="03_blender_real_tube_liquid_close.mp4",
        ),
    ]
    summary["model"] = str(model_path)
    summary["videos"] = [str(path) for path in videos]
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
