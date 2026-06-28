#!/usr/bin/env python3
"""Render full-pipette centrifuge-tube liquid demos using AutoBio meshplane."""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
import json
from pathlib import Path

import imageio.v2 as imageio
import mujoco
import numpy as np
import trimesh

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
AUTOBIO_ROOT = Path("/data/tianang/projects/AutoBio/autobio")
AUTOBIO_ASSETS = AUTOBIO_ROOT / "assets"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(AUTOBIO_ROOT) not in sys.path:
    sys.path.insert(0, str(AUTOBIO_ROOT))

from aero_tasks.liquid import ContainerState, PipetteLiquidController, PipetteTipState, PlungerModel  # noqa: E402
from liquid import ContainerDefinition  # noqa: E402
from meshplane import MeshPlane  # noqa: E402


DEFAULT_BASE_MODEL = PROJECT_ROOT / "models/piper_aero_hand/scenes/pipette_liquid_transfer_demo.xml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/debug_rollouts/meshplane_pipette_centrifuge_liquid"
TUBE_MESH = AUTOBIO_ASSETS / "container/centrifuge_1500ul_no_lid_vis/visual.obj"

TUBE_POS = np.array([0.0, 0.0, 0.0], dtype=np.float64)
GRAVITY_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float64)
LIQUID_COLOR = (0.18, 0.62, 1.0, 0.72)
TIP_SITE_LOCAL_Z = -0.053
LIQUID_POLY_TRIANGLES = 64


@dataclass(frozen=True)
class DemoFrame:
    stage: str
    qpos_m: float
    tip_depth_m: float
    tube_roll_rad: float = 0.0
    tube_pitch_rad: float = 0.0
    acceleration_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float64))
    source_name: str | None = None
    target_name: str | None = None
    camera_mode: str = "close"


@dataclass
class MeshPlaneTube:
    definition: ContainerDefinition
    meshplane: MeshPlane
    previous_distance: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-model", type=Path, default=DEFAULT_BASE_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--initial-tube-volume-ul", type=float, default=900.0)
    parser.add_argument("--tip-capacity-ul", type=float, default=200.0)
    parser.add_argument("--stroke-volume-ul", type=float, default=200.0)
    parser.add_argument("--hold-frames", type=int, default=36)
    parser.add_argument("--motion-frames", type=int, default=150)
    parser.add_argument("--move-frames", type=int, default=70)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def smoothstep(alpha: float) -> float:
    alpha = min(max(float(alpha), 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def interp(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [end]
    return [start + (end - start) * smoothstep(i / (count - 1)) for i in range(count)]


def quat_from_axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = np.asarray(axis, dtype=np.float64)
    axis /= max(float(np.linalg.norm(axis)), 1e-12)
    half = 0.5 * float(angle)
    return np.array([np.cos(half), *(np.sin(half) * axis)], dtype=np.float64)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float64,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=np.float64)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def matrix_to_quat(matrix: np.ndarray) -> np.ndarray:
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quat


def tube_quat(roll_rad: float, pitch_rad: float) -> np.ndarray:
    return quat_multiply(
        quat_from_axis_angle(np.array([0.0, 1.0, 0.0]), pitch_rad),
        quat_from_axis_angle(np.array([1.0, 0.0, 0.0]), roll_rad),
    )


def prepare_model_xml(base_model: Path, out_dir: Path) -> Path:
    xml = base_model.read_text(encoding="utf-8")
    surface_proxy_xml = "\n".join(
        f"""    <body name="liquid_patch_body_{index:02d}" pos="0 0 0">
      <geom name="liquid_patch_{index:02d}" type="box" size="0.00001 0.00001 0.00001" material="meshplane_liquid" contype="0" conaffinity="0" group="2" />
    </body>"""
        for index in range(LIQUID_POLY_TRIANGLES)
    )
    asset_xml = f"""
    <mesh name="centrifuge_tube_visual" file="{TUBE_MESH}" scale="0.001 0.001 0.001" />
    <material name="centrifuge_tube_clear" rgba="0.70 0.92 1.0 0.18" specular="0.5" shininess="0.5" />
    <material name="meshplane_liquid" rgba="{LIQUID_COLOR[0]} {LIQUID_COLOR[1]} {LIQUID_COLOR[2]} {LIQUID_COLOR[3]}" specular="0.85" shininess="0.8" />
    <material name="surface_normal" rgba="1.0 0.38 0.18 0.85" />
"""
    body_xml = f"""
    <body name="centrifuge_tube" pos="0 0 0">
      <geom name="centrifuge_tube_visual" type="mesh" mesh="centrifuge_tube_visual" material="centrifuge_tube_clear" contype="0" conaffinity="0" group="2" />
    </body>
{surface_proxy_xml}
    <body name="surface_normal_arrow" pos="0 0 0.04">
      <geom name="surface_normal_arrow" type="cylinder" size="0.00028 0.012" material="surface_normal" contype="0" conaffinity="0" group="2" />
    </body>
    <body name="tip_submerged_marker" pos="0 0 0.02">
      <geom name="tip_submerged_marker" type="sphere" size="0.0014" rgba="1 0.1 0.05 1" contype="0" conaffinity="0" group="2" />
    </body>
"""
    body_xml = body_xml.replace('material="meshplane_liquid" contype=', 'material="meshplane_liquid" density="0" contype=')
    xml = xml.replace("\n    <material name=\"pipette_plastic\"", asset_xml + "\n    <material name=\"pipette_plastic\"", 1)
    xml = xml.replace("\n  </worldbody>", body_xml + "\n  </worldbody>", 1)
    out_path = out_dir / "generated_meshplane_pipette_centrifuge_liquid.xml"
    out_path.write_text(xml, encoding="utf-8")
    return out_path


def build_meshplane_tube() -> MeshPlaneTube:
    mesh = trimesh.load(TUBE_MESH)
    mesh.apply_scale(0.001)
    definition = ContainerDefinition.from_object_mesh(mesh, split_top=True, split_bottom=False, opening="top")
    return MeshPlaneTube(definition=definition, meshplane=MeshPlane(definition.interior))


def make_container(volume_ul: float) -> ContainerState:
    return ContainerState(
        name="centrifuge_tube",
        geometry=None,  # MeshPlaneTube is the authoritative volume/height geometry for this demo.
        volume_ul=volume_ul,
        capacity_ul=1500.0,
        sample_id="tube_sample",
        liquid_color=LIQUID_COLOR,
    )


def make_controller(args: argparse.Namespace) -> PipetteLiquidController:
    tip = PipetteTipState(capacity_ul=args.tip_capacity_ul, liquid_color=LIQUID_COLOR)
    plunger = PlungerModel(qpos_rest_m=0.0, qpos_pressed_m=-0.008, stroke_volume_ul=args.stroke_volume_ul)
    return PipetteLiquidController.from_initial_qpos(tip=tip, plunger=plunger, qpos_m=-0.008)


def mj_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"Missing MuJoCo object {name!r}")
    return int(obj_id)


def hide_original_source_target(model: mujoco.MjModel) -> None:
    for name in ("source_tube_wall", "source_liquid", "target_well_wall", "target_liquid"):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        if geom_id >= 0:
            model.geom_rgba[geom_id, 3] = 0.0


def meshplane_surface(meshplane_tube: MeshPlaneTube, volume_ul: float, tube_R: np.ndarray, acceleration_world: np.ndarray) -> dict[str, object]:
    effective_up = -(GRAVITY_WORLD - acceleration_world)
    normal_world = effective_up / max(float(np.linalg.norm(effective_up)), 1e-12)
    normal_local = tube_R.T @ normal_world
    meshplane_tube.meshplane.set_plane_normal(*normal_local)
    if meshplane_tube.previous_distance is None:
        low, high, _ = meshplane_tube.meshplane.get_plane_distance_range()
        meshplane_tube.previous_distance = 0.5 * (low + high)
    distance = meshplane_tube.meshplane.solve_plane_distance(volume_ul * 1e-9, meshplane_tube.previous_distance)
    meshplane_tube.previous_distance = distance
    result = meshplane_tube.meshplane.calculate_plane(distance)
    world_center = tube_R @ result.center + TUBE_POS
    world_frame = tube_R @ result.frame
    return {
        "distance": float(distance),
        "valid": bool(result.half_width > 0.0 and result.half_height > 0.0),
        "center_world": world_center,
        "frame_world": world_frame,
        "normal_world": normal_world,
        "half_width": float(result.half_width),
        "half_height": float(result.half_height),
    }


def set_tip_liquid(model: mujoco.MjModel, controller: PipetteLiquidController) -> None:
    segment_ids: list[int] = []
    weights: list[float] = []
    for index in range(24):
        geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tip_liquid_seg_{index:02d}")
        if geom_id < 0:
            continue
        segment_ids.append(geom_id)
        radius = float(model.geom_size[geom_id, 0])
        halfheight = float(model.geom_size[geom_id, 1])
        weights.append(radius * radius * 2.0 * halfheight)
    fractions = controller.tip.liquid_column_segment_fractions(weights)
    for geom_id, fraction in zip(segment_ids, fractions, strict=True):
        model.geom_rgba[geom_id, :3] = controller.tip.liquid_color[:3]
        model.geom_rgba[geom_id, 3] = min(0.82, controller.tip.liquid_color[3] + 0.12) if fraction > 1e-6 else 0.0


def set_liquid_polygon_patches(
    model: mujoco.MjModel,
    patch_body_ids: list[int],
    patch_geom_ids: list[int],
    surface: dict[str, object],
    color: tuple[float, float, float, float],
) -> None:
    center = np.asarray(surface["center_world"], dtype=np.float64)
    normal = np.asarray(surface["normal_world"], dtype=np.float64)
    frame = np.asarray(surface["frame_world"], dtype=np.float64)
    x_axis = frame[:, 0] / max(float(np.linalg.norm(frame[:, 0])), 1e-12)
    y_axis = frame[:, 1] / max(float(np.linalg.norm(frame[:, 1])), 1e-12)
    half_width = max(float(surface["half_width"]), 1e-6)
    half_height = max(float(surface["half_height"]), 1e-6)
    sample_count = len(patch_geom_ids)
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=False)
    boundary = np.asarray([
        center + half_width * np.cos(angle) * x_axis + half_height * np.sin(angle) * y_axis
        for angle in angles
    ])

    for index, (body_id, geom_id) in enumerate(zip(patch_body_ids, patch_geom_ids, strict=True)):
        p0 = boundary[index]
        p1 = boundary[(index + 1) % sample_count]
        edge_mid = 0.5 * (p0 + p1)
        radial_vec = edge_mid - center
        radial_len = float(np.linalg.norm(radial_vec))
        edge_vec = p1 - p0
        edge_len = float(np.linalg.norm(edge_vec))
        if radial_len < 1e-7 or edge_len < 1e-7:
            model.geom_rgba[geom_id, 3] = 0.0
            continue
        x_axis = radial_vec / radial_len
        z_axis = normal / max(float(np.linalg.norm(normal)), 1e-12)
        y_axis = np.cross(z_axis, x_axis)
        y_axis /= max(float(np.linalg.norm(y_axis)), 1e-12)
        x_axis = np.cross(y_axis, z_axis)
        x_axis /= max(float(np.linalg.norm(x_axis)), 1e-12)
        rotation = np.column_stack([x_axis, y_axis, z_axis])
        model.body_pos[body_id] = 0.5 * (center + edge_mid)
        model.body_quat[body_id] = matrix_to_quat(rotation)
        model.geom_size[geom_id] = np.array([max(0.5 * radial_len, 1e-6), max(0.55 * edge_len, 1e-6), 0.00018])
        model.geom_rgba[geom_id, :3] = color[:3]
        model.geom_rgba[geom_id, 3] = color[3]


def set_scene(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    ids: dict[str, int],
    meshplane_tube: MeshPlaneTube,
    container: ContainerState,
    controller: PipetteLiquidController,
    frame: DemoFrame,
) -> dict[str, object]:
    tube_q = tube_quat(frame.tube_roll_rad, frame.tube_pitch_rad)
    tube_R = quat_to_matrix(tube_q)
    model.body_pos[ids["tube_body"]] = TUBE_POS
    model.body_quat[ids["tube_body"]] = tube_q
    surface = meshplane_surface(meshplane_tube, container.volume_ul, tube_R, frame.acceleration_world)

    set_liquid_polygon_patches(
        model,
        ids["surface_patch_bodies"],
        ids["surface_patch_geoms"],
        surface,
        container.liquid_color if container.volume_ul > 1e-6 else (*container.liquid_color[:3], 0.0),
    )

    normal = surface["normal_world"]
    model.body_pos[ids["normal_body"]] = surface["center_world"] + normal * 0.020
    model.body_quat[ids["normal_body"]] = matrix_to_quat(surface["frame_world"])

    tip_target = surface["center_world"] - normal * frame.tip_depth_m
    pipette_pos = tip_target - np.array([0.0, 0.0, TIP_SITE_LOCAL_Z], dtype=np.float64)
    model.body_pos[ids["pipette_body"]] = pipette_pos
    model.body_quat[ids["pipette_body"]] = np.array([1.0, 0.0, 0.0, 0.0])
    data.qpos[ids["button_qpos_adr"]] = frame.qpos_m
    set_tip_liquid(model, controller)
    mujoco.mj_forward(model, data)

    tip_pos = data.site_xpos[ids["tip_site"]].copy()
    local_tip = tube_R.T @ (tip_pos - TUBE_POS)
    signed_depth = float(np.dot(surface["center_world"] - tip_pos, normal))
    radial = float(np.linalg.norm(local_tip[:2]))
    in_tube = bool(radial < 0.0047 and -0.001 <= local_tip[2] <= 0.043)
    submerged = bool(signed_depth > 0.0 and in_tube)
    model.body_pos[ids["marker_body"]] = tip_pos
    model.geom_rgba[ids["marker_geom"]] = np.array([0.0, 0.9, 0.2, 1.0]) if submerged else np.array([1.0, 0.12, 0.05, 1.0])
    mujoco.mj_forward(model, data)
    return {
        "surface_valid": surface["valid"],
        "surface_distance": surface["distance"],
        "surface_center_world": surface["center_world"].tolist(),
        "surface_normal_world": normal.tolist(),
        "surface_half_width": surface["half_width"],
        "surface_half_height": surface["half_height"],
        "surface_boundary_vertices": LIQUID_POLY_TRIANGLES,
        "tip_site_world": tip_pos.tolist(),
        "tip_signed_depth_m": signed_depth,
        "tip_radial_m": radial,
        "tip_in_tube": in_tube,
        "tip_submerged": submerged,
    }


def annotate(image: np.ndarray, frame: DemoFrame, container: ContainerState, controller: PipetteLiquidController, diagnostics: dict[str, object]) -> np.ndarray:
    if Image is None or ImageDraw is None or ImageFont is None:
        return image
    pil = Image.fromarray(image)
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    normal = np.asarray(diagnostics["surface_normal_world"], dtype=np.float64)
    lines = [
        frame.stage,
        f"tube {container.volume_ul:7.1f}/1500 uL   tip {controller.tip.volume_ul:6.1f}/{controller.tip.capacity_ul:.0f} uL   qpos {frame.qpos_m:+.4f} m",
        f"tip_submerged={diagnostics['tip_submerged']}   signed_depth={diagnostics['tip_signed_depth_m']:+.4f} m   radial={diagnostics['tip_radial_m']:.4f} m",
        f"normal [{normal[0]:+.2f}, {normal[1]:+.2f}, {normal[2]:+.2f}]   accel [{frame.acceleration_world[0]:+.1f}, {frame.acceleration_world[1]:+.1f}, {frame.acceleration_world[2]:+.1f}] m/s^2",
        "green marker = tip_site below meshplane liquid surface; red = not submerged",
    ]
    line_height = 18
    box_w = 790
    box_h = 20 + line_height * len(lines)
    draw.rounded_rectangle((18, 18, 18 + box_w, 18 + box_h), radius=6, fill=(250, 250, 250, 218))
    for i, line in enumerate(lines):
        draw.text((30, 30 + i * line_height), line, fill=(25, 35, 40, 255), font=font)
    return np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"))


def exchange_frames(args: argparse.Namespace, *, camera_mode: str) -> list[DemoFrame]:
    frames: list[DemoFrame] = []
    frames += [DemoFrame("Tip inserted below meshplane surface", -0.008, 0.004, camera_mode=camera_mode) for _ in range(args.hold_frames)]
    for qpos in interp(-0.008, 0.0, args.motion_frames):
        frames.append(DemoFrame("Aspirating from centrifuge tube", qpos, 0.004, source_name="centrifuge_tube", camera_mode=camera_mode))
    for depth in interp(0.004, -0.022, args.move_frames):
        frames.append(DemoFrame("Lifting full tip above liquid", 0.0, depth, tube_roll_rad=np.deg2rad(12), acceleration_world=np.array([2.5, 0.0, 0.0]), camera_mode=camera_mode))
    for depth in interp(-0.022, 0.004, args.move_frames):
        frames.append(DemoFrame("Re-inserting tip for dispense", 0.0, depth, tube_roll_rad=np.deg2rad(-10), acceleration_world=np.array([-2.0, 0.0, 0.0]), camera_mode=camera_mode))
    for qpos in interp(0.0, -0.008, args.motion_frames):
        frames.append(DemoFrame("Dispensing back into centrifuge tube", qpos, 0.004, target_name="centrifuge_tube", camera_mode=camera_mode))
    frames += [DemoFrame("Exchange complete: volume restored", -0.008, 0.004, camera_mode=camera_mode) for _ in range(args.hold_frames)]
    return frames


def tilt_accel_frames(args: argparse.Namespace) -> list[DemoFrame]:
    frames: list[DemoFrame] = []
    for index in range(args.motion_frames):
        phase = 2 * np.pi * index / max(args.motion_frames - 1, 1)
        frames.append(
            DemoFrame(
                "Tube tilt + lateral acceleration meshplane surface",
                -0.008,
                -0.020,
                tube_roll_rad=np.deg2rad(26) * np.sin(phase),
                tube_pitch_rad=np.deg2rad(13) * np.sin(phase + 0.9),
                acceleration_world=np.array([4.0 * np.sin(phase), 2.0 * np.sin(phase + 1.2), 0.0]),
                camera_mode="close",
            )
        )
    return frames


def configure_camera(camera: mujoco.MjvCamera, frame: DemoFrame) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if frame.camera_mode == "full":
        camera.lookat[:] = np.array([0.0, 0.0, 0.120])
        camera.distance = 0.48
        camera.azimuth = 142.0
        camera.elevation = -7.0
    else:
        camera.lookat[:] = np.array([0.0, 0.0, 0.030])
        camera.distance = 0.13
        camera.azimuth = 142.0
        camera.elevation = -14.0


def write_demo(args: argparse.Namespace, model_path: Path, name: str, frames: list[DemoFrame], jsonl_path: Path) -> dict[str, object]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    hide_original_source_target(model)
    ids = {
        "tube_body": mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "centrifuge_tube"),
        "surface_patch_bodies": [
            mj_id(model, mujoco.mjtObj.mjOBJ_BODY, f"liquid_patch_body_{index:02d}")
            for index in range(LIQUID_POLY_TRIANGLES)
        ],
        "surface_patch_geoms": [
            mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"liquid_patch_{index:02d}")
            for index in range(LIQUID_POLY_TRIANGLES)
        ],
        "normal_body": mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "surface_normal_arrow"),
        "pipette_body": mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "pipette"),
        "button_joint": mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "pipette_button"),
        "tip_site": mj_id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site"),
        "marker_body": mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "tip_submerged_marker"),
        "marker_geom": mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "tip_submerged_marker"),
    }
    ids["button_qpos_adr"] = int(model.jnt_qposadr[ids["button_joint"]])
    meshplane_tube = build_meshplane_tube()
    container = make_container(args.initial_tube_volume_ul)
    controller = make_controller(args)
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    video_path = args.out_dir / f"{name}.mp4"
    snapshots: list[dict[str, object]] = []
    try:
        with imageio.get_writer(video_path, fps=args.fps, codec="libx264", ffmpeg_params=["-crf", "18"]) as writer:
            with jsonl_path.open("a", encoding="utf-8") as jsonl:
                for frame_index, frame in enumerate(frames):
                    pre = set_scene(model, data, ids, meshplane_tube, container, controller, frame)
                    events = controller.update(
                        frame.qpos_m,
                        source=container if frame.source_name == container.name else None,
                        target=container if frame.target_name == container.name else None,
                        tip_in_liquid=bool(pre["tip_submerged"]),
                        tip_in_target=bool(pre["tip_in_tube"]),
                    )
                    diagnostics = set_scene(model, data, ids, meshplane_tube, container, controller, frame)
                    configure_camera(camera, frame)
                    renderer.update_scene(data, camera=camera)
                    writer.append_data(annotate(renderer.render(), frame, container, controller, diagnostics))
                    snapshot = {
                        "demo": name,
                        "frame_index": frame_index,
                        "stage": frame.stage,
                        "tube": container.as_json(),
                        "tip": controller.tip.as_json(),
                        "events": [event.as_json() for event in events],
                        "qpos_m": frame.qpos_m,
                        "tip_depth_command_m": frame.tip_depth_m,
                        "acceleration_world": frame.acceleration_world.tolist(),
                        **diagnostics,
                    }
                    jsonl.write(json.dumps(snapshot, sort_keys=True) + "\n")
                    snapshots.append(snapshot)
    finally:
        renderer.close()
    final = snapshots[-1]
    return {
        "video": str(video_path),
        "frames": len(frames),
        "final_tube_volume_ul": final["tube"]["volume_ul"],
        "final_tip_volume_ul": final["tip"]["volume_ul"],
    }


def main() -> None:
    args = parse_args()
    args.base_model = resolve_path(args.base_model)
    args.out_dir = resolve_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    model_path = prepare_model_xml(args.base_model, args.out_dir)
    jsonl_path = args.out_dir / "wet_state.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()
    demos = [
        ("01_meshplane_tilt_accel_surface", tilt_accel_frames(args)),
        ("02_full_pipette_meshplane_exchange", exchange_frames(args, camera_mode="full")),
        ("03_close_tip_submersion_exchange", exchange_frames(args, camera_mode="close")),
    ]
    summaries = [write_demo(args, model_path, name, frames, jsonl_path) for name, frames in demos]
    summary = {
        "base_model": str(args.base_model),
        "render_model": str(model_path),
        "tube_mesh": str(TUBE_MESH),
        "liquid_surface": "AutoBio ContainerDefinition + meshplane.MeshPlane solve_plane_distance/calculate_plane",
        "tip_submersion": "computed from actual MuJoCo tip_site against meshplane surface plane and tube radial bounds",
        "initial_tube_volume_ul": args.initial_tube_volume_ul,
        "tip_capacity_ul": args.tip_capacity_ul,
        "stroke_volume_ul": args.stroke_volume_ul,
        "wet_state_jsonl": str(jsonl_path),
        "demos": summaries,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
