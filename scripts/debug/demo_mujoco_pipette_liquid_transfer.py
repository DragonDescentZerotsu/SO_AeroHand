#!/usr/bin/env python3
"""Render MuJoCo demos for semantic pipette liquid transfer."""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

import imageio.v2 as imageio
import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {exc}") from exc

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
DEFAULT_MODEL = PROJECT_ROOT / "models/piper_aero_hand/scenes/pipette_liquid_transfer_demo.xml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/debug_rollouts/mujoco_pipette_liquid_transfer"

SOURCE_X_M = -0.105
TARGET_X_M = 0.105
PIPETTE_Z_M = 0.1534
LIQUID_BOTTOM_Z_M = 0.096
SOURCE_RADIUS_M = 0.006
TARGET_RADIUS_M = 0.0045
SOURCE_INITIAL_UL = 500.0
SOURCE_CAPACITY_UL = 1500.0
TARGET_CAPACITY_UL = 300.0
LIQUID_STYLES = {
    "blue": (0.08, 0.36, 0.95, 0.68),
    "pale_highlight": (0.70, 0.92, 1.0, 0.24),
}
TIP_FRUSTUM_SEGMENT_COUNT = 64
TIP_FRUSTUM_SIDES = 20
TIP_LIQUID_BOTTOM_Z_M = -0.0495
TIP_LIQUID_TOP_Z_M = -0.0135
TIP_LIQUID_BOTTOM_RADIUS_M = 0.00018
TIP_LIQUID_TOP_RADIUS_M = 0.0019


from aero_tasks.liquid import (  # noqa: E402
    ContainerState,
    CylindricalGeometry,
    FrustumSegment,
    PipetteLiquidController,
    PipetteTipState,
    PlungerModel,
    WetLabLiquidState,
)


@dataclass(frozen=True)
class DemoFrame:
    stage: str
    pipette_x_m: float
    button_qpos_m: float
    source_name: str | None = None
    target_name: str | None = None
    tip_in_liquid: bool = False
    tip_in_target: bool = False
    camera_mode: str = "close"


@dataclass(frozen=True)
class TipSegmentProxy:
    geom_id: int
    bottom_z_m: float
    top_z_m: float
    segment: FrustumSegment

    @property
    def height_m(self) -> float:
        return self.top_z_m - self.bottom_z_m

    @property
    def lower_radius_m(self) -> float:
        return self.segment.lower_radius_m

    @property
    def upper_radius_m(self) -> float:
        return self.segment.upper_radius_m


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--tip-capacity-ul", type=float, default=200.0)
    parser.add_argument("--stroke-volume-ul", type=float, default=200.0)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--liquid-style", choices=sorted(LIQUID_STYLES), default="blue")
    parser.add_argument("--motion-frames", type=int, default=130)
    parser.add_argument("--move-frames", type=int, default=90)
    parser.add_argument("--hold-frames", type=int, default=35)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def smoothstep(alpha: float) -> float:
    alpha = min(max(float(alpha), 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def interp(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [float(end)]
    return [float(start + (end - start) * smoothstep(i / (count - 1))) for i in range(count)]


def hold(stage: str, *, pipette_x_m: float, button_qpos_m: float, count: int, camera_mode: str = "close") -> list[DemoFrame]:
    return [DemoFrame(stage, pipette_x_m, button_qpos_m, camera_mode=camera_mode) for _ in range(count)]


def aspirate_frames(args: argparse.Namespace, *, camera_mode: str = "close") -> list[DemoFrame]:
    return [
        DemoFrame(
            "Aspirating from source tube",
            SOURCE_X_M,
            qpos,
            source_name="source_tube",
            tip_in_liquid=True,
            camera_mode=camera_mode,
        )
        for qpos in interp(-0.008, 0.0, args.motion_frames)
    ]


def dispense_frames(args: argparse.Namespace, *, camera_mode: str = "close") -> list[DemoFrame]:
    return [
        DemoFrame(
            "Dispensing into target well",
            TARGET_X_M,
            qpos,
            target_name="target_well",
            tip_in_target=True,
            camera_mode=camera_mode,
        )
        for qpos in interp(0.0, -0.008, args.motion_frames)
    ]


def move_frames(args: argparse.Namespace) -> list[DemoFrame]:
    return [
        DemoFrame("Moving full tip to target", x, 0.0, camera_mode="wide")
        for x in interp(SOURCE_X_M, TARGET_X_M, args.move_frames)
    ]


def make_state(args: argparse.Namespace, *, initial_qpos_m: float) -> WetLabLiquidState:
    source = ContainerState(
        name="source_tube",
        geometry=CylindricalGeometry(bottom_z_m=LIQUID_BOTTOM_Z_M, radius_m=SOURCE_RADIUS_M),
        volume_ul=SOURCE_INITIAL_UL,
        capacity_ul=SOURCE_CAPACITY_UL,
        sample_id="blue_sample",
        liquid_color=LIQUID_STYLES[args.liquid_style],
    )
    target = ContainerState(
        name="target_well",
        geometry=CylindricalGeometry(bottom_z_m=LIQUID_BOTTOM_Z_M, radius_m=TARGET_RADIUS_M),
        volume_ul=0.0,
        capacity_ul=TARGET_CAPACITY_UL,
        liquid_color=LIQUID_STYLES[args.liquid_style],
    )
    tip = PipetteTipState(capacity_ul=args.tip_capacity_ul)
    plunger = PlungerModel(qpos_rest_m=0.0, qpos_pressed_m=-0.008, stroke_volume_ul=args.stroke_volume_ul)
    return WetLabLiquidState(
        containers={"source_tube": source, "target_well": target},
        pipette=PipetteLiquidController.from_initial_qpos(tip=tip, plunger=plunger, qpos_m=initial_qpos_m),
    )


def mj_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"MuJoCo model is missing {name!r}.")
    return int(obj_id)


def make_tip_segment_specs(segment_count: int = TIP_FRUSTUM_SEGMENT_COUNT) -> list[TipSegmentProxy]:
    height = TIP_LIQUID_TOP_Z_M - TIP_LIQUID_BOTTOM_Z_M
    proxies: list[TipSegmentProxy] = []
    for index in range(segment_count):
        lower_alpha = index / segment_count
        upper_alpha = (index + 1) / segment_count
        z0 = TIP_LIQUID_BOTTOM_Z_M + height * lower_alpha
        z1 = TIP_LIQUID_BOTTOM_Z_M + height * upper_alpha
        r0 = TIP_LIQUID_BOTTOM_RADIUS_M + (TIP_LIQUID_TOP_RADIUS_M - TIP_LIQUID_BOTTOM_RADIUS_M) * lower_alpha
        r1 = TIP_LIQUID_BOTTOM_RADIUS_M + (TIP_LIQUID_TOP_RADIUS_M - TIP_LIQUID_BOTTOM_RADIUS_M) * upper_alpha
        proxies.append(
            TipSegmentProxy(
                geom_id=-1,
                bottom_z_m=z0,
                top_z_m=z1,
                segment=FrustumSegment(lower_radius_m=r0, upper_radius_m=r1, height_m=z1 - z0),
            )
        )
    return proxies


def frustum_mesh_xml(name: str, *, lower_radius_m: float, upper_radius_m: float, height_m: float, sides: int = TIP_FRUSTUM_SIDES) -> str:
    vertices: list[tuple[float, float, float]] = []
    z0 = -0.5 * height_m
    z1 = 0.5 * height_m
    for z, radius in ((z0, lower_radius_m), (z1, upper_radius_m)):
        for index in range(sides):
            theta = 2.0 * np.pi * index / sides
            vertices.append((radius * float(np.cos(theta)), radius * float(np.sin(theta)), z))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, z0))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, z1))

    faces: list[tuple[int, int, int]] = []
    for index in range(sides):
        nxt = (index + 1) % sides
        faces.append((index, nxt, sides + nxt))
        faces.append((index, sides + nxt, sides + index))
        faces.append((bottom_center, nxt, index))
        faces.append((top_center, sides + index, sides + nxt))

    vertex_text = " ".join(f"{coord:.9g}" for vertex in vertices for coord in vertex)
    face_text = " ".join(str(index) for face in faces for index in face)
    return f'    <mesh name="{name}" vertex="{vertex_text}" face="{face_text}" />'


def generated_tip_liquid_assets_and_geoms(segment_count: int = TIP_FRUSTUM_SEGMENT_COUNT) -> tuple[str, str]:
    assets: list[str] = []
    geoms: list[str] = []
    for index, proxy in enumerate(make_tip_segment_specs(segment_count)):
        mesh_name = f"tip_liquid_seg_{index:02d}_mesh"
        geom_name = f"tip_liquid_seg_{index:02d}"
        center_z = 0.5 * (proxy.bottom_z_m + proxy.top_z_m)
        assets.append(
            frustum_mesh_xml(
                mesh_name,
                lower_radius_m=proxy.lower_radius_m,
                upper_radius_m=proxy.upper_radius_m,
                height_m=proxy.height_m,
            )
        )
        geoms.append(
            f'        <geom name="{geom_name}" type="mesh" mesh="{mesh_name}" pos="0 0 {center_z:.9g}" '
            'contype="0" conaffinity="0" group="2" material="liquid_blue" rgba="0.08 0.36 0.95 0" />'
        )
    geoms.append(
        '        <geom name="tip_liquid_boundary" type="cylinder" size="0.0001 0.000001" '
        f'pos="0 0 {TIP_LIQUID_BOTTOM_Z_M:.9g}" contype="0" conaffinity="0" group="2" material="liquid_blue" rgba="0.08 0.36 0.95 0" />'
    )
    return "\n".join(assets), "\n".join(geoms)


def prepare_frustum_model_xml(model_path: Path, out_dir: Path) -> Path:
    xml_text = model_path.read_text(encoding="utf-8")
    assets_xml, geoms_xml = generated_tip_liquid_assets_and_geoms()
    xml_text = xml_text.replace("\n    <material name=\"pipette_plastic\"", f"\n{assets_xml}\n\n    <material name=\"pipette_plastic\"", 1)
    xml_text, count = re.subn(
        r'\n\s*<geom name="tip_liquid_seg_00"[^>]*?/>\n(?:\s*<geom name="tip_liquid_seg_\d\d"[^>]*?/>\n)*',
        "\n" + geoms_xml + "\n",
        xml_text,
        count=1,
    )
    if count != 1:
        raise RuntimeError("Could not replace the cylinder tip liquid proxy block in the demo MJCF.")
    generated_path = out_dir / "generated_pipette_liquid_transfer_frustum.xml"
    generated_path.write_text(xml_text, encoding="utf-8")
    return generated_path


def load_model(model_path: Path) -> tuple[mujoco.MjModel, mujoco.MjData, dict[str, int | list[int]]]:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    ids: dict[str, int | list[int]] = {
        "pipette_body": mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "pipette"),
        "button_joint": mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, "pipette_button"),
        "source_liquid": mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "source_liquid"),
        "target_liquid": mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_liquid"),
        "tip_site": mj_id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site"),
        "tip_boundary": mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "tip_liquid_boundary"),
        "tip_segments": [
            mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, f"tip_liquid_seg_{index:02d}")
            for index in range(TIP_FRUSTUM_SEGMENT_COUNT)
        ],
    }
    return model, data, ids


def get_tip_segment_proxies(model: mujoco.MjModel, segment_ids: list[int]) -> list[TipSegmentProxy]:
    proxies = make_tip_segment_specs(len(segment_ids))
    return [
        TipSegmentProxy(
            geom_id=geom_id,
            bottom_z_m=proxy.bottom_z_m,
            top_z_m=proxy.top_z_m,
            segment=proxy.segment,
        )
        for geom_id, proxy in zip(segment_ids, proxies, strict=True)
    ]


def set_container_liquid_geom(
    model: mujoco.MjModel,
    geom_id: int,
    container: ContainerState,
    *,
    rgba: tuple[float, float, float, float],
) -> None:
    height = max(0.0, container.surface().height_m - LIQUID_BOTTOM_Z_M)
    visible = height > 1e-6 and container.volume_ul > 1e-6
    halfheight = max(height * 0.5, 0.00005)
    model.geom_size[geom_id, 1] = halfheight
    model.geom_pos[geom_id, 2] = LIQUID_BOTTOM_Z_M + halfheight
    model.geom_rgba[geom_id, :3] = rgba[:3]
    model.geom_rgba[geom_id, 3] = rgba[3] if visible else 0.0


def set_tip_liquid_geoms(model: mujoco.MjModel, segment_proxies: list[TipSegmentProxy], boundary_geom_id: int, tip: PipetteTipState) -> float:
    fractions = tip.liquid_column_frustum_height_fractions([segment.segment for segment in segment_proxies])
    liquid_height_m = TIP_LIQUID_BOTTOM_Z_M
    boundary_set = False
    for segment, fraction in zip(segment_proxies, fractions, strict=True):
        full = fraction >= 1.0 - 1e-6
        model.geom_rgba[segment.geom_id, :3] = tip.liquid_color[:3]
        model.geom_rgba[segment.geom_id, 3] = min(0.82, tip.liquid_color[3] + 0.12) if full else 0.0
        if full:
            liquid_height_m = segment.top_z_m
        elif fraction > 1e-6 and not boundary_set:
            partial_height = segment.height_m * fraction
            liquid_height_m = segment.bottom_z_m + partial_height
            radius = segment.lower_radius_m + (segment.upper_radius_m - segment.lower_radius_m) * fraction
            model.geom_size[boundary_geom_id, 0] = max(radius, 1e-6)
            model.geom_size[boundary_geom_id, 1] = max(0.5 * partial_height, 1e-6)
            model.geom_pos[boundary_geom_id, 2] = segment.bottom_z_m + 0.5 * partial_height
            model.geom_rgba[boundary_geom_id, :3] = tip.liquid_color[:3]
            model.geom_rgba[boundary_geom_id, 3] = min(0.82, tip.liquid_color[3] + 0.12)
            boundary_set = True
    if not boundary_set:
        model.geom_rgba[boundary_geom_id, :3] = tip.liquid_color[:3]
        model.geom_rgba[boundary_geom_id, 3] = 0.0
        model.geom_size[boundary_geom_id, 1] = 1e-6
        model.geom_pos[boundary_geom_id, 2] = TIP_LIQUID_BOTTOM_Z_M
    return liquid_height_m


def apply_visual_state(model: mujoco.MjModel, ids: dict[str, int | list[int]], segment_proxies: list[TipSegmentProxy], state: WetLabLiquidState) -> float:
    tip_liquid_height_m = set_tip_liquid_geoms(model, segment_proxies, int(ids["tip_boundary"]), state.pipette.tip)
    set_container_liquid_geom(model, ids["source_liquid"], state.containers["source_tube"], rgba=state.containers["source_tube"].liquid_color)  # type: ignore[arg-type]
    set_container_liquid_geom(model, ids["target_liquid"], state.containers["target_well"], rgba=state.containers["target_well"].liquid_color)  # type: ignore[arg-type]
    return tip_liquid_height_m


def configure_camera(camera: mujoco.MjvCamera, data: mujoco.MjData, tip_site_id: int, frame: DemoFrame) -> None:
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    if frame.camera_mode == "full_tool":
        camera.lookat[:] = np.array([frame.pipette_x_m, 0.0, 0.215], dtype=np.float64)
        camera.distance = 0.36
        camera.azimuth = 142.0
        camera.elevation = -8.0
        return
    if frame.camera_mode == "wide":
        camera.lookat[:] = np.array([0.0, 0.0, 0.115], dtype=np.float64)
        camera.distance = 0.34
        camera.azimuth = 138.0
        camera.elevation = -20.0
        return
    tip_pos = data.site_xpos[tip_site_id]
    camera.lookat[:] = tip_pos + np.array([0.0, 0.0, 0.012], dtype=np.float64)
    camera.distance = 0.16
    camera.azimuth = 142.0
    camera.elevation = -16.0


def annotate_frame(image: np.ndarray, state: WetLabLiquidState, frame: DemoFrame) -> np.ndarray:
    if Image is None or ImageDraw is None or ImageFont is None:
        return image
    pil_image = Image.fromarray(image)
    overlay = Image.new("RGBA", pil_image.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    source = state.containers["source_tube"]
    target = state.containers["target_well"]
    tip = state.pipette.tip
    depression = state.pipette.plunger.depression(frame.button_qpos_m)
    lines = [
        frame.stage,
        f"button qpos {frame.button_qpos_m:+.4f} m, depression {100.0 * depression:4.0f}%",
        f"source {source.volume_ul:6.1f} uL   tip {tip.volume_ul:6.1f} / {tip.capacity_ul:.0f} uL   target {target.volume_ul:6.1f} uL",
        f"tip liquid is fixed in local tip frame; source/target surfaces update from volume",
    ]
    line_height = 18
    box_w = 620
    box_h = 20 + line_height * len(lines)
    draw.rounded_rectangle((18, 18, 18 + box_w, 18 + box_h), radius=6, fill=(250, 250, 250, 215))
    for i, line in enumerate(lines):
        draw.text((30, 30 + i * line_height), line, fill=(25, 35, 40, 255), font=font)
    return np.asarray(Image.alpha_composite(pil_image.convert("RGBA"), overlay).convert("RGB"))


def write_demo(
    *,
    video_path: Path,
    jsonl_path: Path,
    model_path: Path,
    frames: list[DemoFrame],
    state: WetLabLiquidState,
    args: argparse.Namespace,
    prefill_tip: bool = False,
) -> dict[str, object]:
    model, data, ids = load_model(model_path)
    pipette_body_id = int(ids["pipette_body"])
    button_joint_id = int(ids["button_joint"])
    button_qpos_adr = int(model.jnt_qposadr[button_joint_id])
    tip_site_id = int(ids["tip_site"])
    segment_proxies = get_tip_segment_proxies(model, ids["tip_segments"])  # type: ignore[arg-type]

    if prefill_tip:
        source = state.containers["source_tube"]
        source_sample_id = source.sample_id
        source_color = source.liquid_color
        removed = source.remove(args.tip_capacity_ul)
        state.pipette.tip.add_liquid(removed, sample_id=source_sample_id, color=source_color)

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    video_path.parent.mkdir(parents=True, exist_ok=True)
    frame_summaries: list[dict[str, object]] = []

    try:
        with imageio.get_writer(video_path, fps=args.fps, codec="libx264", ffmpeg_params=["-crf", "18"]) as writer:
            with jsonl_path.open("a", encoding="utf-8") as jsonl:
                for frame in frames:
                    model.body_pos[pipette_body_id] = np.array([frame.pipette_x_m, 0.0, PIPETTE_Z_M], dtype=np.float64)
                    data.qpos[button_qpos_adr] = frame.button_qpos_m
                    events = state.update_pipette_from_plunger(
                        frame.button_qpos_m,
                        source_name=frame.source_name,
                        target_name=frame.target_name,
                        tip_in_liquid=frame.tip_in_liquid,
                        tip_in_target=frame.tip_in_target,
                    )
                    tip_liquid_height_m = apply_visual_state(model, ids, segment_proxies, state)
                    mujoco.mj_forward(model, data)
                    configure_camera(camera, data, tip_site_id, frame)
                    renderer.update_scene(data, camera=camera)
                    image = renderer.render()
                    image = annotate_frame(image, state, frame)
                    writer.append_data(image)

                    snapshot = state.as_json()
                    snapshot["stage"] = frame.stage
                    snapshot["pipette_x_m"] = frame.pipette_x_m
                    snapshot["button_qpos_m"] = frame.button_qpos_m
                    snapshot["tip_liquid_height_local_z_m"] = tip_liquid_height_m
                    snapshot["new_events"] = [event.as_json() for event in events]
                    jsonl.write(json.dumps(snapshot, sort_keys=True) + "\n")
                    frame_summaries.append(snapshot)
    finally:
        renderer.close()

    final = frame_summaries[-1]
    return {
        "video": str(video_path),
        "frames": len(frames),
        "final_source_ul": final["containers"]["source_tube"]["volume_ul"],
        "final_tip_ul": final["pipette"]["tip"]["volume_ul"],
        "final_target_ul": final["containers"]["target_well"]["volume_ul"],
    }


def build_aspirate_demo(args: argparse.Namespace) -> tuple[list[DemoFrame], WetLabLiquidState, bool]:
    frames = hold("Pressed tip in source liquid", pipette_x_m=SOURCE_X_M, button_qpos_m=-0.008, count=args.hold_frames)
    frames += aspirate_frames(args)
    frames += hold("After aspiration", pipette_x_m=SOURCE_X_M, button_qpos_m=0.0, count=args.hold_frames)
    return frames, make_state(args, initial_qpos_m=-0.008), False


def build_dispense_demo(args: argparse.Namespace) -> tuple[list[DemoFrame], WetLabLiquidState, bool]:
    frames = hold("Full tip above target well", pipette_x_m=TARGET_X_M, button_qpos_m=0.0, count=args.hold_frames)
    frames += dispense_frames(args)
    frames += hold("After dispense", pipette_x_m=TARGET_X_M, button_qpos_m=-0.008, count=args.hold_frames)
    return frames, make_state(args, initial_qpos_m=0.0), True


def build_full_demo(args: argparse.Namespace) -> tuple[list[DemoFrame], WetLabLiquidState, bool]:
    frames = hold("Pressed tip in source liquid", pipette_x_m=SOURCE_X_M, button_qpos_m=-0.008, count=args.hold_frames, camera_mode="wide")
    frames += aspirate_frames(args, camera_mode="wide")
    frames += move_frames(args)
    frames += dispense_frames(args, camera_mode="wide")
    frames += hold("Complete tube-to-well transfer", pipette_x_m=TARGET_X_M, button_qpos_m=-0.008, count=args.hold_frames, camera_mode="wide")
    return frames, make_state(args, initial_qpos_m=-0.008), False


def build_plunger_button_demo(args: argparse.Namespace) -> tuple[list[DemoFrame], WetLabLiquidState, bool]:
    frames = hold("Full pipette view: plunger pressed in source", pipette_x_m=SOURCE_X_M, button_qpos_m=-0.008, count=args.hold_frames, camera_mode="full_tool")
    frames += aspirate_frames(args, camera_mode="full_tool")
    frames += move_frames(args)
    frames = [
        DemoFrame(
            stage=frame.stage,
            pipette_x_m=frame.pipette_x_m,
            button_qpos_m=frame.button_qpos_m,
            source_name=frame.source_name,
            target_name=frame.target_name,
            tip_in_liquid=frame.tip_in_liquid,
            tip_in_target=frame.tip_in_target,
            camera_mode="full_tool",
        )
        for frame in frames
    ]
    frames += dispense_frames(args, camera_mode="full_tool")
    frames += hold("Full pipette view: transfer complete", pipette_x_m=TARGET_X_M, button_qpos_m=-0.008, count=args.hold_frames, camera_mode="full_tool")
    return frames, make_state(args, initial_qpos_m=-0.008), False


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    render_model_path = prepare_frustum_model_xml(model_path, out_dir)
    jsonl_path = out_dir / "wet_state.jsonl"
    if jsonl_path.exists():
        jsonl_path.unlink()

    demos = [
        ("01_mujoco_tip_aspirate_close.mp4", build_aspirate_demo(args)),
        ("02_mujoco_tip_dispense_close.mp4", build_dispense_demo(args)),
        ("03_mujoco_full_tube_to_well_transfer.mp4", build_full_demo(args)),
        ("04_mujoco_plunger_button_full_view.mp4", build_plunger_button_demo(args)),
    ]
    summaries = []
    for filename, (frames, state, prefill_tip) in demos:
        summaries.append(
            write_demo(
                video_path=out_dir / filename,
                jsonl_path=jsonl_path,
                model_path=render_model_path,
                frames=frames,
                state=state,
                args=args,
                prefill_tip=prefill_tip,
            )
        )

    summary = {
        "source_model": str(model_path),
        "render_model": str(render_model_path),
        "tip_capacity_ul": args.tip_capacity_ul,
        "stroke_volume_ul": args.stroke_volume_ul,
        "source_initial_ul": SOURCE_INITIAL_UL,
        "source_capacity_ul": SOURCE_CAPACITY_UL,
        "target_capacity_ul": TARGET_CAPACITY_UL,
        "liquid_style": args.liquid_style,
        "liquid_rgba": list(LIQUID_STYLES[args.liquid_style]),
        "tip_liquid_proxy": "64 MuJoCo frustum mesh segments attached to pipette_tip local frame plus one clipped boundary cylinder; segment height is derived by inverting frustum volume",
        "container_liquid_proxy": "MuJoCo cylinder height updated from semantic container volume",
        "wet_state_jsonl": str(jsonl_path),
        "demos": summaries,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
