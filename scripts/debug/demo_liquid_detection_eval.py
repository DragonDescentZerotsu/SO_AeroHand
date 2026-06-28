#!/usr/bin/env python3
"""Render a minimal semantic liquid detection and BCS demo."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
import sys

import imageio.v2 as imageio
import mujoco
import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError:  # pragma: no cover
    Image = None
    ImageDraw = None
    ImageFont = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.liquid import CylindricalGeometry, ContainerState, M3_TO_UL, PipetteLiquidController, PipetteTipState, PlungerModel  # noqa: E402
from aero_tasks.liquid_detection import CircularContainerRegion, detect_tip_in_circular_container  # noqa: E402
from aero_tasks.liquid_eval import WellVolumeExpectation, evaluate_bcs  # noqa: E402


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/debug_rollouts/liquid_detection_eval"
GRAVITY_WORLD = np.array([0.0, 0.0, -9.81], dtype=np.float64)
SOURCE_POS = np.array([-0.065, 0.0, 0.0], dtype=np.float64)
TARGET_POS = np.array([0.065, 0.0, 0.0], dtype=np.float64)
CONTAINER_RADIUS_M = 0.0040
CONTAINER_TOP_M = 0.035
LIQUID_RGBA = (0.18, 0.62, 1.0, 0.64)


@dataclass(frozen=True)
class DemoFrame:
    stage: str
    tip_pos_world: np.ndarray
    qpos_m: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--source-volume-ul", type=float, default=900.0)
    parser.add_argument("--stroke-volume-ul", type=float, default=200.0)
    parser.add_argument("--tip-capacity-ul", type=float, default=200.0)
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    return path if path.is_absolute() else PROJECT_ROOT / path


def smoothstep(alpha: float) -> float:
    alpha = min(max(float(alpha), 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def lerp_vec(start: np.ndarray, end: np.ndarray, count: int) -> list[np.ndarray]:
    if count <= 1:
        return [end.copy()]
    return [start + (end - start) * smoothstep(i / (count - 1)) for i in range(count)]


def lerp_float(start: float, end: float, count: int) -> list[float]:
    if count <= 1:
        return [float(end)]
    return [start + (end - start) * smoothstep(i / (count - 1)) for i in range(count)]


def build_xml() -> str:
    return f"""
<mujoco model="liquid_detection_eval_demo">
  <compiler angle="radian"/>
  <option timestep="0.01"/>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <material name="container_clear" rgba="0.72 0.92 1.0 0.20" specular="0.45" shininess="0.55"/>
    <material name="liquid" rgba="{LIQUID_RGBA[0]} {LIQUID_RGBA[1]} {LIQUID_RGBA[2]} {LIQUID_RGBA[3]}" specular="0.85" shininess="0.85"/>
    <material name="tip" rgba="0.05 0.06 0.07 0.92" specular="0.25" shininess="0.4"/>
    <material name="table" rgba="0.84 0.84 0.80 1"/>
  </asset>
  <worldbody>
    <light pos="0 -0.45 0.5" dir="0 1 -1" diffuse="0.9 0.9 0.9"/>
    <geom name="table" type="box" pos="0 0 -0.002" size="0.12 0.055 0.002" material="table" contype="0" conaffinity="0"/>
    <body name="source_wall" pos="{SOURCE_POS[0]} {SOURCE_POS[1]} 0.0175">
      <geom type="cylinder" size="0.006 0.0175" material="container_clear" contype="0" conaffinity="0"/>
    </body>
    <body name="target_wall" pos="{TARGET_POS[0]} {TARGET_POS[1]} 0.0175">
      <geom type="cylinder" size="0.006 0.0175" material="container_clear" contype="0" conaffinity="0"/>
    </body>
    <body name="source_liquid_body" pos="{SOURCE_POS[0]} {SOURCE_POS[1]} 0.001">
      <geom name="source_liquid" type="cylinder" size="{CONTAINER_RADIUS_M} 0.001" material="liquid" contype="0" conaffinity="0"/>
    </body>
    <body name="target_liquid_body" pos="{TARGET_POS[0]} {TARGET_POS[1]} 0.001">
      <geom name="target_liquid" type="cylinder" size="{CONTAINER_RADIUS_M} 0.001" material="liquid" contype="0" conaffinity="0"/>
    </body>
    <body name="tip_body" pos="0 0 0.06">
      <geom name="tip_body" type="capsule" fromto="0 0 0 0 0 0.040" size="0.0012" material="tip" contype="0" conaffinity="0"/>
      <site name="tip_site" pos="0 0 0" size="0.0015" rgba="1 0.15 0.08 1"/>
    </body>
    <body name="tip_liquid_body" pos="0 0 0.02">
      <geom name="tip_liquid" type="cylinder" size="0.00085 0.001" material="liquid" contype="0" conaffinity="0"/>
    </body>
  </worldbody>
</mujoco>
"""


def make_container(name: str, pos: np.ndarray, volume_ul: float) -> tuple[ContainerState, CircularContainerRegion]:
    geometry = CylindricalGeometry(
        radius_m=CONTAINER_RADIUS_M,
        bottom_z_m=0.0,
    )
    capacity_ul = float(np.pi * CONTAINER_RADIUS_M * CONTAINER_RADIUS_M * CONTAINER_TOP_M * M3_TO_UL)
    container = ContainerState(
        name=name,
        geometry=geometry,
        volume_ul=volume_ul,
        capacity_ul=capacity_ul,
        sample_id="sample_A" if volume_ul > 0.0 else None,
        liquid_color=LIQUID_RGBA,
    )
    region = CircularContainerRegion(
        name=name,
        center_world=tuple(float(v) for v in pos),
        radius_m=0.006,
        bottom_z_m=-0.001,
        top_z_m=CONTAINER_TOP_M,
    )
    return container, region


def make_controller(args: argparse.Namespace) -> PipetteLiquidController:
    tip = PipetteTipState(capacity_ul=args.tip_capacity_ul, liquid_color=LIQUID_RGBA)
    plunger = PlungerModel(qpos_rest_m=0.0, qpos_pressed_m=-0.008, stroke_volume_ul=args.stroke_volume_ul)
    return PipetteLiquidController.from_initial_qpos(tip=tip, plunger=plunger, qpos_m=-0.008)


def make_frames() -> list[DemoFrame]:
    source_submerged = SOURCE_POS + np.array([0.0, 0.0, 0.010])
    source_above = SOURCE_POS + np.array([0.0, 0.0, 0.055])
    target_submerged = TARGET_POS + np.array([0.0, 0.0, 0.010])
    target_above = TARGET_POS + np.array([0.0, 0.0, 0.055])
    frames: list[DemoFrame] = []
    frames += [DemoFrame("Tip below source liquid; plunger starts pressed", source_submerged, -0.008) for _ in range(36)]
    for qpos in lerp_float(-0.008, 0.0, 96):
        frames.append(DemoFrame("Aspirate: source volume drops, tip volume rises", source_submerged, qpos))
    for pos in lerp_vec(source_submerged, source_above, 40):
        frames.append(DemoFrame("Lift tip out of source", pos, 0.0))
    for pos in lerp_vec(source_above, target_above, 74):
        frames.append(DemoFrame("Move full tip toward target well", pos, 0.0))
    for pos in lerp_vec(target_above, target_submerged, 40):
        frames.append(DemoFrame("Insert tip into target well", pos, 0.0))
    for qpos in lerp_float(0.0, -0.008, 96):
        frames.append(DemoFrame("Dispense: target reaches expected sample volume", target_submerged, qpos))
    frames += [DemoFrame("BCS succeeds after correct sample and volume", target_submerged, -0.008) for _ in range(48)]
    return frames


def mj_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise RuntimeError(f"Missing MuJoCo object {name!r}")
    return int(obj_id)


def update_liquid_geom(model: mujoco.MjModel, body_id: int, geom_id: int, pos_xy: np.ndarray, container: ContainerState) -> None:
    height = max(float(container.geometry.volume_to_height_m(container.volume_ul)), 0.0)
    if height <= 1e-7 or container.volume_ul <= 1e-7:
        model.geom_rgba[geom_id, 3] = 0.0
        model.geom_size[geom_id] = np.array([CONTAINER_RADIUS_M, 1e-5, 0.0])
        model.body_pos[body_id] = np.array([pos_xy[0], pos_xy[1], 1e-5])
        return
    model.geom_rgba[geom_id] = np.array(container.liquid_color, dtype=np.float64)
    model.geom_size[geom_id] = np.array([CONTAINER_RADIUS_M, 0.5 * height, 0.0])
    model.body_pos[body_id] = np.array([pos_xy[0], pos_xy[1], 0.5 * height])


def update_tip_liquid(model: mujoco.MjModel, body_id: int, geom_id: int, tip_pos: np.ndarray, controller: PipetteLiquidController) -> None:
    fill = max(0.0, min(1.0, controller.tip.volume_ul / max(controller.tip.capacity_ul, 1e-9)))
    height = 0.030 * fill
    if height <= 1e-7:
        model.geom_rgba[geom_id, 3] = 0.0
        model.geom_size[geom_id] = np.array([0.00085, 1e-5, 0.0])
        model.body_pos[body_id] = tip_pos + np.array([0.0, 0.0, 1e-5])
        return
    model.geom_rgba[geom_id] = np.array(controller.tip.liquid_color, dtype=np.float64)
    model.geom_size[geom_id] = np.array([0.00085, 0.5 * height, 0.0])
    model.body_pos[body_id] = tip_pos + np.array([0.0, 0.0, 0.5 * height])


def annotate(
    image: np.ndarray,
    frame: DemoFrame,
    source: ContainerState,
    target: ContainerState,
    controller: PipetteLiquidController,
    source_in_liquid: bool,
    target_in_container: bool,
    bcs: dict[str, object],
) -> np.ndarray:
    if Image is None or ImageDraw is None or ImageFont is None:
        return image
    pil = Image.fromarray(image)
    overlay = Image.new("RGBA", pil.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = ImageFont.load_default()
    lines = [
        frame.stage,
        f"source {source.volume_ul:6.1f} uL   tip {controller.tip.volume_ul:6.1f} uL   target {target.volume_ul:6.1f} uL",
        f"tip_in_source_liquid={source_in_liquid}   tip_in_target_region={target_in_container}   qpos {frame.qpos_m:+.4f} m",
        f"BCS score={bcs['score']:.0f}  sample_ok={bcs['sample_ok']}  volume_ok={bcs['volume_ok']}  no_contamination={bcs['no_contamination']}",
    ]
    line_height = 18
    draw.rounded_rectangle((18, 18, 710, 32 + line_height * len(lines)), radius=6, fill=(250, 250, 250, 220))
    for index, line in enumerate(lines):
        draw.text((30, 30 + index * line_height), line, fill=(25, 35, 40, 255), font=font)
    return np.asarray(Image.alpha_composite(pil.convert("RGBA"), overlay).convert("RGB"))


def main() -> None:
    args = parse_args()
    args.out_dir = resolve_path(args.out_dir)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_string(build_xml())
    data = mujoco.MjData(model)
    ids = {
        "source_liquid_body": mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "source_liquid_body"),
        "source_liquid_geom": mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "source_liquid"),
        "target_liquid_body": mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "target_liquid_body"),
        "target_liquid_geom": mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "target_liquid"),
        "tip_body": mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "tip_body"),
        "tip_site": mj_id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_site"),
        "tip_liquid_body": mj_id(model, mujoco.mjtObj.mjOBJ_BODY, "tip_liquid_body"),
        "tip_liquid_geom": mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, "tip_liquid"),
    }

    source, source_region = make_container("source_tube", SOURCE_POS, args.source_volume_ul)
    target, target_region = make_container("target_well", TARGET_POS, 0.0)
    controller = make_controller(args)
    expectations = [
        WellVolumeExpectation(
            container_name="target_well",
            sample_id="sample_A",
            min_volume_ul=args.stroke_volume_ul - 5.0,
            max_volume_ul=args.stroke_volume_ul + 5.0,
        )
    ]
    frames = make_frames()
    video_path = args.out_dir / "01_liquid_detection_bcs_transfer.mp4"
    wet_state_path = args.out_dir / "wet_state.jsonl"
    if wet_state_path.exists():
        wet_state_path.unlink()

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.0, 0.0, 0.030])
    camera.distance = 0.22
    camera.azimuth = 145.0
    camera.elevation = -16.0
    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    snapshots: list[dict[str, object]] = []
    try:
        with imageio.get_writer(video_path, fps=args.fps, codec="libx264", ffmpeg_params=["-crf", "18"]) as writer:
            with wet_state_path.open("w", encoding="utf-8") as wet_state:
                for frame_index, frame in enumerate(frames):
                    model.body_pos[ids["tip_body"]] = frame.tip_pos_world
                    mujoco.mj_forward(model, data)
                    tip_site_world = data.site_xpos[ids["tip_site"]].copy()
                    source_hit = detect_tip_in_circular_container(
                        tip_site_world,
                        container=source,
                        region=source_region,
                        gravity_world=GRAVITY_WORLD,
                    )
                    target_hit = detect_tip_in_circular_container(
                        tip_site_world,
                        container=target,
                        region=target_region,
                        gravity_world=GRAVITY_WORLD,
                    )
                    events = controller.update(
                        frame.qpos_m,
                        source=source if source_hit.tip_in_liquid else None,
                        target=target if target_hit.tip_in_container else None,
                        tip_in_liquid=source_hit.tip_in_liquid,
                        tip_in_target=target_hit.tip_in_container,
                    )
                    update_liquid_geom(model, ids["source_liquid_body"], ids["source_liquid_geom"], SOURCE_POS[:2], source)
                    update_liquid_geom(model, ids["target_liquid_body"], ids["target_liquid_geom"], TARGET_POS[:2], target)
                    update_tip_liquid(model, ids["tip_liquid_body"], ids["tip_liquid_geom"], tip_site_world, controller)
                    mujoco.mj_forward(model, data)
                    bcs = evaluate_bcs(
                        {"source_tube": source, "target_well": target},
                        expectations=expectations,
                        tip=controller.tip,
                    ).as_json()
                    renderer.update_scene(data, camera=camera)
                    writer.append_data(
                        annotate(
                            renderer.render(),
                            frame,
                            source,
                            target,
                            controller,
                            source_hit.tip_in_liquid,
                            target_hit.tip_in_container,
                            bcs,
                        )
                    )
                    snapshot = {
                        "frame_index": frame_index,
                        "stage": frame.stage,
                        "qpos_m": frame.qpos_m,
                        "source": source.as_json(),
                        "target": target.as_json(),
                        "tip": controller.tip.as_json(),
                        "source_hit": source_hit.as_json(),
                        "target_hit": target_hit.as_json(),
                        "events": [event.as_json() for event in events],
                        "bcs": bcs,
                    }
                    wet_state.write(json.dumps(snapshot, sort_keys=True) + "\n")
                    snapshots.append(snapshot)
    finally:
        renderer.close()

    summary = {
        "video": str(video_path),
        "wet_state_jsonl": str(wet_state_path),
        "frames": len(frames),
        "stroke_volume_ul": args.stroke_volume_ul,
        "final_source_volume_ul": snapshots[-1]["source"]["volume_ul"],
        "final_tip_volume_ul": snapshots[-1]["tip"]["volume_ul"],
        "final_target_volume_ul": snapshots[-1]["target"]["volume_ul"],
        "final_bcs": snapshots[-1]["bcs"],
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
