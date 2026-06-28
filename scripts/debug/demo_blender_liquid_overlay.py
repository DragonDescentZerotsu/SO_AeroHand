#!/usr/bin/env python3
"""Create a small qpos + wet-state demo for the Blender liquid renderer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.blender_render import BlenderRenderConfig, prepare_blender_render, run_blender_render  # noqa: E402
from aero_tasks.liquid import CylindricalGeometry, ContainerState, M3_TO_UL, PipetteLiquidController, PipetteTipState, PlungerModel  # noqa: E402


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/debug_rollouts/blender_liquid_overlay_demo"
SOURCE_POS = np.array([-0.06, 0.0, 0.0], dtype=np.float64)
TARGET_POS = np.array([0.06, 0.0, 0.0], dtype=np.float64)
RADIUS_M = 0.004
HEIGHT_M = 0.035
LIQUID_COLOR = (0.70, 0.92, 1.0, 0.42)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--render", action="store_true", help="Run Blender if available; otherwise only prepare files.")
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def smoothstep(alpha: float) -> float:
    alpha = min(max(float(alpha), 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def lerp(start: np.ndarray, end: np.ndarray, count: int) -> list[np.ndarray]:
    return [start + (end - start) * smoothstep(index / max(count - 1, 1)) for index in range(count)]


def lerp_scalar(start: float, end: float, count: int) -> list[float]:
    return [start + (end - start) * smoothstep(index / max(count - 1, 1)) for index in range(count)]


def write_model(path: Path) -> None:
    xml = f"""
<mujoco model="blender_liquid_overlay_demo">
  <compiler angle="radian"/>
  <option timestep="0.01"/>
  <asset>
    <material name="table" rgba="0.78 0.78 0.74 1"/>
    <material name="container" rgba="0.72 0.92 1 0.25"/>
    <material name="tip" rgba="0.92 0.94 0.96 1"/>
    <material name="tip_site" rgba="1 0.12 0.04 1"/>
  </asset>
  <worldbody>
    <geom name="table" type="box" pos="0 0 -0.002" size="0.12 0.055 0.002" material="table" contype="0" conaffinity="0"/>
    <geom name="source_wall" type="cylinder" pos="{SOURCE_POS[0]} 0 0.0175" size="0.006 0.0175" material="container" contype="0" conaffinity="0"/>
    <geom name="target_wall" type="cylinder" pos="{TARGET_POS[0]} 0 0.0175" size="0.006 0.0175" material="container" contype="0" conaffinity="0"/>
    <body name="demo_tip" pos="{SOURCE_POS[0]} 0 0.055">
      <freejoint name="demo_tip_free"/>
      <geom name="tip_shaft" type="capsule" fromto="0 0 0.000 0 0 0.052" size="0.0014" material="tip" contype="0" conaffinity="0"/>
      <geom name="tip_collar" type="cylinder" pos="0 0 0.058" size="0.0036 0.006" material="tip" contype="0" conaffinity="0"/>
      <site name="tip_site" pos="0 0 0" size="0.0022" rgba="1 0.12 0.04 1"/>
    </body>
  </worldbody>
</mujoco>
"""
    path.write_text(xml, encoding="utf-8")


def make_container(name: str, volume_ul: float) -> ContainerState:
    geometry = CylindricalGeometry(radius_m=RADIUS_M, bottom_z_m=0.0)
    capacity_ul = float(np.pi * RADIUS_M * RADIUS_M * HEIGHT_M * M3_TO_UL)
    return ContainerState(
        name=name,
        geometry=geometry,
        volume_ul=volume_ul,
        capacity_ul=capacity_ul,
        sample_id="sample_A" if volume_ul > 0 else None,
        liquid_color=LIQUID_COLOR,
    )


def surface_json(container: ContainerState, pos: np.ndarray) -> dict[str, object]:
    surface = container.surface(container_pos_world=pos).as_json()
    surface["half_width_m"] = RADIUS_M
    surface["half_height_m"] = RADIUS_M
    return surface


def build_frames() -> tuple[np.ndarray, list[dict[str, object]]]:
    source = make_container("source_tube", 900.0)
    target = make_container("target_well", 0.0)
    controller = PipetteLiquidController.from_initial_qpos(
        tip=PipetteTipState(capacity_ul=200.0, liquid_color=LIQUID_COLOR),
        plunger=PlungerModel(qpos_rest_m=0.0, qpos_pressed_m=-0.008, stroke_volume_ul=200.0),
        qpos_m=-0.008,
    )
    source_in = SOURCE_POS + np.array([0.0, 0.0, 0.010])
    source_above = SOURCE_POS + np.array([0.0, 0.0, 0.060])
    target_above = TARGET_POS + np.array([0.0, 0.0, 0.060])
    target_in = TARGET_POS + np.array([0.0, 0.0, 0.010])
    positions: list[np.ndarray] = []
    qpos_m: list[float] = []
    stages: list[str] = []
    for value in lerp_scalar(-0.008, 0.0, 80):
        positions.append(source_in)
        qpos_m.append(value)
        stages.append("aspirate")
    for pos in lerp(source_in, source_above, 28) + lerp(source_above, target_above, 54) + lerp(target_above, target_in, 28):
        positions.append(pos)
        qpos_m.append(0.0)
        stages.append("move")
    for value in lerp_scalar(0.0, -0.008, 80):
        positions.append(target_in)
        qpos_m.append(value)
        stages.append("dispense")

    qpos = np.zeros((len(positions), 7), dtype=np.float64)
    wet_records: list[dict[str, object]] = []
    for index, (pos, plunger_qpos, stage) in enumerate(zip(positions, qpos_m, stages, strict=True)):
        qpos[index, :3] = pos
        qpos[index, 3:7] = np.array([1.0, 0.0, 0.0, 0.0])
        in_source = stage == "aspirate"
        in_target = stage == "dispense"
        events = controller.update(
            plunger_qpos,
            source=source if in_source else None,
            target=target if in_target else None,
            tip_in_liquid=in_source,
            tip_in_target=in_target,
        )
        wet_records.append(
            {
                "frame_index": index,
                "stage": stage,
                "tip_site_world": pos.tolist(),
                "source": source.as_json(),
                "target": target.as_json(),
                "tip": controller.tip.as_json(),
                "source_hit": {
                    "container_name": "source_tube",
                    "tip_in_container": bool(in_source),
                    "tip_in_liquid": bool(in_source),
                    "surface": surface_json(source, SOURCE_POS),
                },
                "target_hit": {
                    "container_name": "target_well",
                    "tip_in_container": bool(in_target),
                    "tip_in_liquid": bool(in_target and target.volume_ul > 0),
                    "surface": surface_json(target, TARGET_POS),
                },
                "events": [event.as_json() for event in events],
            }
        )
    return qpos, wet_records


def main() -> None:
    args = parse_args()
    out_dir = args.out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "synthetic_liquid_transfer.xml"
    trajectory_path = out_dir / "synthetic_liquid_transfer.npz"
    wet_state_path = out_dir / "wet_state.jsonl"
    write_model(model_path)
    qpos, wet_records = build_frames()
    np.savez_compressed(trajectory_path, qpos=qpos, model=str(model_path))
    with wet_state_path.open("w", encoding="utf-8") as handle:
        for record in wet_records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    config = BlenderRenderConfig(
        trajectory=trajectory_path,
        model=model_path,
        wet_state=wet_state_path,
        out_dir=out_dir,
        output_name="01_blender_liquid_overlay.mp4",
        camera="table_overview",
        fps=args.fps,
        max_frames=args.max_frames,
        blender=args.blender,
        engine="BLENDER_EEVEE_NEXT",
        samples=64,
    )
    if args.render:
        try:
            output = run_blender_render(config)
            print(f"Wrote Blender video: {output}")
            return
        except FileNotFoundError as exc:
            print(str(exc))
    manifest, command = prepare_blender_render(config)
    summary = {
        "model": str(model_path),
        "trajectory": str(trajectory_path),
        "wet_state": str(wet_state_path),
        "manifest": str(manifest),
        "render_command": str(out_dir / "render_command.sh"),
        "command": command,
        "frames": int(qpos.shape[0]),
        "final_source_volume_ul": wet_records[-1]["source"]["volume_ul"],
        "final_target_volume_ul": wet_records[-1]["target"]["volume_ul"],
        "final_tip_volume_ul": wet_records[-1]["tip"]["volume_ul"],
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
