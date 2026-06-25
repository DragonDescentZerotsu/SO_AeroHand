"""Render preview frames and dump camera-frame diagnostics for LeRobot cameras."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.lerobot_export import DEFAULT_HANDOFF_CAMERAS, MujocoTrajectoryRenderer  # noqa: E402

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {exc}") from exc


DEFAULT_TRAJECTORY = PROJECT_ROOT / "outputs/lerobot/piper_pipette_handoff/v1_camera_fixed_2ep/raw/episode_000000/piper_gripper_pipette_handoff_expert.npz"
DEFAULT_MODEL = PROJECT_ROOT / "models/piper_aero_hand/scenes/Piper_dual_pipette_rack_table.xml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--out-dir", default=str(PROJECT_ROOT / "outputs/lerobot/camera_preview"))
    parser.add_argument("--width", type=int, default=480)
    parser.add_argument("--height", type=int, default=360)
    parser.add_argument("--frame", type=int, action="append", default=None)
    return parser.parse_args()


def camera_diagnostics(model, data) -> list[dict[str, object]]:
    diagnostics: list[dict[str, object]] = []
    for spec in DEFAULT_HANDOFF_CAMERAS:
        record: dict[str, object] = {
            "name": spec.name,
            "mode": spec.mode,
        }
        if spec.mode == "world":
            record["eye_world"] = list(spec.eye_offset_world)
            record["target_world"] = list(spec.lookat)
        if spec.body is not None:
            body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body)
            body_pos = data.xpos[body_id].copy()
            body_R = data.xmat[body_id].reshape(3, 3).copy()
            eye = body_pos + body_R @ np.asarray(spec.eye_offset_local, dtype=np.float64)
            target = body_pos + body_R @ np.asarray(spec.target_offset_local, dtype=np.float64)
            up = body_R @ np.asarray(spec.up_axis_local, dtype=np.float64)
            record.update(
                {
                    "body": spec.body,
                    "body_pos_world": body_pos.tolist(),
                    "body_R_rows": body_R.tolist(),
                    "body_local_axes_world_columns": {
                        "+X": body_R[:, 0].tolist(),
                        "+Y": body_R[:, 1].tolist(),
                        "+Z": body_R[:, 2].tolist(),
                    },
                    "eye_offset_local": list(spec.eye_offset_local),
                    "target_offset_local": list(spec.target_offset_local),
                    "up_axis_local": list(spec.up_axis_local),
                    "eye_world": eye.tolist(),
                    "target_world": target.tolist(),
                    "up_hint_world": up.tolist(),
                    "eye_minus_body_world": (eye - body_pos).tolist(),
                    "target_minus_body_world": (target - body_pos).tolist(),
                }
            )
        diagnostics.append(record)
    return diagnostics


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir).expanduser()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    trajectory_path = Path(args.trajectory).expanduser()
    if not trajectory_path.is_absolute():
        trajectory_path = PROJECT_ROOT / trajectory_path

    model = mujoco.MjModel.from_xml_path(str(model_path))
    qpos = np.load(trajectory_path)["qpos"]
    frame_ids = args.frame or [0, qpos.shape[0] // 2, qpos.shape[0] - 1]
    frame_ids = [min(max(0, int(frame_id)), qpos.shape[0] - 1) for frame_id in frame_ids]

    renderer = MujocoTrajectoryRenderer(model, DEFAULT_HANDOFF_CAMERAS, width=args.width, height=args.height)
    diagnostics: dict[str, object] = {
        "model": str(model_path),
        "trajectory": str(trajectory_path),
        "frames": {},
    }
    try:
        for frame_id in frame_ids:
            images = renderer.render(qpos[frame_id])
            frame_key = f"{frame_id:06d}"
            for name, image in images.items():
                image_path = out_dir / f"{name}_{frame_key}.png"
                imageio.imwrite(image_path, image)
            diagnostics["frames"][frame_key] = camera_diagnostics(model, renderer.data)
    finally:
        renderer.close()

    debug_path = out_dir / "camera_debug.json"
    debug_path.write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    print(f"Wrote previews and diagnostics: {out_dir}")
    print(f"Wrote camera debug JSON: {debug_path}")


if __name__ == "__main__":
    main()
