"""Run a minimal detachable-tip attach/eject smoke demo."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import imageio.v2 as imageio
import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {exc}") from exc


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.tip_attachment import TipAttachmentController  # noqa: E402


DEFAULT_MODEL = PROJECT_ROOT / "models/piper_aero_hand/scenes/tip_attach_demo.xml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/debug_rollouts/tip_attachment_demo"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--duration", type=float, default=4.0)
    parser.add_argument("--no-video", action="store_true")
    return parser.parse_args()


def resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def smoothstep(alpha: float) -> float:
    alpha = min(max(float(alpha), 0.0), 1.0)
    return alpha * alpha * (3.0 - 2.0 * alpha)


def lerp(a: float, b: float, alpha: float) -> float:
    return (1.0 - alpha) * float(a) + alpha * float(b)


def pipette_z_at_time(t: float) -> float:
    if t < 0.8:
        return lerp(0.105, 0.066, smoothstep(t / 0.8))
    if t < 1.15:
        return 0.066
    if t < 1.85:
        return lerp(0.066, 0.12, smoothstep((t - 1.15) / 0.70))
    return 0.12


def ejector_ctrl_at_time(t: float) -> float:
    if t < 2.35:
        return 0.0
    if t > 3.15:
        return -0.0095
    return lerp(0.0, -0.0095, smoothstep((t - 2.35) / 0.80))


def set_freejoint_pose(model: mujoco.MjModel, data: mujoco.MjData, joint_name: str, pos: np.ndarray, quat: np.ndarray) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise RuntimeError(f"Missing joint: {joint_name}")
    qpos_adr = int(model.jnt_qposadr[joint_id])
    qvel_adr = int(model.jnt_dofadr[joint_id])
    data.qpos[qpos_adr : qpos_adr + 3] = pos
    data.qpos[qpos_adr + 3 : qpos_adr + 7] = quat
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0


def make_camera(model: mujoco.MjModel) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
    camera.fixedcamid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "demo_camera")
    return camera


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    controller = TipAttachmentController(model)
    controller.reset(data)

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pipette_ejector_position")
    if actuator_id < 0:
        raise RuntimeError("Missing actuator: pipette_ejector_position")

    dt = float(model.opt.timestep)
    frame_count = int(round(float(args.duration) * int(args.fps)))
    steps_per_frame = max(1, int(round(1.0 / (int(args.fps) * dt))))
    camera = make_camera(model)
    renderer = None if args.no_video else mujoco.Renderer(model, height=args.height, width=args.width)

    frames: list[np.ndarray] = []
    events: list[dict[str, object]] = []
    diagnostics: list[dict[str, object]] = []
    pipette_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    try:
        step_index = 0
        for _frame_index in range(frame_count):
            for _ in range(steps_per_frame):
                t = float(data.time)
                set_freejoint_pose(
                    model,
                    data,
                    "pipette_free",
                    np.array([0.0, 0.0, pipette_z_at_time(t)], dtype=np.float64),
                    pipette_quat,
                )
                data.ctrl[actuator_id] = ejector_ctrl_at_time(t)
                mujoco.mj_forward(model, data)
                for event in controller.update(data, step_index=step_index):
                    diag = controller.diagnostics(data)
                    events.append({"event": event, "time_s": t, "step": step_index, "diagnostics": diag})
                mujoco.mj_step(model, data)
                step_index += 1

            if renderer is not None:
                renderer.update_scene(data, camera=camera)
                frames.append(renderer.render())
            if _frame_index % max(1, int(args.fps // 10)) == 0:
                diag = controller.diagnostics(data)
                diag["time_s"] = float(data.time)
                diagnostics.append(diag)
    finally:
        if renderer is not None:
            renderer.close()

    video_path = None
    if frames:
        video_path = out_dir / "tip_attachment_demo.mp4"
        imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=1)

    summary = {
        "model": str(model_path),
        "video": None if video_path is None else str(video_path),
        "events": events,
        "final": controller.diagnostics(data),
        "diagnostics": diagnostics,
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary["final"], indent=2))
    print(f"Wrote summary: {summary_path}")
    if video_path is not None:
        print(f"Wrote video: {video_path}")


if __name__ == "__main__":
    main()
