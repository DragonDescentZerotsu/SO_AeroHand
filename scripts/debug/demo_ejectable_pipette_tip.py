"""Render a simplified pipette ejector demo.

The demo keeps the pipette body fixed, models the tip as a free body initially
welded to the socket, then releases the weld and gives the tip an outward
velocity once the ejector slide reaches its threshold.
"""

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
DEFAULT_MODEL = PROJECT_ROOT / "models/piper_aero_hand/scenes/ejectable_pipette_tip_demo.xml"
DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/debug_rollouts/ejectable_pipette_tip"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--duration", type=float, default=2.4)
    parser.add_argument("--press-start", type=float, default=0.35)
    parser.add_argument("--press-end", type=float, default=1.2)
    parser.add_argument("--release-threshold", type=float, default=-0.008)
    parser.add_argument("--eject-speed", type=float, default=0.9)
    parser.add_argument("--pipette-euler-deg", nargs=3, type=float, default=(0.0, 0.0, 0.0))
    return parser.parse_args()


def resolve_path(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def interp_ejector_ctrl(t: float, *, press_start: float, press_end: float) -> float:
    if t <= press_start:
        return 0.0
    if t >= press_end:
        return -0.0095
    alpha = (t - press_start) / max(press_end - press_start, 1e-9)
    smooth = alpha * alpha * (3.0 - 2.0 * alpha)
    return -0.0092 * smooth


def euler_xyz_to_quat_wxyz(euler_deg: tuple[float, float, float] | list[float]) -> np.ndarray:
    roll, pitch, yaw = np.deg2rad(np.asarray(euler_deg, dtype=np.float64))
    cr, sr = np.cos(0.5 * roll), np.sin(0.5 * roll)
    cp, sp = np.cos(0.5 * pitch), np.sin(0.5 * pitch)
    cy, sy = np.cos(0.5 * yaw), np.sin(0.5 * yaw)
    return np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float64,
    )


def main() -> None:
    args = parse_args()
    model_path = resolve_path(args.model)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    pipette_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pipette")
    if pipette_body_id < 0:
        raise RuntimeError("Demo model is missing the pipette body.")
    model.body_quat[pipette_body_id] = euler_xyz_to_quat_wxyz(args.pipette_euler_deg)

    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "pipette_ejector_position")
    ejector_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pipette_ejector")
    tip_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "tip_free")
    weld_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, "tip_lock")
    socket_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "tip_socket_site")
    if min(actuator_id, ejector_joint_id, tip_joint_id, weld_id, socket_site_id) < 0:
        raise RuntimeError("Demo model is missing a required joint, actuator, equality, or site.")

    ejector_qpos_adr = int(model.jnt_qposadr[ejector_joint_id])
    tip_qvel_adr = int(model.jnt_dofadr[tip_joint_id])
    tip_qpos_adr = int(model.jnt_qposadr[tip_joint_id])
    ejector_stiffness = float(model.jnt_stiffness[ejector_joint_id])
    ejector_springref = float(model.qpos_spring[ejector_qpos_adr])
    ejector_range = model.jnt_range[ejector_joint_id].copy()
    mujoco.mj_forward(model, data)
    socket_pos = data.site_xpos[socket_site_id].copy()
    socket_quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(socket_quat, data.site_xmat[socket_site_id])
    data.qpos[tip_qpos_adr : tip_qpos_adr + 3] = socket_pos
    data.qpos[tip_qpos_adr + 3 : tip_qpos_adr + 7] = socket_quat
    mujoco.mj_forward(model, data)
    frame_count = int(round(args.duration * args.fps))
    dt = float(model.opt.timestep)
    steps_per_frame = max(1, int(round(1.0 / (args.fps * dt))))

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = np.array([0.0, 0.0, 0.21], dtype=np.float64)
    camera.distance = 0.58
    camera.azimuth = 135.0
    camera.elevation = -12.0

    renderer = mujoco.Renderer(model, height=args.height, width=args.width)
    video_path = out_dir / "ejectable_pipette_tip_demo.mp4"
    summary_path = out_dir / "summary.json"
    frames: list[np.ndarray] = []
    released = False
    release_time: float | None = None
    release_velocity_world: list[float] | None = None
    release_axis_world: list[float] | None = None

    try:
        for frame_index in range(frame_count):
            for _ in range(steps_per_frame):
                t = float(data.time)
                data.ctrl[actuator_id] = interp_ejector_ctrl(
                    t,
                    press_start=float(args.press_start),
                    press_end=float(args.press_end),
                )
                if (not released) and float(data.qpos[ejector_qpos_adr]) <= float(args.release_threshold):
                    data.eq_active[weld_id] = 0
                    socket_rotation = data.site_xmat[socket_site_id].reshape(3, 3)
                    eject_axis_world = -socket_rotation[:, 2].copy()
                    eject_velocity_world = float(args.eject_speed) * eject_axis_world
                    data.qvel[tip_qvel_adr : tip_qvel_adr + 3] = eject_velocity_world
                    released = True
                    release_time = t
                    release_axis_world = eject_axis_world.tolist()
                    release_velocity_world = eject_velocity_world.tolist()
                mujoco.mj_step(model, data)

            renderer.update_scene(data, camera=camera)
            frames.append(renderer.render())
    finally:
        renderer.close()

    imageio.mimsave(video_path, frames, fps=args.fps, macro_block_size=1)

    summary = {
        "model": str(model_path),
        "video": str(video_path),
        "released": released,
        "release_time_s": release_time,
        "final_time_s": float(data.time),
        "final_ejector_qpos_m": float(data.qpos[ejector_qpos_adr]),
        "ejector_springref_m": ejector_springref,
        "ejector_stiffness_n_per_m": ejector_stiffness,
        "estimated_force_at_rest_n": ejector_stiffness * abs(0.0 - ejector_springref),
        "estimated_force_at_bottom_n": ejector_stiffness * abs(float(ejector_range[0]) - ejector_springref),
        "pipette_euler_deg": list(args.pipette_euler_deg),
        "final_tip_qpos": data.qpos[tip_qpos_adr : tip_qpos_adr + 7].tolist(),
        "release_threshold_m": float(args.release_threshold),
        "eject_speed_m_s": float(args.eject_speed),
        "release_axis_world": release_axis_world,
        "release_velocity_world_m_s": release_velocity_world,
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote video: {video_path}")
    print(f"Wrote summary: {summary_path}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
