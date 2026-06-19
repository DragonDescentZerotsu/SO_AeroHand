"""Benchmark Piper + Aero Hand pose tracking with the production OSQP IK."""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")

import imageio.v2 as imageio
import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.arm_teleop import (
    DampedLeastSquaresIK,
    VelocityTeleopConfig,
    VelocityTeleopController,
    joint_qpos,
    joint_ranges,
)
from aero_quest.osqp_ik import OSQPIKConfig, OSQPVelocityIK


DEFAULT_MODEL = PROJECT_ROOT / "models/piper_aero_hand/Piper_aerohand.xml"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "outputs/piper_ik_benchmark"
ARM_JOINTS = ["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"]
HOME_QPOS = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float64)
JOINT_MOTION_WEIGHTS = np.array([0.7, 1.0, 1.0, 0.35, 0.22, 0.08], dtype=np.float64)
TASK_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.2, 1.2, 1.2], dtype=np.float64)
ACTUATOR_KP = np.array([140.0, 140.0, 140.0, 100.0, 90.0, 90.0], dtype=np.float64)
ACTUATOR_KV = np.array([10.0, 10.0, 10.0, 7.0, 5.0, 5.0], dtype=np.float64)

# Targets are converted to Cartesian poses by FK before the benchmark starts.
# This guarantees that every requested palm pose is reachable by construction.
TARGET_JOINT_CONFIGS = (
    ("forward_left", np.array([0.35, 1.30, -1.22, 0.30, 0.28, 0.45])),
    ("forward_right", np.array([-0.32, 1.38, -1.48, -0.28, 0.36, -0.55])),
    ("high_wrist_roll", np.array([0.12, 1.72, -1.55, 0.42, -0.35, 1.05])),
    ("low_wrist_flex", np.array([-0.12, 1.48, -1.16, -0.48, 0.48, -0.85])),
    ("return_home", HOME_QPOS.copy()),
)


@dataclass(frozen=True)
class PoseTarget:
    name: str
    position: np.ndarray
    rotation: np.ndarray
    source_qpos: np.ndarray


@dataclass
class TargetResult:
    name: str
    success: bool
    settle_time_s: float | None
    final_position_error_m: float
    final_orientation_error_deg: float
    minimum_position_error_m: float
    minimum_orientation_error_deg: float
    max_abs_qdot_rad_s: float
    max_abs_qacc_rad_s2: float
    mean_osqp_wall_time_ms: float
    p95_osqp_wall_time_ms: float
    max_osqp_wall_time_ms: float
    mean_osqp_iterations: float
    min_jacobian_singular_value: float
    max_effective_damping: float
    osqp_failures: int


def rotation_angle(target: np.ndarray, current: np.ndarray) -> float:
    delta = np.asarray(target).reshape(3, 3) @ np.asarray(current).reshape(3, 3).T
    cosine = 0.5 * (float(np.trace(delta)) - 1.0)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def set_arm_state(model, data, joint_ids: list[int], qpos: np.ndarray) -> None:
    for joint_id, value in zip(joint_ids, np.asarray(qpos, dtype=np.float64)):
        data.qpos[model.jnt_qposadr[joint_id]] = float(value)
        data.qvel[model.jnt_dofadr[joint_id]] = 0.0
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id),
        )
        if actuator_id >= 0:
            data.ctrl[actuator_id] = float(value)


def set_actuator_gains(model, joint_ids: list[int]) -> None:
    for index, joint_id in enumerate(joint_ids):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        model.actuator_gainprm[actuator_id, 0] = ACTUATOR_KP[index]
        model.actuator_biasprm[actuator_id, 1] = -ACTUATOR_KP[index]
        model.actuator_biasprm[actuator_id, 2] = -ACTUATOR_KV[index]


def apply_arm_targets(model, data, joint_ids: list[int], qtarget: np.ndarray) -> None:
    for joint_id, value in zip(joint_ids, np.asarray(qtarget, dtype=np.float64)):
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        lo, hi = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = float(np.clip(value, lo, hi))


def robot_subtree_gravity_compensation(model, root_name: str = "base_link") -> None:
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_name)
    for body_id in range(1, model.nbody):
        cursor = body_id
        while cursor:
            if cursor == root_id:
                model.body_gravcomp[body_id] = 1.0
                break
            cursor = int(model.body_parentid[cursor])


def generate_targets(model, data, ik: DampedLeastSquaresIK) -> list[PoseTarget]:
    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()
    targets = []
    for name, qpos in TARGET_JOINT_CONFIGS:
        set_arm_state(model, data, ik.joint_ids, qpos)
        mujoco.mj_forward(model, data)
        position, rotation = ik.ee_pose(data)
        targets.append(PoseTarget(name, position, rotation, qpos.copy()))
    data.qpos[:] = original_qpos
    data.qvel[:] = original_qvel
    mujoco.mj_forward(model, data)
    return targets


def add_target_visuals(scene, target: PoseTarget) -> None:
    if scene.ngeom >= scene.maxgeom - 4:
        return
    sphere = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(
        sphere,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([0.012, 0.0, 0.0]),
        target.position,
        np.eye(3).reshape(-1),
        np.array([1.0, 0.85, 0.1, 0.85], dtype=np.float32),
    )
    scene.ngeom += 1
    colors = (
        np.array([1.0, 0.1, 0.1, 0.9], dtype=np.float32),
        np.array([0.1, 1.0, 0.1, 0.9], dtype=np.float32),
        np.array([0.1, 0.4, 1.0, 0.9], dtype=np.float32),
    )
    for axis, color in enumerate(colors):
        geom = scene.geoms[scene.ngeom]
        start = target.position
        end = target.position + 0.07 * target.rotation[:, axis]
        mujoco.mjv_connector(geom, mujoco.mjtGeom.mjGEOM_CAPSULE, 0.003, start, end)
        geom.rgba[:] = color
        scene.ngeom += 1


def annotate_frame(
    frame: np.ndarray,
    target: PoseTarget,
    sim_time: float,
    position_error: float,
    orientation_error: float,
    status: str,
) -> np.ndarray:
    import cv2

    image = np.asarray(frame).copy()
    lines = (
        f"target: {target.name}",
        f"time: {sim_time:5.2f}s",
        f"position error: {position_error * 1000:6.2f} mm",
        f"orientation error: {np.degrees(orientation_error):5.2f} deg",
        f"OSQP: {status}",
    )
    for index, text in enumerate(lines):
        y = 28 + 26 * index
        cv2.putText(image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (0, 0, 0), 4, cv2.LINE_AA)
        cv2.putText(image, text, (18, y), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (245, 245, 245), 1, cv2.LINE_AA)
    return image


def percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), q)) if values else float("nan")


def run_target(
    model,
    data,
    ik: DampedLeastSquaresIK,
    solver: OSQPVelocityIK,
    velocity_controller: VelocityTeleopController,
    target: PoseTarget,
    timeout_s: float,
    position_tolerance_m: float,
    orientation_tolerance_rad: float,
    hold_s: float,
    renderer,
    render_camera,
    video_writer,
    video_fps: int,
    min_command_lead: float,
    max_command_lead: float,
) -> TargetResult:
    dt = float(model.opt.timestep)
    lower, upper = joint_ranges(model, ik.joint_ids)
    required_hold_steps = max(1, int(round(hold_s / dt)))
    hold_steps = 0
    settle_time = None
    qdot_previous = np.zeros(len(ik.joint_ids), dtype=np.float64)
    min_position_error = float("inf")
    min_orientation_error = float("inf")
    max_abs_qdot = 0.0
    max_abs_qacc = 0.0
    wall_times = []
    iterations = []
    min_singular = float("inf")
    max_damping = 0.0
    failures = 0
    frame_period = 1.0 / float(video_fps)
    next_frame_time = 0.0
    last_status = "initializing"
    total_steps = int(np.ceil(timeout_s / dt))
    command_qpos = joint_qpos(model, data, ik.joint_ids)

    solver.reset()
    for step in range(total_steps):
        sim_time = step * dt
        current_position, current_rotation = ik.ee_pose(data)
        position_error = float(np.linalg.norm(target.position - current_position))
        orientation_error = rotation_angle(target.rotation, current_rotation)
        min_position_error = min(min_position_error, position_error)
        min_orientation_error = min(min_orientation_error, orientation_error)

        q_current = joint_qpos(model, data, ik.joint_ids)
        if position_error <= 0.006 and orientation_error <= np.radians(2.0):
            qdot = np.zeros_like(qdot_previous)
            command_qpos = q_current.copy()
            solver.reset()
            last_status = "pose_captured"
        else:
            command = velocity_controller.compute(
                target.position,
                target.rotation,
                current_position,
                current_rotation,
            )
            task_velocity = command.xdot.copy()
            task_velocity[3:] *= 1.5
            jacobian = ik.jacobian(data, control_orientation=True)
            try:
                result = solver.solve(jacobian, task_velocity, q_current, lower, upper, dt)
                qdot = result.qdot
                last_status = result.status
                wall_times.append(result.wall_time_s)
                iterations.append(result.iterations)
                min_singular = min(min_singular, result.min_singular)
                max_damping = max(max_damping, result.effective_damping)
            except RuntimeError:
                failures += 1
                qdot = np.zeros_like(qdot_previous)
                last_status = "failed"

        qacc = (qdot - qdot_previous) / dt
        max_abs_qdot = max(max_abs_qdot, float(np.max(np.abs(qdot))))
        max_abs_qacc = max(max_abs_qacc, float(np.max(np.abs(qacc))))
        qdot_previous = qdot.copy()
        command_qpos = np.clip(command_qpos + qdot * dt, lower, upper)
        error_scale = max(
            position_error / 0.10,
            orientation_error / np.radians(30.0),
        )
        command_lead = abs(float(min_command_lead)) + (
            abs(float(max_command_lead)) - abs(float(min_command_lead))
        ) * float(np.clip(error_scale, 0.0, 1.0))
        if position_error < 0.025 and orientation_error < np.radians(8.0):
            command_lead = abs(float(min_command_lead))
        command_qpos = np.clip(
            command_qpos,
            q_current - command_lead,
            q_current + command_lead,
        )
        apply_arm_targets(model, data, ik.joint_ids, command_qpos)
        mujoco.mj_step(model, data)

        inside_tolerance = (
            position_error <= position_tolerance_m
            and orientation_error <= orientation_tolerance_rad
        )
        hold_steps = hold_steps + 1 if inside_tolerance else 0
        if hold_steps >= required_hold_steps:
            settle_time = sim_time - hold_s + dt

        if renderer is not None and sim_time + 1e-12 >= next_frame_time:
            renderer.update_scene(data, camera=render_camera)
            add_target_visuals(renderer.scene, target)
            frame = renderer.render()
            frame = annotate_frame(
                frame,
                target,
                sim_time,
                position_error,
                orientation_error,
                last_status,
            )
            video_writer.append_data(frame)
            next_frame_time += frame_period

        if settle_time is not None:
            break

    final_position, final_rotation = ik.ee_pose(data)
    final_position_error = float(np.linalg.norm(target.position - final_position))
    final_orientation_error = rotation_angle(target.rotation, final_rotation)
    success = (
        settle_time is not None
        and final_position_error <= position_tolerance_m
        and final_orientation_error <= orientation_tolerance_rad
        and failures == 0
    )
    return TargetResult(
        name=target.name,
        success=success,
        settle_time_s=settle_time,
        final_position_error_m=final_position_error,
        final_orientation_error_deg=float(np.degrees(final_orientation_error)),
        minimum_position_error_m=min_position_error,
        minimum_orientation_error_deg=float(np.degrees(min_orientation_error)),
        max_abs_qdot_rad_s=max_abs_qdot,
        max_abs_qacc_rad_s2=max_abs_qacc,
        mean_osqp_wall_time_ms=1000.0 * float(np.mean(wall_times)),
        p95_osqp_wall_time_ms=1000.0 * percentile(wall_times, 95),
        max_osqp_wall_time_ms=1000.0 * float(np.max(wall_times)),
        mean_osqp_iterations=float(np.mean(iterations)),
        min_jacobian_singular_value=min_singular,
        max_effective_damping=max_damping,
        osqp_failures=failures,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=4.0)
    parser.add_argument("--hold-seconds", type=float, default=0.15)
    parser.add_argument("--max-settle-seconds", type=float, default=3.0)
    parser.add_argument("--max-osqp-p95-ms", type=float, default=1.0)
    parser.add_argument("--min-command-lead", type=float, default=0.08)
    parser.add_argument("--max-command-lead", type=float, default=0.20)
    parser.add_argument("--record-video", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--video-fps", type=int, default=30)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=720)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model = mujoco.MjModel.from_xml_path(str(Path(args.model)))
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])
    robot_subtree_gravity_compensation(model)
    ik = DampedLeastSquaresIK(
        model,
        ee_site="aero_wrist_site",
        joint_names=ARM_JOINTS,
        damping=0.035,
        max_joint_speed=5.0,
    )
    set_actuator_gains(model, ik.joint_ids)
    set_arm_state(model, data, ik.joint_ids, HOME_QPOS)
    mujoco.mj_forward(model, data)
    targets = generate_targets(model, data, ik)
    set_arm_state(model, data, ik.joint_ids, HOME_QPOS)
    mujoco.mj_forward(model, data)

    solver = OSQPVelocityIK(
        joint_count=6,
        task_dimension=6,
        joint_motion_weights=JOINT_MOTION_WEIGHTS,
        task_weights=TASK_WEIGHTS,
        config=OSQPIKConfig(
            base_damping=0.035,
            accel_weight=0.04,
            max_joint_speed=5.0,
            max_joint_accel=120.0,
            singular_damping_threshold=0.10,
            singular_damping_gain=0.10,
        ),
    )
    velocity_controller = VelocityTeleopController(
        VelocityTeleopConfig(
            kp_pos=12.0,
            kp_rot=5.0,
            max_linear_speed=0.65,
            max_angular_speed=3.0,
            control_orientation=True,
        )
    )

    renderer = None
    render_camera = None
    video_writer = None
    video_path = output_dir / "piper_osqp_ik_benchmark.mp4"
    if args.record_video:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, args.width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, args.height)
        renderer = mujoco.Renderer(model, height=args.height, width=args.width)
        render_camera = mujoco.MjvCamera()
        render_camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        render_camera.fixedcamid = -1
        render_camera.lookat[:] = np.array([0.18, 0.0, 0.30], dtype=np.float64)
        render_camera.distance = 1.35
        render_camera.azimuth = 135.0
        render_camera.elevation = -18.0
        video_writer = imageio.get_writer(
            video_path,
            fps=args.video_fps,
            codec="libx264",
            quality=8,
            macro_block_size=1,
        )

    results = []
    start = time.perf_counter()
    try:
        for target in targets:
            result = run_target(
                model=model,
                data=data,
                ik=ik,
                solver=solver,
                velocity_controller=velocity_controller,
                target=target,
                timeout_s=args.timeout,
                position_tolerance_m=args.position_tolerance_mm / 1000.0,
                orientation_tolerance_rad=np.radians(args.orientation_tolerance_deg),
                hold_s=args.hold_seconds,
                renderer=renderer,
                render_camera=render_camera,
                video_writer=video_writer,
                video_fps=args.video_fps,
                min_command_lead=args.min_command_lead,
                max_command_lead=args.max_command_lead,
            )
            results.append(result)
            print(
                f"{result.name}: success={result.success} settle={result.settle_time_s} "
                f"pos={result.final_position_error_m * 1000:.2f}mm "
                f"rot={result.final_orientation_error_deg:.2f}deg "
                f"osqp_p95={result.p95_osqp_wall_time_ms:.3f}ms"
            )
    finally:
        if video_writer is not None:
            video_writer.close()
        if renderer is not None:
            renderer.close()

    elapsed = time.perf_counter() - start
    all_success = all(result.success for result in results)
    settle_times = [result.settle_time_s for result in results if result.settle_time_s is not None]
    p95_solve_ms = max(result.p95_osqp_wall_time_ms for result in results)
    pass_criteria = {
        "all_targets_reached": all_success,
        "max_settle_time_s": max(settle_times, default=float("inf")) <= args.max_settle_seconds,
        "osqp_p95_wall_time_ms": p95_solve_ms <= args.max_osqp_p95_ms,
        "zero_osqp_failures": sum(result.osqp_failures for result in results) == 0,
    }
    passed = all(pass_criteria.values())
    summary = {
        "passed": passed,
        "model": str(Path(args.model).resolve()),
        "elapsed_wall_time_s": elapsed,
        "criteria": {
            "position_tolerance_mm": args.position_tolerance_mm,
            "orientation_tolerance_deg": args.orientation_tolerance_deg,
            "hold_seconds": args.hold_seconds,
            "max_settle_seconds": args.max_settle_seconds,
            "max_osqp_p95_ms": args.max_osqp_p95_ms,
            "min_command_lead_rad": args.min_command_lead,
            "max_command_lead_rad": args.max_command_lead,
        },
        "pass_criteria": pass_criteria,
        "targets": [
            {
                **asdict(result),
                "target_position": targets[index].position.tolist(),
                "target_rotation": targets[index].rotation.tolist(),
                "source_qpos": targets[index].source_qpos.tolist(),
            }
            for index, result in enumerate(results)
        ],
        "video": str(video_path.resolve()) if args.record_video else None,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"summary={summary_path}")
    if args.record_video:
        print(f"video={video_path}")
    print(f"benchmark_passed={passed}")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
