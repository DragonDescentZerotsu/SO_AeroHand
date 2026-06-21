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
TRAJECTORY_ANCHOR_QPOS = np.array([0.18, 1.48, -1.30, 0.82, 0.62, 0.30], dtype=np.float64)
WRIST_SWEEP_ANCHOR_QPOS = np.array([0.18, 1.48, -1.30, 0.45, 0.35, 0.30], dtype=np.float64)
JOINT_MOTION_WEIGHTS = np.array([0.7, 1.0, 1.0, 0.35, 0.22, 0.08], dtype=np.float64)
TASK_WEIGHTS = np.array([1.0, 1.0, 1.0, 1.2, 1.2, 1.2], dtype=np.float64)
ACTUATOR_KP = np.array([140.0, 140.0, 140.0, 100.0, 90.0, 90.0], dtype=np.float64)
ACTUATOR_KV = np.array([10.0, 10.0, 10.0, 7.0, 5.0, 5.0], dtype=np.float64)

# Targets are converted to Cartesian poses by FK before the benchmark starts,
# which guarantees that every requested palm pose is reachable by construction.
SMOKE_TARGET_JOINT_CONFIGS = (
    ("forward_left", np.array([0.35, 1.30, -1.22, 0.30, 0.28, 0.45])),
    ("forward_right", np.array([-0.32, 1.38, -1.48, -0.28, 0.36, -0.55])),
    ("high_wrist_roll", np.array([0.12, 1.72, -1.55, 0.42, -0.35, 1.05])),
    ("low_wrist_flex", np.array([-0.12, 1.48, -1.16, -0.48, 0.48, -0.85])),
    ("return_home", HOME_QPOS.copy()),
)

FULL_CHALLENGE_JOINT_CONFIGS = (
    ("center_forward", np.array([0.00, 1.28, -1.12, 0.00, 0.00, 0.00])),
    ("left_mid", np.array([0.48, 1.36, -1.30, 0.18, 0.18, 0.35])),
    ("right_mid", np.array([-0.48, 1.36, -1.30, -0.18, 0.18, -0.35])),
    ("left_high", np.array([0.55, 1.82, -1.62, 0.28, -0.22, 0.55])),
    ("right_high", np.array([-0.55, 1.82, -1.62, -0.28, -0.22, -0.55])),
    ("left_low", np.array([0.38, 1.16, -1.02, -0.34, 0.38, 0.25])),
    ("right_low", np.array([-0.38, 1.16, -1.02, 0.34, 0.38, -0.25])),
    ("far_reach", np.array([0.00, 0.82, -0.58, 0.10, 0.12, 0.00])),
    ("compact_reach", np.array([0.00, 2.08, -2.08, -0.18, 0.22, 0.00])),
    ("wrist_roll_positive", np.array([0.12, 1.58, -1.42, 0.10, -0.18, 1.55])),
    ("wrist_roll_negative", np.array([-0.12, 1.58, -1.42, -0.10, -0.18, -1.55])),
    ("wrist_flex_positive", np.array([0.18, 1.48, -1.30, 0.82, 0.62, 0.30])),
    ("wrist_flex_negative", np.array([-0.18, 1.48, -1.30, -0.82, 0.62, -0.30])),
    ("combined_orientation_a", np.array([0.34, 1.70, -1.58, 0.62, -0.58, 1.10])),
    ("combined_orientation_b", np.array([-0.34, 1.70, -1.58, -0.62, -0.58, -1.10])),
    ("near_singular_extension", np.array([0.05, 0.62, -0.34, 0.08, 0.05, 0.40])),
)

RANDOM_QPOS_MIN = np.array([-0.75, 0.90, -1.90, -0.70, -0.60, -1.40], dtype=np.float64)
RANDOM_QPOS_MAX = np.array([0.75, 1.90, -0.70, 0.70, 0.60, 1.40], dtype=np.float64)


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


@dataclass
class TrajectoryResult:
    name: str
    success: bool
    duration_s: float
    rms_position_error_m: float
    p95_position_error_m: float
    max_position_error_m: float
    rms_orientation_error_deg: float
    p95_orientation_error_deg: float
    max_orientation_error_deg: float
    max_abs_qdot_rad_s: float
    max_abs_qacc_rad_s2: float
    mean_osqp_wall_time_ms: float
    p95_osqp_wall_time_ms: float
    max_osqp_wall_time_ms: float
    min_jacobian_singular_value: float
    max_effective_damping: float
    osqp_failures: int


@dataclass
class OrientationSweepResult:
    name: str
    success: bool
    duration_s: float
    max_position_error_m: float
    final_position_error_m: float
    max_orientation_error_deg: float
    final_orientation_error_deg: float
    min_jacobian_singular_value: float
    min_orientation_scale: float
    max_abs_qdot_rad_s: float
    max_abs_qacc_rad_s2: float
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


def random_joint_configs(count: int, seed: int) -> tuple[tuple[str, np.ndarray], ...]:
    rng = np.random.default_rng(seed)
    samples = rng.uniform(RANDOM_QPOS_MIN, RANDOM_QPOS_MAX, size=(max(0, count), 6))
    return tuple((f"random_{index + 1:03d}", qpos) for index, qpos in enumerate(samples))


def generate_targets(model, data, ik: DampedLeastSquaresIK, configs) -> list[PoseTarget]:
    original_qpos = data.qpos.copy()
    original_qvel = data.qvel.copy()
    targets = []
    for name, qpos in configs:
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


def mean_or_nan(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def max_or_nan(values: list[float]) -> float:
    return float(np.max(values)) if values else float("nan")


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
        mean_osqp_wall_time_ms=1000.0 * mean_or_nan(wall_times),
        p95_osqp_wall_time_ms=1000.0 * percentile(wall_times, 95),
        max_osqp_wall_time_ms=1000.0 * max_or_nan(wall_times),
        mean_osqp_iterations=mean_or_nan(iterations),
        min_jacobian_singular_value=min_singular,
        max_effective_damping=max_damping,
        osqp_failures=failures,
    )


def rotation_from_rotvec(rotvec: np.ndarray) -> np.ndarray:
    rotvec = np.asarray(rotvec, dtype=np.float64)
    angle = float(np.linalg.norm(rotvec))
    if angle < 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def run_continuous_trajectory(
    model,
    data,
    ik: DampedLeastSquaresIK,
    solver: OSQPVelocityIK,
    velocity_controller: VelocityTeleopController,
    anchor_target: PoseTarget,
    duration_s: float,
    period_s: float,
    renderer,
    render_camera,
    video_writer,
    video_fps: int,
    min_command_lead: float,
    max_command_lead: float,
    p95_position_limit_m: float,
    p95_orientation_limit_rad: float,
) -> TrajectoryResult:
    dt = float(model.opt.timestep)
    lower, upper = joint_ranges(model, ik.joint_ids)
    total_steps = int(np.ceil(duration_s / dt))
    command_qpos = joint_qpos(model, data, ik.joint_ids)
    qdot_previous = np.zeros(len(ik.joint_ids), dtype=np.float64)
    position_errors = []
    orientation_errors = []
    wall_times = []
    min_singular = float("inf")
    max_damping = 0.0
    max_abs_qdot = 0.0
    max_abs_qacc = 0.0
    failures = 0
    frame_period = 1.0 / float(video_fps)
    next_frame_time = 0.0
    solver.reset()

    for step in range(total_steps):
        sim_time = step * dt
        phase = 2.0 * np.pi * sim_time / period_s
        target_position = anchor_target.position + np.array(
            [
                0.055 * np.sin(phase),
                0.045 * np.sin(2.0 * phase),
                0.030 * (1.0 - np.cos(phase)),
            ],
            dtype=np.float64,
        )
        target_rotation = rotation_from_rotvec(
            np.array(
                [
                    0.24 * np.sin(phase),
                    0.28 * np.sin(2.0 * phase),
                    0.42 * np.sin(phase),
                ],
                dtype=np.float64,
            )
        ) @ anchor_target.rotation

        current_position, current_rotation = ik.ee_pose(data)
        position_error = float(np.linalg.norm(target_position - current_position))
        orientation_error = rotation_angle(target_rotation, current_rotation)
        position_errors.append(position_error)
        orientation_errors.append(orientation_error)

        command = velocity_controller.compute(
            target_position,
            target_rotation,
            current_position,
            current_rotation,
        )
        task_velocity = command.xdot.copy()
        task_velocity[3:] *= 1.5
        jacobian = ik.jacobian(data, control_orientation=True)
        q_current = joint_qpos(model, data, ik.joint_ids)
        try:
            result = solver.solve(jacobian, task_velocity, q_current, lower, upper, dt)
            qdot = result.qdot
            wall_times.append(result.wall_time_s)
            min_singular = min(min_singular, result.min_singular)
            max_damping = max(max_damping, result.effective_damping)
            status = result.status
        except RuntimeError:
            qdot = np.zeros_like(qdot_previous)
            failures += 1
            status = "failed"

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
        command_qpos = np.clip(
            command_qpos,
            q_current - command_lead,
            q_current + command_lead,
        )
        apply_arm_targets(model, data, ik.joint_ids, command_qpos)
        mujoco.mj_step(model, data)

        if renderer is not None and sim_time + 1e-12 >= next_frame_time:
            target = PoseTarget(
                name="continuous_6d",
                position=target_position,
                rotation=target_rotation,
                source_qpos=anchor_target.source_qpos,
            )
            renderer.update_scene(data, camera=render_camera)
            add_target_visuals(renderer.scene, target)
            frame = renderer.render()
            frame = annotate_frame(
                frame,
                target,
                sim_time,
                position_error,
                orientation_error,
                status,
            )
            video_writer.append_data(frame)
            next_frame_time += frame_period

    position_errors = np.asarray(position_errors, dtype=np.float64)
    orientation_errors = np.asarray(orientation_errors, dtype=np.float64)
    p95_position = float(np.percentile(position_errors, 95))
    p95_orientation = float(np.percentile(orientation_errors, 95))
    success = bool(
        p95_position <= p95_position_limit_m
        and p95_orientation <= p95_orientation_limit_rad
        and failures == 0
    )
    return TrajectoryResult(
        name="continuous_6d",
        success=success,
        duration_s=duration_s,
        rms_position_error_m=float(np.sqrt(np.mean(position_errors**2))),
        p95_position_error_m=p95_position,
        max_position_error_m=float(np.max(position_errors)),
        rms_orientation_error_deg=float(np.degrees(np.sqrt(np.mean(orientation_errors**2)))),
        p95_orientation_error_deg=float(np.degrees(p95_orientation)),
        max_orientation_error_deg=float(np.degrees(np.max(orientation_errors))),
        max_abs_qdot_rad_s=max_abs_qdot,
        max_abs_qacc_rad_s2=max_abs_qacc,
        mean_osqp_wall_time_ms=1000.0 * mean_or_nan(wall_times),
        p95_osqp_wall_time_ms=1000.0 * percentile(wall_times, 95),
        max_osqp_wall_time_ms=1000.0 * max_or_nan(wall_times),
        min_jacobian_singular_value=min_singular,
        max_effective_damping=max_damping,
        osqp_failures=failures,
    )


def run_fixed_position_orientation_sweep(
    model,
    data,
    ik: DampedLeastSquaresIK,
    solver: OSQPVelocityIK,
    velocity_controller: VelocityTeleopController,
    anchor_target: PoseTarget,
    renderer,
    render_camera,
    video_writer,
    video_fps: int,
    min_command_lead: float,
    max_command_lead: float,
    max_position_limit_m: float,
    final_position_limit_m: float,
) -> OrientationSweepResult:
    dt = float(model.opt.timestep)
    duration_s = 10.0
    lower, upper = joint_ranges(model, ik.joint_ids)
    command_qpos = joint_qpos(model, data, ik.joint_ids)
    qdot_previous = np.zeros(len(ik.joint_ids), dtype=np.float64)
    position_errors = []
    orientation_errors = []
    min_singular = float("inf")
    min_orientation_scale = 1.0
    max_abs_qdot = 0.0
    max_abs_qacc = 0.0
    failures = 0
    frame_period = 1.0 / float(video_fps)
    next_frame_time = 0.0
    solver.reset()

    for step in range(int(np.ceil(duration_s / dt))):
        sim_time = step * dt
        if sim_time < 2.0:
            angle_deg = 70.0 * sim_time / 2.0
        elif sim_time < 3.0:
            angle_deg = 70.0
        elif sim_time < 8.0:
            angle_deg = 70.0 - 140.0 * (sim_time - 3.0) / 5.0
        else:
            angle_deg = -70.0
        target_rotation = anchor_target.rotation @ rotation_from_rotvec(
            np.array([0.0, np.radians(angle_deg), 0.0], dtype=np.float64)
        )

        current_position, current_rotation = ik.ee_pose(data)
        position_error = float(np.linalg.norm(anchor_target.position - current_position))
        orientation_error = rotation_angle(target_rotation, current_rotation)
        position_errors.append(position_error)
        orientation_errors.append(orientation_error)

        command = velocity_controller.compute(
            anchor_target.position,
            target_rotation,
            current_position,
            current_rotation,
        )
        task_velocity = command.xdot.copy()
        task_velocity[3:] *= 1.5
        q_current = joint_qpos(model, data, ik.joint_ids)
        try:
            result = solver.solve(
                ik.jacobian(data, control_orientation=True),
                task_velocity,
                q_current,
                lower,
                upper,
                dt,
            )
            qdot = result.qdot
            min_singular = min(min_singular, result.min_singular)
            min_orientation_scale = min(min_orientation_scale, result.orientation_scale)
            status = f"{result.status} ori_scale={result.orientation_scale:.2f}"
        except RuntimeError:
            qdot = np.zeros_like(qdot_previous)
            failures += 1
            status = "failed"

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
        command_qpos = np.clip(
            command_qpos,
            q_current - command_lead,
            q_current + command_lead,
        )
        apply_arm_targets(model, data, ik.joint_ids, command_qpos)
        mujoco.mj_step(model, data)

        if renderer is not None and sim_time + 1e-12 >= next_frame_time:
            target = PoseTarget(
                name=f"fixed_position_wrist_pitch_{angle_deg:+.0f}deg",
                position=anchor_target.position,
                rotation=target_rotation,
                source_qpos=anchor_target.source_qpos,
            )
            renderer.update_scene(data, camera=render_camera)
            add_target_visuals(renderer.scene, target)
            frame = annotate_frame(
                renderer.render(),
                target,
                sim_time,
                position_error,
                orientation_error,
                status,
            )
            video_writer.append_data(frame)
            next_frame_time += frame_period

    max_position_error = max(position_errors)
    final_position_error = position_errors[-1]
    max_orientation_error = max(orientation_errors)
    final_orientation_error = orientation_errors[-1]
    success = bool(
        max_position_error <= max_position_limit_m
        and final_position_error <= final_position_limit_m
        and min_orientation_scale < 1.0
        and failures == 0
    )
    return OrientationSweepResult(
        name="fixed_position_wrist_pitch_plus70_to_minus70",
        success=success,
        duration_s=duration_s,
        max_position_error_m=max_position_error,
        final_position_error_m=final_position_error,
        max_orientation_error_deg=float(np.degrees(max_orientation_error)),
        final_orientation_error_deg=float(np.degrees(final_orientation_error)),
        min_jacobian_singular_value=min_singular,
        min_orientation_scale=min_orientation_scale,
        max_abs_qdot_rad_s=max_abs_qdot,
        max_abs_qacc_rad_s2=max_abs_qacc,
        osqp_failures=failures,
    )


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--suite", choices=("smoke", "full"), default="full")
    parser.add_argument("--random-targets", type=int, default=100)
    parser.add_argument("--random-seed", type=int, default=20260619)
    parser.add_argument("--timeout", type=float, default=4.0)
    parser.add_argument("--position-tolerance-mm", type=float, default=10.0)
    parser.add_argument("--orientation-tolerance-deg", type=float, default=4.0)
    parser.add_argument("--hold-seconds", type=float, default=0.15)
    parser.add_argument("--max-settle-seconds", type=float, default=3.0)
    parser.add_argument("--min-random-success-rate", type=float, default=0.95)
    parser.add_argument("--max-osqp-p95-ms", type=float, default=1.0)
    parser.add_argument("--trajectory-duration", type=float, default=16.0)
    parser.add_argument("--trajectory-period", type=float, default=8.0)
    parser.add_argument("--trajectory-p95-position-mm", type=float, default=25.0)
    parser.add_argument("--trajectory-p95-orientation-deg", type=float, default=8.0)
    parser.add_argument("--wrist-sweep-max-position-mm", type=float, default=40.0)
    parser.add_argument("--wrist-sweep-final-position-mm", type=float, default=10.0)
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
    fixed_configs = SMOKE_TARGET_JOINT_CONFIGS if args.suite == "smoke" else FULL_CHALLENGE_JOINT_CONFIGS
    fixed_targets = generate_targets(model, data, ik, fixed_configs)
    random_targets = (
        []
        if args.suite == "smoke"
        else generate_targets(
            model,
            data,
            ik,
            random_joint_configs(args.random_targets, args.random_seed),
        )
    )
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

    fixed_results = []
    random_results = []
    video_results = []
    trajectory_result = None
    orientation_sweep_result = None
    start = time.perf_counter()
    try:
        for target in fixed_targets:
            if args.suite == "full":
                set_arm_state(model, data, ik.joint_ids, HOME_QPOS)
                mujoco.mj_forward(model, data)
                solver.reset()
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
                renderer=renderer if args.suite == "smoke" else None,
                render_camera=render_camera if args.suite == "smoke" else None,
                video_writer=video_writer if args.suite == "smoke" else None,
                video_fps=args.video_fps,
                min_command_lead=args.min_command_lead,
                max_command_lead=args.max_command_lead,
            )
            fixed_results.append(result)
            print(
                f"fixed/{result.name}: success={result.success} settle={result.settle_time_s} "
                f"pos={result.final_position_error_m * 1000:.2f}mm "
                f"rot={result.final_orientation_error_deg:.2f}deg "
                f"osqp_p95={result.p95_osqp_wall_time_ms:.3f}ms"
            )

        if args.suite == "full":
            if args.record_video:
                video_targets = generate_targets(model, data, ik, SMOKE_TARGET_JOINT_CONFIGS)
                set_arm_state(model, data, ik.joint_ids, HOME_QPOS)
                mujoco.mj_forward(model, data)
                for target in video_targets:
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
                    video_results.append(result)
                    print(
                        f"video/{result.name}: success={result.success} settle={result.settle_time_s} "
                        f"pos={result.final_position_error_m * 1000:.2f}mm "
                        f"rot={result.final_orientation_error_deg:.2f}deg"
                    )

            trajectory_target = generate_targets(
                model,
                data,
                ik,
                (("trajectory_anchor", TRAJECTORY_ANCHOR_QPOS.copy()),),
            )[0]
            if args.record_video:
                anchor_result = run_target(
                    model=model,
                    data=data,
                    ik=ik,
                    solver=solver,
                    velocity_controller=velocity_controller,
                    target=trajectory_target,
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
                video_results.append(anchor_result)
                print(
                    f"video/{anchor_result.name}: success={anchor_result.success} "
                    f"settle={anchor_result.settle_time_s} "
                    f"pos={anchor_result.final_position_error_m * 1000:.2f}mm "
                    f"rot={anchor_result.final_orientation_error_deg:.2f}deg"
                )
            else:
                set_arm_state(model, data, ik.joint_ids, TRAJECTORY_ANCHOR_QPOS)
                mujoco.mj_forward(model, data)
            trajectory_result = run_continuous_trajectory(
                model=model,
                data=data,
                ik=ik,
                solver=solver,
                velocity_controller=velocity_controller,
                anchor_target=trajectory_target,
                duration_s=args.trajectory_duration,
                period_s=args.trajectory_period,
                renderer=renderer,
                render_camera=render_camera,
                video_writer=video_writer,
                video_fps=args.video_fps,
                min_command_lead=args.min_command_lead,
                max_command_lead=args.max_command_lead,
                p95_position_limit_m=args.trajectory_p95_position_mm / 1000.0,
                p95_orientation_limit_rad=np.radians(args.trajectory_p95_orientation_deg),
            )
            print(
                f"trajectory/{trajectory_result.name}: success={trajectory_result.success} "
                f"pos_rms={trajectory_result.rms_position_error_m * 1000:.2f}mm "
                f"pos_p95={trajectory_result.p95_position_error_m * 1000:.2f}mm "
                f"rot_rms={trajectory_result.rms_orientation_error_deg:.2f}deg "
                f"rot_p95={trajectory_result.p95_orientation_error_deg:.2f}deg"
            )

            sweep_target = generate_targets(
                model,
                data,
                ik,
                (("wrist_sweep_anchor", WRIST_SWEEP_ANCHOR_QPOS.copy()),),
            )[0]
            set_arm_state(model, data, ik.joint_ids, WRIST_SWEEP_ANCHOR_QPOS)
            mujoco.mj_forward(model, data)
            orientation_sweep_result = run_fixed_position_orientation_sweep(
                model=model,
                data=data,
                ik=ik,
                solver=solver,
                velocity_controller=velocity_controller,
                anchor_target=sweep_target,
                renderer=renderer,
                render_camera=render_camera,
                video_writer=video_writer,
                video_fps=args.video_fps,
                min_command_lead=args.min_command_lead,
                max_command_lead=args.max_command_lead,
                max_position_limit_m=args.wrist_sweep_max_position_mm / 1000.0,
                final_position_limit_m=args.wrist_sweep_final_position_mm / 1000.0,
            )
            print(
                f"stress/{orientation_sweep_result.name}: success={orientation_sweep_result.success} "
                f"max_pos={orientation_sweep_result.max_position_error_m * 1000:.2f}mm "
                f"final_pos={orientation_sweep_result.final_position_error_m * 1000:.2f}mm "
                f"max_rot={orientation_sweep_result.max_orientation_error_deg:.2f}deg "
                f"final_rot={orientation_sweep_result.final_orientation_error_deg:.2f}deg "
                f"min_sv={orientation_sweep_result.min_jacobian_singular_value:.6f} "
                f"min_ori_scale={orientation_sweep_result.min_orientation_scale:.3f}"
            )

            for target in random_targets:
                set_arm_state(model, data, ik.joint_ids, HOME_QPOS)
                mujoco.mj_forward(model, data)
                solver.reset()
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
                    renderer=None,
                    render_camera=None,
                    video_writer=None,
                    video_fps=args.video_fps,
                    min_command_lead=args.min_command_lead,
                    max_command_lead=args.max_command_lead,
                )
                random_results.append(result)
                print(
                    f"random/{result.name}: success={result.success} settle={result.settle_time_s} "
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
    all_results = fixed_results + random_results + video_results
    fixed_success = all(result.success for result in fixed_results)
    random_success_rate = (
        sum(result.success for result in random_results) / len(random_results)
        if random_results
        else 1.0
    )
    settle_times = [result.settle_time_s for result in all_results if result.settle_time_s is not None]
    p95_solve_candidates = [result.p95_osqp_wall_time_ms for result in all_results]
    if trajectory_result is not None:
        p95_solve_candidates.append(trajectory_result.p95_osqp_wall_time_ms)
    p95_solve_ms = max(p95_solve_candidates, default=float("inf"))
    pass_criteria = {
        "all_fixed_targets_reached": fixed_success,
        "random_success_rate": random_success_rate >= args.min_random_success_rate,
        "continuous_trajectory": trajectory_result is None or trajectory_result.success,
        "fixed_position_wrist_sweep": (
            orientation_sweep_result is None or orientation_sweep_result.success
        ),
        "video_sequence": not video_results or all(result.success for result in video_results),
        "max_settle_time_s": max(settle_times, default=float("inf")) <= args.max_settle_seconds,
        "osqp_p95_wall_time_ms": p95_solve_ms <= args.max_osqp_p95_ms,
        "zero_osqp_failures": (
            sum(result.osqp_failures for result in all_results)
            + (trajectory_result.osqp_failures if trajectory_result is not None else 0)
            + (
                orientation_sweep_result.osqp_failures
                if orientation_sweep_result is not None
                else 0
            )
        )
        == 0,
    }
    passed = all(pass_criteria.values())
    summary = {
        "passed": passed,
        "suite": args.suite,
        "model": str(Path(args.model).resolve()),
        "elapsed_wall_time_s": elapsed,
        "criteria": {
            "position_tolerance_mm": args.position_tolerance_mm,
            "orientation_tolerance_deg": args.orientation_tolerance_deg,
            "hold_seconds": args.hold_seconds,
            "max_settle_seconds": args.max_settle_seconds,
            "min_random_success_rate": args.min_random_success_rate,
            "max_osqp_p95_ms": args.max_osqp_p95_ms,
            "trajectory_p95_position_mm": args.trajectory_p95_position_mm,
            "trajectory_p95_orientation_deg": args.trajectory_p95_orientation_deg,
            "wrist_sweep_max_position_mm": args.wrist_sweep_max_position_mm,
            "wrist_sweep_final_position_mm": args.wrist_sweep_final_position_mm,
            "min_command_lead_rad": args.min_command_lead,
            "max_command_lead_rad": args.max_command_lead,
        },
        "pass_criteria": pass_criteria,
        "coverage": {
            "fixed_target_count": len(fixed_targets),
            "fixed_challenge_count": (
                len(SMOKE_TARGET_JOINT_CONFIGS) - 1
                if args.suite == "smoke"
                else len(FULL_CHALLENGE_JOINT_CONFIGS)
            ),
            "random_target_count": len(random_targets),
            "random_seed": args.random_seed,
            "random_success_rate": random_success_rate,
            "trajectory_duration_s": args.trajectory_duration if trajectory_result is not None else 0.0,
        },
        "aggregate": {
            "fixed_success_count": sum(result.success for result in fixed_results),
            "fixed_total_count": len(fixed_results),
            "random_success_count": sum(result.success for result in random_results),
            "random_total_count": len(random_results),
            "random_success_rate": random_success_rate,
            "successful_settle_mean_s": mean_or_nan(settle_times),
            "successful_settle_p95_s": percentile(settle_times, 95),
            "successful_settle_max_s": max_or_nan(settle_times),
            "fixed_max_final_position_error_mm": 1000.0
            * max((result.final_position_error_m for result in fixed_results), default=float("nan")),
            "fixed_max_final_orientation_error_deg": max(
                (result.final_orientation_error_deg for result in fixed_results),
                default=float("nan"),
            ),
            "random_p95_final_position_error_mm": 1000.0
            * percentile([result.final_position_error_m for result in random_results], 95),
            "random_p95_final_orientation_error_deg": percentile(
                [result.final_orientation_error_deg for result in random_results],
                95,
            ),
            "max_osqp_p95_wall_time_ms": p95_solve_ms,
        },
        "fixed_targets": [
            {
                **asdict(result),
                "target_position": fixed_targets[index].position.tolist(),
                "target_rotation": fixed_targets[index].rotation.tolist(),
                "source_qpos": fixed_targets[index].source_qpos.tolist(),
            }
            for index, result in enumerate(fixed_results)
        ],
        "random_targets": [
            {
                **asdict(result),
                "target_position": random_targets[index].position.tolist(),
                "target_rotation": random_targets[index].rotation.tolist(),
                "source_qpos": random_targets[index].source_qpos.tolist(),
            }
            for index, result in enumerate(random_results)
        ],
        "trajectory": asdict(trajectory_result) if trajectory_result is not None else None,
        "orientation_sweep": (
            asdict(orientation_sweep_result)
            if orientation_sweep_result is not None
            else None
        ),
        "video_sequence": [asdict(result) for result in video_results],
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
