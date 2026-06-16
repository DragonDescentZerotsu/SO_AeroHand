import argparse
import queue
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit("Missing runtime dependency. Install with: pip install mujoco numpy") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.arm_teleop import (
    DampedLeastSquaresIK,
    VelocityTeleopConfig,
    VelocityTeleopController,
    joint_qpos,
    joint_ranges,
)
from aero_quest.quest_hand_frame import RelativeWristArmController, palm_frame_from_landmarks_wrist, quat_xyzw_to_matrix
from aero_quest.retargeting import AeroHandRetargetingWrapper
from aero_quest.so101_aero_control import normalized_aero_hand_to_ctrl, print_combined_actuator_info
from scripts.legacy.quest_so101_aero_split_wrist_teleop import (
    DEFAULT_MODEL,
    DEFAULT_R_BQ,
    SAFE_OPEN_HAND,
    ctrl_midpoints,
    drain_latest,
    parse_matrix,
    parse_vec3,
    resolve_model,
    start_quest_receiver,
)


DEFAULT_ARM_JOINTS = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll"


def quest_orientation_frame_Q(frame, orientation_source: str) -> np.ndarray:
    R_wrist_Q = quat_xyzw_to_matrix(frame.wrist_quat_world)
    if orientation_source == "wrist_pose":
        return R_wrist_Q
    if orientation_source == "palm_landmarks":
        return R_wrist_Q @ palm_frame_from_landmarks_wrist(frame.landmarks_wrist)
    raise ValueError(f"Unsupported orientation_source: {orientation_source!r}")


def absolute_orientation_target_B(frame, R_BQ: np.ndarray, orientation_source: str) -> np.ndarray:
    """Map current Quest wrist/palm orientation directly into robot base frame."""
    return np.asarray(R_BQ, dtype=np.float64).reshape(3, 3) @ quest_orientation_frame_Q(frame, orientation_source)


def solve_weighted_task_space_ik(
    ik: DampedLeastSquaresIK,
    data,
    xdot_cmd: np.ndarray,
    dt: float,
    position_weight: float,
    orientation_weight: float,
    control_orientation: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Solve a weighted resolved-rate IK step using the selected arm joints.

    This keeps position and orientation in one least-squares problem so wrist
    joints are not overwritten by a separate orientation controller.
    """
    xdot_cmd = np.asarray(xdot_cmd, dtype=np.float64)
    if control_orientation:
        if xdot_cmd.shape != (6,):
            raise ValueError(f"Expected 6D task velocity, got {xdot_cmd.shape}")
        J = ik.jacobian(data, control_orientation=True)
        weights = np.array(
            [position_weight, position_weight, position_weight, orientation_weight, orientation_weight, orientation_weight],
            dtype=np.float64,
        )
    else:
        if xdot_cmd.shape != (3,):
            raise ValueError(f"Expected 3D task velocity, got {xdot_cmd.shape}")
        J = ik.jacobian(data, control_orientation=False)
        weights = np.array([position_weight, position_weight, position_weight], dtype=np.float64)

    weights = np.sqrt(np.clip(weights, 0.0, np.inf))
    Jw = weights[:, None] * J
    xw = weights * xdot_cmd
    JJt = Jw @ Jw.T
    qdot = Jw.T @ np.linalg.solve(JJt + (ik.damping**2) * np.eye(JJt.shape[0]), xw)
    qdot = np.clip(qdot, -ik.max_joint_speed, ik.max_joint_speed)

    q_current = joint_qpos(ik.model, data, ik.joint_ids)
    q_target = q_current + qdot * float(dt)
    lo, hi = joint_ranges(ik.model, ik.joint_ids)
    q_target = np.clip(q_target, lo, hi)
    if ik.smoothing_alpha > 0.0 and ik.prev_qtarget is not None:
        q_target = ik.smoothing_alpha * ik.prev_qtarget + (1.0 - ik.smoothing_alpha) * q_target
    ik.prev_qtarget = q_target.copy()
    return q_target, qdot


def make_key_callback(arm_channel, latest_frame_ref, ik, data, state, args, R_BQ):
    def on_key(keycode):
        try:
            key = chr(keycode).lower()
        except (TypeError, ValueError):
            return
        if key == "r":
            frame = latest_frame_ref["frame"]
            if frame is None:
                print("Re-zero requested, but no Quest hand frame has arrived yet.")
                return
            ee_pos_B, ee_R_B = ik.ee_pose(data)
            arm_channel.set_teleop_zero(
                frame.wrist_pos_world,
                frame.wrist_quat_world,
                ee_pos_B,
                ee_R_B,
                landmarks_wrist=frame.landmarks_wrist,
            )
            state["target_pos_B"] = ee_pos_B.copy()
            state["target_R_B"] = (
                absolute_orientation_target_B(frame, R_BQ, args.orientation_source)
                if args.control_orientation and args.orientation_tracking == "absolute"
                else ee_R_B.copy()
            )
            state["last_qtarget"] = joint_qpos(ik.model, data, ik.joint_ids)
            print("Re-zeroed weighted IK: current Quest wrist maps to current SO101 end-effector pose.")
        elif key == "p":
            state["paused"] = not state["paused"]
            print(f"paused={state['paused']}")

    return on_key


def parse_args():
    parser = argparse.ArgumentParser(
        description="Quest wrist pose -> weighted SO101 5-joint IK, landmarks -> Aero Hand."
    )
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hand", choices=["right", "left", "any"], default="right")
    parser.add_argument("--ee-site", default="aero_wrist_site")
    parser.add_argument("--ee-body", default=None)
    parser.add_argument("--arm-joint-names", default=DEFAULT_ARM_JOINTS)
    parser.add_argument("--arm-joint-prefix", default=None)
    parser.add_argument("--scale", type=float, default=0.9)
    parser.add_argument("--R_BQ", type=parse_matrix, default=None)
    parser.add_argument("--workspace-min", nargs=3, default=["0.05", "-0.35", "0.03"])
    parser.add_argument("--workspace-max", nargs=3, default=["0.55", "0.35", "0.60"])
    parser.add_argument("--deadzone", type=float, default=0.005)
    parser.add_argument("--target-smoothing-alpha", type=float, default=0.10)
    parser.add_argument("--kp-pos", type=float, default=10.0)
    parser.add_argument("--kp-rot", type=float, default=1.2)
    parser.add_argument("--max-linear-speed", type=float, default=0.45)
    parser.add_argument("--max-angular-speed", type=float, default=0.8)
    parser.add_argument("--ik-damping", type=float, default=0.05)
    parser.add_argument("--max-joint-speed", type=float, default=3.0)
    parser.add_argument("--joint-target-smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--control-orientation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--orientation-source", choices=["palm_landmarks", "wrist_pose"], default="palm_landmarks")
    parser.add_argument("--orientation-tracking", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=0.03)
    parser.add_argument("--hand-smoothing-alpha", type=float, default=0.25)
    parser.add_argument("--disable-hand-retargeting", action="store_true")
    parser.add_argument("--timeout", type=float, default=0.30)
    parser.add_argument("--debug-interval", type=float, default=0.25)
    parser.add_argument("--disable-gravity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    workspace_min = parse_vec3(args.workspace_min, "--workspace-min")
    workspace_max = parse_vec3(args.workspace_max, "--workspace-max")
    R_BQ = DEFAULT_R_BQ.copy() if args.R_BQ is None else np.asarray(args.R_BQ, dtype=np.float64).reshape(3, 3)
    model_path = resolve_model(args.model)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if args.disable_gravity:
        model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    data.ctrl[:] = ctrl_midpoints(model)
    mujoco.mj_forward(model, data)

    joint_names = [name.strip() for name in args.arm_joint_names.split(",") if name.strip()] if args.arm_joint_names else None
    ik = DampedLeastSquaresIK(
        model,
        ee_site=args.ee_site,
        ee_body=args.ee_body,
        joint_names=joint_names,
        joint_prefix=args.arm_joint_prefix,
        damping=args.ik_damping,
        max_joint_speed=args.max_joint_speed,
        smoothing_alpha=args.joint_target_smoothing_alpha,
    )
    vel_controller = VelocityTeleopController(
        VelocityTeleopConfig(
            kp_pos=args.kp_pos,
            kp_rot=args.kp_rot,
            max_linear_speed=args.max_linear_speed,
            max_angular_speed=args.max_angular_speed,
            control_orientation=args.control_orientation,
        )
    )
    arm_channel = RelativeWristArmController(
        scale=args.scale,
        R_BQ=R_BQ,
        deadzone=args.deadzone,
        smoothing_alpha=args.target_smoothing_alpha,
        control_orientation=args.control_orientation,
        orientation_source=args.orientation_source,
    )
    hand_retargeter = AeroHandRetargetingWrapper(
        args.hand_smoothing_alpha,
        disabled=args.disable_hand_retargeting,
        initial_action=SAFE_OPEN_HAND,
    )

    ee_pos_B, ee_R_B = ik.ee_pose(data)
    print("Before running:")
    print(f"  1. Confirm: adb reverse tcp:{args.port} tcp:{args.port}")
    print(f"  2. Quest HTS: TCP, localhost, port {args.port}")
    print("  3. Weighted IK: wrist relative motion and orientation -> selected SO101 joints.")
    print("  4. Hand Channel: wrist-relative landmarks -> Aero Hand retargeting.")
    print("  5. Press R to re-zero, P to pause/resume arm.")
    print(f"model={model_path}")
    print(f"ee_site={args.ee_site} arm_joints={ik.joint_names}")
    print(f"ee_start_B={np.array2string(ee_pos_B, precision=5)}")
    print(f"workspace_min={np.array2string(workspace_min, precision=3)} workspace_max={np.array2string(workspace_max, precision=3)}")
    print(f"scale={args.scale} R_BQ=\n{np.array2string(R_BQ, precision=4, suppress_small=True)}")
    print(
        f"control_orientation={args.control_orientation} orientation_source={args.orientation_source} "
        f"orientation_tracking={args.orientation_tracking} "
        f"position_weight={args.position_weight} orientation_weight={args.orientation_weight}"
    )
    print(f"disable_hand_retargeting={args.disable_hand_retargeting}")
    print_combined_actuator_info(model)

    if args.dry_run:
        _raw_hand, filtered_hand = hand_retargeter(np.zeros((21, 3), dtype=np.float32))
        print(f"dry_run=true q_arm={np.array2string(joint_qpos(model, data, ik.joint_ids), precision=6)}")
        print(f"dry_run_hand={np.array2string(filtered_hand, precision=4, suppress_small=True)}")
        return

    frame_queue = queue.Queue()
    start_quest_receiver(args, frame_queue)
    print(f"Waiting for Quest TCP connection on {args.host}:{args.port}...")

    latest_frame_ref = {"frame": None}
    state = {
        "paused": False,
        "target_pos_B": ee_pos_B.copy(),
        "target_R_B": ee_R_B.copy(),
        "last_qtarget": joint_qpos(model, data, ik.joint_ids),
    }
    last_frame_time = 0.0
    last_debug_time = 0.0
    last_control_time = time.time()
    last_frame_id = None
    last_delta_p_Q = np.zeros(3, dtype=np.float64)
    last_velocity = np.zeros(3, dtype=np.float64)
    last_control_dt = float(model.opt.timestep)
    sim_step_dt = float(model.opt.timestep)
    sim_accumulator = 0.0
    last_sim_steps = 0
    last_error = np.zeros(3, dtype=np.float64)
    last_rot_error = np.zeros(3, dtype=np.float64)
    last_qdot = np.zeros(len(ik.joint_ids), dtype=np.float64)
    filtered_hand = SAFE_OPEN_HAND.copy()

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=make_key_callback(arm_channel, latest_frame_ref, ik, data, state, args, R_BQ),
    ) as viewer:
        while viewer.is_running():
            latest, drained = drain_latest(frame_queue)
            got_new_frame = latest is not None
            if latest is not None:
                frame_time, quest_frame = latest
                latest_frame_ref["frame"] = quest_frame
                last_frame_time = frame_time
                last_frame_id = quest_frame.frame_id
                if not arm_channel.is_calibrated:
                    ee_pos_B, ee_R_B = ik.ee_pose(data)
                    arm_channel.set_teleop_zero(
                        quest_frame.wrist_pos_world,
                        quest_frame.wrist_quat_world,
                        ee_pos_B,
                        ee_R_B,
                        landmarks_wrist=quest_frame.landmarks_wrist,
                    )
                    state["target_pos_B"] = ee_pos_B.copy()
                    state["target_R_B"] = (
                        absolute_orientation_target_B(quest_frame, R_BQ, args.orientation_source)
                        if args.control_orientation and args.orientation_tracking == "absolute"
                        else ee_R_B.copy()
                    )
                    state["last_qtarget"] = joint_qpos(model, data, ik.joint_ids)
                    print("Teleop zero set for weighted IK: current Quest wrist maps to current SO101 end-effector pose.")

                target = arm_channel.compute_target(quest_frame)
                state["target_pos_B"] = np.clip(target.target_pos_B, workspace_min, workspace_max)
                if args.control_orientation and args.orientation_tracking == "absolute":
                    state["target_R_B"] = absolute_orientation_target_B(quest_frame, R_BQ, args.orientation_source)
                elif target.target_R_B is not None:
                    state["target_R_B"] = target.target_R_B.copy()
                last_delta_p_Q = target.delta_p_Q

                _raw_hand, filtered_hand = hand_retargeter(quest_frame.landmarks_wrist)

            now = time.time()
            raw_control_dt = max(0.0, now - last_control_time)
            last_control_dt = float(np.clip(raw_control_dt, 0.001, 0.03))
            last_control_time = now
            sim_accumulator += min(raw_control_dt, 0.05)
            stale = last_frame_time == 0.0 or now - last_frame_time > args.timeout
            ee_pos_B, ee_R_B = ik.ee_pose(data)
            if arm_channel.is_calibrated and not stale and not state["paused"]:
                target_R_B = state["target_R_B"] if args.control_orientation else None
                cmd = vel_controller.compute(state["target_pos_B"], target_R_B, ee_pos_B, ee_R_B)
                last_velocity = cmd.xdot
                last_error = cmd.position_error
                last_rot_error = np.zeros(3, dtype=np.float64) if cmd.rotation_error is None else cmd.rotation_error
                qtarget, last_qdot = solve_weighted_task_space_ik(
                    ik,
                    data,
                    cmd.xdot,
                    dt=last_control_dt,
                    position_weight=args.position_weight,
                    orientation_weight=args.orientation_weight,
                    control_orientation=args.control_orientation,
                )
                state["last_qtarget"] = qtarget.copy()
                ik.apply_position_targets(data, qtarget)
            else:
                last_velocity = np.zeros(6 if args.control_orientation else 3, dtype=np.float64)
                last_error = state["target_pos_B"] - ee_pos_B
                last_rot_error = np.zeros(3, dtype=np.float64)
                last_qdot = np.zeros(len(ik.joint_ids), dtype=np.float64)
                ik.apply_position_targets(data, state["last_qtarget"])
                if stale:
                    alpha = float(np.clip(args.hand_smoothing_alpha, 0.0, 1.0))
                    filtered_hand = alpha * filtered_hand + (1.0 - alpha) * SAFE_OPEN_HAND
                    hand_retargeter.prev_action = filtered_hand.astype(np.float32)

            normalized_aero_hand_to_ctrl(model, filtered_hand, ctrl=data.ctrl)
            last_sim_steps = 0
            while sim_accumulator >= sim_step_dt and last_sim_steps < 10:
                mujoco.mj_step(model, data)
                sim_accumulator -= sim_step_dt
                last_sim_steps += 1
            if last_sim_steps >= 10:
                sim_accumulator = 0.0
            viewer.sync()

            if now - last_debug_time >= args.debug_interval:
                print(
                    "debug "
                    f"frame_id={last_frame_id} new={got_new_frame} drained={drained} stale={stale} paused={state['paused']} "
                    f"delta_p_Q={np.array2string(last_delta_p_Q, precision=4, suppress_small=True)} "
                    f"target_B={np.array2string(state['target_pos_B'], precision=4, suppress_small=True)} "
                    f"ee_B={np.array2string(ee_pos_B, precision=4, suppress_small=True)} "
                    f"err={np.array2string(last_error, precision=4, suppress_small=True)} "
                    f"rot_err={np.array2string(last_rot_error, precision=4, suppress_small=True)} "
                    f"xdot={np.array2string(last_velocity, precision=4, suppress_small=True)} "
                    f"qdot={np.array2string(last_qdot, precision=4, suppress_small=True)} "
                    f"dt={last_control_dt:.4f} "
                    f"sim_steps={last_sim_steps} "
                    f"hand={np.array2string(filtered_hand, precision=3, suppress_small=True)}"
                )
                last_debug_time = now

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
