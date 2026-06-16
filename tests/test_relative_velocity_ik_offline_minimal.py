import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("Install required dependency with: pip install mujoco numpy") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.arm_teleop import (
    DampedLeastSquaresIK,
    RelativePoseMapper,
    VelocityTeleopConfig,
    VelocityTeleopController,
    WorkspaceLimiter,
    joint_qpos,
)


DEFAULT_MODEL = PROJECT_ROOT / "models/so101_aero_hand/SO101_aerohand.xml"
DEFAULT_ARM_JOINTS = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll"


def obj_name(model, obj_type, idx):
    return mujoco.mj_id2name(model, obj_type, int(idx)) or f"<unnamed_{idx}>"


def parse_vec3(values, name: str) -> np.ndarray:
    if isinstance(values, (list, tuple)):
        text = " ".join(str(value) for value in values)
    else:
        text = str(values)
    parsed = [float(v) for v in text.replace(",", " ").split()]
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError(f"{name} expected 3 floats, got {text!r}")
    return np.asarray(parsed, dtype=np.float64)


def resolve_model(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {path}")
    return path


def ctrl_midpoints(model) -> np.ndarray:
    return 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])


def is_hand_like(name: str) -> bool:
    text = name.lower()
    return any(k in text for k in ("thumb", "index", "middle", "ring", "pinky", "little", "aero", "right_", "tendon"))


def actuator_for_joint(model, joint_id: int) -> int | None:
    matches = []
    for actuator_id in range(model.nu):
        if (
            int(model.actuator_trntype[actuator_id]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[actuator_id, 0]) == int(joint_id)
        ):
            matches.append(actuator_id)
    if len(matches) == 1:
        return int(matches[0])
    return None


def actuator_target(model, actuator_id: int) -> str:
    trn_type = int(model.actuator_trntype[actuator_id])
    trn_id = int(model.actuator_trnid[actuator_id, 0])
    if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT) and trn_id >= 0:
        return f"joint:{obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, trn_id)}"
    if trn_type == int(mujoco.mjtTrn.mjTRN_TENDON) and trn_id >= 0:
        return f"tendon:{obj_name(model, mujoco.mjtObj.mjOBJ_TENDON, trn_id)}"
    return f"trntype={trn_type} trnid={model.actuator_trnid[actuator_id].tolist()}"


def fake_hand_position(step: int, total_steps: int, start: np.ndarray, delta: np.ndarray) -> np.ndarray:
    if total_steps <= 1:
        phase = 1.0
    else:
        phase = float(step) / float(total_steps - 1)
    phase = np.clip(phase, 0.0, 1.0)
    smooth_phase = 0.5 - 0.5 * np.cos(np.pi * phase)
    return start + smooth_phase * delta


def parse_args():
    parser = argparse.ArgumentParser(description="Offline relative-position mapping + task-space velocity IK smoke test.")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--ee_site", default="grasp_site")
    parser.add_argument("--ee_body", default=None)
    parser.add_argument("--arm_joint_names", default=DEFAULT_ARM_JOINTS)
    parser.add_argument("--arm_joint_prefix", default=None)
    parser.add_argument("--hand_delta", nargs="+", default=["0.02", "0.0", "0.0"], help="Fake Quest palm xyz delta.")
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--kp_pos", type=float, default=5.0)
    parser.add_argument("--max_linear_speed", type=float, default=0.05)
    parser.add_argument("--ik_damping", type=float, default=0.05)
    parser.add_argument("--max_joint_speed", type=float, default=1.0)
    parser.add_argument("--target_smoothing_alpha", type=float, default=0.0)
    parser.add_argument("--joint_target_smoothing_alpha", type=float, default=0.0)
    parser.add_argument("--workspace_min", nargs=3, type=float, default=[-0.5, -0.5, 0.02])
    parser.add_argument("--workspace_max", nargs=3, type=float, default=[0.5, 0.5, 0.7])
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--disable_gravity", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    hand_delta = parse_vec3(args.hand_delta, "--hand_delta")
    model_path = resolve_model(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if args.disable_gravity:
        model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    data.ctrl[:] = ctrl_midpoints(model)
    mujoco.mj_forward(model, data)

    joint_names = [name.strip() for name in args.arm_joint_names.split(",") if name.strip()] if args.arm_joint_names else None
    if joint_names:
        handish = [name for name in joint_names if is_hand_like(name)]
        if handish:
            raise SystemExit(f"Refusing to include likely Aero Hand joints in arm IK: {handish}")

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

    arm_actuator_ids = []
    for joint_id in ik.joint_ids:
        actuator_id = actuator_for_joint(model, joint_id)
        if actuator_id is None:
            joint_name = obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise SystemExit(f"No unique joint actuator found for arm joint {joint_name}")
        arm_actuator_ids.append(actuator_id)
    non_arm_actuator_ids = [idx for idx in range(model.nu) if idx not in set(arm_actuator_ids)]

    ee_start, ee_rot_start = ik.ee_pose(data)
    hand_start = np.zeros(3, dtype=np.float64)
    hand_rot = np.eye(3, dtype=np.float64)
    workspace = WorkspaceLimiter(
        minimum=np.asarray(args.workspace_min, dtype=np.float64),
        maximum=np.asarray(args.workspace_max, dtype=np.float64),
    )
    mapper = RelativePoseMapper(
        scale=float(args.scale),
        R_align=np.eye(3, dtype=np.float64),
        workspace=workspace,
        smoothing_alpha=float(args.target_smoothing_alpha),
    )
    mapper.calibrate(hand_start, hand_rot, ee_start, ee_rot_start)
    vel_controller = VelocityTeleopController(
        VelocityTeleopConfig(
            kp_pos=float(args.kp_pos),
            max_linear_speed=float(args.max_linear_speed),
            control_orientation=False,
        )
    )

    ctrl_start = data.ctrl.copy()
    q_start = joint_qpos(model, data, ik.joint_ids)
    target_position = ee_start.copy()
    velocity_cmd = np.zeros(3, dtype=np.float64)
    ik_qdot = np.zeros(len(ik.joint_ids), dtype=np.float64)
    arm_qtarget = q_start.copy()
    max_error = 0.0

    print(f"model={model_path}")
    print(f"gravity={np.array2string(model.opt.gravity, precision=6)}")
    print(f"ee_kind={ik.ee_kind} ee_name={args.ee_site or args.ee_body}")
    print(f"arm_joints={ik.joint_names}")
    print(f"hand_start={np.array2string(hand_start, precision=6)} hand_delta={np.array2string(hand_delta, precision=6)} scale={args.scale}")
    print(f"ee_start={np.array2string(ee_start, precision=6)} expected_final_target={np.array2string(ee_start + args.scale * hand_delta, precision=6)}")
    print("arm_actuators:")
    for actuator_id in arm_actuator_ids:
        print(f"  {actuator_id}: {obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)} -> {actuator_target(model, actuator_id)}")
    print("non_arm_actuators_held_at_midpoint:")
    for actuator_id in non_arm_actuator_ids:
        print(f"  {actuator_id}: {obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)} -> {actuator_target(model, actuator_id)}")

    for step in range(int(args.steps)):
        hand_position = fake_hand_position(step, int(args.steps), hand_start, hand_delta)
        target_position, _target_rotation = mapper.target_pose(hand_position, hand_rot, control_orientation=False)
        ee_position, ee_rotation = ik.ee_pose(data)
        cmd = vel_controller.compute(target_position, None, ee_position, ee_rotation)
        velocity_cmd = cmd.xdot
        arm_qtarget, ik_qdot = ik.solve(
            data,
            velocity_cmd,
            dt=float(model.opt.timestep),
            control_orientation=False,
        )

        data.ctrl[non_arm_actuator_ids] = ctrl_start[non_arm_actuator_ids]
        ik.apply_position_targets(data, arm_qtarget)
        mujoco.mj_step(model, data)
        max_error = max(max_error, float(np.linalg.norm(cmd.position_error)))

        one_based = step + 1
        if args.print_every > 0 and (one_based == 1 or one_based % int(args.print_every) == 0 or one_based == int(args.steps)):
            ee_now, _ = ik.ee_pose(data)
            print(
                f"step={one_based} "
                f"hand={np.array2string(hand_position, precision=6)} "
                f"target={np.array2string(target_position, precision=6)} "
                f"ee={np.array2string(ee_now, precision=6)} "
                f"err_norm={float(np.linalg.norm(target_position - ee_now)):.9f} "
                f"v_cmd={np.array2string(velocity_cmd, precision=6)} "
                f"qdot={np.array2string(ik_qdot, precision=6)}"
            )

    ee_final, _ = ik.ee_pose(data)
    q_final = joint_qpos(model, data, ik.joint_ids)
    ctrl_final = data.ctrl.copy()
    non_arm_ctrl_delta = ctrl_final[non_arm_actuator_ids] - ctrl_start[non_arm_actuator_ids]

    print("summary:")
    print(f"  calibrated={mapper.is_calibrated}")
    print(f"  p_hand0={np.array2string(mapper.p_hand0, precision=9)}")
    print(f"  p_ee0={np.array2string(mapper.p_ee0, precision=9)}")
    print(f"  final_target={np.array2string(target_position, precision=9)}")
    print(f"  ee_final={np.array2string(ee_final, precision=9)}")
    print(f"  ee_actual_delta={np.array2string(ee_final - ee_start, precision=9)}")
    print(f"  target_delta_from_start={np.array2string(target_position - ee_start, precision=9)}")
    print(f"  final_error_norm={float(np.linalg.norm(target_position - ee_final)):.9f}")
    print(f"  max_error_norm={max_error:.9f}")
    print(f"  arm_q_start={np.array2string(q_start, precision=9)}")
    print(f"  arm_q_final={np.array2string(q_final, precision=9)}")
    print(f"  arm_qtarget_last={np.array2string(arm_qtarget, precision=9)}")
    print(f"  ik_qdot_last={np.array2string(ik_qdot, precision=9)}")
    print(f"  ee_velocity_cmd_last={np.array2string(velocity_cmd, precision=9)}")
    print(f"  non_arm_ctrl_max_abs_delta={float(np.max(np.abs(non_arm_ctrl_delta))) if len(non_arm_ctrl_delta) else 0.0:.12f}")
    print(f"  qpos_finite={bool(np.all(np.isfinite(data.qpos)))} qvel_finite={bool(np.all(np.isfinite(data.qvel)))}")


if __name__ == "__main__":
    main()
