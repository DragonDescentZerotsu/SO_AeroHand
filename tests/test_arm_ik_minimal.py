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

from aero_quest.arm_teleop import DampedLeastSquaresIK, clamp_norm, joint_qpos


DEFAULT_MODEL = PROJECT_ROOT / "models/so101_aero_hand/scene.xml"
DEFAULT_ARM_JOINTS = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll"


def obj_name(model, obj_type, idx):
    return mujoco.mj_id2name(model, obj_type, int(idx)) or f"<unnamed_{idx}>"


def resolve_model(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {path}")
    return path


def parse_vec3(values) -> np.ndarray:
    if isinstance(values, (list, tuple)):
        text = " ".join(str(value) for value in values)
    else:
        text = str(values)
    values = [float(v) for v in text.replace(",", " ").split()]
    if len(values) != 3:
        raise argparse.ArgumentTypeError(f"Expected 3 floats, got {text!r}")
    return np.asarray(values, dtype=np.float64)


def is_hand_like(name: str) -> bool:
    text = name.lower()
    return any(k in text for k in ("thumb", "index", "middle", "ring", "pinky", "little", "aero", "right_", "tendon"))


def ctrl_midpoints(model) -> np.ndarray:
    return 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])


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


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal closed-loop DLS IK test for arm-only MuJoCo control.")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--ee_site", default="grasp_site")
    parser.add_argument("--ee_body", default=None)
    parser.add_argument("--arm_joint_names", default=DEFAULT_ARM_JOINTS)
    parser.add_argument("--arm_joint_prefix", default=None)
    parser.add_argument("--delta", nargs="+", default=["0.01", "0.0", "0.0"], help="Target EE xyz delta, e.g. --delta 0.01 0 0")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--kp_pos", type=float, default=5.0)
    parser.add_argument("--max_linear_speed", type=float, default=0.05)
    parser.add_argument("--ik_damping", type=float, default=0.05)
    parser.add_argument("--max_joint_speed", type=float, default=1.0)
    parser.add_argument("--print_every", type=int, default=50)
    parser.add_argument("--disable_gravity", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    args.delta = parse_vec3(args.delta)
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
    )

    arm_actuator_ids = []
    for joint_id in ik.joint_ids:
        actuator_id = actuator_for_joint(model, joint_id)
        if actuator_id is None:
            joint_name = obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            raise SystemExit(f"No unique joint actuator found for arm joint {joint_name}")
        arm_actuator_ids.append(actuator_id)
    non_arm_actuator_ids = [idx for idx in range(model.nu) if idx not in set(arm_actuator_ids)]

    ee_start, _ = ik.ee_pose(data)
    target = ee_start + np.asarray(args.delta, dtype=np.float64)
    q_start = joint_qpos(model, data, ik.joint_ids)
    ctrl_start = data.ctrl.copy()

    print(f"model={model_path}")
    print(f"gravity={np.array2string(model.opt.gravity, precision=6)}")
    print(f"ee_kind={ik.ee_kind} ee_name={args.ee_site or args.ee_body} ee_start={np.array2string(ee_start, precision=6)}")
    print(f"target_delta={np.array2string(args.delta, precision=6)} target={np.array2string(target, precision=6)}")
    print(f"arm_joints={ik.joint_names}")
    print("arm_actuators:")
    for actuator_id in arm_actuator_ids:
        print(f"  {actuator_id}: {obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)} -> {actuator_target(model, actuator_id)}")
    print("non_arm_actuators_held_at_midpoint:")
    for actuator_id in non_arm_actuator_ids:
        print(f"  {actuator_id}: {obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)} -> {actuator_target(model, actuator_id)}")

    last_qdot = np.zeros(len(ik.joint_ids), dtype=np.float64)
    for step in range(1, int(args.steps) + 1):
        ee_pos, _ = ik.ee_pose(data)
        error = target - ee_pos
        v_cmd = clamp_norm(float(args.kp_pos) * error, float(args.max_linear_speed))
        q_target, qdot = ik.solve(data, v_cmd, dt=float(model.opt.timestep), control_orientation=False)

        data.ctrl[non_arm_actuator_ids] = ctrl_start[non_arm_actuator_ids]
        ik.apply_position_targets(data, q_target)
        mujoco.mj_step(model, data)
        last_qdot = qdot

        if args.print_every > 0 and (step == 1 or step % int(args.print_every) == 0 or step == int(args.steps)):
            current, _ = ik.ee_pose(data)
            print(
                f"step={step} "
                f"ee={np.array2string(current, precision=6)} "
                f"err_norm={float(np.linalg.norm(target - current)):.9f} "
                f"v_cmd={np.array2string(v_cmd, precision=6)} "
                f"qdot={np.array2string(qdot, precision=6)}"
            )

    ee_final, _ = ik.ee_pose(data)
    q_final = joint_qpos(model, data, ik.joint_ids)
    ctrl_final = data.ctrl.copy()
    non_arm_ctrl_delta = ctrl_final[non_arm_actuator_ids] - ctrl_start[non_arm_actuator_ids]

    print("summary:")
    print(f"  ee_start={np.array2string(ee_start, precision=9)}")
    print(f"  ee_target={np.array2string(target, precision=9)}")
    print(f"  ee_final={np.array2string(ee_final, precision=9)}")
    print(f"  ee_actual_delta={np.array2string(ee_final - ee_start, precision=9)}")
    print(f"  final_error_norm={float(np.linalg.norm(target - ee_final)):.9f}")
    print(f"  arm_q_start={np.array2string(q_start, precision=9)}")
    print(f"  arm_q_final={np.array2string(q_final, precision=9)}")
    print(f"  last_ik_qdot={np.array2string(last_qdot, precision=9)}")
    print(f"  non_arm_ctrl_max_abs_delta={float(np.max(np.abs(non_arm_ctrl_delta))) if len(non_arm_ctrl_delta) else 0.0:.12f}")
    print(f"  qpos_finite={bool(np.all(np.isfinite(data.qpos)))} qvel_finite={bool(np.all(np.isfinite(data.qvel)))}")


if __name__ == "__main__":
    main()
