import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("Install required dependency with: pip install mujoco") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = PROJECT_ROOT / "models/so101_aero_hand/scene.xml"


def obj_name(model, obj_type, idx):
    return mujoco.mj_id2name(model, obj_type, idx) or f"<unnamed_{idx}>"


def resolve_model(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {path}")
    return path


def is_hand_like(name: str) -> bool:
    text = name.lower()
    return any(k in text for k in ("thumb", "index", "middle", "ring", "pinky", "little", "aero", "right_", "tendon"))


def actuator_target_text(model, actuator_id):
    trn_type = int(model.actuator_trntype[actuator_id])
    trn_id = int(model.actuator_trnid[actuator_id, 0])
    if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT) and trn_id >= 0:
        return f"joint:{obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, trn_id)}"
    if trn_type == int(mujoco.mjtTrn.mjTRN_TENDON) and trn_id >= 0:
        return f"tendon:{obj_name(model, mujoco.mjtObj.mjOBJ_TENDON, trn_id)}"
    return f"trntype={trn_type} trnid={model.actuator_trnid[actuator_id].tolist()}"


def print_actuator_mapping(model):
    print("available_actuator_mapping:")
    for actuator_id in range(model.nu):
        actuator_name = obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        ctrlrange = model.actuator_ctrlrange[actuator_id].tolist()
        print(f"  {actuator_id:3d} {actuator_name:32s} target={actuator_target_text(model, actuator_id):36s} ctrlrange={ctrlrange}")


def find_joint_actuator(model, joint_id, joint_name):
    candidates = []
    for actuator_id in range(model.nu):
        if (
            int(model.actuator_trntype[actuator_id]) == int(mujoco.mjtTrn.mjTRN_JOINT)
            and int(model.actuator_trnid[actuator_id, 0]) == int(joint_id)
        ):
            candidates.append(actuator_id)
    if len(candidates) == 1:
        return candidates[0]

    same_name = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
    if same_name >= 0 and same_name not in candidates:
        candidates.append(int(same_name))
    if len(candidates) == 1:
        return candidates[0]
    return None


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal single-arm-joint position-control test.")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--joint_name", required=True)
    parser.add_argument("--delta", type=float, default=0.05)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--allow_hand_joint", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model(args.model)
    print(f"model={model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, args.joint_name)
    if joint_id < 0:
        available = [obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(model.njnt)]
        raise SystemExit(f"Joint not found: {args.joint_name}\navailable_joints={available}")
    if is_hand_like(args.joint_name) and not args.allow_hand_joint:
        raise SystemExit(
            f"Refusing to control likely Aero Hand joint for arm test: {args.joint_name}\n"
            "Pass an arm joint such as shoulder_pan/shoulder_lift/elbow_flex/wrist_flex/wrist_roll."
        )

    actuator_id = find_joint_actuator(model, joint_id, args.joint_name)
    if actuator_id is None:
        print(f"No unique joint actuator found for joint={args.joint_name}.")
        print_actuator_mapping(model)
        raise SystemExit("Cannot run single-joint control test without an unambiguous joint actuator.")

    qposadr = int(model.jnt_qposadr[joint_id])
    q_before = float(data.qpos[qposadr])
    target = q_before + float(args.delta)
    lo, hi = model.actuator_ctrlrange[actuator_id]
    target_clipped = float(np.clip(target, lo, hi))

    data.ctrl[:] = 0.0
    for idx in range(model.nu):
        ctrl_lo, ctrl_hi = model.actuator_ctrlrange[idx]
        data.ctrl[idx] = 0.5 * (ctrl_lo + ctrl_hi)
    data.ctrl[actuator_id] = target_clipped

    actuator_name = obj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
    print(f"joint={args.joint_name} joint_id={joint_id} qposadr={qposadr}")
    print(f"actuator_id={actuator_id} actuator_name={actuator_name} target={actuator_target_text(model, actuator_id)}")
    print(f"q_before={q_before:.9f} requested_target={target:.9f} clipped_target={target_clipped:.9f}")

    for _ in range(int(args.steps)):
        mujoco.mj_step(model, data)

    q_after = float(data.qpos[qposadr])
    print(f"steps={int(args.steps)}")
    print(f"q_after={q_after:.9f} q_delta={q_after - q_before:.9f}")
    print(f"moved={bool(abs(q_after - q_before) > 1e-6)}")
    print(f"qpos_finite={bool(np.all(np.isfinite(data.qpos)))} qvel_finite={bool(np.all(np.isfinite(data.qvel)))}")


if __name__ == "__main__":
    main()
