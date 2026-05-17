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


def name(model, obj_type, idx):
    return mujoco.mj_id2name(model, obj_type, idx) or f"<unnamed_{idx}>"


def joint_type_name(value):
    mapping = {
        int(mujoco.mjtJoint.mjJNT_FREE): "free",
        int(mujoco.mjtJoint.mjJNT_BALL): "ball",
        int(mujoco.mjtJoint.mjJNT_SLIDE): "slide",
        int(mujoco.mjtJoint.mjJNT_HINGE): "hinge",
    }
    return mapping.get(int(value), f"unknown({int(value)})")


def actuator_target(model, actuator_id):
    trn_type = int(model.actuator_trntype[actuator_id])
    trn_id = int(model.actuator_trnid[actuator_id, 0])
    if trn_id < 0:
        return f"trntype={trn_type} target=<none>"
    if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT):
        return f"joint:{name(model, mujoco.mjtObj.mjOBJ_JOINT, trn_id)}"
    if trn_type == int(mujoco.mjtTrn.mjTRN_TENDON):
        return f"tendon:{name(model, mujoco.mjtObj.mjOBJ_TENDON, trn_id)}"
    if trn_type == int(mujoco.mjtTrn.mjTRN_SITE):
        return f"site:{name(model, mujoco.mjtObj.mjOBJ_SITE, trn_id)}"
    if trn_type == int(mujoco.mjtTrn.mjTRN_BODY):
        return f"body:{name(model, mujoco.mjtObj.mjOBJ_BODY, trn_id)}"
    return f"trntype={trn_type} trnid={model.actuator_trnid[actuator_id].tolist()}"


def looks_like_arm(text):
    text = text.lower()
    keywords = ("shoulder", "elbow", "wrist", "so101", "arm", "pan", "lift", "flex", "roll")
    hand_keywords = ("thumb", "index", "middle", "ring", "pinky", "little", "tendon", "aero", "finger")
    return any(k in text for k in keywords) and not any(k in text for k in hand_keywords)


def looks_like_aero_hand(text):
    text = text.lower()
    keywords = (
        "aero",
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
        "little",
        "right_",
        "if_",
        "mf_",
        "rf_",
        "pf_",
        "th_",
        "tendon",
    )
    return any(k in text for k in keywords)


def looks_like_ee(text):
    text = text.lower()
    keywords = ("ee", "end_effector", "gripperframe", "gripper", "wrist_site", "palm", "attach", "tool", "tcp", "grasp")
    return any(k in text for k in keywords) and "tendon" not in text


def print_guess(label, values):
    print(f"\nGuessed {label}:")
    if values:
        for value in values:
            print(f"  {value}")
    else:
        print("  <none>")


def parse_args():
    parser = argparse.ArgumentParser(description="Inspect MuJoCo model joints, actuators, bodies, and sites.")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    model = mujoco.MjModel.from_xml_path(str(model_path))
    print(f"model: {model_path}")
    print(
        f"nq={model.nq} nv={model.nv} nu={model.nu} "
        f"njnt={model.njnt} nbody={model.nbody} nsite={model.nsite} ngeom={model.ngeom}"
    )

    print("\nJoints:")
    joint_names = []
    for idx in range(model.njnt):
        joint_name = name(model, mujoco.mjtObj.mjOBJ_JOINT, idx)
        joint_names.append(joint_name)
        joint_type = joint_type_name(model.jnt_type[idx])
        qposadr = int(model.jnt_qposadr[idx])
        dofadr = int(model.jnt_dofadr[idx])
        limited = bool(model.jnt_limited[idx])
        joint_range = model.jnt_range[idx].tolist() if limited else None
        print(f"  {idx:3d} {joint_name:32s} type={joint_type} qposadr={qposadr} dofadr={dofadr} range={joint_range}")

    print("\nActuators:")
    actuator_names = []
    for idx in range(model.nu):
        actuator_name = name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
        actuator_names.append(actuator_name)
        ctrlrange = model.actuator_ctrlrange[idx].tolist()
        target = actuator_target(model, idx)
        print(f"  {idx:3d} {actuator_name:32s} target={target:36s} ctrlrange={ctrlrange}")

    print("\nBodies:")
    body_names = []
    for idx in range(model.nbody):
        body_name = name(model, mujoco.mjtObj.mjOBJ_BODY, idx)
        body_names.append(body_name)
        print(f"  {idx:3d} {body_name}")

    print("\nSites:")
    site_names = []
    if model.nsite == 0:
        print("  当前模型没有 site，IK 最好先给末端 body 或添加 ee_site。")
    else:
        for idx in range(model.nsite):
            site_name = name(model, mujoco.mjtObj.mjOBJ_SITE, idx)
            site_names.append(site_name)
            print(f"  {idx:3d} {site_name}")

    print("\nGeoms:")
    geom_names = []
    for idx in range(model.ngeom):
        geom_name = name(model, mujoco.mjtObj.mjOBJ_GEOM, idx)
        geom_names.append(geom_name)
        print(f"  {idx:3d} {geom_name}")

    arm_joints = [joint_name for joint_name in joint_names if looks_like_arm(joint_name)]
    hand_joints = [joint_name for joint_name in joint_names if looks_like_aero_hand(joint_name)]
    arm_actuators = [
        actuator_name
        for actuator_name in actuator_names
        if looks_like_arm(actuator_name)
    ]
    hand_actuators = [
        actuator_name
        for actuator_name in actuator_names
        if looks_like_aero_hand(actuator_name)
    ]
    ee_sites = [site_name for site_name in site_names if looks_like_ee(site_name)]
    ee_bodies = [body_name for body_name in body_names if looks_like_ee(body_name)]

    print_guess("arm joints", arm_joints)
    print_guess("Aero Hand joints", hand_joints)
    print_guess("arm actuators", arm_actuators)
    print_guess("Aero Hand actuators", hand_actuators)
    print_guess("possible end-effector sites", ee_sites)
    print_guess("possible end-effector bodies", ee_bodies)

    if model.nu:
        midpoint = np.mean(model.actuator_ctrlrange, axis=1)
        print(f"\nctrl midpoint shape={midpoint.shape} values={np.array2string(midpoint, precision=6, suppress_small=True)}")


if __name__ == "__main__":
    main()
