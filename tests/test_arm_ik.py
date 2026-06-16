import argparse
import sys
from pathlib import Path

try:
    import mujoco
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install required dependencies with: pip install mujoco numpy") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.arm_teleop import DampedLeastSquaresIK, joint_qpos


DEFAULT_MODEL = PROJECT_ROOT / "models/so101_aero_hand/SO101_aerohand.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test resolved-rate arm IK on a MuJoCo model.")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--ee_site", default="grasp_site")
    parser.add_argument("--ee_body", default=None)
    parser.add_argument("--arm_joint_prefix", default=None)
    parser.add_argument("--arm_joint_names", default="shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll")
    parser.add_argument("--ik_damping", type=float, default=0.05)
    parser.add_argument("--max_joint_speed", type=float, default=1.5)
    parser.add_argument("--dt", type=float, default=0.005)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
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
    )
    ee_position, _ = ik.ee_pose(data)
    print(f"model={model_path}")
    print(f"ee_kind={ik.ee_kind} ee_id={ik.ee_id} ee_position={np.array2string(ee_position, precision=5)}")
    print(f"arm_joints={ik.joint_names}")
    for label, xdot in (
        ("+x", np.array([0.02, 0.0, 0.0])),
        ("+y", np.array([0.0, 0.02, 0.0])),
        ("+z", np.array([0.0, 0.0, 0.02])),
    ):
        q_before = joint_qpos(model, data, ik.joint_ids)
        q_target, qdot = ik.solve(data, xdot, dt=args.dt, control_orientation=False)
        print(
            f"{label}: xdot={np.array2string(xdot, precision=4)} "
            f"qdot={np.array2string(qdot, precision=4)} "
            f"q_before={np.array2string(q_before, precision=4)} "
            f"q_target={np.array2string(q_target, precision=4)} "
            f"finite={bool(np.all(np.isfinite(qdot)) and np.all(np.isfinite(q_target)))}"
        )


if __name__ == "__main__":
    main()
