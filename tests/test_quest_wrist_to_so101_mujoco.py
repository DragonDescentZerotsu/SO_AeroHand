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

from aero_quest.arm_teleop import MuJoCoSO101Adapter, QuestArmTeleopController
from aero_quest.so101_aero_control import ctrl_midpoints


DEFAULT_MODEL = PROJECT_ROOT / "models/so101_aero_hand/scene.xml"
DEFAULT_ARM_JOINTS = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll"


def make_landmarks(wrist_xyz) -> np.ndarray:
    """Create a simple valid 21-landmark hand with a configurable wrist."""
    wrist = np.asarray(wrist_xyz, dtype=np.float64)
    P = np.zeros((21, 3), dtype=np.float64)
    P[0] = wrist
    P[5] = wrist + np.array([0.04, 0.04, 0.0], dtype=np.float64)
    P[9] = wrist + np.array([0.00, 0.08, 0.0], dtype=np.float64)
    P[17] = wrist + np.array([-0.04, 0.04, 0.0], dtype=np.float64)

    # Fill the unused landmarks with plausible nonzero finger points so shape
    # checks pass and debug dumps look like a hand frame.
    for base, ids in ((P[5], [6, 7, 8]), (P[9], [10, 11, 12]), (wrist + [0.02, 0.065, 0.0], [13, 14, 15, 16]), (P[17], [18, 19, 20])):
        for k, idx in enumerate(ids, start=1):
            P[idx] = base + np.array([0.0, 0.025 * k, -0.004 * k], dtype=np.float64)
    P[1] = wrist + np.array([-0.035, 0.015, -0.004], dtype=np.float64)
    P[2] = wrist + np.array([-0.055, 0.035, -0.008], dtype=np.float64)
    P[3] = wrist + np.array([-0.070, 0.055, -0.012], dtype=np.float64)
    P[4] = wrist + np.array([-0.085, 0.075, -0.016], dtype=np.float64)
    return P


def resolve_model(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Smoke-test Quest wrist delta -> SO101 MuJoCo EE motion through QuestArmTeleopController.")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--ee-site", default="grasp_site")
    parser.add_argument("--ee-body", default=None)
    parser.add_argument("--arm-joint-names", default=DEFAULT_ARM_JOINTS)
    parser.add_argument("--arm-joint-prefix", default=None)
    parser.add_argument("--wrist-delta", nargs=3, type=float, default=[0.04, 0.0, 0.0])
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--print-every", type=int, default=10)
    parser.add_argument("--max-ee-step", type=float, default=0.02)
    parser.add_argument("--max-joint-step", type=float, default=0.03)
    parser.add_argument("--disable-gravity", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    if args.disable_gravity:
        model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    data.ctrl[:] = ctrl_midpoints(model)
    mujoco.mj_forward(model, data)
    for _ in range(int(args.settle_steps)):
        mujoco.mj_step(model, data)

    joint_names = [name.strip() for name in args.arm_joint_names.split(",") if name.strip()] if args.arm_joint_names else None
    robot = MuJoCoSO101Adapter(
        model,
        data,
        ee_site=args.ee_site,
        ee_body=args.ee_body,
        joint_names=joint_names,
        joint_prefix=args.arm_joint_prefix,
        damping=0.05,
        max_iters=50,
    )
    controller = QuestArmTeleopController(
        scale=args.scale,
        R_robot_from_quest=np.eye(3),
        use_orientation=False,
        position_alpha=1.0,
        deadzone=0.0,
        max_ee_step=args.max_ee_step,
        max_joint_step=args.max_joint_step,
        workspace_bounds=None,
    )

    P0 = make_landmarks([0.0, 0.0, 0.0])
    P1 = make_landmarks(args.wrist_delta)
    q_start = robot.get_joint_positions()
    ee_start = robot.get_ee_position()
    reset_debug = controller.reset(P0, robot)

    print(f"model={model_path}")
    print(f"arm_joints={robot.joint_names}")
    print(f"ee_start={np.array2string(ee_start, precision=6)}")
    print(f"wrist_delta={np.array2string(np.asarray(args.wrist_delta), precision=6)} scale={args.scale}")
    print(f"reset_initialized={reset_debug['initialized']}")

    last_debug = None
    for step in range(1, int(args.steps) + 1):
        last_debug = controller.update(P1, robot)
        mujoco.mj_step(model, data)
        if args.print_every > 0 and (step == 1 or step % int(args.print_every) == 0 or step == int(args.steps)):
            ee_now = robot.get_ee_position()
            print(
                f"step={step} "
                f"delta_p_hand={np.array2string(last_debug['delta_p_hand'], precision=6)} "
                f"p_ee_target={np.array2string(last_debug['p_ee_target'], precision=6)} "
                f"ik_success={last_debug['ik_success']} "
                f"q_cmd={np.array2string(last_debug['q_cmd'], precision=6)} "
                f"ee_now={np.array2string(ee_now, precision=6)}"
            )

    ee_final = robot.get_ee_position()
    q_final = robot.get_joint_positions()
    expected_first_target = ee_start + args.scale * np.asarray(args.wrist_delta, dtype=np.float64)
    actual_ee_delta = ee_final - ee_start
    q_delta_norm = float(np.linalg.norm(q_final - q_start))
    ee_delta_norm = float(np.linalg.norm(actual_ee_delta))
    ik_success = bool(last_debug["ik_success"]) if last_debug is not None else False

    print("summary:")
    print(f"  chain=P[0] wrist -> delta_p_hand -> p_ee_target -> IK -> q_target -> MuJoCo")
    print(f"  expected_unlimited_target={np.array2string(expected_first_target, precision=9)}")
    print(f"  final_p_ee_target={np.array2string(last_debug['p_ee_target'], precision=9)}")
    print(f"  ee_final={np.array2string(ee_final, precision=9)}")
    print(f"  ee_actual_delta={np.array2string(actual_ee_delta, precision=9)}")
    print(f"  q_start={np.array2string(q_start, precision=9)}")
    print(f"  q_final={np.array2string(q_final, precision=9)}")
    print(f"  q_delta_norm={q_delta_norm:.9f}")
    print(f"  ee_delta_norm={ee_delta_norm:.9f}")
    print(f"  ik_success={ik_success}")
    print(f"  qpos_finite={bool(np.all(np.isfinite(data.qpos)))} qvel_finite={bool(np.all(np.isfinite(data.qvel)))}")

    assert ik_success, "IK did not report success"
    assert q_delta_norm > 1e-6, "Arm joint command did not change"
    assert ee_delta_norm > 1e-5, "MuJoCo end-effector did not move"


if __name__ == "__main__":
    main()
