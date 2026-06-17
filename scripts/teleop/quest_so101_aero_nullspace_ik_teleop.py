import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit("Missing runtime dependency. Install with: pip install mujoco numpy") from exc

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
from aero_quest.quest_hand_frame import (
    QuestHandFrame,
    RelativeWristArmController,
    palm_frame_from_landmarks_wrist,
    quat_xyzw_to_matrix,
    quest_hand_frame_from_sdk,
)
from aero_quest.retargeting import AeroHandRetargetingWrapper
from aero_quest.so101_aero_control import normalized_aero_hand_to_ctrl, print_combined_actuator_info


DEFAULT_DESCRIPTION = "Quest wrist pose -> position-priority SO101 IK with nullspace orientation, landmarks -> Aero Hand."
DEFAULT_MODEL = PROJECT_ROOT / "models/so101_aero_hand/SO101_aerohand.xml"
DEFAULT_ARM_JOINTS = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll"
DEFAULT_EE_SITE = "aero_wrist_site"
DEFAULT_SCALE = 0.9
DEFAULT_WORKSPACE_MIN = ["0.05", "-0.35", "0.03"]
DEFAULT_WORKSPACE_MAX = ["0.55", "0.35", "1.35"]
DEFAULT_KP_POS = 10.0
DEFAULT_KP_ROT = 1.2
DEFAULT_MAX_LINEAR_SPEED = 0.45
DEFAULT_MAX_ANGULAR_SPEED = 0.8
DEFAULT_MAX_JOINT_SPEED = 3.0
DEFAULT_IK_MODE = "position_nullspace"
DEFAULT_ROBOT_GRAVITY_ROOT = "base"
SAFE_OPEN_HAND = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)

# Quest/Unity Q: +X right, +Y up, +Z forward.
# SO101/MuJoCo debug B: +X forward, +Y left, +Z up.
DEFAULT_R_BQ = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_vec3(values, name):
    if isinstance(values, (list, tuple, np.ndarray)):
        text = " ".join(str(value) for value in values)
    else:
        text = str(values)
    parsed = [float(v) for v in text.replace(",", " ").split()]
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError(f"{name} expected 3 floats, got {text!r}")
    return np.asarray(parsed, dtype=np.float64)


def parse_matrix(text):
    values = [float(v) for v in str(text).replace(",", " ").split()]
    if len(values) != 9:
        raise argparse.ArgumentTypeError("--R_BQ expects 9 floats, row-major")
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def resolve_model(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {path}")
    return path


def ctrl_midpoints(model):
    return 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])


def body_name(model, body_id: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(body_id)) or f"<body_{body_id}>"


def body_subtree_ids(model, root_body_name: str) -> list[int]:
    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
    if root_id < 0:
        raise ValueError(f"Robot gravity compensation root body not found: {root_body_name!r}")

    selected = []
    for body_id in range(1, model.nbody):
        cursor = body_id
        while cursor != 0:
            if cursor == root_id:
                selected.append(body_id)
                break
            cursor = int(model.body_parentid[cursor])
    return selected


def apply_robot_gravity_compensation(model, root_body_name: str, value: float = 1.0) -> list[str]:
    body_ids = body_subtree_ids(model, root_body_name)
    model.body_gravcomp[body_ids] = float(value)
    return [body_name(model, body_id) for body_id in body_ids]


def initialize_viewer_camera(viewer, model) -> None:
    """Initialize the passive viewer free camera from MJCF statistic/global options."""
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    viewer.cam.fixedcamid = -1
    viewer.cam.lookat[:] = np.asarray(model.stat.center, dtype=np.float64)
    viewer.cam.distance = max(0.1, 1.5 * float(model.stat.extent))
    viewer.cam.azimuth = float(model.vis.global_.azimuth)
    viewer.cam.elevation = float(model.vis.global_.elevation)


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


def pose_xyz(pose) -> np.ndarray | None:
    if pose is None or not all(hasattr(pose, attr) for attr in ("x", "y", "z")):
        return None
    return np.asarray([pose.x, pose.y, pose.z], dtype=np.float64)


def pose_quat_xyzw(pose) -> np.ndarray | None:
    if pose is None or not all(hasattr(pose, attr) for attr in ("qx", "qy", "qz", "qw")):
        return None
    quat = np.asarray([pose.qx, pose.qy, pose.qz, pose.qw], dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        return None
    return quat / norm


def matrix_to_quat_xyzw(R: np.ndarray) -> np.ndarray:
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(R))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (R[2, 1] - R[1, 2]) / s
        qy = (R[0, 2] - R[2, 0]) / s
        qz = (R[1, 0] - R[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(R)))
        if idx == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            qw = (R[2, 1] - R[1, 2]) / s
            qx = 0.25 * s
            qy = (R[0, 1] + R[1, 0]) / s
            qz = (R[0, 2] + R[2, 0]) / s
        elif idx == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            qw = (R[0, 2] - R[2, 0]) / s
            qx = (R[0, 1] + R[1, 0]) / s
            qy = 0.25 * s
            qz = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            qw = (R[1, 0] - R[0, 1]) / s
            qx = (R[0, 2] + R[2, 0]) / s
            qy = (R[1, 2] + R[2, 1]) / s
            qz = 0.25 * s
    quat = np.asarray([qx, qy, qz, qw], dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1e-8)


def head_pose_from_frame(head_frame) -> tuple[np.ndarray, np.ndarray] | None:
    head = getattr(head_frame, "head", None)
    pos = pose_xyz(head)
    quat = pose_quat_xyzw(head)
    if pos is None or quat is None:
        return None
    return pos, quat


def frame_in_head_reference(
    frame: QuestHandFrame,
    head_pos_Q: np.ndarray,
    head_anchor_quat_Q: np.ndarray,
) -> QuestHandFrame:
    """Express wrist pose in the head-anchor frame, with current head translation removed."""
    R_HQ = quat_xyzw_to_matrix(head_anchor_quat_Q).T
    p_wrist_H = R_HQ @ (np.asarray(frame.wrist_pos_world, dtype=np.float64) - np.asarray(head_pos_Q, dtype=np.float64))
    R_wrist_H = R_HQ @ quat_xyzw_to_matrix(frame.wrist_quat_world)
    return QuestHandFrame(
        hand_side=frame.hand_side,
        timestamp_ns=frame.timestamp_ns,
        frame_id=frame.frame_id,
        wrist_pos_world=p_wrist_H,
        wrist_quat_world=matrix_to_quat_xyzw(R_wrist_H),
        landmarks_wrist=frame.landmarks_wrist,
    )


def hand_matches(frame, hand: str) -> bool:
    if hand in {"any", "both"}:
        return True
    side = getattr(frame, "side", None)
    side_value = getattr(side, "value", side)
    return str(side_value).lower() == hand


def start_quest_receiver(args, event_queue):
    try:
        from hand_tracking_sdk import HandFrame, HeadFrame, HTSClient, HTSClientConfig, StreamOutput, TransportMode
    except ImportError as exc:
        raise SystemExit("Quest TCP streaming requires: pip install hand-tracking-sdk") from exc

    def run():
        client = HTSClient(
            HTSClientConfig(
                transport_mode=TransportMode.TCP_SERVER,
                host=args.host,
                port=args.port,
                output=StreamOutput.FRAMES,
            )
        )
        for frame in client.iter_events():
            now = time.time()
            if isinstance(frame, HeadFrame):
                head_pose = head_pose_from_frame(frame)
                if head_pose is not None:
                    event_queue.put((now, "head", getattr(frame, "sequence_id", None), head_pose))
                continue
            if isinstance(frame, HandFrame) and hand_matches(frame, args.hand):
                try:
                    quest_frame = quest_hand_frame_from_sdk(frame)
                except ValueError as exc:
                    print(f"Skipping invalid Quest SDK frame: {exc}")
                    continue
                event_queue.put((now, "hand", getattr(frame, "sequence_id", None), quest_frame))

    thread = threading.Thread(target=run, name="quest-nullspace-head-reference-receiver", daemon=True)
    thread.start()
    return thread


def drain_latest_events(event_queue):
    latest_head = None
    latest_hand = None
    count = 0
    while True:
        try:
            item = event_queue.get_nowait()
            count += 1
        except queue.Empty:
            return latest_head, latest_hand, count
        if item[1] == "head":
            latest_head = item
        elif item[1] == "hand":
            latest_hand = item


def _damped_right_pinv(J: np.ndarray, damping: float) -> np.ndarray:
    J = np.asarray(J, dtype=np.float64)
    JJt = J @ J.T
    return J.T @ np.linalg.solve(JJt + (float(damping) ** 2) * np.eye(JJt.shape[0]), np.eye(JJt.shape[0]))


def _scale_secondary_to_joint_speed_budget(primary: np.ndarray, secondary: np.ndarray, max_joint_speed: float) -> np.ndarray:
    primary = np.asarray(primary, dtype=np.float64)
    secondary = np.asarray(secondary, dtype=np.float64)
    max_joint_speed = abs(float(max_joint_speed))
    if max_joint_speed <= 0.0:
        return np.zeros_like(secondary)

    alpha = 1.0
    for qdot_primary, qdot_secondary in zip(primary, secondary):
        if abs(qdot_secondary) < 1e-12:
            continue
        if qdot_secondary > 0.0:
            limit = (max_joint_speed - qdot_primary) / qdot_secondary
        else:
            limit = (-max_joint_speed - qdot_primary) / qdot_secondary
        alpha = min(alpha, max(0.0, float(limit)))
    return float(np.clip(alpha, 0.0, 1.0)) * secondary


def solve_nullspace_task_space_ik(
    ik: DampedLeastSquaresIK,
    data,
    xdot_cmd: np.ndarray,
    dt: float,
    orientation_gain: float,
    control_orientation: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve position-priority resolved-rate IK with nullspace orientation.

    The 3D position task is solved first. Orientation is then projected into the
    position nullspace, so it can only use joint motion that does not change the
    end-effector position to first order.
    """
    xdot_cmd = np.asarray(xdot_cmd, dtype=np.float64)
    J_pos = ik.jacobian(data, control_orientation=False)
    J_pos_pinv = _damped_right_pinv(J_pos, ik.damping)

    if control_orientation:
        if xdot_cmd.shape != (6,):
            raise ValueError(f"Expected 6D task velocity, got {xdot_cmd.shape}")
        linear_cmd = xdot_cmd[:3]
        angular_cmd = xdot_cmd[3:]
    else:
        if xdot_cmd.shape != (3,):
            raise ValueError(f"Expected 3D task velocity, got {xdot_cmd.shape}")
        linear_cmd = xdot_cmd
        angular_cmd = None

    qdot_pos = J_pos_pinv @ linear_cmd
    qdot_pos = np.clip(qdot_pos, -ik.max_joint_speed, ik.max_joint_speed)

    qdot_null = np.zeros_like(qdot_pos)
    if control_orientation and angular_cmd is not None and abs(float(orientation_gain)) > 0.0:
        J_full = ik.jacobian(data, control_orientation=True)
        J_rot = J_full[3:, :]
        null_projector = np.eye(J_pos.shape[1], dtype=np.float64) - np.linalg.pinv(J_pos, rcond=1e-4) @ J_pos
        angular_residual = angular_cmd - J_rot @ qdot_pos
        J_rot_null = J_rot @ null_projector
        JJN = J_rot_null @ J_rot_null.T
        secondary = null_projector @ (
            J_rot_null.T
            @ np.linalg.solve(JJN + (ik.damping**2) * np.eye(JJN.shape[0]), angular_residual)
        )
        qdot_null = float(orientation_gain) * secondary
        qdot_null = _scale_secondary_to_joint_speed_budget(qdot_pos, qdot_null, ik.max_joint_speed)

    qdot = qdot_pos + qdot_null

    q_current = joint_qpos(ik.model, data, ik.joint_ids)
    q_target = q_current + qdot * float(dt)
    lo, hi = joint_ranges(ik.model, ik.joint_ids)
    q_target = np.clip(q_target, lo, hi)
    if ik.smoothing_alpha > 0.0 and ik.prev_qtarget is not None:
        q_target = ik.smoothing_alpha * ik.prev_qtarget + (1.0 - ik.smoothing_alpha) * q_target
    ik.prev_qtarget = q_target.copy()
    return q_target, qdot, qdot_pos, qdot_null


def solve_full_task_space_ik(
    ik: DampedLeastSquaresIK,
    data,
    xdot_cmd: np.ndarray,
    dt: float,
    control_orientation: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Solve resolved-rate IK with position and orientation in one task.

    This is the better default for 6-DoF arms such as Piper. SO101 keeps using
    the nullspace mode because it has only 5 arm joints and cannot independently
    satisfy an arbitrary 6D end-effector pose.
    """
    xdot_cmd = np.asarray(xdot_cmd, dtype=np.float64)
    if control_orientation:
        if xdot_cmd.shape != (6,):
            raise ValueError(f"Expected 6D task velocity, got {xdot_cmd.shape}")
        J = ik.jacobian(data, control_orientation=True)
    else:
        if xdot_cmd.shape != (3,):
            raise ValueError(f"Expected 3D task velocity, got {xdot_cmd.shape}")
        J = ik.jacobian(data, control_orientation=False)

    qdot = _damped_right_pinv(J, ik.damping) @ xdot_cmd
    qdot = np.clip(qdot, -ik.max_joint_speed, ik.max_joint_speed)

    q_current = joint_qpos(ik.model, data, ik.joint_ids)
    q_target = q_current + qdot * float(dt)
    lo, hi = joint_ranges(ik.model, ik.joint_ids)
    q_target = np.clip(q_target, lo, hi)
    if ik.smoothing_alpha > 0.0 and ik.prev_qtarget is not None:
        q_target = ik.smoothing_alpha * ik.prev_qtarget + (1.0 - ik.smoothing_alpha) * q_target
    ik.prev_qtarget = q_target.copy()
    return q_target, qdot, qdot.copy(), np.zeros_like(qdot)


def make_key_callback(arm_channel, latest_frame_ref, latest_head_ref, ik, data, state, args, R_BQ):
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
            head_pose = latest_head_ref["pose"]
            if args.reference_frame == "head":
                if head_pose is None:
                    print("Re-zero requested, but no Quest head frame has arrived yet.")
                    return
                head_pos_Q, head_quat_Q = head_pose
                state["head_anchor_quat_Q"] = head_quat_Q.copy()
                frame = frame_in_head_reference(frame, head_pos_Q, state["head_anchor_quat_Q"])
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
            state["raw_wrist_zero_Q"] = np.asarray(latest_frame_ref["frame"].wrist_pos_world, dtype=np.float64).copy()
            print(f"Re-zeroed IK in {args.reference_frame} frame: current Quest wrist maps to current robot end-effector pose.")
        elif key == "p":
            state["paused"] = not state["paused"]
            print(f"paused={state['paused']}")

    return on_key


def parse_args():
    parser = argparse.ArgumentParser(description=DEFAULT_DESCRIPTION)
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hand", choices=["right", "left", "any"], default="right")
    parser.add_argument("--ee-site", default=DEFAULT_EE_SITE)
    parser.add_argument("--ee-body", default=None)
    parser.add_argument("--arm-joint-names", default=DEFAULT_ARM_JOINTS)
    parser.add_argument("--arm-joint-prefix", default=None)
    parser.add_argument("--scale", type=float, default=DEFAULT_SCALE)
    parser.add_argument("--R_BQ", type=parse_matrix, default=None)
    parser.add_argument("--workspace-min", nargs=3, default=DEFAULT_WORKSPACE_MIN)
    parser.add_argument("--workspace-max", nargs=3, default=DEFAULT_WORKSPACE_MAX)
    parser.add_argument("--deadzone", type=float, default=0.005)
    parser.add_argument("--target-smoothing-alpha", type=float, default=0.10)
    parser.add_argument("--kp-pos", type=float, default=DEFAULT_KP_POS)
    parser.add_argument("--kp-rot", type=float, default=DEFAULT_KP_ROT)
    parser.add_argument("--max-linear-speed", type=float, default=DEFAULT_MAX_LINEAR_SPEED)
    parser.add_argument("--max-angular-speed", type=float, default=DEFAULT_MAX_ANGULAR_SPEED)
    parser.add_argument("--ik-damping", type=float, default=0.05)
    parser.add_argument("--max-joint-speed", type=float, default=DEFAULT_MAX_JOINT_SPEED)
    parser.add_argument("--joint-target-smoothing-alpha", type=float, default=0.0)
    parser.add_argument(
        "--ik-mode",
        choices=["position_nullspace", "full_pose"],
        default=DEFAULT_IK_MODE,
        help="position_nullspace keeps position primary and uses remaining motion for orientation; full_pose solves the selected task rows together.",
    )
    parser.add_argument("--control-orientation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--orientation-source", choices=["palm_landmarks", "wrist_pose"], default="palm_landmarks")
    parser.add_argument("--orientation-tracking", choices=["absolute", "relative"], default="relative")
    parser.add_argument("--position-weight", type=float, default=1.0)
    parser.add_argument("--orientation-weight", type=float, default=1.0)
    parser.add_argument("--reference-frame", choices=["head", "quest"], default="head")
    parser.add_argument("--hand-smoothing-alpha", type=float, default=0.25)
    parser.add_argument("--disable-hand-retargeting", action="store_true")
    parser.add_argument("--timeout", type=float, default=0.30)
    parser.add_argument("--debug-interval", type=float, default=0.25)
    parser.add_argument(
        "--disable-gravity",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable MuJoCo gravity at runtime. By default, keep the gravity defined in the XML model.",
    )
    parser.add_argument(
        "--robot-gravity-compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Compensate gravity for the robot body subtree only. Objects such as pipettes remain affected by gravity.",
    )
    parser.add_argument("--robot-gravity-root", default=DEFAULT_ROBOT_GRAVITY_ROOT, help="Root body of the robot subtree for gravity compensation.")
    parser.add_argument(
        "--viewer-camera-init-seconds",
        type=float,
        default=1.0,
        help="Seconds to keep applying the MJCF viewer camera after launch, avoiding passive viewer startup overrides.",
    )
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
    gravity_compensated_bodies = []
    if args.robot_gravity_compensation:
        gravity_compensated_bodies = apply_robot_gravity_compensation(model, args.robot_gravity_root)
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
    print(f"  3. IK mode: {args.ik_mode}.")
    print("  4. Hand Channel: wrist-relative landmarks -> Aero Hand retargeting.")
    print("  5. Press R to re-zero, P to pause/resume arm.")
    print(f"model={model_path}")
    print(f"ee_site={args.ee_site} arm_joints={ik.joint_names}")
    print(f"ee_start_B={np.array2string(ee_pos_B, precision=5)}")
    print(f"workspace_min={np.array2string(workspace_min, precision=3)} workspace_max={np.array2string(workspace_max, precision=3)}")
    if np.any(ee_pos_B < workspace_min) or np.any(ee_pos_B > workspace_max):
        print("WARNING: ee_start_B is outside the workspace; teleop targets will be clipped until the workspace is adjusted.")
    print(f"scale={args.scale} R_BQ=\n{np.array2string(R_BQ, precision=4, suppress_small=True)}")
    print(
        f"viewer_camera lookat={np.array2string(model.stat.center, precision=4, suppress_small=True)} "
        f"distance={1.5 * float(model.stat.extent):.4f} "
        f"azimuth={float(model.vis.global_.azimuth):.2f} elevation={float(model.vis.global_.elevation):.2f} "
        f"init_seconds={args.viewer_camera_init_seconds:.2f}"
    )
    print(f"gravity={np.array2string(model.opt.gravity, precision=4, suppress_small=True)} disable_gravity={args.disable_gravity}")
    print(
        f"robot_gravity_compensation={args.robot_gravity_compensation} "
        f"root={args.robot_gravity_root!r} compensated_bodies={len(gravity_compensated_bodies)}"
    )
    print(
        f"control_orientation={args.control_orientation} orientation_source={args.orientation_source} "
        f"orientation_tracking={args.orientation_tracking} "
        f"position_weight={args.position_weight} orientation_weight={args.orientation_weight} "
        f"reference_frame={args.reference_frame}"
    )
    print(f"disable_hand_retargeting={args.disable_hand_retargeting}")
    print_combined_actuator_info(model, arm_actuator_names=ik.joint_names)

    if args.dry_run:
        _raw_hand, filtered_hand = hand_retargeter(np.zeros((21, 3), dtype=np.float32))
        print(f"dry_run=true q_arm={np.array2string(joint_qpos(model, data, ik.joint_ids), precision=6)}")
        print(f"dry_run_hand={np.array2string(filtered_hand, precision=4, suppress_small=True)}")
        return

    frame_queue = queue.Queue()
    start_quest_receiver(args, frame_queue)
    print(f"Waiting for Quest TCP connection on {args.host}:{args.port}...")

    latest_frame_ref = {"frame": None}
    latest_head_ref = {"pose": None}
    state = {
        "paused": False,
        "target_pos_B": ee_pos_B.copy(),
        "target_R_B": ee_R_B.copy(),
        "last_qtarget": joint_qpos(model, data, ik.joint_ids),
        "head_anchor_quat_Q": None,
        "raw_wrist_zero_Q": None,
    }
    last_frame_time = 0.0
    last_debug_time = 0.0
    last_control_time = time.time()
    last_frame_id = None
    last_delta_p_Q = np.zeros(3, dtype=np.float64)
    last_delta_p_ref = np.zeros(3, dtype=np.float64)
    last_velocity = np.zeros(3, dtype=np.float64)
    last_control_dt = float(model.opt.timestep)
    sim_step_dt = float(model.opt.timestep)
    sim_accumulator = 0.0
    last_sim_steps = 0
    last_error = np.zeros(3, dtype=np.float64)
    last_rot_error = np.zeros(3, dtype=np.float64)
    last_qdot = np.zeros(len(ik.joint_ids), dtype=np.float64)
    last_qdot_pos = np.zeros(len(ik.joint_ids), dtype=np.float64)
    last_qdot_null = np.zeros(len(ik.joint_ids), dtype=np.float64)
    filtered_hand = SAFE_OPEN_HAND.copy()

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=make_key_callback(arm_channel, latest_frame_ref, latest_head_ref, ik, data, state, args, R_BQ),
    ) as viewer:
        viewer_camera_init_until = time.time() + max(0.0, float(args.viewer_camera_init_seconds))
        initialize_viewer_camera(viewer, model)
        viewer.sync()
        while viewer.is_running():
            latest_head, latest_hand, drained = drain_latest_events(frame_queue)
            got_new_frame = latest_hand is not None
            if latest_head is not None:
                _head_time, _kind, _head_sequence_id, head_pose = latest_head
                latest_head_ref["pose"] = head_pose
            if latest_hand is not None:
                frame_time, _kind, _hand_sequence_id, quest_frame_raw = latest_hand
                latest_frame_ref["frame"] = quest_frame_raw
                last_frame_time = frame_time
                last_frame_id = quest_frame_raw.frame_id
                quest_frame = quest_frame_raw
                if args.reference_frame == "head":
                    head_pose = latest_head_ref["pose"]
                    if head_pose is None:
                        quest_frame = None
                    else:
                        head_pos_Q, head_quat_Q = head_pose
                        if state["head_anchor_quat_Q"] is None:
                            state["head_anchor_quat_Q"] = head_quat_Q.copy()
                        quest_frame = frame_in_head_reference(quest_frame_raw, head_pos_Q, state["head_anchor_quat_Q"])
                if quest_frame is not None:
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
                        state["raw_wrist_zero_Q"] = np.asarray(quest_frame_raw.wrist_pos_world, dtype=np.float64).copy()
                        print(f"Teleop zero set in {args.reference_frame} frame: current Quest wrist maps to current robot end-effector pose.")

                    target = arm_channel.compute_target(quest_frame)
                    state["target_pos_B"] = np.clip(target.target_pos_B, workspace_min, workspace_max)
                    if args.control_orientation and args.orientation_tracking == "absolute":
                        state["target_R_B"] = absolute_orientation_target_B(quest_frame, R_BQ, args.orientation_source)
                    elif target.target_R_B is not None:
                        state["target_R_B"] = target.target_R_B.copy()
                    last_delta_p_ref = target.delta_p_Q
                    if state["raw_wrist_zero_Q"] is not None:
                        last_delta_p_Q = np.asarray(quest_frame_raw.wrist_pos_world, dtype=np.float64) - state["raw_wrist_zero_Q"]

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
                if args.ik_mode == "full_pose":
                    weighted_xdot = cmd.xdot.copy()
                    weighted_xdot[:3] *= float(args.position_weight)
                    if args.control_orientation:
                        weighted_xdot[3:] *= float(args.orientation_weight)
                    qtarget, last_qdot, last_qdot_pos, last_qdot_null = solve_full_task_space_ik(
                        ik,
                        data,
                        weighted_xdot,
                        dt=last_control_dt,
                        control_orientation=args.control_orientation,
                    )
                else:
                    weighted_xdot = cmd.xdot.copy()
                    weighted_xdot[:3] *= float(args.position_weight)
                    qtarget, last_qdot, last_qdot_pos, last_qdot_null = solve_nullspace_task_space_ik(
                        ik,
                        data,
                        weighted_xdot,
                        dt=last_control_dt,
                        orientation_gain=args.orientation_weight,
                        control_orientation=args.control_orientation,
                    )
                state["last_qtarget"] = qtarget.copy()
                ik.apply_position_targets(data, qtarget)
            else:
                last_velocity = np.zeros(6 if args.control_orientation else 3, dtype=np.float64)
                last_error = state["target_pos_B"] - ee_pos_B
                last_rot_error = np.zeros(3, dtype=np.float64)
                last_qdot = np.zeros(len(ik.joint_ids), dtype=np.float64)
                last_qdot_pos = np.zeros(len(ik.joint_ids), dtype=np.float64)
                last_qdot_null = np.zeros(len(ik.joint_ids), dtype=np.float64)
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
            if time.time() <= viewer_camera_init_until:
                initialize_viewer_camera(viewer, model)
            viewer.sync()

            if now - last_debug_time >= args.debug_interval:
                print(
                    "debug "
                    f"frame_id={last_frame_id} new={got_new_frame} drained={drained} stale={stale} paused={state['paused']} "
                    f"delta_p_Q={np.array2string(last_delta_p_Q, precision=4, suppress_small=True)} "
                    f"delta_p_ref={np.array2string(last_delta_p_ref, precision=4, suppress_small=True)} "
                    f"head_ref={latest_head_ref['pose'] is not None} "
                    f"target_B={np.array2string(state['target_pos_B'], precision=4, suppress_small=True)} "
                    f"ee_B={np.array2string(ee_pos_B, precision=4, suppress_small=True)} "
                    f"err={np.array2string(last_error, precision=4, suppress_small=True)} "
                    f"rot_err={np.array2string(last_rot_error, precision=4, suppress_small=True)} "
                    f"xdot={np.array2string(last_velocity, precision=4, suppress_small=True)} "
                    f"qdot={np.array2string(last_qdot, precision=4, suppress_small=True)} "
                    f"qdot_pos={np.array2string(last_qdot_pos, precision=4, suppress_small=True)} "
                    f"qdot_null={np.array2string(last_qdot_null, precision=4, suppress_small=True)} "
                    f"dt={last_control_dt:.4f} "
                    f"sim_steps={last_sim_steps} "
                    f"hand={np.array2string(filtered_hand, precision=3, suppress_small=True)}"
                )
                last_debug_time = now

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
