import argparse
import queue
import sys
import threading
import time
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
    import numpy as np
    from hand_tracking_sdk import (
        HandFrame,
        HTSClient,
        HTSClientConfig,
        StreamOutput,
        TransportMode,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing runtime dependency. Install required packages with: "
        "pip install numpy mujoco hand-tracking-sdk"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.arm_teleop import (
    DampedLeastSquaresIK,
    RelativePoseMapper,
    TeleopStateMachine,
    VelocityTeleopConfig,
    VelocityTeleopController,
    WorkspaceLimiter,
    index_thumb_alignment,
    joint_qpos,
)
from aero_quest.retargeting import estimate_palm_pose, get_quest_points_21
from aero_quest.so101_aero_control import ctrl_midpoints


DEFAULT_MODEL = PROJECT_ROOT / "mujoco_menagerie/so101_aero_hand/scene.xml"
DEFAULT_ARM_JOINTS = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll"


def str_to_bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in ("1", "true", "yes", "y", "on"):
        return True
    if value in ("0", "false", "no", "n", "off"):
        return False
    raise argparse.ArgumentTypeError(f"Expected true/false, got {value!r}")


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


def is_hand_like(name: str) -> bool:
    text = name.lower()
    return any(k in text for k in ("thumb", "index", "middle", "ring", "pinky", "little", "aero", "right_", "tendon"))


def pose_xyz(pose):
    if pose is None:
        return None
    if all(hasattr(pose, attr) for attr in ("x", "y", "z")):
        return np.asarray([float(pose.x), float(pose.y), float(pose.z)], dtype=np.float64)
    return None


def pose_quat_xyzw(pose):
    if pose is None:
        return None
    if all(hasattr(pose, attr) for attr in ("qx", "qy", "qz", "qw")):
        quat = np.asarray([float(pose.qx), float(pose.qy), float(pose.qz), float(pose.qw)], dtype=np.float64)
        norm = float(np.linalg.norm(quat))
        if norm > 1e-8:
            return quat / norm
    return None


def quat_xyzw_to_matrix(quat):
    x, y, z, w = np.asarray(quat, dtype=np.float64)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def select_hand_rotation(frame, palm_rotation, orientation_source):
    if orientation_source == "wrist_pose":
        wrist_quat = pose_quat_xyzw(getattr(frame, "wrist", None))
        if wrist_quat is not None:
            return quat_xyzw_to_matrix(wrist_quat)
        print("Warning: frame.wrist quaternion missing; falling back to palm_from_landmarks.")
        return palm_rotation
    return palm_rotation


def select_hand_position(frame, landmarks, palm_position, pose_source):
    if pose_source == "wrist_pose":
        wrist_pose = pose_xyz(getattr(frame, "wrist", None))
        if wrist_pose is not None:
            return wrist_pose
        print("Warning: frame.wrist pose missing; falling back to palm_from_landmarks.")
        return palm_position
    if pose_source == "wrist_landmark":
        return np.asarray(landmarks[0], dtype=np.float64)
    return palm_position


def hand_matches(frame, hand):
    if hand == "any":
        return True
    side = getattr(frame, "side", None)
    side_value = getattr(side, "value", side)
    return str(side_value).lower() == hand


def start_quest_receiver(args, frame_queue):
    def run():
        client = HTSClient(
            HTSClientConfig(
                transport_mode=TransportMode.TCP_SERVER,
                host=args.tcp_host,
                port=args.tcp_port,
                output=StreamOutput.FRAMES,
            )
        )
        for frame in client.iter_events():
            if isinstance(frame, HandFrame) and hand_matches(frame, args.hand):
                frame_queue.put((time.time(), getattr(frame, "sequence_id", None), frame))

    thread = threading.Thread(target=run, name="quest-arm-ik-receiver", daemon=True)
    thread.start()
    return thread


def drain_latest_frame(frame_queue):
    latest = None
    count = 0
    while True:
        try:
            latest = frame_queue.get_nowait()
            count += 1
        except queue.Empty:
            return latest, count


def make_key_callback(state, mapper, pending_calibration):
    def on_key(keycode):
        try:
            key = chr(keycode).lower()
        except (TypeError, ValueError):
            return
        if key == "c":
            pending_calibration["requested"] = True
            print("Calibration requested.")
        elif key == "p":
            state.toggle_pause()
            print(f"Paused={state.paused}")
        elif key == "r":
            mapper.reset()
            state.set_calibrated(False)
            pending_calibration["requested"] = True
            print("Calibration reset; waiting for next valid hand frame.")
        elif key == "e":
            state.emergency_stop()
            print("Emergency stop set. Restart script to clear it.")

    return on_key


def maybe_print_debug(now, last_debug_time, args, state, frame_age, seq, drained, hand_pos, target_pos, ee_pos, velocity_cmd, ik_qdot):
    if not args.debug or now - last_debug_time < args.debug_interval:
        return last_debug_time
    print(
        "debug "
        f"mode={state.mode} calibrated={state.calibrated} paused={state.paused} "
        f"frame_age={frame_age:.3f}s seq={seq} drained={drained} "
        f"hand={np.array2string(hand_pos, precision=3, suppress_small=True) if hand_pos is not None else None} "
        f"target={np.array2string(target_pos, precision=3, suppress_small=True)} "
        f"ee={np.array2string(ee_pos, precision=3, suppress_small=True)} "
        f"xdot={np.array2string(velocity_cmd, precision=3, suppress_small=True)} "
        f"qdot={np.array2string(ik_qdot, precision=3, suppress_small=True)}"
    )
    return now


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Quest palm/wrist relative-position velocity IK teleop for arm only.")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--tcp_host", "--host", dest="tcp_host", default="0.0.0.0")
    parser.add_argument("--tcp_port", "--port", dest="tcp_port", type=int, default=8000)
    parser.add_argument("--hand", choices=["right", "left", "any"], default="right")
    parser.add_argument("--pose_source", choices=["wrist_pose", "palm_landmarks", "wrist_landmark"], default="wrist_pose")
    parser.add_argument("--ee_site", default="grasp_site")
    parser.add_argument("--ee_body", default=None)
    parser.add_argument("--arm_joint_names", default=DEFAULT_ARM_JOINTS)
    parser.add_argument("--arm_joint_prefix", default=None)
    parser.add_argument("--scale", type=float, default=0.5)
    parser.add_argument("--direction_alignment", choices=["index_thumb", "identity"], default="index_thumb")
    parser.add_argument("--robot_forward_axis", choices=["+x", "-x", "+y", "-y", "+z", "-z"], default="+x")
    parser.add_argument("--robot_up_axis", choices=["+x", "-x", "+y", "-y", "+z", "-z"], default="+z")
    parser.add_argument("--control_orientation", type=str_to_bool, default=False)
    parser.add_argument("--orientation_source", choices=["wrist_pose", "palm_landmarks"], default="wrist_pose")
    parser.add_argument("--kp_pos", type=float, default=5.0)
    parser.add_argument("--kp_rot", type=float, default=2.0)
    parser.add_argument("--max_linear_speed", type=float, default=0.05)
    parser.add_argument("--max_angular_speed", type=float, default=0.5)
    parser.add_argument("--ik_damping", type=float, default=0.05)
    parser.add_argument("--max_joint_speed", type=float, default=1.0)
    parser.add_argument("--target_smoothing_alpha", type=float, default=0.0)
    parser.add_argument("--joint_target_smoothing_alpha", type=float, default=0.0)
    parser.add_argument("--workspace_min", nargs=3, type=float, default=[-0.5, -0.5, 0.02])
    parser.add_argument("--workspace_max", nargs=3, type=float, default=[0.5, 0.5, 0.7])
    parser.add_argument("--timeout", type=float, default=0.3)
    parser.add_argument("--auto_calibrate", type=str_to_bool, default=True)
    parser.add_argument("--teleop_enabled", type=str_to_bool, default=True)
    parser.add_argument("--disable_gravity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--debug_interval", type=float, default=0.5)
    parser.add_argument("--dry_run", action="store_true", help="Load model and initialize controllers, then exit before Quest/viewer.")
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model(args.model)
    workspace_min = parse_vec3(args.workspace_min, "--workspace_min")
    workspace_max = parse_vec3(args.workspace_max, "--workspace_max")

    print("Before running:")
    print(f"  1. Confirm: adb reverse tcp:{args.tcp_port} tcp:{args.tcp_port}")
    print(f"  2. Quest HTS: TCP, localhost, port {args.tcp_port}")
    print("  3. Keys: C calibrate, P pause, R reset calibration, E emergency stop.")
    print("  4. This script controls only SO101 arm actuators. Aero Hand actuators stay at midpoint.")
    print(f"Loading model: {model_path}")

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
    workspace = WorkspaceLimiter(minimum=workspace_min, maximum=workspace_max)
    mapper = RelativePoseMapper(
        scale=args.scale,
        workspace=workspace,
        smoothing_alpha=args.target_smoothing_alpha,
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
    state = TeleopStateMachine(enabled=args.teleop_enabled)
    pending_calibration = {"requested": bool(args.auto_calibrate)}

    print(f"gravity={np.array2string(model.opt.gravity, precision=6)}")
    print(f"ee_kind={ik.ee_kind} ee_name={args.ee_site or args.ee_body}")
    print(f"arm_joints={ik.joint_names}")
    print(
        f"pose_source={args.pose_source} scale={args.scale} "
        f"direction_alignment={args.direction_alignment} "
        f"robot_forward_axis={args.robot_forward_axis} robot_up_axis={args.robot_up_axis} "
        f"control_orientation={args.control_orientation} orientation_source={args.orientation_source}"
    )
    print(f"ctrl_shape={data.ctrl.shape} ctrl_home={np.array2string(ctrl_midpoints(model), precision=6, suppress_small=True)}")
    if args.dry_run:
        ee_position, _ = ik.ee_pose(data)
        print(f"dry_run=true ee_position={np.array2string(ee_position, precision=6)}")
        return

    frame_queue = queue.Queue()
    start_quest_receiver(args, frame_queue)
    print(f"Waiting for Quest TCP connection on {args.tcp_host}:{args.tcp_port}...")

    ctrl_home = ctrl_midpoints(model)
    last_frame_time = 0.0
    last_sequence_id = None
    have_hand = False
    latest_points = None
    hand_position = None
    hand_rotation = np.eye(3, dtype=np.float64)
    target_position = ik.ee_pose(data)[0]
    velocity_cmd = np.zeros(3 if not args.control_orientation else 6, dtype=np.float64)
    ik_qdot = np.zeros(len(ik.joint_ids), dtype=np.float64)
    arm_qtarget = joint_qpos(model, data, ik.joint_ids)
    last_debug_time = 0.0

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=make_key_callback(state, mapper, pending_calibration),
    ) as viewer:
        while viewer.is_running():
            latest, drained = drain_latest_frame(frame_queue)
            if latest is not None:
                frame_time, last_sequence_id, frame = latest
                try:
                    points = get_quest_points_21(frame)
                    latest_points = points
                    palm_position, palm_rotation = estimate_palm_pose(points)
                    hand_rotation = select_hand_rotation(frame, palm_rotation, args.orientation_source)
                    hand_position = select_hand_position(frame, points, palm_position, args.pose_source)
                    last_frame_time = frame_time
                    have_hand = True
                except ValueError as exc:
                    print(f"Skipping invalid Quest frame: {exc}")

            now = time.time()
            frame_age = float("inf") if not have_hand else now - last_frame_time
            timed_out = (not have_hand) or (frame_age > args.timeout)

            ee_position, ee_rotation = ik.ee_pose(data)
            if pending_calibration["requested"] and have_hand and not timed_out:
                if args.direction_alignment == "index_thumb":
                    try:
                        mapper.R_align = index_thumb_alignment(
                            latest_points,
                            robot_forward_axis=args.robot_forward_axis,
                            robot_up_axis=args.robot_up_axis,
                        )
                        print("Direction alignment set from initial index/thumb frame:")
                        print(np.array2string(mapper.R_align, precision=4, suppress_small=True))
                    except ValueError as exc:
                        print(f"Direction alignment failed ({exc}); using identity alignment.")
                        mapper.R_align = np.eye(3, dtype=np.float64)
                else:
                    mapper.R_align = np.eye(3, dtype=np.float64)
                mapper.calibrate(hand_position, hand_rotation, ee_position, ee_rotation)
                state.set_calibrated(True)
                pending_calibration["requested"] = False
                print(
                    "Calibrated: "
                    f"hand0={np.array2string(hand_position, precision=4)} "
                    f"ee0={np.array2string(ee_position, precision=4)}"
                )

            data.ctrl[:] = ctrl_home
            if state.can_control() and have_hand and not timed_out:
                target_position, target_rotation = mapper.target_pose(
                    hand_position,
                    hand_rotation,
                    control_orientation=args.control_orientation,
                )
                cmd = vel_controller.compute(target_position, target_rotation, ee_position, ee_rotation)
                velocity_cmd = cmd.xdot
                arm_qtarget, ik_qdot = ik.solve(
                    data,
                    velocity_cmd,
                    dt=float(model.opt.timestep),
                    control_orientation=args.control_orientation,
                )
                ik.apply_position_targets(data, arm_qtarget)
            else:
                target_position = ee_position.copy()
                velocity_cmd = np.zeros(3 if not args.control_orientation else 6, dtype=np.float64)
                ik_qdot = np.zeros(len(ik.joint_ids), dtype=np.float64)
                arm_qtarget = joint_qpos(model, data, ik.joint_ids)

            mujoco.mj_step(model, data)
            viewer.sync()

            last_debug_time = maybe_print_debug(
                now,
                last_debug_time,
                args,
                state,
                frame_age,
                last_sequence_id,
                drained,
                hand_position,
                target_position,
                ee_position,
                velocity_cmd,
                ik_qdot,
            )
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
