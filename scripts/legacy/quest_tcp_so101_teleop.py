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
    import yaml
    from hand_tracking_sdk import (
        HandFrame,
        HeadFrame,
        HTSClient,
        HTSClientConfig,
        StreamOutput,
        TransportMode,
    )
except ImportError as exc:
    raise SystemExit(
        "Missing runtime dependency. Install required packages with: "
        "pip install numpy mujoco pyyaml hand-tracking-sdk"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.envs.so101_mujoco_env import SO101MujocoEnv
from aero_quest.retargeting import get_quest_points_21, quest_points_to_action_7d


DEFAULT_CONFIG = PROJECT_ROOT / "configs/env/so101_mujoco.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Quest TCP wrist/hand teleop for SO101 MuJoCo.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hand", choices=["right", "left", "any"], default="right")
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--timeout", type=float, default=0.3)
    parser.add_argument("--debug-interval", type=float, default=0.5)
    parser.add_argument("--use-head-reference", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--x-gain", type=float, default=4.0, help="Quest lateral wrist delta to shoulder_pan.")
    parser.add_argument("--y-gain", type=float, default=3.0, help="Quest forward wrist delta to elbow_flex.")
    parser.add_argument("--z-gain", type=float, default=4.0, help="Quest vertical wrist delta to shoulder_lift.")
    parser.add_argument("--wrist-gain", type=float, default=0.8, help="Palm orientation proxy to wrist_flex/roll.")
    parser.add_argument("--gripper-open", type=float, default=1.0, help="Normalized SO101 gripper action for open hand.")
    parser.add_argument("--gripper-closed", type=float, default=-1.0, help="Normalized SO101 gripper action for fist.")
    parser.add_argument("--action-limit", type=float, default=1.0)
    parser.add_argument("--arm-deadzone", type=float, default=0.015)
    parser.add_argument("--max-wrist-delta", type=float, default=0.25)
    parser.add_argument("--invert-x", action="store_true")
    parser.add_argument("--invert-y", action="store_true")
    parser.add_argument("--invert-z", action="store_true")
    parser.add_argument("--invert-wrist-flex", action="store_true")
    parser.add_argument("--invert-wrist-roll", action="store_true")
    parser.add_argument("--wrist-rotation-source", choices=["wrist-pose", "landmarks"], default="wrist-pose")
    parser.add_argument("--recalibrate-keyboard", action="store_true", help="Reserved; close/restart to recalibrate for now.")
    return parser.parse_args()


def resolve_path(path):
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def hand_matches(frame, hand):
    if hand == "any":
        return True
    side = getattr(frame, "side", None)
    side_value = getattr(side, "value", side)
    return str(side_value).lower() == hand


def pose_xyz(pose):
    if pose is None:
        return None
    if all(hasattr(pose, attr) for attr in ("x", "y", "z")):
        return np.asarray([float(pose.x), float(pose.y), float(pose.z)], dtype=np.float32)
    return None


def pose_quat_xyzw(pose):
    if pose is None:
        return None
    if all(hasattr(pose, attr) for attr in ("qx", "qy", "qz", "qw")):
        quat = np.asarray([float(pose.qx), float(pose.qy), float(pose.qz), float(pose.qw)], dtype=np.float32)
        norm = float(np.linalg.norm(quat))
        if norm > 1e-8:
            return quat / norm
    return None


def quat_conjugate_xyzw(q):
    return np.asarray([-q[0], -q[1], -q[2], q[3]], dtype=np.float32)


def quat_multiply_xyzw(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.asarray(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        dtype=np.float32,
    )


def quat_to_roll_pitch_yaw_xyzw(q):
    x, y, z, w = q
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return float(roll), float(pitch), float(yaw)


def start_quest_receiver(args, event_queue):
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
                event_queue.put((now, "head", getattr(frame, "sequence_id", None), frame))
            elif isinstance(frame, HandFrame) and hand_matches(frame, args.hand):
                event_queue.put((now, "hand", getattr(frame, "sequence_id", None), frame))

    thread = threading.Thread(target=run, name="quest-so101-tcp-receiver", daemon=True)
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


def hand_position(frame, points):
    wrist_pose = pose_xyz(getattr(frame, "wrist", None))
    if wrist_pose is not None:
        return wrist_pose
    return np.asarray(points[0], dtype=np.float32)


def palm_orientation_proxy(points):
    points = np.asarray(points, dtype=np.float32)
    index_side = points[5] - points[17]
    middle_dir = points[9] - points[0]
    index_norm = np.linalg.norm(index_side)
    middle_norm = np.linalg.norm(middle_dir)
    if index_norm < 1e-6 or middle_norm < 1e-6:
        return 0.0, 0.0
    index_side = index_side / index_norm
    middle_dir = middle_dir / middle_norm
    palm_normal = np.cross(index_side, middle_dir)
    normal_norm = np.linalg.norm(palm_normal)
    if normal_norm < 1e-6:
        return 0.0, 0.0
    palm_normal = palm_normal / normal_norm
    wrist_flex_proxy = float(np.clip(-middle_dir[2], -1.0, 1.0))
    wrist_roll_proxy = float(np.clip(palm_normal[0], -1.0, 1.0))
    return wrist_flex_proxy, wrist_roll_proxy


def apply_deadzone(value, deadzone):
    if abs(float(value)) < float(deadzone):
        return 0.0
    return float(value)


def signed_axis(value, invert=False):
    return -float(value) if invert else float(value)


def wrist_rotation_proxy(points, wrist_quat, neutral_wrist_quat, args):
    if args.wrist_rotation_source == "wrist-pose" and wrist_quat is not None and neutral_wrist_quat is not None:
        q_rel = quat_multiply_xyzw(wrist_quat, quat_conjugate_xyzw(neutral_wrist_quat))
        q_rel = q_rel / max(float(np.linalg.norm(q_rel)), 1e-8)
        roll, pitch, _yaw = quat_to_roll_pitch_yaw_xyzw(q_rel)
        return float(np.clip(pitch / 1.2, -1.0, 1.0)), float(np.clip(roll / 1.2, -1.0, 1.0))
    return palm_orientation_proxy(points)


def quest_hand_to_so101_action(points, hand_pos, head_pos, neutral_ref, wrist_quat, neutral_wrist_quat, args):
    ref_pos = hand_pos
    if args.use_head_reference and head_pos is not None:
        ref_pos = hand_pos - head_pos
    delta = ref_pos - neutral_ref
    delta = np.clip(delta, -float(args.max_wrist_delta), float(args.max_wrist_delta))
    delta = np.asarray([apply_deadzone(v, args.arm_deadzone) for v in delta], dtype=np.float32)

    formula_action = quest_points_to_action_7d(points)
    finger_curl = float(np.mean(formula_action[3:7]))
    gripper_action = (1.0 - finger_curl) * args.gripper_open + finger_curl * args.gripper_closed
    wrist_flex_proxy, wrist_roll_proxy = wrist_rotation_proxy(points, wrist_quat, neutral_wrist_quat, args)
    wrist_flex_proxy = signed_axis(wrist_flex_proxy, args.invert_wrist_flex)
    wrist_roll_proxy = signed_axis(wrist_roll_proxy, args.invert_wrist_roll)

    action = np.zeros(6, dtype=np.float32)
    action[0] = args.x_gain * signed_axis(delta[0], args.invert_x)
    action[1] = args.z_gain * signed_axis(delta[2], args.invert_z)
    action[2] = -args.y_gain * signed_axis(delta[1], args.invert_y)
    action[3] = args.wrist_gain * wrist_flex_proxy
    action[4] = args.wrist_gain * wrist_roll_proxy
    action[5] = gripper_action
    return np.clip(action, -args.action_limit, args.action_limit).astype(np.float32), formula_action, delta


def maybe_print_debug(now, last_debug_time, args, raw_action, filtered_action, wrist_delta, timed_out, have_hand, frame_age, seq, drained):
    if now - last_debug_time < args.debug_interval:
        return last_debug_time
    print(
        "debug "
        f"wrist_delta={np.array2string(wrist_delta, precision=3, suppress_small=True)} "
        f"raw_action={np.array2string(raw_action, precision=3, suppress_small=True)} "
        f"filtered_action={np.array2string(filtered_action, precision=3, suppress_small=True)} "
        f"timeout={timed_out} have_hand={have_hand} frame_age={frame_age:.3f}s seq={seq} drained={drained}"
    )
    return now


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.model_path is not None:
        config["model_path"] = str(resolve_path(args.model_path))
    else:
        config["model_path"] = str(resolve_path(config.get("model_path")))

    print("Before running:")
    print(f"  1. Confirm: adb reverse tcp:{args.port} tcp:{args.port}")
    print(f"  2. Quest HTS: TCP, localhost, port {args.port}")
    print("  3. Quest wrist/head/hand pose -> SO101 normalized action -> MuJoCo viewer")

    env = SO101MujocoEnv(
        model_path=config.get("model_path"),
        control_dt=config.get("control_dt", 0.02),
        physics_dt=config.get("physics_dt"),
        render_width=config.get("render_width", 640),
        render_height=config.get("render_height", 480),
        camera_names=config.get("camera_names") or [],
        action_scale=config.get("action_scale", 1.0),
        episode_len=10**9,
        init_qpos=config.get("init_qpos"),
        reward_type=config.get("reward_type", "zero"),
        ee_site_name=config.get("ee_site_name", "gripperframe"),
        print_model_info=True,
    )
    env.reset()

    event_queue = queue.Queue()
    start_quest_receiver(args, event_queue)
    print(f"Waiting for Quest TCP connection on {args.host}:{args.port}...")

    latest_head_pos = None
    neutral_ref = None
    neutral_wrist_quat = None
    raw_action = np.zeros(env.model.nu, dtype=np.float32)
    filtered_action = np.zeros(env.model.nu, dtype=np.float32)
    wrist_delta = np.zeros(3, dtype=np.float32)
    last_hand_time = 0.0
    last_sequence_id = None
    last_debug_time = 0.0
    have_hand = False

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        while viewer.is_running():
            latest_head, latest_hand, drained = drain_latest_events(event_queue)
            if latest_head is not None:
                _, _, _, head_frame = latest_head
                latest_head_pos = pose_xyz(getattr(head_frame, "head", None))
            if latest_hand is not None:
                frame_time, _, last_sequence_id, hand_frame = latest_hand
                try:
                    points = get_quest_points_21(hand_frame)
                    wrist_pose = getattr(hand_frame, "wrist", None)
                    hand_pos = hand_position(hand_frame, points)
                    wrist_quat = pose_quat_xyzw(wrist_pose)
                    ref_pos = hand_pos - latest_head_pos if args.use_head_reference and latest_head_pos is not None else hand_pos
                    if neutral_ref is None:
                        neutral_ref = ref_pos.copy()
                        neutral_wrist_quat = wrist_quat.copy() if wrist_quat is not None else None
                        print(f"Calibrated neutral Quest reference: {np.array2string(neutral_ref, precision=4)}")
                    raw_action, _formula_action, wrist_delta = quest_hand_to_so101_action(
                        points, hand_pos, latest_head_pos, neutral_ref, wrist_quat, neutral_wrist_quat, args
                    )
                    have_hand = True
                    last_hand_time = frame_time
                except ValueError as exc:
                    print(f"Skipping invalid Quest hand frame: {exc}")

            now = time.time()
            frame_age = float("inf") if not have_hand else now - last_hand_time
            timed_out = (not have_hand) or (frame_age > args.timeout)
            target_action = np.zeros(env.model.nu, dtype=np.float32) if timed_out else raw_action
            alpha = float(np.clip(args.alpha, 0.0, 1.0))
            filtered_action = ((1.0 - alpha) * filtered_action + alpha * target_action).astype(np.float32)

            env.step(filtered_action)
            viewer.sync()
            last_debug_time = maybe_print_debug(
                now,
                last_debug_time,
                args,
                raw_action,
                filtered_action,
                wrist_delta,
                timed_out,
                have_hand,
                frame_age,
                last_sequence_id,
                drained,
            )
            time.sleep(env.model.opt.timestep)

    env.close()


if __name__ == "__main__":
    main()
