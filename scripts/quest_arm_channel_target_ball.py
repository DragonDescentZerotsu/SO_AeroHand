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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.quest_hand_frame import RelativeWristArmController, quest_hand_frame_from_sdk


DEFAULT_MODEL = PROJECT_ROOT / "models/quest_arm_channel_target_ball/scene.xml"
DEFAULT_TARGET_BODY = "quest_arm_target"

# Default axis map for validating "up / forward / left":
# Quest/Unity Q: +X right, +Y up, +Z forward.
# MuJoCo debug B: +X forward, +Y left, +Z up.
# Therefore delta_B = R_BQ @ delta_Q:
#   hand up      (+Y_Q) -> +Z_B
#   hand forward (+Z_Q) -> +X_B
#   hand left    (-X_Q) -> +Y_B
DEFAULT_R_BQ = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_vec3(values, name):
    arr = np.asarray([float(v) for v in values], dtype=np.float64)
    if arr.shape != (3,):
        raise argparse.ArgumentTypeError(f"{name} expected 3 floats")
    return arr


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


def parse_args():
    parser = argparse.ArgumentParser(description="Validate Quest Arm Channel by moving a MuJoCo target sphere.")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--target-body", default=DEFAULT_TARGET_BODY)
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hand", choices=["right", "left", "any"], default="right")
    parser.add_argument("--origin-pos-B", nargs=3, default=["0.35", "0.0", "0.25"], help="Target sphere zero position in MuJoCo/debug B frame.")
    parser.add_argument("--scale", type=float, default=1.0, help="Meters of target motion per meter of Quest wrist motion.")
    parser.add_argument("--deadzone", type=float, default=0.005)
    parser.add_argument("--smoothing-alpha", type=float, default=0.15, help="0=no smoothing, larger=more smoothing.")
    parser.add_argument("--R_BQ", type=parse_matrix, default=None, help="Optional 3x3 row-major Quest-Q to MuJoCo-B axis map.")
    parser.add_argument("--debug-interval", type=float, default=0.25)
    parser.add_argument("--disable-gravity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true", help="Load model and print mapping, then exit before Quest/viewer.")
    return parser.parse_args()


def hand_matches(quest_frame, hand):
    return hand == "any" or quest_frame.hand_side.lower() == hand


def start_quest_receiver(args, frame_queue):
    try:
        from hand_tracking_sdk import HandFrame, HTSClient, HTSClientConfig, StreamOutput, TransportMode
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
            if not isinstance(frame, HandFrame):
                continue
            try:
                quest_frame = quest_hand_frame_from_sdk(frame)
            except ValueError as exc:
                print(f"Skipping invalid Quest SDK frame: {exc}")
                continue
            if hand_matches(quest_frame, args.hand):
                frame_queue.put((time.time(), quest_frame))

    thread = threading.Thread(target=run, name="quest-arm-channel-target-receiver", daemon=True)
    thread.start()
    return thread


def drain_latest(frame_queue):
    latest = None
    count = 0
    while True:
        try:
            latest = frame_queue.get_nowait()
            count += 1
        except queue.Empty:
            return latest, count


def make_key_callback(controller, latest_frame_ref, origin_pos_B):
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
            controller.set_teleop_zero(
                frame.wrist_pos_world,
                frame.wrist_quat_world,
                ee_pos_B=origin_pos_B,
            )
            print("Re-zeroed Arm Channel at current Quest wrist pose.")

    return on_key


def resolve_mocap_body(model, body_name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, idx) for idx in range(model.nbody)]
        raise ValueError(f"Target body {body_name!r} not found. Available bodies: {names}")
    mocap_id = int(model.body_mocapid[body_id])
    if mocap_id < 0:
        raise ValueError(f"Target body {body_name!r} exists but is not mocap=true")
    return body_id, mocap_id


def print_mapping(R_BQ):
    print("Axis mapping for this debug B frame:")
    print("  B +X = forward in MuJoCo viewer/debug space")
    print("  B +Y = left")
    print("  B +Z = up")
    print("R_BQ maps Quest/Unity Q deltas into B:")
    print(np.array2string(R_BQ, precision=4, suppress_small=True))
    print("Expected with default mapping:")
    print("  hand up      (+Y_Q) -> target +Z_B")
    print("  hand forward (+Z_Q) -> target +X_B")
    print("  hand left    (-X_Q) -> target +Y_B")


def main():
    args = parse_args()
    origin_pos_B = parse_vec3(args.origin_pos_B, "--origin-pos-B")
    R_BQ = DEFAULT_R_BQ.copy() if args.R_BQ is None else np.asarray(args.R_BQ, dtype=np.float64).reshape(3, 3)
    model_path = resolve_model(args.model)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if args.disable_gravity:
        model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    _target_body_id, target_mocap_id = resolve_mocap_body(model, args.target_body)
    data.mocap_pos[target_mocap_id] = origin_pos_B

    print("Before running:")
    print(f"  1. Confirm: adb reverse tcp:{args.port} tcp:{args.port}")
    print(f"  2. Quest HTS: TCP, localhost, port {args.port}")
    print("  3. This standalone scene only contains the Arm Channel target ball.")
    print("  4. Move wrist/root: up -> sphere up, forward -> sphere forward, left -> sphere left.")
    print("  5. Press R in the MuJoCo viewer to re-zero at the current wrist pose.")
    print(f"model={model_path}")
    print(f"target_body={args.target_body}")
    print(f"origin_pos_B={np.array2string(origin_pos_B, precision=4)} scale={args.scale}")
    print_mapping(R_BQ)

    if args.dry_run:
        print("dry_run=true")
        return

    controller = RelativeWristArmController(
        scale=args.scale,
        R_BQ=R_BQ,
        deadzone=args.deadzone,
        smoothing_alpha=args.smoothing_alpha,
    )
    latest_frame_ref = {"frame": None}
    frame_queue = queue.Queue()
    start_quest_receiver(args, frame_queue)
    print(f"Waiting for Quest TCP connection on {args.host}:{args.port}...")

    target_pos_B = origin_pos_B.copy()
    last_debug_time = 0.0
    last_frame_time = 0.0
    last_frame_id = None

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=make_key_callback(controller, latest_frame_ref, origin_pos_B),
    ) as viewer:
        while viewer.is_running():
            latest, drained = drain_latest(frame_queue)
            if latest is not None:
                frame_time, quest_frame = latest
                latest_frame_ref["frame"] = quest_frame
                last_frame_time = frame_time
                last_frame_id = quest_frame.frame_id
                if not controller.is_calibrated:
                    controller.set_teleop_zero(
                        quest_frame.wrist_pos_world,
                        quest_frame.wrist_quat_world,
                        ee_pos_B=origin_pos_B,
                    )
                    print("Teleop zero set from first Quest wrist pose.")
                target = controller.compute_target(quest_frame)
                target_pos_B = target.target_pos_B
                data.mocap_pos[target_mocap_id] = target_pos_B

                now = time.time()
                if now - last_debug_time >= args.debug_interval:
                    print(
                        "debug "
                        f"frame_id={last_frame_id} drained={drained} "
                        f"wrist_pos_Q={np.array2string(quest_frame.wrist_pos_world, precision=4, suppress_small=True)} "
                        f"delta_p_Q={np.array2string(target.delta_p_Q, precision=4, suppress_small=True)} "
                        f"target_pos_B={np.array2string(target_pos_B, precision=4, suppress_small=True)}"
                    )
                    last_debug_time = now

            data.mocap_pos[target_mocap_id] = target_pos_B
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)

    if last_frame_time == 0.0:
        print("Viewer closed before any Quest hand frame arrived.")


if __name__ == "__main__":
    main()
