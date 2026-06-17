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

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.retargeting import (
    GeometricRetargeter,
    get_quest_points_21,
    map_7d_to_mujoco_ctrl,
    print_actuator_info,
)


DEFAULT_MODEL_PATH = PROJECT_ROOT / "mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml"
SAFE_OPEN_ACTION = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


def is_right_hand_frame(frame):
    side = getattr(frame, "side", None)
    side_value = getattr(side, "value", side)
    return str(side_value).lower() == "right"


def parse_args():
    parser = argparse.ArgumentParser(description="Formula-based Quest 21-landmark to Aero Hand teleop.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL_PATH))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--timeout", type=float, default=0.3)
    parser.add_argument("--debug-interval", type=float, default=0.5)
    return parser.parse_args()


def start_quest_receiver(args, frame_queue):
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
            if isinstance(frame, HandFrame) and is_right_hand_frame(frame):
                frame_queue.put((time.time(), getattr(frame, "sequence_id", None), frame))

    thread = threading.Thread(target=run, name="quest-tcp-receiver", daemon=True)
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


def maybe_print_debug(
    now,
    last_debug_time,
    args,
    raw_action,
    filtered_action,
    timed_out,
    have_frame,
    got_new_frame,
    frame_age,
    sequence_id,
    drained_count,
):
    if now - last_debug_time < args.debug_interval:
        return last_debug_time

    raw_text = np.array2string(raw_action, precision=3, suppress_small=True)
    filtered_text = np.array2string(filtered_action, precision=3, suppress_small=True)
    print(
        "debug "
        f"raw_action={raw_text} filtered_action={filtered_text} "
        f"timeout={timed_out} have_frame={have_frame} new_frame={got_new_frame} "
        f"frame_age={frame_age:.3f}s seq={sequence_id} drained={drained_count}"
    )
    return now


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser()

    print("Before running:")
    print(f"  1. Confirm: adb reverse tcp:{args.port} tcp:{args.port}")
    print(f"  2. Quest HTS: TCP, localhost, port {args.port}")
    print("  3. Formula retargeting: Quest 21 landmarks -> Aero 7D action")

    print(f"Loading model: {model_path}")
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    print_actuator_info(model)

    retargeter = GeometricRetargeter(alpha=args.alpha, initial_action=SAFE_OPEN_ACTION)
    raw_action = SAFE_OPEN_ACTION.copy()
    filtered_action = SAFE_OPEN_ACTION.copy()
    last_frame_time = 0.0
    last_sequence_id = None
    last_debug_time = 0.0
    have_frame = False

    frame_queue = queue.Queue()
    start_quest_receiver(args, frame_queue)
    print(f"Waiting for Quest TCP connection on {args.host}:{args.port}...")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            latest, drained_count = drain_latest_frame(frame_queue)
            got_new_frame = latest is not None
            if latest is not None:
                frame_recv_time, last_sequence_id, frame = latest
                try:
                    quest_points = get_quest_points_21(frame)
                    raw_action, filtered_action = retargeter(quest_points)
                    last_frame_time = frame_recv_time
                    have_frame = True
                except ValueError as exc:
                    print(f"Skipping invalid Quest frame: {exc}")

            now = time.time()
            frame_age = float("inf") if not have_frame else now - last_frame_time
            timed_out = (not have_frame) or (frame_age > args.timeout)
            if timed_out:
                alpha = float(np.clip(args.alpha, 0.0, 1.0))
                filtered_action = ((1.0 - alpha) * filtered_action + alpha * SAFE_OPEN_ACTION).astype(np.float32)
                retargeter.prev_action = filtered_action.copy()

            data.ctrl[:] = map_7d_to_mujoco_ctrl(filtered_action, model)
            mujoco.mj_step(model, data)
            viewer.sync()

            last_debug_time = maybe_print_debug(
                now,
                last_debug_time,
                args,
                raw_action,
                filtered_action,
                timed_out,
                have_frame,
                got_new_frame,
                frame_age,
                last_sequence_id,
                drained_count,
            )
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
