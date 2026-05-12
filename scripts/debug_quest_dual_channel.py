import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.quest_hand_frame import (
    LandmarkHandRetargeter,
    QuestPacketParseError,
    RelativeWristArmController,
    parse_quest_hand_frames,
    quest_hand_frame_from_sdk,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Debug the mixed-frame Quest dual-channel hand pipeline.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hand", choices=["right", "left", "any"], default="right")
    parser.add_argument("--sample-packet", default=None, help="Parse a text packet file instead of opening TCP.")
    parser.add_argument("--debug-interval", type=float, default=0.5)
    parser.add_argument("--scale", type=float, default=1.0)
    parser.add_argument("--deadzone", type=float, default=0.0)
    parser.add_argument("--smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--no-auto-zero", action="store_true")
    return parser.parse_args()


def hand_matches(quest_frame, hand: str) -> bool:
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
            if isinstance(frame, HandFrame):
                try:
                    quest_frame = quest_hand_frame_from_sdk(frame)
                except ValueError as exc:
                    print(f"Skipping invalid SDK frame: {exc}")
                    continue
                if hand_matches(quest_frame, args.hand):
                    frame_queue.put((time.time(), quest_frame))

    thread = threading.Thread(target=run, name="quest-dual-channel-debug-receiver", daemon=True)
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


def print_frame(quest_frame, hand_command, arm_target, drained):
    print(
        "debug "
        f"side={quest_frame.hand_side} frame_id={quest_frame.frame_id} timestamp_ns={quest_frame.timestamp_ns} drained={drained}"
    )
    print(f"  wrist_pos_world={np.array2string(quest_frame.wrist_pos_world, precision=5, suppress_small=True)}")
    print(f"  wrist_quat_world_xyzw={np.array2string(quest_frame.wrist_quat_world, precision=5, suppress_small=True)}")
    print(f"  landmarks_wrist_shape={quest_frame.landmarks_wrist.shape}")
    print(f"  landmarks_wrist_first3={np.array2string(quest_frame.landmarks_wrist[:3], precision=5, suppress_small=True)}")
    feature_text = ", ".join(f"{key}={value:.4f}" for key, value in sorted(hand_command.features.items()))
    print(f"  hand_features={feature_text}")
    if hand_command.aero_action_7d is not None:
        print(f"  aero_action_7d={np.array2string(hand_command.aero_action_7d, precision=4, suppress_small=True)}")
    if arm_target is None:
        print("  arm_target_B=<teleop zero not set>")
    else:
        print(f"  arm_delta_p_Q={np.array2string(arm_target.delta_p_Q, precision=5, suppress_small=True)}")
        print(f"  arm_target_pos_B={np.array2string(arm_target.target_pos_B, precision=5, suppress_small=True)}")


def run_sample_packet(args):
    text = Path(args.sample_packet).read_text(encoding="utf-8")
    try:
        frames = parse_quest_hand_frames(text)
    except QuestPacketParseError as exc:
        raise SystemExit(f"Could not parse sample packet: {exc}") from exc
    retargeter = LandmarkHandRetargeter()
    arm = RelativeWristArmController(scale=args.scale, deadzone=args.deadzone, smoothing_alpha=args.smoothing_alpha)
    for quest_frame in frames:
        if not hand_matches(quest_frame, args.hand):
            continue
        if not args.no_auto_zero and not arm.is_calibrated:
            arm.set_teleop_zero(
                quest_frame.wrist_pos_world,
                quest_frame.wrist_quat_world,
                ee_pos_B=np.zeros(3, dtype=np.float64),
            )
        arm_target = arm.compute_target(quest_frame) if arm.is_calibrated else None
        print_frame(quest_frame, retargeter(quest_frame.landmarks_wrist), arm_target, drained=0)


def main():
    args = parse_args()
    if args.sample_packet:
        run_sample_packet(args)
        return

    print("Before running:")
    print(f"  1. Confirm: adb reverse tcp:{args.port} tcp:{args.port}")
    print(f"  2. Quest HTS: TCP, localhost, port {args.port}")
    print("  3. This prints Arm Channel wrist pose and Hand Channel wrist-local landmarks.")

    retargeter = LandmarkHandRetargeter()
    arm = RelativeWristArmController(scale=args.scale, deadzone=args.deadzone, smoothing_alpha=args.smoothing_alpha)
    frame_queue = queue.Queue()
    start_quest_receiver(args, frame_queue)
    print(f"Waiting for Quest TCP connection on {args.host}:{args.port}...")

    last_print = 0.0
    try:
        while True:
            latest, drained = drain_latest(frame_queue)
            if latest is None:
                time.sleep(0.01)
                continue
            now, quest_frame = latest
            if now - last_print < args.debug_interval:
                continue
            if not args.no_auto_zero and not arm.is_calibrated:
                arm.set_teleop_zero(
                    quest_frame.wrist_pos_world,
                    quest_frame.wrist_quat_world,
                    ee_pos_B=np.zeros(3, dtype=np.float64),
                )
                print("Teleop zero set from first received wrist pose; debug EE origin is [0, 0, 0] in B.")
            arm_target = arm.compute_target(quest_frame) if arm.is_calibrated else None
            print_frame(quest_frame, retargeter(quest_frame.landmarks_wrist), arm_target, drained=drained)
            last_print = now
    except KeyboardInterrupt:
        print("Exiting Quest dual-channel debug.")


if __name__ == "__main__":
    main()
