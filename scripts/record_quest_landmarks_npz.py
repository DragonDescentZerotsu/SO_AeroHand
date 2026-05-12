import argparse
import sys
import time
from pathlib import Path

try:
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
        "Missing recording dependency. Install required packages with: "
        "pip install numpy hand-tracking-sdk"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.retargeting import get_quest_points_21


def parse_args():
    parser = argparse.ArgumentParser(description="Record Quest 21 hand landmarks to a compressed NPZ file.")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--output", default="data/quest_landmarks.npz")
    parser.add_argument("--num-frames", type=int, default=3000)
    parser.add_argument("--hand", choices=["right", "left", "any"], default="right")
    parser.add_argument("--debug-interval", type=float, default=0.5)
    return parser.parse_args()


def resolve_output(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def hand_matches(frame, hand):
    if hand == "any":
        return True
    side = getattr(frame, "side", None)
    side_value = getattr(side, "value", side)
    return str(side_value).lower() == hand


def main():
    args = parse_args()
    output_path = resolve_output(args.output)

    print("Before running:")
    print(f"  1. Confirm: adb reverse tcp:{args.port} tcp:{args.port}")
    print(f"  2. Quest HTS: TCP, localhost, port {args.port}")
    print(f"  3. Recording {args.hand} hand landmarks to {output_path}")

    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host=args.host,
            port=args.port,
            output=StreamOutput.FRAMES,
        )
    )

    landmarks = []
    sequence_ids = []
    wall_times = []
    start_time = time.time()
    last_debug_time = 0.0

    print(f"Waiting for Quest TCP connection on {args.host}:{args.port}...")
    for frame in client.iter_events():
        if not isinstance(frame, HandFrame):
            continue
        if not hand_matches(frame, args.hand):
            continue
        if len(landmarks) >= args.num_frames:
            break

        try:
            points = get_quest_points_21(frame)
        except ValueError as exc:
            print(f"Skipping invalid Quest frame: {exc}")
            continue

        now = time.time()
        landmarks.append(points)
        sequence_ids.append(int(getattr(frame, "sequence_id", -1)))
        wall_times.append(float(now))

        if now - last_debug_time >= args.debug_interval:
            print(
                f"recorded={len(landmarks)}/{args.num_frames} "
                f"seq={sequence_ids[-1]} elapsed={now - start_time:.2f}s"
            )
            last_debug_time = now

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        P_human=np.asarray(landmarks, dtype=np.float32),
        landmarks=np.asarray(landmarks, dtype=np.float32),
        sequence_id=np.asarray(sequence_ids, dtype=np.int64),
        wall_time=np.asarray(wall_times, dtype=np.float64),
        hand=np.asarray(args.hand),
    )
    print(f"Saved {len(landmarks)} frames to {output_path}")


if __name__ == "__main__":
    main()

