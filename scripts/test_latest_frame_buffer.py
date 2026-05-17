#!/usr/bin/env python
"""Exercise the live Quest receiver plus latest-frame buffer."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.quest_frame_buffer import LatestFrameBuffer, QuestReceiverThread


def parse_args():
    parser = argparse.ArgumentParser(description="Test live Quest latest-frame buffering.")
    parser.add_argument("--transport", choices=["tcp", "udp"], default="tcp")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hand", default="right")
    parser.add_argument("--rate-hz", type=float, default=60.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    buffer = LatestFrameBuffer()
    receiver_thread = QuestReceiverThread(
        buffer=buffer,
        transport=args.transport,
        host=args.host,
        port=args.port,
        hand=args.hand,
    )
    receiver_thread.start()
    period = 1.0 / args.rate_hz
    print(f"Reading latest Quest frame at {args.rate_hz:.1f} Hz. Press Ctrl-C to stop.")
    try:
        while True:
            time.sleep(period)
            latest = buffer.get_latest()
            stats = buffer.stats()
            if receiver_thread.error is not None:
                raise receiver_thread.error
            if latest is None:
                print(f"no frame yet stats={stats}")
                continue
            print(
                f"age_ms={stats['last_frame_age_ms']:.1f} valid={latest.valid} "
                f"frame_id={latest.frame_id} sequence_id={latest.sequence_id} stats={stats}"
            )
    except KeyboardInterrupt:
        print("Stopping receiver thread.")
    finally:
        receiver_thread.stop()


if __name__ == "__main__":
    main()
