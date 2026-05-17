#!/usr/bin/env python
"""Record Quest wrist pose and wrist-local landmarks to JSONL."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.quest_logger import QuestJsonlLogger
from aero_quest.quest_receiver import QuestTelemetryReceiver


def parse_args():
    parser = argparse.ArgumentParser(description="Record Quest dual-channel telemetry to JSONL.")
    parser.add_argument("--transport", choices=["tcp", "udp"], default="tcp")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hand", default="right")
    parser.add_argument("--out", default="logs/quest_session.jsonl")
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    receiver = QuestTelemetryReceiver(args.transport, args.host, args.port, args.hand)
    total = valid = 0
    start = time.monotonic()
    print(f"Recording Quest {args.transport.upper()} telemetry on {args.host}:{args.port} -> {out}")
    try:
        with QuestJsonlLogger(out) as logger:
            for frame in receiver.iter_frames():
                total += 1
                valid += int(frame.valid)
                logger.write(frame)
                if total % max(1, args.print_every) == 0:
                    elapsed = max(1e-9, time.monotonic() - start)
                    fps = total / elapsed
                    print(
                        f"frames={total} valid={valid} fps={fps:.1f} "
                        f"wrist_pos_world={np.array2string(frame.wrist_pos_world, precision=4, suppress_small=True)} "
                        f"landmarks_shape={frame.landmarks_wrist.shape}"
                    )
                if args.duration is not None and (time.monotonic() - start) >= args.duration:
                    break
    except KeyboardInterrupt:
        print("Stopping recording.")
    finally:
        receiver.close()
        print(f"Wrote {total} frames ({valid} valid) to {out}")


if __name__ == "__main__":
    main()
