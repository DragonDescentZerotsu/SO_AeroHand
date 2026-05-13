#!/usr/bin/env python
"""Replay Quest dual-channel JSONL logs."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.quest_replay import QuestReplay


def parse_args():
    parser = argparse.ArgumentParser(description="Replay Quest dual-channel JSONL logs.")
    parser.add_argument("--log", required=True)
    parser.add_argument("--realtime", action="store_true")
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--max-frames", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    replay = QuestReplay(args.log, realtime=args.realtime, speed=args.speed)
    for index, frame in enumerate(replay.iter_frames(), start=1):
        print(
            f"frame_id={frame.frame_id} sequence_id={frame.sequence_id} "
            f"recv_ts_ns={frame.recv_ts_ns} source_ts_ns={frame.source_ts_ns} "
            f"wrist_pos_world={np.array2string(frame.wrist_pos_world, precision=4, suppress_small=True)} "
            f"landmarks_shape={frame.landmarks_wrist.shape}"
        )
        if args.max_frames is not None and index >= args.max_frames:
            break


if __name__ == "__main__":
    main()
