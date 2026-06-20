from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from quest_io.data_logger import JsonlHandDataLogger
from quest_io.hts_receiver import MockHTSReceiver


def main() -> None:
    """Record mock Quest data until a real HTS receiver is wired in."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(PROJECT_DIR / "data/raw/mock_quest_hand.jsonl"))
    parser.add_argument("--num-frames", type=int, default=120)
    args = parser.parse_args()
    frames = list(MockHTSReceiver(num_frames=args.num_frames).iter_frames())
    JsonlHandDataLogger(args.out).write_frames(frames)
    print(f"wrote {len(frames)} frames to {args.out}")


if __name__ == "__main__":
    main()

