from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_DIR / "src"))

from quest_io.data_loader import load_jsonl_frames


def main() -> None:
    """Replay a JSONL recording by printing basic frame information."""
    parser = argparse.ArgumentParser()
    parser.add_argument("path")
    args = parser.parse_args()
    frames = load_jsonl_frames(args.path)
    for frame in frames[:5]:
        print(frame.to_dict())
    print(f"loaded {len(frames)} frames")


if __name__ == "__main__":
    main()

