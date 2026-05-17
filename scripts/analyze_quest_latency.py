#!/usr/bin/env python
"""Analyze recorded Quest JSONL latency and quality."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.quest_data_quality import summarize_quality
from aero_quest.quest_logger import load_quest_jsonl


def parse_args():
    parser = argparse.ArgumentParser(description="Analyze Quest dual-channel JSONL logs.")
    parser.add_argument("--log", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frames = load_quest_jsonl(args.log)
    summary = summarize_quality(frames)
    print(f"Quest telemetry quality: {args.log}")
    for key, value in summary.items():
        if isinstance(value, float):
            print(f"{key:34s} {value:.3f}")
        else:
            print(f"{key:34s} {value}")
    print()
    _print_warnings(summary)


def _print_warnings(summary: dict) -> None:
    warnings = []
    valid_ratio = summary.get("valid_ratio")
    if valid_ratio is not None and valid_ratio < 0.95:
        warnings.append(f"valid ratio is low ({valid_ratio:.1%})")
    mean = summary.get("mean_frame_interval_ms")
    std = summary.get("std_frame_interval_ms")
    if mean and std and std > max(5.0, mean * 0.25):
        warnings.append(f"FPS is unstable (interval std {std:.1f} ms)")
    p95 = summary.get("p95_frame_interval_ms")
    if p95 is not None and p95 > 50.0:
        warnings.append(f"p95 frame interval is high ({p95:.1f} ms)")
    if summary.get("landmark_bad_shape_count", 0) > 0:
        warnings.append(f"bad landmark shapes: {summary['landmark_bad_shape_count']}")
    if summary.get("wrist_position_jump_count", 0) > 0:
        warnings.append(f"wrist position jumps: {summary['wrist_position_jump_count']}")
    if warnings:
        print("Highlights:")
        for warning in warnings:
            print(f"  - {warning}")
    else:
        print("Highlights: no obvious quality warnings.")


if __name__ == "__main__":
    main()
