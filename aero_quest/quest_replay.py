"""Replay Quest dual-channel JSONL recordings."""

from __future__ import annotations

import time
from collections.abc import Iterator
from pathlib import Path

from aero_quest.quest_dual_channel import QuestDualChannelFrame
from aero_quest.quest_logger import load_quest_jsonl


class QuestReplay:
    def __init__(self, path: str | Path, realtime: bool = True, speed: float = 1.0, loop: bool = False):
        if speed <= 0.0:
            raise ValueError("speed must be > 0")
        self.path = Path(path)
        self.realtime = bool(realtime)
        self.speed = float(speed)
        self.loop = bool(loop)
        self.frames = load_quest_jsonl(self.path)

    def iter_frames(self) -> Iterator[QuestDualChannelFrame]:
        if not self.frames:
            return
        while True:
            previous_ts = None
            for frame in self.frames:
                ts = _timestamp_ns(frame)
                if self.realtime and previous_ts is not None and ts is not None:
                    delay_s = max(0.0, (ts - previous_ts) / 1_000_000_000.0 / self.speed)
                    time.sleep(delay_s)
                previous_ts = ts
                yield frame
            if not self.loop:
                break


def _timestamp_ns(frame: QuestDualChannelFrame) -> int | None:
    return frame.source_ts_ns if frame.source_ts_ns is not None else frame.recv_ts_ns
