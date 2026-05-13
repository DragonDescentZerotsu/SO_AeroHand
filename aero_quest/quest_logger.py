"""JSONL logging for Quest dual-channel frames."""

from __future__ import annotations

import json
from pathlib import Path

from aero_quest.quest_dual_channel import (
    QuestDualChannelFrame,
    frame_from_json_dict,
    frame_to_json_dict,
)


class QuestJsonlLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self.path.open("a", encoding="utf-8")

    def write(self, frame: QuestDualChannelFrame) -> None:
        self._file.write(json.dumps(frame_to_json_dict(frame), separators=(",", ":")) + "\n")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "QuestJsonlLogger":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()


def load_quest_jsonl(path: str | Path) -> list[QuestDualChannelFrame]:
    frames: list[QuestDualChannelFrame] = []
    with Path(path).open("r", encoding="utf-8") as file:
        for line_no, line in enumerate(file, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                frames.append(frame_from_json_dict(json.loads(stripped)))
            except Exception as exc:
                raise ValueError(f"Could not parse {path}:{line_no}: {exc}") from exc
    return frames
