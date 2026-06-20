from __future__ import annotations

import json
from pathlib import Path

from .hts_receiver import HandFrame


class JsonlHandDataLogger:
    """Append minimal hand frames to a JSONL file."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write_frames(self, frames: list[HandFrame]) -> None:
        """Write all frames to disk in JSON lines format."""
        with self.path.open("w", encoding="utf-8") as f:
            for frame in frames:
                f.write(json.dumps(frame.to_dict()) + "\n")

