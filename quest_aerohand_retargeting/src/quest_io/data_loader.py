from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .hts_receiver import HandFrame


def load_jsonl_frames(path: str | Path) -> list[HandFrame]:
    """Load hand frames from a JSONL recording."""
    frames: list[HandFrame] = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                frames.append(HandFrame.from_dict(json.loads(line)))
    return frames


def load_quest_dual_channel_jsonl(path: str | Path, valid_only: bool = True) -> list[HandFrame]:
    """Load existing ``aero_quest`` dual-channel JSONL as scaffold frames.

    The source recording keeps landmarks in the Quest wrist/root-local frame.
    This adapter preserves that frame and exposes the key fingertip fields used
    by the baseline/evaluation scaffold.
    """
    from aero_quest.quest_logger import load_quest_jsonl

    frames: list[HandFrame] = []
    for frame in load_quest_jsonl(path):
        if valid_only and not frame.valid:
            continue
        landmarks_wrist = np.asarray(frame.landmarks_wrist, dtype=np.float64).reshape(21, 3)
        timestamp = _timestamp_seconds(frame.recv_ts_ns, frame.source_ts_ns, len(frames))
        wrist_pose = np.concatenate(
            [
                np.asarray(frame.wrist_pos_world, dtype=np.float64).reshape(3),
                np.asarray(frame.wrist_quat_world, dtype=np.float64).reshape(4),
            ]
        )
        frames.append(
            HandFrame(
                timestamp=timestamp,
                wrist_pose=wrist_pose,
                thumb_tip=landmarks_wrist[4],
                thumb_ip=landmarks_wrist[3],
                thumb_mcp=landmarks_wrist[2],
                index_tip=landmarks_wrist[8],
                index_pip=landmarks_wrist[6],
                index_mcp=landmarks_wrist[5],
                middle_tip=landmarks_wrist[12],
                ring_tip=landmarks_wrist[16],
                pinky_tip=landmarks_wrist[20],
                landmarks_wrist=landmarks_wrist,
            )
        )
    return frames


def _timestamp_seconds(recv_ts_ns: int | None, source_ts_ns: int | None, fallback_index: int) -> float:
    """Return a stable seconds timestamp for recorded frames."""
    timestamp_ns = recv_ts_ns if recv_ts_ns is not None else source_ts_ns
    if timestamp_ns is None:
        return float(fallback_index)
    return float(timestamp_ns) * 1e-9
