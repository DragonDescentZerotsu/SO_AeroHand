"""Quality and latency analysis for Quest dual-channel telemetry."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from aero_quest.quest_dual_channel import QuestDualChannelFrame


def compute_frame_intervals(frames: Sequence[QuestDualChannelFrame]) -> np.ndarray:
    timestamps = [_timestamp_ns(frame) for frame in frames]
    timestamps = [ts for ts in timestamps if ts is not None]
    if len(timestamps) < 2:
        return np.asarray([], dtype=np.float64)
    return np.diff(np.asarray(timestamps, dtype=np.float64)) / 1_000_000.0


def compute_fps_stats(frames: Sequence[QuestDualChannelFrame]) -> dict[str, float | None]:
    intervals_ms = compute_frame_intervals(frames)
    positive = intervals_ms[intervals_ms > 0.0]
    if positive.size == 0:
        return {"average_fps": None, "active_average_fps": None, "instant_fps_mean": None, "nominal_fps_p50": None}
    active = positive[positive <= 100.0]
    active_average = float(1000.0 / np.mean(active)) if active.size else None
    return {
        "average_fps": float(1000.0 / np.mean(positive)),
        "active_average_fps": active_average,
        "instant_fps_mean": float(np.mean(1000.0 / positive)),
        "nominal_fps_p50": float(1000.0 / np.percentile(positive, 50)),
    }


def compute_jitter_stats(frames: Sequence[QuestDualChannelFrame]) -> dict[str, float | None]:
    intervals_ms = compute_frame_intervals(frames)
    if intervals_ms.size == 0:
        return _empty_interval_stats()
    return {
        "min_frame_interval_ms": float(np.min(intervals_ms)),
        "max_frame_interval_ms": float(np.max(intervals_ms)),
        "mean_frame_interval_ms": float(np.mean(intervals_ms)),
        "std_frame_interval_ms": float(np.std(intervals_ms)),
        "p50_frame_interval_ms": float(np.percentile(intervals_ms, 50)),
        "p90_frame_interval_ms": float(np.percentile(intervals_ms, 90)),
        "p95_frame_interval_ms": float(np.percentile(intervals_ms, 95)),
        "p99_frame_interval_ms": float(np.percentile(intervals_ms, 99)),
    }


def count_bad_frames(frames: Sequence[QuestDualChannelFrame]) -> dict[str, int]:
    invalid = sum(1 for frame in frames if not frame.valid)
    return {"valid_frames": len(frames) - invalid, "invalid_frames": invalid}


def detect_position_jumps(
    frames: Sequence[QuestDualChannelFrame],
    threshold_m: float = 0.20,
) -> list[tuple[int, float]]:
    jumps: list[tuple[int, float]] = []
    previous: np.ndarray | None = None
    for index, frame in enumerate(frames):
        if not frame.valid:
            continue
        pos = np.asarray(frame.wrist_pos_world, dtype=np.float64)
        if pos.shape != (3,) or not np.all(np.isfinite(pos)):
            continue
        if previous is not None:
            distance = float(np.linalg.norm(pos - previous))
            if distance > float(threshold_m):
                jumps.append((index, distance))
        previous = pos
    return jumps


def quaternion_norm_stats(frames: Sequence[QuestDualChannelFrame]) -> dict[str, float | None]:
    norms = []
    for frame in frames:
        quat = np.asarray(frame.wrist_quat_world, dtype=np.float64)
        if quat.shape == (4,) and np.all(np.isfinite(quat)):
            norms.append(float(np.linalg.norm(quat)))
    if not norms:
        return {"quat_norm_mean": None, "quat_norm_std": None, "quat_norm_min": None, "quat_norm_max": None}
    values = np.asarray(norms, dtype=np.float64)
    return {
        "quat_norm_mean": float(np.mean(values)),
        "quat_norm_std": float(np.std(values)),
        "quat_norm_min": float(np.min(values)),
        "quat_norm_max": float(np.max(values)),
    }


def landmark_shape_stats(frames: Sequence[QuestDualChannelFrame]) -> dict[str, int]:
    bad = sum(1 for frame in frames if np.asarray(frame.landmarks_wrist).shape != (21, 3))
    return {"landmark_bad_shape_count": bad}


def summarize_quality(frames: Sequence[QuestDualChannelFrame]) -> dict[str, Any]:
    bad_counts = count_bad_frames(frames)
    intervals = compute_jitter_stats(frames)
    fps = compute_fps_stats(frames)
    out_of_order = _count_out_of_order_timestamps(frames)
    dropped = _estimate_dropped_frames(frames)
    jumps = detect_position_jumps(frames)
    summary: dict[str, Any] = {
        "total_frames": len(frames),
        **bad_counts,
        "valid_ratio": (bad_counts["valid_frames"] / len(frames)) if frames else None,
        **fps,
        **intervals,
        "estimated_dropped_frames": dropped,
        "sequence_reset_count": _count_sequence_resets(frames),
        "out_of_order_timestamp_count": out_of_order,
        "burst_interval_count_le_1ms": int(np.sum(compute_frame_intervals(frames) <= 1.0)),
        "long_gap_count_gt_33ms": int(np.sum(compute_frame_intervals(frames) > 33.0)),
        "long_gap_count_gt_100ms": int(np.sum(compute_frame_intervals(frames) > 100.0)),
        "long_gap_count_gt_1000ms": int(np.sum(compute_frame_intervals(frames) > 1000.0)),
        "wrist_position_jump_count": len(jumps),
        "wrist_position_jump_max_m": max((distance for _, distance in jumps), default=0.0),
        **quaternion_norm_stats(frames),
        **landmark_shape_stats(frames),
    }
    return summary


def _timestamp_ns(frame: QuestDualChannelFrame) -> int | None:
    return frame.source_ts_ns if frame.source_ts_ns is not None else frame.recv_ts_ns


def _count_out_of_order_timestamps(frames: Sequence[QuestDualChannelFrame]) -> int:
    count = 0
    previous: int | None = None
    for frame in frames:
        ts = _timestamp_ns(frame)
        if ts is None:
            continue
        if previous is not None and ts < previous:
            count += 1
        previous = ts
    return count


def _estimate_dropped_frames(frames: Sequence[QuestDualChannelFrame]) -> int | None:
    ids = []
    for frame in frames:
        value = frame.sequence_id if frame.sequence_id is not None else frame.frame_id
        try:
            ids.append(int(value))
        except (TypeError, ValueError):
            continue
    if len(ids) < 2:
        return None
    dropped = 0
    previous = ids[0]
    for value in ids[1:]:
        if value > previous + 1:
            dropped += value - previous - 1
        previous = value
    return int(dropped)


def _count_sequence_resets(frames: Sequence[QuestDualChannelFrame]) -> int:
    count = 0
    previous: int | None = None
    for frame in frames:
        value = frame.sequence_id
        if value is None:
            continue
        if previous is not None and int(value) < previous:
            count += 1
        previous = int(value)
    return count


def _empty_interval_stats() -> dict[str, None]:
    return {
        "min_frame_interval_ms": None,
        "max_frame_interval_ms": None,
        "mean_frame_interval_ms": None,
        "std_frame_interval_ms": None,
        "p50_frame_interval_ms": None,
        "p90_frame_interval_ms": None,
        "p95_frame_interval_ms": None,
        "p99_frame_interval_ms": None,
    }
