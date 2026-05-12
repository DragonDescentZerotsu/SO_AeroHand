"""Quality filters for recorded Quest 21-landmark hand frames."""

from __future__ import annotations

from collections import Counter

import numpy as np

from aero_quest.closure_features import SEGMENT_PAIRS


PALM_SCALE_PAIR = (0, 9)
PALM_ANCHOR_PAIRS = ((0, 5), (0, 9), (0, 17))
QUALITY_SEGMENT_PAIRS = tuple(SEGMENT_PAIRS) + PALM_ANCHOR_PAIRS


def _as_landmark_sequence(landmarks: np.ndarray) -> np.ndarray:
    """Return landmarks as a float64 ``(T, 21, 3)`` array."""
    arr = np.asarray(landmarks, dtype=np.float64)
    if arr.ndim != 3 or arr.shape[1:] != (21, 3):
        raise ValueError(f"Expected landmarks shape (T, 21, 3), got {arr.shape}")
    return arr


def palm_scale(points: np.ndarray) -> float:
    """Return wrist-to-middle-proximal scale for one 21-landmark frame."""
    points = np.asarray(points, dtype=np.float64)
    return float(np.linalg.norm(points[PALM_SCALE_PAIR[1]] - points[PALM_SCALE_PAIR[0]]))


def segment_lengths(points: np.ndarray, pairs=QUALITY_SEGMENT_PAIRS) -> np.ndarray:
    """Return selected hand segment lengths for one 21-landmark frame."""
    points = np.asarray(points, dtype=np.float64)
    return np.asarray([np.linalg.norm(points[j] - points[i]) for i, j in pairs], dtype=np.float64)


def robust_reference_lengths(landmarks: np.ndarray, pairs=QUALITY_SEGMENT_PAIRS) -> np.ndarray:
    """Estimate stable reference segment lengths from the whole recording."""
    landmarks = _as_landmark_sequence(landmarks)
    lengths = np.stack([segment_lengths(frame, pairs=pairs) for frame in landmarks], axis=0)
    finite = np.isfinite(lengths)
    safe_lengths = np.where(finite, lengths, np.nan)
    ref = np.nanmedian(safe_lengths, axis=0)
    ref = np.nan_to_num(ref, nan=1.0, posinf=1.0, neginf=1.0)
    return np.maximum(ref, 1e-8)


def quality_check_frame(
    points: np.ndarray,
    *,
    prev_points: np.ndarray | None = None,
    frame_index: int = 0,
    reference_scale: float | None = None,
    reference_lengths: np.ndarray | None = None,
    warmup_frames: int = 0,
    min_palm_scale: float = 0.03,
    max_palm_scale: float = 0.25,
    min_scale_ratio: float = 0.55,
    max_scale_ratio: float = 1.80,
    min_segment_length: float = 0.005,
    min_segment_ratio: float = 0.35,
    max_segment_ratio: float = 2.50,
    max_wrist_step: float = 0.15,
    max_point_step: float = 0.12,
) -> tuple[bool, str]:
    """Return ``(keep, reason)`` for one Quest landmark frame.

    The filter catches non-finite frames, implausible palm scale, distorted bone
    lengths, and sudden inter-frame jumps. Thresholds are in meters for world or
    wrist-local Quest coordinates.
    """
    points = np.asarray(points, dtype=np.float64)
    if points.shape != (21, 3):
        return False, "bad_shape"
    if frame_index < int(warmup_frames):
        return False, "warmup"
    if not np.all(np.isfinite(points)):
        return False, "non_finite"

    scale = palm_scale(points)
    if not np.isfinite(scale) or scale < min_palm_scale or scale > max_palm_scale:
        return False, "palm_scale_abs"
    if reference_scale is not None and reference_scale > 1e-8:
        scale_ratio = scale / float(reference_scale)
        if scale_ratio < min_scale_ratio or scale_ratio > max_scale_ratio:
            return False, "palm_scale_ratio"

    lengths = segment_lengths(points)
    if not np.all(np.isfinite(lengths)) or float(np.min(lengths)) < min_segment_length:
        return False, "segment_length_abs"
    if reference_lengths is not None:
        ratios = lengths / np.maximum(np.asarray(reference_lengths, dtype=np.float64), 1e-8)
        if np.any((ratios < min_segment_ratio) | (ratios > max_segment_ratio)):
            return False, "segment_length_ratio"

    if prev_points is not None and np.asarray(prev_points).shape == (21, 3):
        wrist_step = float(np.linalg.norm(points[0] - prev_points[0]))
        if wrist_step > max_wrist_step:
            return False, "wrist_jump"
        point_step = float(np.max(np.linalg.norm(points - prev_points, axis=1)))
        if point_step > max_point_step:
            return False, "point_jump"

    return True, "ok"


def build_quality_mask(
    landmarks: np.ndarray,
    *,
    warmup_frames: int = 0,
    min_palm_scale: float = 0.03,
    max_palm_scale: float = 0.25,
    min_scale_ratio: float = 0.55,
    max_scale_ratio: float = 1.80,
    min_segment_length: float = 0.005,
    min_segment_ratio: float = 0.35,
    max_segment_ratio: float = 2.50,
    max_wrist_step: float = 0.15,
    max_point_step: float = 0.12,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Build a keep mask, per-frame reason strings, and summary counts."""
    landmarks = _as_landmark_sequence(landmarks)
    scales = np.asarray([palm_scale(frame) for frame in landmarks], dtype=np.float64)
    finite_scales = scales[np.isfinite(scales)]
    reference_scale = float(np.median(finite_scales)) if finite_scales.size else 1.0
    reference_lengths = robust_reference_lengths(landmarks)

    keep = np.zeros(len(landmarks), dtype=bool)
    reasons: list[str] = []
    prev_good = None
    for idx, points in enumerate(landmarks):
        frame_keep, reason = quality_check_frame(
            points,
            prev_points=prev_good,
            frame_index=idx,
            reference_scale=reference_scale,
            reference_lengths=reference_lengths,
            warmup_frames=warmup_frames,
            min_palm_scale=min_palm_scale,
            max_palm_scale=max_palm_scale,
            min_scale_ratio=min_scale_ratio,
            max_scale_ratio=max_scale_ratio,
            min_segment_length=min_segment_length,
            min_segment_ratio=min_segment_ratio,
            max_segment_ratio=max_segment_ratio,
            max_wrist_step=max_wrist_step,
            max_point_step=max_point_step,
        )
        keep[idx] = frame_keep
        reasons.append(reason)
        if frame_keep:
            prev_good = points

    counts = dict(Counter(reasons))
    counts["total"] = int(len(landmarks))
    counts["kept"] = int(np.sum(keep))
    counts["dropped"] = int(len(landmarks) - np.sum(keep))
    counts["reference_scale"] = float(reference_scale)
    return keep, np.asarray(reasons, dtype=object), counts
