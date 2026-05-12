"""Closure feature extraction for 21-landmark hands."""

from __future__ import annotations

import numpy as np


TIP_INDICES = [4, 8, 12, 16, 20]
BEND_TRIPLES = [
    (0, 1, 2),
    (1, 2, 3),
    (2, 3, 4),
    (5, 6, 7),
    (6, 7, 8),
    (9, 10, 11),
    (10, 11, 12),
    (13, 14, 15),
    (14, 15, 16),
    (17, 18, 19),
    (18, 19, 20),
]
BEND_NAMES = [
    "thumb_base",
    "thumb_mid",
    "thumb_tip",
    "index_proximal",
    "index_distal",
    "middle_proximal",
    "middle_distal",
    "ring_proximal",
    "ring_distal",
    "little_proximal",
    "little_distal",
]
SEGMENT_PAIRS = [
    (1, 2),
    (2, 3),
    (3, 4),
    (5, 6),
    (6, 7),
    (7, 8),
    (9, 10),
    (10, 11),
    (11, 12),
    (13, 14),
    (14, 15),
    (15, 16),
    (17, 18),
    (18, 19),
    (19, 20),
]


def _as_points(points: np.ndarray) -> np.ndarray:
    """Validate and return 21x3 points as float64."""
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected points shape (21, 3), got {arr.shape}")
    return np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)


def joint_bend(points, a, b, c, eps=1e-8) -> float:
    """Return bend angle ``pi - theta`` for triplet ``a-b-c``."""
    points = _as_points(points)
    v1 = points[a] - points[b]
    v2 = points[c] - points[b]
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < eps or n2 < eps:
        return 0.0
    dot = float(np.dot(v1, v2) / (n1 * n2))
    theta = float(np.arccos(np.clip(dot, -1.0, 1.0)))
    bend = float(np.pi - theta)
    return bend if np.isfinite(bend) else 0.0


def segment_direction(points, i, j, eps=1e-8) -> np.ndarray:
    """Return unit direction from landmark ``i`` to ``j`` or zero if degenerate."""
    points = _as_points(points)
    v = points[j] - points[i]
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros(3, dtype=np.float64)
    direction = v / norm
    return np.nan_to_num(direction, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def extract_closure_features(points_local) -> dict:
    """Extract bend, fingertip, segment-direction, and flat closure features."""
    points = _as_points(points_local)
    bends = np.asarray([joint_bend(points, *triple) for triple in BEND_TRIPLES], dtype=np.float64)
    tip_positions = points[TIP_INDICES].astype(np.float64)
    segment_dirs = np.asarray(
        [segment_direction(points, i, j) for i, j in SEGMENT_PAIRS],
        dtype=np.float64,
    )
    flat = np.concatenate([bends, tip_positions.reshape(-1), segment_dirs.reshape(-1)]).astype(np.float64)
    return {
        "bends": np.nan_to_num(bends, nan=0.0, posinf=0.0, neginf=0.0),
        "tip_positions": np.nan_to_num(tip_positions, nan=0.0, posinf=0.0, neginf=0.0),
        "segment_dirs": np.nan_to_num(segment_dirs, nan=0.0, posinf=0.0, neginf=0.0),
        "flat": np.nan_to_num(flat, nan=0.0, posinf=0.0, neginf=0.0),
    }

