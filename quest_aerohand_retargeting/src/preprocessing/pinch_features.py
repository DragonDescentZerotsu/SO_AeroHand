from __future__ import annotations

import numpy as np


def pinch_distance(keypoints: dict[str, np.ndarray]) -> float:
    """Return thumb-tip to index-tip distance in meters or normalized units."""
    return float(np.linalg.norm(np.asarray(keypoints["thumb_tip"]) - np.asarray(keypoints["index_tip"])))


def pinch_direction(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    """Return unit direction from thumb tip toward index tip."""
    vector = np.asarray(keypoints["index_tip"], dtype=np.float64) - np.asarray(keypoints["thumb_tip"], dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-8:
        return np.zeros(3, dtype=np.float64)
    return vector / norm


def pinch_strength(distance: float, closed_m: float = 0.025, open_m: float = 0.085) -> float:
    """Map pinch distance to ``[0, 1]`` where 1 means closed pinch."""
    denom = max(float(open_m) - float(closed_m), 1e-8)
    return float(np.clip((float(open_m) - float(distance)) / denom, 0.0, 1.0))


def detect_pinch_event(distance: float, threshold_m: float = 0.03) -> bool:
    """Return whether thumb and index are close enough to count as a pinch."""
    return bool(float(distance) <= float(threshold_m))


def extract_pinch_features(keypoints: dict[str, np.ndarray], closed_m: float = 0.025, open_m: float = 0.085) -> dict:
    """Compute distance, direction, strength, and binary pinch event."""
    dist = pinch_distance(keypoints)
    return {
        "pinch_distance": dist,
        "pinch_direction": pinch_direction(keypoints),
        "pinch_strength": pinch_strength(dist, closed_m=closed_m, open_m=open_m),
        "pinch_event": detect_pinch_event(dist, threshold_m=closed_m * 1.2),
    }

