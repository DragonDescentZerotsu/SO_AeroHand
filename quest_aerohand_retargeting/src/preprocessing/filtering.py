from __future__ import annotations

import numpy as np


class LowPassFilter:
    """Simple exponential low-pass filter for numeric arrays."""

    def __init__(self, alpha: float = 0.25):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        self.previous: dict[str, np.ndarray] | None = None

    def apply(self, keypoints: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Filter a dictionary of keypoints while preserving keys."""
        current = {name: np.asarray(point, dtype=np.float64) for name, point in keypoints.items()}
        if self.previous is None:
            self.previous = {name: point.copy() for name, point in current.items()}
            return current
        filtered = {}
        for name, point in current.items():
            prev = self.previous.get(name, point)
            filtered[name] = (1.0 - self.alpha) * prev + self.alpha * point
        self.previous = {name: point.copy() for name, point in filtered.items()}
        return filtered

