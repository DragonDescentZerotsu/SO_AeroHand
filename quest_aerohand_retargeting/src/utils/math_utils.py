from __future__ import annotations

import numpy as np


def safe_normalize(vector: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a unit vector or zeros if the input is degenerate."""
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < eps:
        return np.zeros_like(vector)
    return vector / norm

