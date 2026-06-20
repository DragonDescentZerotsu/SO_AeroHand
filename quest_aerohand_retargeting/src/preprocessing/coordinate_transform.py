from __future__ import annotations

import numpy as np


def normalize_to_wrist_frame(keypoints: dict[str, np.ndarray], wrist_pose: np.ndarray) -> dict[str, np.ndarray]:
    """Translate keypoints into a wrist-origin frame.

    TODO: Apply the inverse wrist orientation once the exact HTS quaternion
    convention for this scaffold is connected to live Quest frames.
    """
    wrist_pose = np.asarray(wrist_pose, dtype=np.float64).reshape(7)
    origin = wrist_pose[:3]
    return {name: np.asarray(point, dtype=np.float64).reshape(3) - origin for name, point in keypoints.items()}


def scale_normalize(keypoints: dict[str, np.ndarray], reference_m: float = 0.10) -> dict[str, np.ndarray]:
    """Normalize hand scale using the wrist-to-middle-tip distance proxy."""
    middle = np.asarray(keypoints["middle_tip"], dtype=np.float64)
    scale = max(float(np.linalg.norm(middle)), 1e-8)
    factor = float(reference_m) / scale
    return {name: np.asarray(point, dtype=np.float64) * factor for name, point in keypoints.items()}

