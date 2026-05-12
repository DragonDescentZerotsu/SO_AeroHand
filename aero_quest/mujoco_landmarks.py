"""MuJoCo Aero Hand 21-landmark site readers."""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


ROBOT_LANDMARK_SITE_NAMES = [
    "aero_wrist_lm",
    "aero_thumb_metacarpal_lm",
    "aero_thumb_proximal_lm",
    "aero_thumb_distal_lm",
    "aero_thumb_tip_lm",
    "aero_index_proximal_lm",
    "aero_index_intermediate_lm",
    "aero_index_distal_lm",
    "aero_index_tip_lm",
    "aero_middle_proximal_lm",
    "aero_middle_intermediate_lm",
    "aero_middle_distal_lm",
    "aero_middle_tip_lm",
    "aero_ring_proximal_lm",
    "aero_ring_intermediate_lm",
    "aero_ring_distal_lm",
    "aero_ring_tip_lm",
    "aero_little_proximal_lm",
    "aero_little_intermediate_lm",
    "aero_little_distal_lm",
    "aero_little_tip_lm",
]


def _require_mujoco() -> None:
    """Raise a clear error if MuJoCo is not importable."""
    if mujoco is None:
        raise RuntimeError("mujoco is required. Install it with: pip install mujoco")


def _site_names(site_names=None) -> list[str]:
    """Return the requested landmark site names."""
    return list(ROBOT_LANDMARK_SITE_NAMES if site_names is None else site_names)


def get_missing_robot_landmark_sites(model, site_names=None) -> list[str]:
    """Return robot landmark site names missing from ``model``."""
    _require_mujoco()
    missing = []
    for name in _site_names(site_names):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        if site_id < 0:
            missing.append(name)
    return missing


def has_robot_landmark_sites(model, site_names=None) -> bool:
    """Return true when every requested robot landmark site exists."""
    return len(get_missing_robot_landmark_sites(model, site_names)) == 0


def get_robot_landmarks_21(model, data, site_names=None) -> np.ndarray:
    """Read the 21 robot landmark site world positions as ``float64``."""
    _require_mujoco()
    names = _site_names(site_names)
    missing = get_missing_robot_landmark_sites(model, names)
    if missing:
        missing_text = ", ".join(missing)
        raise ValueError(f"Missing robot landmark sites: {missing_text}")

    points = []
    for name in names:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
        points.append(data.site_xpos[site_id].copy())
    arr = np.asarray(points, dtype=np.float64)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected robot landmarks shape (21, 3), got {arr.shape}")
    return arr

