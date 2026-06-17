"""Control helpers for the combined SO101 arm + Aero Hand MuJoCo model."""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


SO101_ARM_ACTUATOR_NAMES = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
)

AERO_HAND_SEMANTIC_NAMES = (
    "thumb_abduction",
    "thumb_flexion_1",
    "thumb_flexion_2",
    "index_curl",
    "middle_curl",
    "ring_curl",
    "little_curl",
)

AERO_HAND_ACTION_MAP = (
    ("thumb_abduction", "right_thumb_A_cmc_abd", True),
    ("thumb_flexion_1", "right_th1_A_tendon", True),
    ("thumb_flexion_2", "right_th2_A_tendon", True),
    ("index_curl", "right_index_A_tendon", True),
    ("middle_curl", "right_middle_A_tendon", True),
    ("ring_curl", "right_ring_A_tendon", True),
    ("little_curl", "right_pinky_A_tendon", True),
)


def _require_mujoco() -> None:
    if mujoco is None:
        raise RuntimeError("mujoco is required. Install it with: pip install mujoco")


def actuator_id(model, name: str) -> int:
    """Return actuator id or raise a clear error."""
    _require_mujoco()
    idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if idx < 0:
        raise RuntimeError(f"Combined SO101+Aero actuator not found: {name}")
    return int(idx)


def ctrl_midpoints(model) -> np.ndarray:
    """Return midpoint ctrl for every actuator."""
    ctrl = np.zeros(model.nu, dtype=np.float64)
    for idx in range(model.nu):
        lo, hi = model.actuator_ctrlrange[idx]
        ctrl[idx] = 0.5 * (lo + hi)
    return ctrl


def normalized_so101_arm_to_ctrl(model, arm_action: np.ndarray, ctrl: np.ndarray | None = None) -> np.ndarray:
    """Map normalized SO101 arm action ``[-1, 1]^5`` to actuator ctrl."""
    arm_action = np.asarray(arm_action, dtype=np.float64)
    if arm_action.shape != (len(SO101_ARM_ACTUATOR_NAMES),):
        raise ValueError(f"Expected SO101 arm action shape ({len(SO101_ARM_ACTUATOR_NAMES)},), got {arm_action.shape}")
    if ctrl is None:
        ctrl = ctrl_midpoints(model)
    for value, name in zip(arm_action, SO101_ARM_ACTUATOR_NAMES):
        idx = actuator_id(model, name)
        lo, hi = model.actuator_ctrlrange[idx]
        value = float(np.clip(value, -1.0, 1.0))
        ctrl[idx] = 0.5 * (lo + hi) + value * 0.5 * (hi - lo)
    return ctrl


def normalized_aero_hand_to_ctrl(model, hand_action: np.ndarray, ctrl: np.ndarray | None = None) -> np.ndarray:
    """Map semantic Aero Hand action ``[0, 1]^7`` to combined model ctrl."""
    hand_action = np.asarray(hand_action, dtype=np.float64)
    if hand_action.shape != (len(AERO_HAND_ACTION_MAP),):
        raise ValueError(f"Expected Aero hand action shape ({len(AERO_HAND_ACTION_MAP)},), got {hand_action.shape}")
    if ctrl is None:
        ctrl = ctrl_midpoints(model)
    for value, (_semantic_name, actuator_name, inverted) in zip(hand_action, AERO_HAND_ACTION_MAP):
        idx = actuator_id(model, actuator_name)
        lo, hi = model.actuator_ctrlrange[idx]
        value = float(np.clip(value, 0.0, 1.0))
        if inverted:
            value = 1.0 - value
        ctrl[idx] = lo + value * (hi - lo)
    return ctrl


def normalized_so101_aero_to_ctrl(model, arm_action: np.ndarray, hand_action: np.ndarray) -> np.ndarray:
    """Return combined ctrl for SO101 arm ``[-1,1]^5`` and Aero hand ``[0,1]^7``."""
    ctrl = ctrl_midpoints(model)
    normalized_so101_arm_to_ctrl(model, arm_action, ctrl=ctrl)
    normalized_aero_hand_to_ctrl(model, hand_action, ctrl=ctrl)
    return ctrl.astype(np.float32)


def apply_so101_aero_action(model, data, arm_action: np.ndarray, hand_action: np.ndarray) -> None:
    """Apply normalized SO101 arm and Aero Hand actions to ``data.ctrl``."""
    data.ctrl[:] = normalized_so101_aero_to_ctrl(model, arm_action, hand_action)


def print_combined_actuator_info(model, arm_actuator_names: list[str] | tuple[str, ...] | None = None) -> None:
    """Print combined model actuator order and semantic mapping."""
    _require_mujoco()
    arm_actuator_names = tuple(SO101_ARM_ACTUATOR_NAMES if arm_actuator_names is None else arm_actuator_names)
    print(f"model.nu = {model.nu}")
    print("Actuators:")
    for idx in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx)
        print(idx, name, model.actuator_ctrlrange[idx])
    print("Robot arm action order [-1, 1]:")
    for idx, name in enumerate(arm_actuator_names):
        print(f"  arm[{idx}] -> {name}")
    print("Aero hand action order [0, 1]:")
    for idx, (semantic, actuator_name, inverted) in enumerate(AERO_HAND_ACTION_MAP):
        direction = "inverted" if inverted else "direct"
        print(f"  hand[{idx}] {semantic} -> {actuator_name} ({direction})")
