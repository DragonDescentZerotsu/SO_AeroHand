"""MuJoCo control helpers for the Aero Hand semantic 7D action."""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None


AERO_ACTION_NAMES = (
    "thumb_abduction",
    "thumb_flexion_1",
    "thumb_flexion_2",
    "index_curl",
    "middle_curl",
    "ring_curl",
    "little_curl",
)

AERO_HAND_ACTION_MAP = (
    # action_name, actuator_name, inverted
    ("thumb_abduction", "right_thumb_A_cmc_abd", True),
    ("thumb_flexion_1", "right_th1_A_tendon", True),
    ("thumb_flexion_2", "right_th2_A_tendon", True),
    ("index_curl", "right_index_A_tendon", True),
    ("middle_curl", "right_middle_A_tendon", True),
    ("ring_curl", "right_ring_A_tendon", True),
    ("little_curl", "right_pinky_A_tendon", True),
)


def require_mujoco() -> None:
    """Raise a clear error if MuJoCo is not importable."""
    if mujoco is None:
        raise RuntimeError("mujoco is required. Install it with: pip install mujoco")


def write_normalized_aero_action_to_ctrl(
    model,
    action: np.ndarray,
    ctrl: np.ndarray,
) -> np.ndarray:
    """Write only Aero Hand actuator targets into an existing ctrl array."""
    require_mujoco()
    action = np.asarray(action, dtype=np.float64)
    if action.shape != (7,):
        raise ValueError(f"Expected semantic Aero action shape (7,), got {action.shape}")
    ctrl = np.asarray(ctrl)
    if ctrl.shape != (model.nu,):
        raise ValueError(f"Expected ctrl shape ({model.nu},), got {ctrl.shape}")

    for value, (action_name, actuator_name, inverted) in zip(action, AERO_HAND_ACTION_MAP):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise RuntimeError(f"Actuator not found for {action_name}: {actuator_name}")
        lo, hi = model.actuator_ctrlrange[actuator_id]
        value = float(np.clip(value, 0.0, 1.0))
        if inverted:
            value = 1.0 - value
        ctrl[actuator_id] = lo + value * (hi - lo)
    return ctrl


def normalized_aero_action_to_ctrl(model, action: np.ndarray) -> np.ndarray:
    """Convert semantic normalized 7D Aero action to a fresh MuJoCo ctrl."""
    ctrl = np.zeros(model.nu, dtype=np.float64)
    for actuator_id in range(model.nu):
        lo, hi = model.actuator_ctrlrange[actuator_id]
        ctrl[actuator_id] = 0.5 * (lo + hi)
    write_normalized_aero_action_to_ctrl(model, action, ctrl)
    return ctrl.astype(np.float32)


def apply_normalized_aero_action(model, data, action: np.ndarray) -> None:
    """Apply semantic normalized 7D Aero action to ``data.ctrl``."""
    data.ctrl[:] = normalized_aero_action_to_ctrl(model, action)


def print_actuator_info(model) -> None:
    """Print actuator ranges and the semantic 7D action mapping."""
    require_mujoco()
    print(f"model.nu = {model.nu}")
    print("Actuator ctrl ranges:")
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        print(actuator_id, name, model.actuator_ctrlrange[actuator_id])
    print("7D semantic mapping:")
    for action_name, actuator_name, inverted in AERO_HAND_ACTION_MAP:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            print(f"  {action_name:16s} -> MISSING {actuator_name}")
            continue
        direction = "inverted" if inverted else "direct"
        print(f"  {action_name:16s} -> ctrl[{actuator_id}] {actuator_name} ({direction})")
