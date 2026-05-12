import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


MODEL_PATH = Path.home() / "Projects/aero_quest_sim/mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml"

ACTION_NAMES = (
    "thumb_abduction",
    "thumb_opposition_or_flexion_1",
    "thumb_curl_or_flexion_2",
    "index_curl",
    "middle_curl",
    "ring_curl",
    "little_curl",
)

AERO_HAND_ACTION_MAP = (
    # action_name, actuator_name, inverted
    # Tendon position actuators close the hand when ctrl moves toward the low end.
    # In this right-hand model, lower CMC abduction ctrl matches the open-thumb
    # side of the Quest thumb/index distance signal.
    ("thumb_abduction", "right_thumb_A_cmc_abd", True),
    ("thumb_opposition_or_flexion_1", "right_th1_A_tendon", True),
    ("thumb_curl_or_flexion_2", "right_th2_A_tendon", True),
    ("index_curl", "right_index_A_tendon", True),
    ("middle_curl", "right_middle_A_tendon", True),
    ("ring_curl", "right_ring_A_tendon", True),
    ("little_curl", "right_pinky_A_tendon", True),
)


def print_actuator_info(model):
    print(f"model.nu = {model.nu}")
    print("Actuator ctrl ranges:")
    for i in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = model.actuator_ctrlrange[i]
        print(f"  [{i:02d}] {name}: ({lo:.6f}, {hi:.6f})")
    print("7D semantic mapping:")
    for action_name, actuator_name, inverted in AERO_HAND_ACTION_MAP:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            print(f"  {action_name:30s} -> MISSING {actuator_name}")
            continue
        if action_name == "thumb_abduction":
            direction = "1=low/open-abducted" if inverted else "1=high/open-abducted"
        else:
            direction = "1=low/closed" if inverted else "1=high/closed"
        print(f"  {action_name:30s} -> ctrl[{actuator_id}] {actuator_name:24s} ({direction})")


def make_demo_action(t):
    """
    Generate normalized [0, 1] 7D action for periodic open-close testing.

    For curl channels: 0 means open, 1 means closed.
    """
    close = 0.5 - 0.5 * np.cos(t)
    return np.array(
        [
            0.45,   # thumb_abduction
            close,  # thumb_opposition_or_flexion_1
            close,  # thumb_curl_or_flexion_2
            close,  # index_curl
            close,  # middle_curl
            close,  # ring_curl
            close,  # little_curl
        ],
        dtype=np.float32,
    )


def map_7d_to_mujoco_ctrl(action_7d, model):
    """
    Map normalized [0, 1] action_7d to MuJoCo ctrl.

    Uses Aero Hand actuator names from right_hand.xml, because the model actuator
    order is index, middle, ring, pinky, thumb_abd, thumb_tendon1, thumb_tendon2.
    TODO: If real hardware uses the opposite convention for a channel, flip that
    channel in AERO_HAND_ACTION_MAP after checking visually.
    """
    action_7d = np.asarray(action_7d, dtype=np.float32)
    if action_7d.shape != (7,):
        raise ValueError(f"Expected action_7d shape (7,), got {action_7d.shape}")

    ctrl = np.zeros(model.nu, dtype=np.float32)
    for i in range(model.nu):
        lo, hi = model.actuator_ctrlrange[i]
        ctrl[i] = 0.5 * (lo + hi)

    for value, (action_name, actuator_name, inverted) in zip(action_7d, AERO_HAND_ACTION_MAP):
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
        if actuator_id < 0:
            raise RuntimeError(f"Actuator not found for {action_name}: {actuator_name}")
        lo, hi = model.actuator_ctrlrange[actuator_id]
        value = float(np.clip(value, 0.0, 1.0))
        if inverted:
            value = 1.0 - value
        ctrl[actuator_id] = lo + value * (hi - lo)
    return ctrl


def main():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    print(f"Loading model: {MODEL_PATH}")
    print_actuator_info(model)

    with mujoco.viewer.launch_passive(model, data) as viewer:
        t0 = time.time()
        while viewer.is_running():
            t = time.time() - t0
            action_7d = make_demo_action(t)
            data.ctrl[:] = map_7d_to_mujoco_ctrl(action_7d, model)
            mujoco.mj_step(model, data)
            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
