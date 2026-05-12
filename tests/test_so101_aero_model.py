import sys
from pathlib import Path

import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.mujoco_landmarks import get_missing_robot_landmark_sites, get_robot_landmarks_21
from aero_quest.so101_aero_control import (
    AERO_HAND_ACTION_MAP,
    SO101_ARM_ACTUATOR_NAMES,
    apply_so101_aero_action,
    normalized_so101_aero_to_ctrl,
)


MODEL_PATH = PROJECT_ROOT / "mujoco_menagerie/so101_aero_hand/scene.xml"


def test_combined_model_loads_and_steps():
    assert MODEL_PATH.exists()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    assert model.nu == 12
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in range(model.nu)]
    expected = list(SO101_ARM_ACTUATOR_NAMES) + [item[1] for item in AERO_HAND_ACTION_MAP]
    assert sorted(actuator_names) == sorted(expected)
    assert get_missing_robot_landmark_sites(model) == []

    arm_action = np.zeros(5, dtype=np.float32)
    hand_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    ctrl = normalized_so101_aero_to_ctrl(model, arm_action, hand_action)
    assert ctrl.shape == (model.nu,)
    assert np.all(np.isfinite(ctrl))
    assert np.all(ctrl >= model.actuator_ctrlrange[:, 0] - 1e-6)
    assert np.all(ctrl <= model.actuator_ctrlrange[:, 1] + 1e-6)

    for _ in range(100):
        apply_so101_aero_action(model, data, arm_action, hand_action)
        mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))
    landmarks = get_robot_landmarks_21(model, data)
    assert landmarks.shape == (21, 3)
    assert np.all(np.isfinite(landmarks))


if __name__ == "__main__":
    test_combined_model_loads_and_steps()
    print("SO101 + Aero Hand combined model smoke test passed")
