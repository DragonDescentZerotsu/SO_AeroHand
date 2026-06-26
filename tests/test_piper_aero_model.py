import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.mujoco_landmarks import get_missing_robot_landmark_sites, get_robot_landmarks_21
from aero_quest.so101_aero_control import AERO_HAND_ACTION_MAP


MODEL_PATH = PROJECT_ROOT / "models/piper_aero_hand/Piper_aerohand.xml"
BLACK_GRIPPER_MODEL_PATH = PROJECT_ROOT / "models/piper_aero_hand/Piper_original_gripper_black.xml"
PIPER_ARM_ACTUATOR_NAMES = ("joint1", "joint2", "joint3", "joint4", "joint5", "joint6")


def test_piper_aero_model_loads_aligns_and_steps():
    assert MODEL_PATH.exists()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    assert model.nu == 13
    actuator_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in range(model.nu)]
    expected = list(PIPER_ARM_ACTUATOR_NAMES) + [item[1] for item in AERO_HAND_ACTION_MAP]
    assert sorted(actuator_names) == sorted(expected)
    assert "gripper" not in actuator_names

    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link7") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link8") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tetheria_mount") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint7") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint8") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "link6") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "link7") < 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MESH, "link8") < 0
    assert get_missing_robot_landmark_sites(model) == []

    link6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link6")
    palm_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "palm")
    joint6_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "joint6")
    attach_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "piper_aero_attach_site")
    wrist_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "aero_wrist_site")
    assert link6_id >= 0
    assert palm_id >= 0
    assert joint6_id >= 0
    assert attach_id >= 0
    assert wrist_id >= 0
    assert not any(int(model.geom_bodyid[geom_id]) == link6_id for geom_id in range(model.ngeom))

    link6_R = data.xmat[link6_id].reshape(3, 3)
    palm_R = data.xmat[palm_id].reshape(3, 3)
    joint6_axis_world = link6_R @ model.jnt_axis[joint6_id]
    palm_wrist_to_fingers_axis_world = palm_R[:, 2]
    assert np.dot(joint6_axis_world, palm_wrist_to_fingers_axis_world) > 1.0 - 1e-6
    assert np.linalg.norm(data.site_xpos[attach_id] - data.site_xpos[wrist_id]) <= 1e-6
    middle_tip_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "aero_middle_tip_site")
    assert np.dot(data.site_xpos[middle_tip_id] - data.site_xpos[wrist_id], joint6_axis_world) > 0.0

    data.ctrl[:] = 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])
    for _ in range(100):
        mujoco.mj_step(model, data)
        assert np.all(np.isfinite(data.qpos))
        assert np.all(np.isfinite(data.qvel))
    landmarks = get_robot_landmarks_21(model, data)
    assert landmarks.shape == (21, 3)
    assert np.all(np.isfinite(landmarks))


def test_original_piper_gripper_visuals_are_black():
    assert BLACK_GRIPPER_MODEL_PATH.exists()
    model = mujoco.MjModel.from_xml_path(str(BLACK_GRIPPER_MODEL_PATH))
    black_mat_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, "black_mat")
    assert black_mat_id >= 0

    for body_name in ("link6", "link7", "link8"):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        assert body_id >= 0
        visual_geom_ids = [
            geom_id
            for geom_id in range(model.ngeom)
            if int(model.geom_bodyid[geom_id]) == body_id and int(model.geom_group[geom_id]) == 2
        ]
        assert visual_geom_ids
        assert all(int(model.geom_matid[geom_id]) == black_mat_id for geom_id in visual_geom_ids)


if __name__ == "__main__":
    test_piper_aero_model_loads_aligns_and_steps()
    test_original_piper_gripper_visuals_are_black()
    print("Piper + Aero Hand combined model smoke test passed")
