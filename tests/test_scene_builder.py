import sys
from pathlib import Path

import mujoco


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.scene_builder import build_task_scene, load_yaml_config


CONFIG_PATH = PROJECT_ROOT / "configs/scenes/pipette_grasp.yaml"


def test_pipette_scene_builds_and_loads():
    config = load_yaml_config(CONFIG_PATH)
    output_path = build_task_scene(config)
    assert output_path.exists()
    model = mujoco.MjModel.from_xml_path(str(output_path))
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pipette_0") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pipette_0_free") >= 0
    assert mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "pipette_0/pipette_button") >= 0


if __name__ == "__main__":
    test_pipette_scene_builds_and_loads()
    print("Scene builder smoke test passed")
