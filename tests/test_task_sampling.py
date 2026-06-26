import sys
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.scene_builder import build_task_scene, load_yaml_config
from aero_tasks.task_sampling import (
    RackBarSampleConfig,
    apply_episode_spec_to_model,
    apply_episode_spec_to_qpos,
    sample_pipette_rack_bar_episode,
)


PIPER_DUAL_RACK_TABLE_CONFIG_PATH = PROJECT_ROOT / "configs/scenes/piper_dual_pipette_rack_table.yaml"


def normalize(vector: np.ndarray) -> np.ndarray:
    return vector / np.linalg.norm(vector)


def test_rack_bar_sampling_uses_bar_center_and_table_centered_rack_pose():
    output_path = build_task_scene(load_yaml_config(PIPER_DUAL_RACK_TABLE_CONFIG_PATH))
    model = mujoco.MjModel.from_xml_path(str(output_path))
    config = RackBarSampleConfig(
        offset_range_m=(0.0, 0.0),
        sample_rack_pose=True,
        rack_center_xy_m=(0.1, 0.05),
        rack_x_range_m=(0.02, 0.02),
        rack_y_range_m=(0.01, 0.01),
        rack_yaw_range_deg=(10.0, 10.0),
    )

    spec = sample_pipette_rack_bar_episode(
        model,
        rng=np.random.default_rng(123),
        seed=123,
        config=config,
    )
    assert config.rack_body in spec.body_poses
    assert config.object_freejoint in spec.freejoint_poses
    assert spec.metadata["offset_reference"] == "rack_bar_center"
    assert spec.metadata["rack_center_xy_m"] == [0.1, 0.05]

    apply_episode_spec_to_model(model, spec)
    qpos = model.qpos0.copy()
    apply_episode_spec_to_qpos(model, qpos, spec)
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    rack_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.rack_body)
    pipette_root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pipette_0")
    assert rack_body >= 0
    assert pipette_root >= 0

    assert np.allclose(data.xpos[rack_body][:2], np.array([0.12, 0.06]), atol=1e-8)

    rack_R = data.xmat[rack_body].reshape(3, 3)
    object_local = rack_R.T @ (data.xpos[pipette_root] - data.xpos[rack_body])
    axis_local = normalize(np.asarray(config.axis_local, dtype=np.float64))
    bar_center_local = np.asarray(config.bar_center_local, dtype=np.float64)
    offset_from_center = float(np.dot(object_local - bar_center_local, axis_local))
    assert abs(offset_from_center) < 1e-8
