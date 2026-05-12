import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.envs.so101_mujoco_env import SO101MujocoEnv, default_so101_model_path


def assert_obs_finite(obs):
    for key, value in obs.items():
        assert np.all(np.isfinite(value)), f"{key} contains NaN or inf"


def test_so101_xml_exists():
    assert default_so101_model_path().exists()


def test_so101_env_reset_step_and_rollout():
    env = SO101MujocoEnv(print_model_info=False, episode_len=200)
    try:
        obs, info = env.reset(seed=0)
        assert set(["qpos", "qvel", "ee_pos", "ee_quat", "gripper"]).issubset(obs)
        assert obs["qpos"].shape == (env.model.nu,)
        assert obs["qvel"].shape == (env.model.nu,)
        assert env.action_space.shape == (env.model.nu,)
        assert_obs_finite(obs)

        rng = np.random.default_rng(1)
        for _ in range(100):
            action = rng.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32)
            next_obs, reward, terminated, truncated, info = env.step(action)
            assert next_obs["qpos"].shape == obs["qpos"].shape
            assert next_obs["qvel"].shape == obs["qvel"].shape
            assert np.isfinite(reward)
            assert not terminated
            assert_obs_finite(next_obs)
            obs = next_obs
    finally:
        env.close()


if __name__ == "__main__":
    test_so101_xml_exists()
    test_so101_env_reset_step_and_rollout()
    print("SO101 MuJoCo env smoke tests passed")
