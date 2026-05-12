import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import yaml
except ImportError as exc:
    raise SystemExit("Install required dependency with: pip install pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.envs.so101_mujoco_env import SO101MujocoEnv


DEFAULT_CONFIG = PROJECT_ROOT / "configs/env/so101_mujoco.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Run random normalized actions in the SO101 MuJoCo env.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", default="data/so101_random_rollout.npz")
    parser.add_argument("--action-scale", type=float, default=None)
    return parser.parse_args()


def resolve_path(path):
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def flatten_obs(obs):
    return np.concatenate(
        [
            obs["qpos"].ravel(),
            obs["qvel"].ravel(),
            obs["ee_pos"].ravel(),
            obs["ee_quat"].ravel(),
            obs["gripper"].ravel(),
        ]
    ).astype(np.float32)


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.model_path is not None:
        config["model_path"] = str(resolve_path(args.model_path))
    else:
        config["model_path"] = str(resolve_path(config.get("model_path")))
    if args.action_scale is not None:
        config["action_scale"] = args.action_scale

    env = SO101MujocoEnv(
        model_path=config.get("model_path"),
        control_dt=config.get("control_dt", 0.02),
        physics_dt=config.get("physics_dt"),
        render_width=config.get("render_width", 640),
        render_height=config.get("render_height", 480),
        camera_names=config.get("camera_names") or [],
        action_scale=config.get("action_scale", 1.0),
        episode_len=max(args.steps, int(config.get("episode_len", args.steps))),
        init_qpos=config.get("init_qpos"),
        reward_type=config.get("reward_type", "zero"),
        ee_site_name=config.get("ee_site_name", "gripperframe"),
        print_model_info=True,
    )
    rng = np.random.default_rng(args.seed)
    obs, info = env.reset(seed=args.seed)
    observations = [flatten_obs(obs)]
    actions = []
    rewards = []
    qpos = [obs["qpos"].copy()]
    ee_pos = [obs["ee_pos"].copy()]
    gripper = [obs["gripper"].copy()]

    for step in range(max(0, args.steps)):
        action = rng.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action)
        if not np.all(np.isfinite(flatten_obs(obs))):
            raise RuntimeError(f"Non-finite observation at step {step}")
        observations.append(flatten_obs(obs))
        actions.append(action)
        rewards.append(float(reward))
        qpos.append(obs["qpos"].copy())
        ee_pos.append(obs["ee_pos"].copy())
        gripper.append(obs["gripper"].copy())
        if terminated or truncated:
            break

    output_path = resolve_path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        observations=np.asarray(observations, dtype=np.float32),
        actions=np.asarray(actions, dtype=np.float32),
        rewards=np.asarray(rewards, dtype=np.float32),
        qpos=np.asarray(qpos, dtype=np.float32),
        ee_pos=np.asarray(ee_pos, dtype=np.float32),
        gripper=np.asarray(gripper, dtype=np.float32),
        joint_names=np.asarray(env.control_joint_names),
        actuator_names=np.asarray(env.actuator_names),
    )
    print(f"Saved {len(actions)} SO101 random rollout steps to {output_path}")
    env.close()


if __name__ == "__main__":
    main()
