import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
    import yaml
except ImportError as exc:
    raise SystemExit("Install required dependencies with: pip install mujoco pyyaml") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.envs.so101_mujoco_env import SO101MujocoEnv


DEFAULT_CONFIG = PROJECT_ROOT / "configs/env/so101_mujoco.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Open a MuJoCo viewer for SO101.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--print-model-info", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--random-actions", action="store_true")
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_model_path(path):
    if path is None:
        return None
    path = Path(path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path


def main():
    args = parse_args()
    config = load_config(args.config)
    if args.model_path is not None:
        config["model_path"] = str(resolve_model_path(args.model_path))
    else:
        config["model_path"] = str(resolve_model_path(config.get("model_path")))

    env = SO101MujocoEnv(
        model_path=config.get("model_path"),
        control_dt=config.get("control_dt", 0.02),
        physics_dt=config.get("physics_dt"),
        render_width=config.get("render_width", 640),
        render_height=config.get("render_height", 480),
        camera_names=config.get("camera_names") or [],
        action_scale=config.get("action_scale", 1.0),
        episode_len=config.get("episode_len", 200),
        init_qpos=config.get("init_qpos"),
        reward_type=config.get("reward_type", "zero"),
        ee_site_name=config.get("ee_site_name", "gripperframe"),
        print_model_info=args.print_model_info,
    )
    env.reset()
    print("Viewer controls the same MuJoCo model used by SO101MujocoEnv.")
    print("Press Ctrl+C or close the viewer to exit.")

    with mujoco.viewer.launch_passive(env.model, env.data) as viewer:
        rng = np.random.default_rng(0)
        action = np.zeros(env.model.nu, dtype=np.float32)
        while viewer.is_running():
            if args.random_actions and env._step_count % 50 == 0:
                action = rng.uniform(-1.0, 1.0, size=env.model.nu).astype(np.float32)
            env.step(action)
            viewer.sync()
            time.sleep(env.model.opt.timestep)
    env.close()


if __name__ == "__main__":
    main()
