import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.scene_builder import build_task_scene, load_yaml_config


DEFAULT_CONFIG = PROJECT_ROOT / "configs/scenes/pipette_grasp.yaml"


def parse_args():
    parser = argparse.ArgumentParser(description="Build a task MuJoCo scene from a YAML recipe.")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config).expanduser()
    if not config_path.is_absolute():
        config_path = PROJECT_ROOT / config_path
    config = load_yaml_config(config_path)
    output_path = build_task_scene(config)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
