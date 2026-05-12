import argparse
import sys
import time
from pathlib import Path

try:
    import mujoco
    import mujoco.viewer
    import numpy as np
except ImportError as exc:
    raise SystemExit("Install required dependency with: pip install mujoco numpy") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.so101_aero_control import apply_so101_aero_action, print_combined_actuator_info


DEFAULT_MODEL = PROJECT_ROOT / "mujoco_menagerie/so101_aero_hand/scene.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="View the combined SO101 arm + Aero Hand MuJoCo model.")
    parser.add_argument("--model-path", default=str(DEFAULT_MODEL))
    parser.add_argument("--random-actions", action="store_true")
    parser.add_argument("--print-actuators", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = Path(args.model_path).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    if args.print_actuators:
        print_combined_actuator_info(model)

    rng = np.random.default_rng(0)
    arm_action = np.zeros(5, dtype=np.float32)
    hand_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    print(f"Loaded combined model: {model_path}")
    print("Close the viewer or press Ctrl+C to exit.")
    with mujoco.viewer.launch_passive(model, data) as viewer:
        step = 0
        while viewer.is_running():
            if args.random_actions and step % 80 == 0:
                arm_action = rng.uniform(-0.5, 0.5, size=5).astype(np.float32)
                hand_action = rng.uniform(0.0, 1.0, size=7).astype(np.float32)
            apply_so101_aero_action(model, data, arm_action, hand_action)
            mujoco.mj_step(model, data)
            viewer.sync()
            step += 1
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
