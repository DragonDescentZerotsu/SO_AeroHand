import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("Install required dependency with: pip install mujoco") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_MODEL = PROJECT_ROOT / "mujoco_menagerie/so101_aero_hand/scene.xml"


def resolve_model(path_text: str) -> Path:
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {path}")
    return path


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal MuJoCo load/create-data/step smoke test without viewer.")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--steps", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    model_path = resolve_model(args.model)
    print(f"model={model_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    print(f"loaded=true nq={model.nq} nv={model.nv} nu={model.nu} timestep={model.opt.timestep}")
    print(f"qpos_shape={data.qpos.shape} qvel_shape={data.qvel.shape} ctrl_shape={data.ctrl.shape}")

    if model.nu > 0:
        data.ctrl[:] = 0.0
        print("ctrl_zero_written=true")
    else:
        print("ctrl_zero_written=false reason=no_actuators")

    mujoco.mj_forward(model, data)
    for _ in range(int(args.steps)):
        mujoco.mj_step(model, data)

    print(f"stepped={int(args.steps)}")
    print(f"qpos_finite={bool(np.all(np.isfinite(data.qpos)))} qvel_finite={bool(np.all(np.isfinite(data.qvel)))}")
    print(f"qpos_min={float(np.min(data.qpos)):.6f} qpos_max={float(np.max(data.qpos)):.6f}")
    print(f"qvel_min={float(np.min(data.qvel)):.6f} qvel_max={float(np.max(data.qvel)):.6f}")


if __name__ == "__main__":
    main()
