from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from sim.aerohand_env import AeroHandSimEnv
from utils.config import load_config


def main() -> None:
    """Run one placeholder AeroHand simulation step."""
    cfg = load_config(PROJECT_DIR / "configs/default.yaml")
    env = AeroHandSimEnv(**cfg["sim"])
    env.reset()
    obs = env.step(np.array([1.0, 0.8, 0.8, 1.0, 0.1, 0.1, 0.1]))
    print(obs)


if __name__ == "__main__":
    main()
