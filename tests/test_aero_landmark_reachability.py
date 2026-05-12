import argparse
import sys
from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError as exc:
    raise SystemExit("mujoco is required. Install it with: pip install mujoco") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.closure_features import BEND_NAMES, TIP_INDICES, extract_closure_features
from aero_quest.mujoco_control import apply_normalized_aero_action
from aero_quest.mujoco_landmarks import get_missing_robot_landmark_sites, get_robot_landmarks_21
from aero_quest.retargeting import palm_localize


DEFAULT_XML = PROJECT_ROOT / "mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="Diagnose Aero Hand landmark reachability.")
    parser.add_argument("--xml", default=str(DEFAULT_XML))
    parser.add_argument("--num-samples", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--settle-steps", type=int, default=50)
    return parser.parse_args()


def main():
    args = parse_args()
    rng = np.random.default_rng(args.seed)
    model = mujoco.MjModel.from_xml_path(str(Path(args.xml).expanduser()))
    missing = get_missing_robot_landmark_sites(model)
    if missing:
        print("Missing robot landmark sites:")
        for name in missing:
            print(f"  - {name}")
        raise SystemExit(1)

    bends = []
    tip_wrist_distances = []
    thumb_tip_distances = []
    min_thumb_pinky = float("inf")
    min_thumb_pinky_action = None

    for _ in range(max(0, args.num_samples)):
        data = mujoco.MjData(model)
        action = rng.random(7)
        apply_normalized_aero_action(model, data, action)
        for _step in range(max(1, args.settle_steps)):
            mujoco.mj_step(model, data)
        robot_world = get_robot_landmarks_21(model, data)
        robot_local = palm_localize(robot_world)
        features = extract_closure_features(robot_local)
        bends.append(features["bends"])
        tips = robot_local[TIP_INDICES]
        tip_wrist_distances.append(np.linalg.norm(tips - robot_local[0], axis=1))
        thumb_to_fingers = np.linalg.norm(tips[1:] - tips[0], axis=1)
        thumb_tip_distances.append(thumb_to_fingers)
        if float(thumb_to_fingers[-1]) < min_thumb_pinky:
            min_thumb_pinky = float(thumb_to_fingers[-1])
            min_thumb_pinky_action = action.copy()

    bends = np.asarray(bends, dtype=np.float64)
    tip_wrist_distances = np.asarray(tip_wrist_distances, dtype=np.float64)
    thumb_tip_distances = np.asarray(thumb_tip_distances, dtype=np.float64)

    print(f"samples={len(bends)} xml={args.xml} settle_steps={args.settle_steps}")
    for idx, name in enumerate(BEND_NAMES):
        print(f"bend {name}: min={bends[:, idx].min():.6f} max={bends[:, idx].max():.6f}")
    for idx, name in enumerate(["thumb", "index", "middle", "ring", "little"]):
        print(
            f"tip_to_wrist {name}: "
            f"min={tip_wrist_distances[:, idx].min():.6f} "
            f"max={tip_wrist_distances[:, idx].max():.6f}"
        )
    for idx, name in enumerate(["index", "middle", "ring", "little"]):
        print(
            f"thumb_tip_to_{name}_tip: "
            f"min={thumb_tip_distances[:, idx].min():.6f} "
            f"max={thumb_tip_distances[:, idx].max():.6f}"
        )
    print(f"thumb-pinky min distance={min_thumb_pinky:.6f}")
    print(f"thumb-pinky min action={np.array2string(min_thumb_pinky_action, precision=4)}")


if __name__ == "__main__":
    main()

