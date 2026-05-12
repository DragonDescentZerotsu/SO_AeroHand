import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.retargeting import AeroHandRetargetingWrapper, estimate_palm_pose, quest_points_to_action_7d


def make_fake_landmarks(curl: float = 0.45) -> np.ndarray:
    """Return a deterministic Quest-like right-hand landmark array, shape (21, 3)."""
    curl = float(np.clip(curl, 0.0, 1.0))
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = [0.0, 0.0, 0.0]

    finger_ids = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    mcp_x = [-0.045, -0.015, 0.015, 0.045]
    segment_y = [0.035, 0.065, 0.092, 0.116]
    for x, ids in zip(mcp_x, finger_ids):
        for k, landmark_id in enumerate(ids):
            bend = curl * max(0, k - 1)
            points[landmark_id] = [x, segment_y[k] - 0.015 * curl * k, -0.025 * bend]

    points[1] = [-0.055, 0.018, -0.004]
    points[2] = [-0.075, 0.038, -0.009 - 0.004 * curl]
    points[3] = [-0.090, 0.060 - 0.016 * curl, -0.014 - 0.018 * curl]
    points[4] = [-0.103, 0.082 - 0.032 * curl, -0.019 - 0.032 * curl]
    return points


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal fake-landmark test for existing Aero Hand retargeting.")
    parser.add_argument("--curl", type=float, default=0.45, help="Fake finger curl in [0, 1].")
    parser.add_argument("--alpha", type=float, default=0.25, help="Wrapper smoothing alpha.")
    return parser.parse_args()


def main():
    args = parse_args()
    landmarks = make_fake_landmarks(args.curl)
    print(f"fake_landmarks_shape={landmarks.shape} dtype={landmarks.dtype}")
    print(f"fake_landmarks_nan={bool(np.isnan(landmarks).any())}")

    raw_action = quest_points_to_action_7d(landmarks)
    retargeter = AeroHandRetargetingWrapper(smoothing_alpha=args.alpha)
    wrapper_raw, filtered_action = retargeter(landmarks)
    palm_position, palm_rotation = estimate_palm_pose(landmarks)

    for label, action in (("raw_action", raw_action), ("wrapper_raw", wrapper_raw), ("filtered_action", filtered_action)):
        print(f"{label}={np.array2string(action, precision=6, suppress_small=True)}")
        print(
            f"{label}_shape={action.shape} "
            f"min={float(np.min(action)):.6f} max={float(np.max(action)):.6f} "
            f"mean={float(np.mean(action)):.6f} has_nan={bool(np.isnan(action).any())}"
        )

    print(f"palm_position={np.array2string(palm_position, precision=6, suppress_small=True)}")
    print(f"palm_rotation_det={float(np.linalg.det(palm_rotation)):.6f}")
    print(f"features={retargeter.last_features}")


if __name__ == "__main__":
    main()
