import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.retargeting import (
    AeroHandRetargetingWrapper,
    estimate_palm_pose,
    quest_points_to_action_7d,
)


def make_fake_open_hand(curl: float = 0.0) -> np.ndarray:
    """Create simple Quest-like 21 landmarks with controllable finger curl."""
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = [0.0, 0.0, 0.0]
    mcp_x = [-0.045, -0.015, 0.015, 0.045]
    finger_ids = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    for x, ids in zip(mcp_x, finger_ids):
        for k, landmark_id in enumerate(ids):
            forward = 0.035 + 0.025 * k
            down = -curl * 0.02 * max(0, k - 1)
            points[landmark_id] = [x, forward - curl * 0.01 * k, down]
    points[1] = [-0.06, 0.015, -0.005]
    points[2] = [-0.075, 0.035, -0.01 - curl * 0.005]
    points[3] = [-0.085, 0.055 - curl * 0.01, -0.015 - curl * 0.015]
    points[4] = [-0.095, 0.075 - curl * 0.025, -0.020 - curl * 0.025]
    return points


def main():
    retargeter = AeroHandRetargetingWrapper(smoothing_alpha=0.25)
    for curl in (0.0, 0.5, 1.0):
        points = make_fake_open_hand(curl)
        raw = quest_points_to_action_7d(points)
        raw2, filtered = retargeter(points)
        palm_pos, palm_R = estimate_palm_pose(points)
        print(
            f"curl={curl:.1f} "
            f"raw={np.array2string(raw, precision=3, suppress_small=True)} "
            f"wrapper_raw={np.array2string(raw2, precision=3, suppress_small=True)} "
            f"filtered={np.array2string(filtered, precision=3, suppress_small=True)} "
            f"in_range={bool(np.all(raw >= 0.0) and np.all(raw <= 1.0))} "
            f"pinch={retargeter.last_features.get('pinch_distance', 0.0):.3f} "
            f"openness={retargeter.last_features.get('palm_openness', 0.0):.3f} "
            f"palm_pos={np.array2string(palm_pos, precision=3, suppress_small=True)} "
            f"detR={np.linalg.det(palm_R):.3f}"
        )


if __name__ == "__main__":
    main()
