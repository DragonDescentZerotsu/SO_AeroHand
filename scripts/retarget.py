from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.retargeting import (  # noqa: F401
    ACTION_NAMES,
    GeometricRetargeter,
    angle_between,
    as_points_array,
    clamp01,
    finger_joint_bends,
    get_quest_points_21,
    joint_bend,
    map_7d_to_mujoco_ctrl,
    normalize,
    normalize_bend,
    points_to_palm_local,
    print_actuator_info,
    quest_points_to_action_7d,
)


class HandRetargeter:
    """Compatibility wrapper for older scripts that import scripts.retarget."""

    def __init__(self, ema_alpha=0.25, debug_interval=0.5, **_kwargs):
        del debug_interval
        self.retargeter = GeometricRetargeter(alpha=ema_alpha)

    def neutral_action(self):
        return self.retargeter.prev_action.copy()

    def __call__(self, landmarks):
        _, filtered_action = self.retargeter(landmarks)
        return filtered_action

