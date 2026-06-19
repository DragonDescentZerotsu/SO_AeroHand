"""Quest teleoperation entry point for the SO101 + Aero Hand model."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.teleop import quest_aero_arm_ik_teleop as teleop


teleop.DEFAULT_DESCRIPTION = (
    "Quest wrist pose -> position-priority SO101 IK with nullspace orientation, "
    "wrist-local landmarks -> Aero Hand."
)
teleop.DEFAULT_MODEL = PROJECT_ROOT / "models/so101_aero_hand/SO101_aerohand.xml"
teleop.DEFAULT_ARM_JOINTS = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll"
teleop.DEFAULT_EE_SITE = "aero_wrist_site"
teleop.DEFAULT_SCALE = 0.9
teleop.DEFAULT_WORKSPACE_MIN = ["0.05", "-0.35", "0.03"]
teleop.DEFAULT_WORKSPACE_MAX = ["0.55", "0.35", "1.35"]
teleop.DEFAULT_KP_POS = 10.0
teleop.DEFAULT_KP_ROT = 1.2
teleop.DEFAULT_MAX_LINEAR_SPEED = 0.45
teleop.DEFAULT_MAX_ANGULAR_SPEED = 0.8
teleop.DEFAULT_IK_DAMPING = 0.05
teleop.DEFAULT_MAX_JOINT_SPEED = 3.0
teleop.DEFAULT_ORIENTATION_SOURCE = "palm_landmarks"
teleop.DEFAULT_ORIENTATION_WEIGHT = 1.0
teleop.DEFAULT_IK_MODE = "position_nullspace"
teleop.DEFAULT_ROBOT_GRAVITY_ROOT = "base"
teleop.DEFAULT_INITIAL_ARM_QPOS = None
teleop.DEFAULT_JOINT_MOTION_WEIGHTS = None
teleop.DEFAULT_ARM_ACTUATOR_KP = None
teleop.DEFAULT_ARM_ACTUATOR_KV = None


if __name__ == "__main__":
    teleop.main()
