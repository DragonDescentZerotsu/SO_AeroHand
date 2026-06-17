"""Quest teleoperation entry point for the Piper + Aero Hand model."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.teleop import quest_so101_aero_nullspace_ik_teleop as teleop


teleop.DEFAULT_DESCRIPTION = "Quest wrist pose -> full-pose Piper IK, landmarks -> Aero Hand."
teleop.DEFAULT_MODEL = PROJECT_ROOT / "models/piper_aero_hand/Piper_aerohand.xml"
teleop.DEFAULT_ARM_JOINTS = "joint1,joint2,joint3,joint4,joint5,joint6"
teleop.DEFAULT_EE_SITE = "aero_wrist_site"
teleop.DEFAULT_SCALE = 0.8
teleop.DEFAULT_WORKSPACE_MIN = ["-0.10", "-0.45", "0.02"]
teleop.DEFAULT_WORKSPACE_MAX = ["0.65", "0.45", "0.75"]
teleop.DEFAULT_KP_POS = 8.0
teleop.DEFAULT_KP_ROT = 2.0
teleop.DEFAULT_MAX_LINEAR_SPEED = 0.40
teleop.DEFAULT_MAX_ANGULAR_SPEED = 1.2
teleop.DEFAULT_MAX_JOINT_SPEED = 2.5
teleop.DEFAULT_IK_MODE = "full_pose"
teleop.DEFAULT_ROBOT_GRAVITY_ROOT = "base_link"


if __name__ == "__main__":
    teleop.main()
