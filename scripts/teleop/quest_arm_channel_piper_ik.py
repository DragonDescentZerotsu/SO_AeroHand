"""Arm-only Quest teleoperation entry point for the 6-DoF Piper."""

from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.teleop import quest_arm_channel_so101_ik as teleop


teleop.DEFAULT_DESCRIPTION = "Quest wrist pose -> Piper 6-DoF full-pose IK."
teleop.DEFAULT_ROBOT_NAME = "Piper"
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
teleop.DEFAULT_IK_SOLVER = "osqp"
teleop.DEFAULT_ARM_HOME_QPOS = "0 1.57 -1.3485 0 0 0"
teleop.DEFAULT_SHOW_TARGET = False


if __name__ == "__main__":
    teleop.main()
