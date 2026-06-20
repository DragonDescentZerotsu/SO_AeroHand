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
teleop.DEFAULT_KP_ROT = 5.0
teleop.DEFAULT_MAX_LINEAR_SPEED = 0.40
teleop.DEFAULT_MAX_ANGULAR_SPEED = 3.0
teleop.DEFAULT_IK_DAMPING = 0.035
teleop.DEFAULT_MAX_JOINT_SPEED = 5.0
teleop.DEFAULT_IK_SOLVER = "osqp"
teleop.DEFAULT_IK_MODE = "full_pose"
teleop.DEFAULT_ORIENTATION_SOURCE = "wrist_pose"
teleop.DEFAULT_ORIENTATION_WEIGHT = 1.5
teleop.DEFAULT_ROBOT_GRAVITY_ROOT = "base_link"
teleop.DEFAULT_ARM_HOME_QPOS = "0 1.57 -1.3485 0 0 0"
teleop.DEFAULT_JOINT_MOTION_WEIGHTS = "0.7 1.0 1.0 0.35 0.22 0.08"
teleop.DEFAULT_ARM_ACTUATOR_KP = "140 140 140 100 90 90"
teleop.DEFAULT_ARM_ACTUATOR_KV = "10 10 10 7 5 5"
teleop.DEFAULT_QP_TASK_WEIGHTS = "1.0 1.0 1.0 1.2 1.2 1.2"
teleop.DEFAULT_QP_ACCEL_WEIGHT = 0.04
teleop.DEFAULT_QP_MAX_JOINT_ACCEL = 120.0
teleop.DEFAULT_QP_SINGULAR_DAMPING_THRESHOLD = 0.10
teleop.DEFAULT_QP_SINGULAR_DAMPING_GAIN = 0.10


if __name__ == "__main__":
    teleop.main()
