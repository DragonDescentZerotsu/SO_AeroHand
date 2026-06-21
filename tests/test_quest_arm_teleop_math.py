import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.arm_teleop import QuestArmTeleopController, compute_palm_pose, orientation_error


class FakeRobot:
    """Small robot double for QuestArmTeleopController math tests."""

    def __init__(self, ee_position=None, q=None):
        self.ee_position = np.asarray(ee_position if ee_position is not None else [0.2, 0.0, 0.2], dtype=np.float64)
        self.ee_rotation = np.eye(3, dtype=np.float64)
        self.q = np.asarray(q if q is not None else np.zeros(5), dtype=np.float64)
        self.q_cmd = self.q.copy()
        self.last_target_pos = None
        self.last_target_rot = None

    def get_ee_position(self):
        return self.ee_position.copy()

    def get_ee_rotation(self):
        return self.ee_rotation.copy()

    def get_joint_positions(self):
        return self.q_cmd.copy()

    def get_joint_limits(self):
        return -np.ones_like(self.q) * 2.0, np.ones_like(self.q) * 2.0

    @property
    def joint_names(self):
        return ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]

    def solve_ik(self, target_pos, target_rot=None, q_seed=None):
        self.last_target_pos = np.asarray(target_pos, dtype=np.float64).copy()
        self.last_target_rot = None if target_rot is None else np.asarray(target_rot, dtype=np.float64).copy()
        q = np.asarray(q_seed if q_seed is not None else self.q_cmd, dtype=np.float64).copy()
        q[0:3] = self.last_target_pos
        return q, True

    def set_joint_positions(self, q_cmd):
        self.q_cmd = np.asarray(q_cmd, dtype=np.float64).copy()


def make_pose_points(wrist=(0.0, 0.0, 0.0)):
    P = np.zeros((21, 3), dtype=np.float64)
    P[0] = wrist
    P[5] = np.asarray(wrist, dtype=np.float64) + np.array([1.0, 0.0, 0.0])
    P[9] = np.asarray(wrist, dtype=np.float64) + np.array([0.0, 1.0, 0.0])
    P[17] = np.asarray(wrist, dtype=np.float64) + np.array([-1.0, 0.0, 0.0])
    return P


def make_oriented_pose_points(wrist, x_axis, y_axis):
    wrist = np.asarray(wrist, dtype=np.float64)
    x_axis = np.asarray(x_axis, dtype=np.float64)
    y_axis = np.asarray(y_axis, dtype=np.float64)
    P = np.zeros((21, 3), dtype=np.float64)
    P[0] = wrist
    P[5] = wrist + x_axis
    P[9] = wrist + y_axis
    P[17] = wrist - x_axis
    return P


def rotation_matrix(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis /= np.linalg.norm(axis)
    cross = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return np.eye(3) + np.sin(angle) * cross + (1.0 - np.cos(angle)) * (cross @ cross)


def test_orientation_error_zero():
    np.testing.assert_allclose(orientation_error(np.eye(3), np.eye(3)), np.zeros(3), atol=1e-12)


def test_orientation_error_world_frame_direction_and_magnitude():
    axis = np.array([0.2, -0.4, 0.7], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = np.radians(90.0)
    error = orientation_error(rotation_matrix(axis, angle), np.eye(3))
    np.testing.assert_allclose(error, axis * angle, atol=1e-10)


def test_orientation_error_remains_large_near_180_degrees():
    axis = np.array([0.3, 0.4, -0.5], dtype=np.float64)
    axis /= np.linalg.norm(axis)
    angle = np.radians(179.9)
    error = orientation_error(rotation_matrix(axis, angle), np.eye(3))
    assert np.isclose(np.linalg.norm(error), angle, atol=1e-9)
    assert np.dot(error, axis) > 0.0


def test_orientation_error_uses_shortest_rotation():
    error = orientation_error(rotation_matrix([0.0, 0.0, 1.0], np.radians(181.0)), np.eye(3))
    np.testing.assert_allclose(error, [0.0, 0.0, -np.radians(179.0)], atol=1e-10)


def test_compute_palm_pose():
    R, p = compute_palm_pose(make_pose_points())
    assert R.shape == (3, 3)
    assert p.shape == (3,)
    np.testing.assert_allclose(R[:, 0], [1.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(R[:, 1], [0.0, 1.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(R[:, 2], [0.0, 0.0, 1.0], atol=1e-7)
    np.testing.assert_allclose(R.T @ R, np.eye(3), atol=1e-7)
    assert abs(float(np.linalg.det(R)) - 1.0) < 1e-7


def test_relative_position_mapping():
    robot = FakeRobot(ee_position=[0.2, 0.0, 0.2])
    controller = QuestArmTeleopController(
        scale=0.5,
        R_robot_from_quest=np.eye(3),
        position_alpha=1.0,
        deadzone=0.0,
        max_ee_step=1.0,
        max_joint_step=1.0,
    )
    controller.reset(make_pose_points([0.0, 0.0, 0.0]), robot)
    debug = controller.update(make_pose_points([0.1, 0.0, 0.0]), robot)
    np.testing.assert_allclose(debug["p_ee_target"], [0.25, 0.0, 0.2], atol=1e-7)


def test_deadzone():
    robot = FakeRobot(ee_position=[0.2, 0.0, 0.2])
    controller = QuestArmTeleopController(
        scale=0.5,
        position_alpha=1.0,
        deadzone=0.005,
        max_ee_step=1.0,
        max_joint_step=1.0,
    )
    controller.reset(make_pose_points([0.0, 0.0, 0.0]), robot)
    debug = controller.update(make_pose_points([0.001, 0.0, 0.0]), robot)
    np.testing.assert_allclose(debug["p_ee_target"], [0.2, 0.0, 0.2], atol=1e-7)


def test_step_limit():
    robot = FakeRobot(ee_position=[0.2, 0.0, 0.2])
    controller = QuestArmTeleopController(
        scale=1.0,
        position_alpha=1.0,
        deadzone=0.0,
        max_ee_step=0.02,
        max_joint_step=1.0,
    )
    controller.reset(make_pose_points([0.0, 0.0, 0.0]), robot)
    debug = controller.update(make_pose_points([1.0, 0.0, 0.0]), robot)
    np.testing.assert_allclose(debug["p_ee_target"], [0.22, 0.0, 0.2], atol=1e-7)


def test_hand_frame_alignment():
    robot = FakeRobot(ee_position=[0.2, 0.0, 0.2])
    controller = QuestArmTeleopController(
        scale=0.5,
        position_alpha=1.0,
        deadzone=0.0,
        max_ee_step=1.0,
        max_joint_step=1.0,
        align_to_hand_on_reset=True,
        robot_control_frame=np.eye(3),
    )
    x_axis_world = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    y_axis_world = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    controller.reset(make_oriented_pose_points([0.0, 0.0, 0.0], x_axis_world, y_axis_world), robot)
    moved_wrist = 0.1 * x_axis_world
    debug = controller.update(make_oriented_pose_points(moved_wrist, x_axis_world, y_axis_world), robot)
    np.testing.assert_allclose(debug["delta_p_hand"], [0.0, 0.1, 0.0], atol=1e-7)
    np.testing.assert_allclose(debug["p_ee_target"], [0.25, 0.0, 0.2], atol=1e-7)


def test_incremental_current_hand_rotation_no_drift_then_current_direction():
    robot = FakeRobot(ee_position=[0.2, 0.0, 0.2])
    controller = QuestArmTeleopController(
        scale=0.5,
        position_alpha=1.0,
        deadzone=0.0,
        max_ee_step=1.0,
        max_joint_step=1.0,
        align_to_hand_on_reset=True,
        robot_control_frame=np.eye(3),
        position_mapping_mode="incremental_current_hand",
    )
    controller.reset(
        make_oriented_pose_points([0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
        robot,
    )

    rotated_x = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    rotated_y = np.array([-1.0, 0.0, 0.0], dtype=np.float64)
    debug = controller.update(make_oriented_pose_points([0.0, 0.0, 0.0], rotated_x, rotated_y), robot)
    np.testing.assert_allclose(debug["p_ee_target"], [0.2, 0.0, 0.2], atol=1e-7)

    debug = controller.update(make_oriented_pose_points(0.1 * rotated_x, rotated_x, rotated_y), robot)
    np.testing.assert_allclose(debug["p_ee_target"], [0.25, 0.0, 0.2], atol=1e-7)


def test_relative_orientation_target():
    robot = FakeRobot(ee_position=[0.2, 0.0, 0.2])
    controller = QuestArmTeleopController(
        scale=0.5,
        position_alpha=1.0,
        orientation_alpha=1.0,
        deadzone=0.0,
        max_ee_step=1.0,
        max_joint_step=1.0,
        use_orientation=True,
        position_mapping_mode="anchored_initial_hand",
    )
    P = make_pose_points([0.0, 0.0, 0.0])
    controller.reset(P, robot, hand_rotation=np.eye(3))
    theta = np.pi / 2.0
    Rz = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    controller.update(P, robot, hand_rotation=Rz)
    np.testing.assert_allclose(controller.prev_R_target, Rz, atol=1e-7)


def test_wrist_rotation_does_not_corrupt_position_frame():
    robot = FakeRobot(ee_position=[0.2, 0.0, 0.2])
    controller = QuestArmTeleopController(
        scale=0.5,
        position_alpha=1.0,
        deadzone=0.0,
        max_ee_step=1.0,
        max_joint_step=1.0,
        align_to_hand_on_reset=True,
        robot_control_frame=np.eye(3),
        position_mapping_mode="incremental_current_hand",
    )
    P0 = make_pose_points([0.0, 0.0, 0.0])
    theta = np.pi / 2.0
    Rz = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    controller.reset(P0, robot, hand_rotation=Rz)
    debug = controller.update(make_pose_points([0.1, 0.0, 0.0]), robot, hand_rotation=Rz)
    np.testing.assert_allclose(debug["p_ee_target"], [0.25, 0.0, 0.2], atol=1e-7)


def test_direct_wrist_joint_overlay():
    robot = FakeRobot(ee_position=[0.2, 0.0, 0.2])
    controller = QuestArmTeleopController(
        scale=0.5,
        position_alpha=1.0,
        orientation_alpha=1.0,
        deadzone=0.0,
        max_ee_step=1.0,
        max_joint_step=1.0,
        use_orientation=True,
        direct_wrist_control=True,
        direct_wrist_mapping="rotvec",
        wrist_flex_axis="+z",
        wrist_roll_axis="+x",
        wrist_flex_gain=1.0,
        wrist_roll_gain=1.0,
        position_mapping_mode="anchored_initial_hand",
    )
    P = make_pose_points([0.0, 0.0, 0.0])
    controller.reset(P, robot, hand_rotation=np.eye(3))
    theta = 0.4
    Rz = np.asarray(
        [
            [np.cos(theta), -np.sin(theta), 0.0],
            [np.sin(theta), np.cos(theta), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    debug = controller.update(P, robot, hand_rotation=Rz)
    np.testing.assert_allclose(debug["q_target"][3], theta, atol=1e-6)


def test_direct_wrist_palm_proxy_overlay():
    robot = FakeRobot(ee_position=[0.2, 0.0, 0.2])
    controller = QuestArmTeleopController(
        scale=0.5,
        position_alpha=1.0,
        orientation_alpha=1.0,
        deadzone=0.0,
        max_ee_step=1.0,
        max_joint_step=1.0,
        use_orientation=True,
        direct_wrist_control=True,
        direct_wrist_mapping="palm_proxy",
        wrist_flex_gain=1.0,
        wrist_roll_gain=1.0,
        position_mapping_mode="anchored_initial_hand",
    )
    P = make_pose_points([0.0, 0.0, 0.0])
    controller.reset(P, robot, hand_rotation=np.eye(3))
    R_tilt = np.eye(3)
    R_tilt[2, 1] = -0.25
    R_tilt[1, 1] = np.sqrt(1.0 - 0.25**2)
    R_tilt[0, 2] = 0.4
    debug = controller.update(P, robot, hand_rotation=R_tilt)
    np.testing.assert_allclose(debug["q_target"][3], 0.25, atol=1e-7)
    np.testing.assert_allclose(debug["q_target"][4], 0.4, atol=1e-7)


def main():
    test_compute_palm_pose()
    test_relative_position_mapping()
    test_deadzone()
    test_step_limit()
    test_hand_frame_alignment()
    test_incremental_current_hand_rotation_no_drift_then_current_direction()
    test_relative_orientation_target()
    test_wrist_rotation_does_not_corrupt_position_frame()
    test_direct_wrist_joint_overlay()
    test_direct_wrist_palm_proxy_overlay()
    print("quest arm teleop math tests passed")


if __name__ == "__main__":
    main()
