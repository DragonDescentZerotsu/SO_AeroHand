import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.quest_hand_frame import (
    QuestPacketParseError,
    RelativeWristArmController,
    convert_landmarks_wrist_to_world,
    palm_frame_from_landmarks_wrist,
    palm_frame_from_quest_frame_world,
    parse_quest_hand_frame,
    parse_quest_hand_frames,
)


def landmark_values():
    return [0.001 * i for i in range(21 * 3)]


def palm_landmark_values(angle=0.0):
    points = np.zeros((21, 3), dtype=np.float64)
    points[5] = [1.0, 0.0, 0.0]
    points[9] = [0.0, 1.0, 0.0]
    points[17] = [-1.0, 0.0, 0.0]
    c = float(np.cos(angle))
    s = float(np.sin(angle))
    Rz = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)
    return (points @ Rz.T).reshape(-1).tolist()


def packet(side="Right", debug=False, values=None, wrist_quat=None):
    if values is None:
        values = landmark_values()
    if wrist_quat is None:
        wrist_quat = [0.0, 0.0, 0.0, 1.0]
    meta = " | f = 123 | t = 123456789012345" if debug else ""
    wrist = f"{side} wrist{meta}:, 1.0, 2.0, 3.0, " + ", ".join(str(v) for v in wrist_quat)
    landmarks = f"{side} landmarks{meta}:, " + ", ".join(str(v) for v in values)
    return wrist + "\n" + landmarks


def test_parse_normal_packet():
    frame = parse_quest_hand_frame(packet("Right"))
    assert frame.hand_side == "Right"
    assert frame.timestamp_ns is None
    assert frame.frame_id is None
    assert frame.wrist_pos_world.shape == (3,)
    assert frame.wrist_quat_world.shape == (4,)
    assert frame.landmarks_wrist.shape == (21, 3)
    np.testing.assert_allclose(frame.wrist_pos_world, [1.0, 2.0, 3.0])


def test_parse_debug_packet_metadata():
    frame = parse_quest_hand_frame(packet("Right", debug=True))
    assert frame.frame_id == 123
    assert frame.timestamp_ns == 123456789012345
    assert frame.landmarks_wrist.shape == (21, 3)


def test_parse_left_and_right_packet():
    frames = parse_quest_hand_frames(packet("Left") + "\n" + packet("Right"))
    assert {frame.hand_side for frame in frames} == {"Left", "Right"}
    left = parse_quest_hand_frame(packet("Left") + "\n" + packet("Right"), hand_side="left")
    assert left.hand_side == "Left"


def test_malformed_packet_is_clear_error():
    try:
        parse_quest_hand_frame("Right nope:, 1, 2, 3")
    except QuestPacketParseError as exc:
        assert "Unrecognized" in str(exc)
    else:
        raise AssertionError("Expected QuestPacketParseError")


def test_missing_landmark_values_are_rejected():
    try:
        parse_quest_hand_frame(packet("Right", values=landmark_values()[:-1]))
    except QuestPacketParseError as exc:
        assert "landmarks expected 63 floats" in str(exc)
    else:
        raise AssertionError("Expected QuestPacketParseError")


def test_convert_landmarks_wrist_to_world_identity_quat():
    landmarks_wrist = np.zeros((21, 3), dtype=np.float64)
    landmarks_wrist[8] = [0.1, 0.2, 0.3]
    world = convert_landmarks_wrist_to_world([1.0, 2.0, 3.0], [0.0, 0.0, 0.0, 1.0], landmarks_wrist)
    np.testing.assert_allclose(world[0], [1.0, 2.0, 3.0])
    np.testing.assert_allclose(world[8], [1.1, 2.2, 3.3])


def test_relative_wrist_arm_controller_position_only():
    frame = parse_quest_hand_frame(packet("Right"))
    controller = RelativeWristArmController(scale=2.0, R_BQ=np.eye(3))
    controller.set_teleop_zero(frame.wrist_pos_world, frame.wrist_quat_world, ee_pos_B=np.array([0.5, 0.0, 0.0]))
    moved = parse_quest_hand_frame(packet("Right").replace("1.0, 2.0, 3.0", "1.1, 2.0, 2.9", 1))
    target = controller.compute_target(moved)
    np.testing.assert_allclose(target.delta_p_Q, [0.1, 0.0, -0.1])
    np.testing.assert_allclose(target.target_pos_B, [0.7, 0.0, -0.2])


def test_relative_wrist_arm_controller_orientation_target():
    frame = parse_quest_hand_frame(packet("Right"))
    controller = RelativeWristArmController(
        scale=1.0,
        R_BQ=np.eye(3),
        control_orientation=True,
        orientation_source="wrist_pose",
    )
    controller.set_teleop_zero(
        frame.wrist_pos_world,
        frame.wrist_quat_world,
        ee_pos_B=np.array([0.0, 0.0, 0.0]),
        ee_R_B=np.eye(3),
    )
    s = np.sqrt(0.5)
    moved = parse_quest_hand_frame(packet("Right", wrist_quat=[0.0, 0.0, s, s]))
    target = controller.compute_target(moved)
    expected_Rz_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(target.target_R_B, expected_Rz_90, atol=1e-7)


def test_relative_wrist_arm_controller_palm_landmark_orientation_target():
    initial = parse_quest_hand_frame(packet("Right", values=palm_landmark_values(0.0)))
    controller = RelativeWristArmController(scale=1.0, R_BQ=np.eye(3), control_orientation=True)
    controller.set_teleop_zero(
        initial.wrist_pos_world,
        initial.wrist_quat_world,
        ee_pos_B=np.array([0.0, 0.0, 0.0]),
        ee_R_B=np.eye(3),
        landmarks_wrist=initial.landmarks_wrist,
    )
    moved = parse_quest_hand_frame(packet("Right", values=palm_landmark_values(np.pi / 2.0)))
    target = controller.compute_target(moved)
    expected_Rz_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(palm_frame_from_landmarks_wrist(initial.landmarks_wrist), np.eye(3), atol=1e-7)
    np.testing.assert_allclose(target.target_R_B, expected_Rz_90, atol=1e-7)


def test_palm_frame_is_expressed_in_quest_world():
    s = np.sqrt(0.5)
    frame = parse_quest_hand_frame(
        packet(
            "Right",
            values=palm_landmark_values(0.0),
            wrist_quat=[0.0, 0.0, s, s],
        )
    )
    expected_Rz_90 = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    np.testing.assert_allclose(palm_frame_from_quest_frame_world(frame), expected_Rz_90, atol=1e-7)


if __name__ == "__main__":
    test_parse_normal_packet()
    test_parse_debug_packet_metadata()
    test_parse_left_and_right_packet()
    test_malformed_packet_is_clear_error()
    test_missing_landmark_values_are_rejected()
    test_convert_landmarks_wrist_to_world_identity_quat()
    test_relative_wrist_arm_controller_position_only()
    test_relative_wrist_arm_controller_orientation_target()
    test_relative_wrist_arm_controller_palm_landmark_orientation_target()
    test_palm_frame_is_expressed_in_quest_world()
    print("quest hand frame tests passed")
