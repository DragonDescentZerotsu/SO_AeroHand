import numpy as np

from aero_quest.quest_dual_channel import (
    QuestDualChannelFrame,
    frame_from_json_dict,
    frame_to_json_dict,
    validate_dual_channel_frame,
)


def fake_frame(**kwargs):
    values = dict(
        hand_side="Right",
        recv_ts_ns=1_000_000_000,
        source_ts_ns=900_000_000,
        frame_id=1,
        sequence_id=1,
        wrist_pos_world=np.array([0.1, 0.2, 0.3]),
        wrist_quat_world=np.array([0.0, 0.0, 0.0, 1.0]),
        landmarks_wrist=np.zeros((21, 3)),
    )
    values.update(kwargs)
    return QuestDualChannelFrame(**values)


def test_valid_frame_passes_validation():
    frame = fake_frame()
    assert validate_dual_channel_frame(frame)
    assert frame.valid
    assert frame.quality_flags == {}


def test_json_roundtrip_preserves_shapes_and_values():
    frame = fake_frame(landmarks_wrist=np.arange(63, dtype=float).reshape(21, 3))
    restored = frame_from_json_dict(frame_to_json_dict(frame))
    assert restored.landmarks_wrist.shape == (21, 3)
    assert restored.wrist_pos_world.shape == (3,)
    assert restored.wrist_quat_world.shape == (4,)
    np.testing.assert_allclose(restored.landmarks_wrist, frame.landmarks_wrist)
    np.testing.assert_allclose(restored.wrist_pos_world, frame.wrist_pos_world)
    np.testing.assert_allclose(restored.wrist_quat_world, frame.wrist_quat_world)


def test_invalid_landmark_shape_is_detected():
    frame = fake_frame(landmarks_wrist=np.zeros((20, 3)))
    assert not validate_dual_channel_frame(frame)
    assert "bad_landmarks_wrist_shape" in frame.quality_flags


def test_invalid_quaternion_is_detected():
    frame = fake_frame(wrist_quat_world=np.zeros(4))
    assert not validate_dual_channel_frame(frame)
    assert "bad_wrist_quat_norm" in frame.quality_flags
