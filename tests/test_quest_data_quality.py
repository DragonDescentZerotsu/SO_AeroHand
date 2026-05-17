import numpy as np

from aero_quest.quest_data_quality import (
    compute_fps_stats,
    compute_frame_intervals,
    detect_position_jumps,
    summarize_quality,
)
from aero_quest.quest_dual_channel import QuestDualChannelFrame


def frame(index: int, timestamp_ns: int, pos=None):
    return QuestDualChannelFrame(
        hand_side="Right",
        recv_ts_ns=timestamp_ns,
        source_ts_ns=None,
        frame_id=index,
        sequence_id=index,
        wrist_pos_world=np.asarray(pos if pos is not None else [0.0, 0.0, 0.0], dtype=float),
        wrist_quat_world=np.array([0.0, 0.0, 0.0, 1.0]),
        landmarks_wrist=np.zeros((21, 3)),
    )


def test_regular_timestamps_fps_and_intervals():
    frames = [frame(i, i * 20_000_000) for i in range(5)]
    np.testing.assert_allclose(compute_frame_intervals(frames), [20.0, 20.0, 20.0, 20.0])
    fps = compute_fps_stats(frames)
    assert abs(fps["average_fps"] - 50.0) < 1e-9
    assert abs(fps["instant_fps_mean"] - 50.0) < 1e-9


def test_position_jump_detected():
    frames = [
        frame(0, 0, [0.0, 0.0, 0.0]),
        frame(1, 20_000_000, [0.01, 0.0, 0.0]),
        frame(2, 40_000_000, [0.50, 0.0, 0.0]),
    ]
    jumps = detect_position_jumps(frames, threshold_m=0.20)
    assert len(jumps) == 1
    assert jumps[0][0] == 2


def test_out_of_order_timestamp_counted():
    frames = [frame(0, 0), frame(1, 30_000_000), frame(2, 20_000_000)]
    summary = summarize_quality(frames)
    assert summary["out_of_order_timestamp_count"] == 1
