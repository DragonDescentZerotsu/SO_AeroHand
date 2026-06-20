import numpy as np

from aero_quest.aero_hand_teleop import (
    AeroHandTeleopChannel,
    AeroHandTeleopConfig,
    SAFE_OPEN_HAND,
)
from test_formula_retargeting import make_hand


def test_hand_channel_consumes_only_wrist_local_landmarks():
    channel = AeroHandTeleopChannel(
        AeroHandTeleopConfig(
            smoothing_alpha=0.0,
            grasp_profile="pipette",
        )
    )
    result = channel.process(make_hand())
    assert result.action.shape == (7,)
    assert np.all(np.isfinite(result.action))
    assert np.isfinite(result.pinch_distance_m)


def test_stale_hand_channel_relaxes_to_safe_open():
    channel = AeroHandTeleopChannel(
        AeroHandTeleopConfig(smoothing_alpha=0.5)
    )
    channel.action = np.ones(7, dtype=np.float32)
    relaxed = channel.relax_to_safe_open()
    assert np.linalg.norm(relaxed - SAFE_OPEN_HAND) < np.linalg.norm(
        np.ones(7, dtype=np.float32) - SAFE_OPEN_HAND
    )
