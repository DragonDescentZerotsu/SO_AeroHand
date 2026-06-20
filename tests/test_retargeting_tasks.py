from pathlib import Path

import numpy as np

from aero_quest.pinch_state import PinchHysteresis
from aero_quest.retargeting_tasks import (
    extract_task_vectors,
    load_vector_tasks,
    vector_matching_loss,
)
from aero_quest.retargeting import AeroHandRetargetingWrapper, apply_hand_grasp_profile
from test_formula_retargeting import make_hand


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_vector_config_and_identical_loss():
    tasks, config = load_vector_tasks(
        PROJECT_ROOT / "configs/retargeting/aero_hand_vector.yaml"
    )
    points = make_hand()
    assert len(tasks) == 9
    assert config["retargeting"]["source_frame"] == "wrist"
    assert vector_matching_loss(points, points, tasks)["total"] < 1e-12
    pinched = vector_matching_loss(points, points, tasks, pinch_active=True)
    assert pinched["weights"]["thumb_index"] == 8.0


def test_vectors_are_translation_and_rotation_invariant_after_canonicalization():
    tasks, _ = load_vector_tasks(
        PROJECT_ROOT / "configs/retargeting/aero_hand_vector.yaml"
    )
    points = make_hand()
    angle = 0.7
    rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    transformed = points @ rotation.T + np.array([2.0, -1.0, 0.5])
    original = extract_task_vectors(points, tasks, source="human")
    changed = extract_task_vectors(transformed, tasks, source="human")
    assert np.allclose(original, changed, atol=1e-6)


def test_pinch_hysteresis_does_not_chatter_between_thresholds():
    state = PinchHysteresis(enter_distance=0.3, exit_distance=0.4)
    assert state.update_distance(0.29)
    assert state.update_distance(0.35)
    assert not state.update_distance(0.41)
    assert not state.update_distance(0.35)


def test_formula_wrapper_exposes_stable_pinch_state_without_changing_action_shape():
    wrapper = AeroHandRetargetingWrapper(smoothing_alpha=0.0)
    raw, filtered = wrapper(make_hand())
    assert raw.shape == (7,)
    assert filtered.shape == (7,)
    assert isinstance(wrapper.last_features["pinch_active"], bool)


def test_pipette_profile_strengthens_thumb_index_and_middle_support():
    action = np.zeros(7, dtype=np.float32)
    boosted = apply_hand_grasp_profile(
        action,
        profile="pipette",
        pinch_active=True,
        pinch_strength=0.2,
    )
    assert boosted[1] >= 0.68 * 0.85
    assert boosted[2] >= 0.82 * 0.85
    assert boosted[3] >= 0.90 * 0.85
    assert boosted[4] >= 0.48 * 0.85
    assert boosted[5] < boosted[4]
    assert np.allclose(
        apply_hand_grasp_profile(action, profile="none"), action
    )
