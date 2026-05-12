import sys
from pathlib import Path

import numpy as np
import mujoco

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.closure_features import extract_closure_features
from aero_quest.closure_loss import closure_matching_loss
from aero_quest.mujoco_landmarks import get_robot_landmarks_21
from aero_quest.quality_filter import build_quality_mask
from aero_quest.retargeting import palm_localize
from test_formula_retargeting import make_hand


def assert_no_nan(arr):
    assert np.all(np.isfinite(arr)), "array contains NaN or inf"


def test_palm_localize():
    points = make_hand()
    local = palm_localize(points)
    assert local.shape == (21, 3)
    assert np.linalg.norm(local[0]) < 1e-6
    assert_no_nan(local)


def test_closure_features_shapes():
    local = palm_localize(make_hand())
    features = extract_closure_features(local)
    assert features["bends"].shape == (11,)
    assert features["tip_positions"].shape == (5, 3)
    assert features["segment_dirs"].shape == (15, 3)
    assert features["flat"].shape == (11 + 15 + 45,)
    for value in features.values():
        assert_no_nan(value)


def test_closure_loss_identical_and_changed():
    human = palm_localize(make_hand())
    same = closure_matching_loss(human, human)
    changed = palm_localize(make_hand(index=1.1))
    different = closure_matching_loss(human, changed)
    assert same["total"] < 1e-9
    assert different["total"] > same["total"]


def test_mujoco_landmark_missing_error_message():
    xml = "<mujoco><worldbody><body name='b'><site name='present'/></body></worldbody></mujoco>"
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    try:
        get_robot_landmarks_21(model, data, site_names=["missing_a", "missing_b"])
    except ValueError as exc:
        message = str(exc)
        assert "missing_a" in message
        assert "missing_b" in message
    else:
        raise AssertionError("Expected ValueError for missing sites")


def test_quality_filter_rejects_bad_frames():
    good = make_hand()
    nan_frame = good.copy()
    nan_frame[4, 0] = np.nan
    jump_frame = good.copy()
    jump_frame[8] += np.array([1.0, 0.0, 0.0], dtype=np.float32)
    tiny_frame = good * 0.01
    landmarks = np.stack([good, good.copy(), nan_frame, jump_frame, tiny_frame], axis=0)
    keep, reasons, summary = build_quality_mask(landmarks, max_palm_scale=2.0, max_point_step=0.2)
    assert keep.shape == (5,)
    assert reasons.shape == (5,)
    assert summary["kept"] >= 2
    assert not keep[2]
    assert not keep[3]
    assert not keep[4]
    assert reasons[2] == "non_finite"


if __name__ == "__main__":
    test_palm_localize()
    test_closure_features_shapes()
    test_closure_loss_identical_and_changed()
    test_mujoco_landmark_missing_error_message()
    test_quality_filter_rejects_bad_frames()
    print("geometry module tests passed")
