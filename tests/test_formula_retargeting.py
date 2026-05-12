import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.retargeting import ACTION_NAMES, GeometricRetargeter, quest_points_to_action_7d


def make_finger(base, curl=0.0, lengths=(0.35, 0.25, 0.2)):
    base = np.asarray(base, dtype=np.float32)
    points = [base]
    angle = 0.0
    for length in lengths:
        angle += float(curl)
        step = np.array([0.0, length * np.cos(angle), -length * np.sin(angle)], dtype=np.float32)
        points.append(points[-1] + step)
    return points


def make_thumb(curl=0.0, abducted=True):
    base = np.array([0.55, 0.35, 0.0], dtype=np.float32)
    direction = np.array([0.75, 0.25, 0.0], dtype=np.float32)
    if not abducted:
        base = np.array([0.28, 0.65, 0.0], dtype=np.float32)
        direction = np.array([0.12, 0.35, 0.0], dtype=np.float32)
    direction = direction / np.linalg.norm(direction)
    points = [base]
    angle = 0.0
    for length in (0.25, 0.2, 0.18):
        angle += float(curl)
        step = length * direction + np.array([0.0, 0.0, -0.25 * np.sin(angle)], dtype=np.float32)
        points.append(points[-1] + step.astype(np.float32))
    return points


def make_hand(index=0.0, middle=0.0, ring=0.0, little=0.0, thumb=0.0, thumb_abducted=True):
    points = [np.array([0.0, 0.0, 0.0], dtype=np.float32)]
    points.extend(make_thumb(curl=thumb, abducted=thumb_abducted))
    points.extend(make_finger([0.35, 1.0, 0.0], curl=index))
    points.extend(make_finger([0.0, 1.1, 0.0], curl=middle))
    points.extend(make_finger([-0.32, 1.0, 0.0], curl=ring))
    points.extend(make_finger([-0.58, 0.85, 0.0], curl=little))
    return np.asarray(points, dtype=np.float32)


def print_action(name, action):
    text = " ".join(f"{action_name}={value:.3f}" for action_name, value in zip(ACTION_NAMES, action))
    print(f"{name}: {text}")


def main():
    open_hand = make_hand(thumb_abducted=True)
    thumb_flex_only = make_hand(thumb=1.0, thumb_abducted=True)
    fist = make_hand(index=1.1, middle=1.1, ring=1.1, little=1.1, thumb=0.9, thumb_abducted=False)
    index_only = make_hand(index=1.1, thumb_abducted=True)

    open_action = quest_points_to_action_7d(open_hand)
    thumb_flex_action = quest_points_to_action_7d(thumb_flex_only)
    fist_action = quest_points_to_action_7d(fist)
    index_action = quest_points_to_action_7d(index_only)

    print_action("open", open_action)
    print_action("thumb_flex_only", thumb_flex_action)
    print_action("fist", fist_action)
    print_action("index_only", index_action)

    assert open_action[3] < 0.2, "open index should be open"
    assert thumb_flex_action[0] > 0.8, "plain thumb flexion should not look like adduction"
    assert thumb_flex_action[1] > open_action[1], "plain thumb flexion should increase thumb flexion"
    assert fist_action[0] < open_action[0], "fist thumb should be less abducted than open thumb"
    assert fist_action[3] > 0.6, "fist index should curl"
    assert fist_action[4] > 0.6, "fist middle should curl"
    assert index_action[3] > 0.6, "index-only index should curl"
    assert index_action[4] < 0.2, "index-only middle should stay open"

    retargeter = GeometricRetargeter(alpha=0.5)
    _, first = retargeter(open_hand)
    _, second = retargeter(fist)
    assert np.all(second >= 0.0) and np.all(second <= 1.0), "filtered action must stay clamped"
    assert second[3] > first[3], "smoothing should still move index curl toward fist"
    print("synthetic formula retargeting tests passed")


if __name__ == "__main__":
    main()
