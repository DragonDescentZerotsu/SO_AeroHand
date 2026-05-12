"""Closure feature matching losses for human and robot palm-local landmarks."""

from __future__ import annotations

import numpy as np

from aero_quest.closure_features import extract_closure_features


def closure_matching_loss(
    human_local,
    robot_local,
    w_bend=5.0,
    w_tip=0.5,
    w_dir=1.0,
) -> dict:
    """Compare human and robot palm-local closure features."""
    human = extract_closure_features(human_local)
    robot = extract_closure_features(robot_local)

    bend_diff = robot["bends"] - human["bends"]
    l_bend = float(np.mean(bend_diff * bend_diff))

    tip_diff = robot["tip_positions"] - human["tip_positions"]
    l_tip = float(np.mean(np.sum(tip_diff * tip_diff, axis=1)))

    dir_losses = []
    for robot_dir, human_dir in zip(robot["segment_dirs"], human["segment_dirs"]):
        robot_norm = float(np.linalg.norm(robot_dir))
        human_norm = float(np.linalg.norm(human_dir))
        if robot_norm < 1e-8 or human_norm < 1e-8:
            continue
        dot = float(np.clip(np.dot(robot_dir, human_dir), -1.0, 1.0))
        dir_losses.append(1.0 - dot)
    l_dir = float(np.mean(dir_losses)) if dir_losses else 0.0

    total = float(w_bend * l_bend + w_tip * l_tip + w_dir * l_dir)
    if not np.isfinite(total):
        total = 0.0
    return {
        "total": float(total),
        "bend": float(l_bend if np.isfinite(l_bend) else 0.0),
        "tip": float(l_tip if np.isfinite(l_tip) else 0.0),
        "direction": float(l_dir if np.isfinite(l_dir) else 0.0),
    }

