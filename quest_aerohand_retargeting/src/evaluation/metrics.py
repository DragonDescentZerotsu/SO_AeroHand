from __future__ import annotations

import numpy as np


def summarize_metrics(records: list[dict], pinch_success_threshold_m: float = 0.03) -> dict[str, float]:
    """Summarize placeholder retargeting metrics over a trajectory."""
    if not records:
        return {}
    human = np.asarray([r["human_pinch_distance"] for r in records], dtype=np.float64)
    robot = np.asarray([r["robot_pinch_distance"] for r in records], dtype=np.float64)
    actions = np.asarray([r["action"] for r in records], dtype=np.float64)
    contact = np.asarray([bool(r["contact"]) for r in records], dtype=bool)
    latency = np.asarray([r.get("latency_s", 0.0) for r in records], dtype=np.float64)
    diffs = np.diff(actions, axis=0) if len(actions) > 1 else np.zeros_like(actions)
    pinch_success = robot <= pinch_success_threshold_m
    contact_given_pinch_success = float(np.mean(contact[pinch_success])) if np.any(pinch_success) else 0.0
    return {
        "human_pinch_distance_mean": float(np.mean(human)),
        "robot_thumb_index_distance_mean": float(np.mean(robot)),
        "robot_thumb_index_distance_median": float(np.median(robot)),
        "robot_thumb_index_distance_min": float(np.min(robot)),
        "pinch_distance_mae": float(np.mean(np.abs(robot - human))),
        "final_fingertip_error": float(abs(human[-1] - robot[-1])),
        "pinch_success_rate": float(np.mean(pinch_success)),
        "contact_success_rate": float(np.mean(contact)),
        "contact_given_pinch_success_rate": contact_given_pinch_success,
        "pinch_success_without_contact_rate": float(np.mean(pinch_success & ~contact)),
        "object_slip_rate": 0.0,
        "control_smoothness": float(np.mean(np.linalg.norm(diffs, axis=1))) if len(diffs) else 0.0,
        "teleoperation_latency_mean_s": float(np.mean(latency)),
    }
