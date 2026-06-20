from __future__ import annotations

import numpy as np
from aero_quest.pinch_state import PinchHysteresis
from aero_quest.mujoco_control import (
    AERO_ACTION_NAMES,
    normalized_aero_action_to_ctrl,
    print_actuator_info,
)


ACTION_NAMES = AERO_ACTION_NAMES

INDEX_IDS = (5, 6, 7, 8)
MIDDLE_IDS = (9, 10, 11, 12)
RING_IDS = (13, 14, 15, 16)
LITTLE_IDS = (17, 18, 19, 20)

def point_to_xyz(point):
    """Convert one SDK/list/array landmark point to xyz floats."""
    if all(hasattr(point, attr) for attr in ("x", "y", "z")):
        return [float(point.x), float(point.y), float(point.z)]
    if isinstance(point, (list, tuple, np.ndarray)) and len(point) >= 3:
        return [float(point[0]), float(point[1]), float(point[2])]
    raise ValueError(f"Unsupported landmark point format: {point!r}")


def as_points_array(points):
    """Convert a 21-point landmark container to a ``(21, 3)`` float array."""
    if hasattr(points, "points"):
        points = points.points
    arr = np.asarray([point_to_xyz(point) for point in points], dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected 21 landmark xyz points with shape (21, 3), got {arr.shape}")
    return arr


def get_quest_points_21(frame):
    """Extract Quest 21 landmark xyz points from a HandFrame-like object."""
    landmarks = getattr(frame, "landmarks", None)
    points = getattr(landmarks, "points", landmarks)
    if points is None:
        raise ValueError("Frame has no landmarks points")
    return as_points_array(points)


def safe_normalize(v: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return a unit vector, or a deterministic zero vector if degenerate."""
    v = np.asarray(v, dtype=np.float32)
    norm = float(np.linalg.norm(v))
    if norm < eps:
        return np.zeros_like(v, dtype=np.float32)
    return v / norm


def palm_localize(points: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Convert 21 landmarks to the palm-local frame with deterministic fallbacks."""
    points = as_points_array(points)
    origin = points[0]
    x_axis = safe_normalize(points[5] - points[17], eps)
    if float(np.linalg.norm(x_axis)) < eps:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float32)

    y_hint = safe_normalize(points[9] - points[0], eps)
    if float(np.linalg.norm(y_hint)) < eps or abs(float(np.dot(x_axis, y_hint))) > 0.98:
        y_hint = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        if abs(float(np.dot(x_axis, y_hint))) > 0.98:
            y_hint = np.array([0.0, 0.0, 1.0], dtype=np.float32)

    z_axis = safe_normalize(np.cross(x_axis, y_hint), eps)
    if float(np.linalg.norm(z_axis)) < eps:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    y_axis = safe_normalize(np.cross(z_axis, x_axis), eps)
    if float(np.linalg.norm(y_axis)) < eps:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    R = np.stack([x_axis, y_axis, z_axis], axis=1)
    scale = float(np.linalg.norm(points[9] - points[0]))
    if scale < eps:
        scale = 1.0
    local = ((points - origin) @ R) / scale
    local = np.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0)
    return local.astype(np.float32)


def angle_between(v1, v2):
    """Return the unsigned angle between two vectors in radians."""
    v1 = np.asarray(v1, dtype=np.float32)
    v2 = np.asarray(v2, dtype=np.float32)
    n1 = float(np.linalg.norm(v1))
    n2 = float(np.linalg.norm(v2))
    if n1 < 1e-8 or n2 < 1e-8:
        return 0.0
    cos_theta = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.arccos(np.clip(cos_theta, -1.0, 1.0)))


def joint_bend(points, a, b, c):
    """Return bend angle ``pi - theta`` at landmark ``b``."""
    theta = angle_between(points[a] - points[b], points[c] - points[b])
    return float(np.pi - theta)


def finger_joint_bends(points, ids):
    """Return proximal and distal bend angles for a four-landmark finger."""
    mcp, pip, dip, tip = ids
    proximal = joint_bend(points, mcp, pip, dip)
    distal = joint_bend(points, pip, dip, tip)
    return proximal, distal


def clamp01(value):
    """Clamp a scalar value into ``[0, 1]``."""
    return float(np.clip(value, 0.0, 1.0))


def normalize_bend(bend, open_angle=0.08, closed_angle=1.35):
    """Normalize a raw bend angle into a clamped curl value."""
    return clamp01((float(bend) - open_angle) / (closed_angle - open_angle))


def thumb_abduction_from_local(points_local):
    """Estimate thumb abduction/adduction from palm-local thumb base geometry."""
    # Use only the thumb base segment for abduction/adduction. The thumb tip
    # moves a lot during plain flexion, so using tip-index distance makes
    # bending look like inward adduction.
    base_lateral = float(points_local[2, 0] - points_local[1, 0])
    proximal_gap = float(points_local[2, 0] - points_local[5, 0])
    base_score = (base_lateral - 0.08) / (0.22 - 0.08)
    gap_score = (proximal_gap - 0.00) / (0.30 - 0.00)
    return clamp01(0.75 * base_score + 0.25 * gap_score)


def quest_points_to_action_7d(points):
    """Convert Quest 21 landmarks to semantic normalized Aero 7D action."""
    local = palm_localize(points)

    thumb_base = joint_bend(local, 0, 1, 2)
    thumb_mid = joint_bend(local, 1, 2, 3)
    thumb_tip = joint_bend(local, 2, 3, 4)
    index_proximal, index_distal = finger_joint_bends(local, INDEX_IDS)
    middle_proximal, middle_distal = finger_joint_bends(local, MIDDLE_IDS)
    ring_proximal, ring_distal = finger_joint_bends(local, RING_IDS)
    little_proximal, little_distal = finger_joint_bends(local, LITTLE_IDS)

    action = np.array(
        [
            thumb_abduction_from_local(local),
            normalize_bend(0.75 * thumb_base + 0.25 * thumb_mid),
            normalize_bend(0.35 * thumb_mid + 0.65 * thumb_tip),
            normalize_bend(0.65 * index_proximal + 0.35 * index_distal),
            normalize_bend(0.65 * middle_proximal + 0.35 * middle_distal),
            normalize_bend(0.65 * ring_proximal + 0.35 * ring_distal),
            normalize_bend(0.65 * little_proximal + 0.35 * little_distal),
        ],
        dtype=np.float32,
    )
    return np.clip(action, 0.0, 1.0).astype(np.float32)


def quest_points_to_hand_features(points):
    """Return interpretable hand features used around the Aero 7D action.

    The action mapping above remains the source of truth for control. These
    features make logging/debugging easier for later shared-autonomy datasets.
    """
    points = as_points_array(points)
    local = palm_localize(points)
    action = quest_points_to_action_7d(points)
    thumb_tip = local[4]
    index_tip = local[8]
    middle_tip = local[12]
    ring_tip = local[16]
    little_tip = local[20]
    pinch_distance = float(np.linalg.norm(thumb_tip - index_tip))
    fingertip_distances = [
        float(np.linalg.norm(index_tip)),
        float(np.linalg.norm(middle_tip)),
        float(np.linalg.norm(ring_tip)),
        float(np.linalg.norm(little_tip)),
    ]
    return {
        "thumb_curl": float(0.5 * (action[1] + action[2])),
        "index_curl": float(action[3]),
        "middle_curl": float(action[4]),
        "ring_curl": float(action[5]),
        "pinky_curl": float(action[6]),
        "thumb_opposition": float(clamp01(1.0 - pinch_distance / 1.2)),
        "pinch_distance": pinch_distance,
        "palm_openness": float(np.clip(np.mean(fingertip_distances) / 1.3, 0.0, 1.0)),
    }


def apply_hand_grasp_profile(
    action_7d: np.ndarray,
    *,
    profile: str = "none",
    pinch_active: bool = False,
    pinch_strength: float = 0.0,
    blend: float = 1.0,
) -> np.ndarray:
    """Apply an optional task-oriented grasp bias to a formula 7D action.

    The ``pipette`` profile keeps thumb/index closure strong while adding a
    modest middle-finger support curl. Hysteretic ``pinch_active`` should come
    from :class:`PinchHysteresis`; ``pinch_strength`` provides proportional
    approach motion before the pinch state latches.
    """
    action = np.asarray(action_7d, dtype=np.float32).reshape(7)
    if profile == "none":
        return action.copy()
    if profile != "pipette":
        raise ValueError(f"Unknown hand grasp profile: {profile!r}")

    strength = float(np.clip(pinch_strength, 0.0, 1.0))
    if pinch_active:
        strength = max(strength, 0.85)
    # [thumb abd, thumb flex 1, thumb flex 2, index, middle, ring, little]
    closure_floor = np.asarray(
        [0.30, 0.68, 0.82, 0.90, 0.48, 0.18, 0.12],
        dtype=np.float32,
    )
    target = np.maximum(action, closure_floor * strength)
    blend = float(np.clip(blend, 0.0, 1.0))
    return np.clip((1.0 - blend) * action + blend * target, 0.0, 1.0).astype(
        np.float32
    )


class GeometricRetargeter:
    """Stateful formula retargeter with exponential action smoothing."""

    def __init__(
        self,
        alpha=0.25,
        initial_action=None,
    ):
        """Create a retargeter with smoothing coefficient ``alpha``."""
        self.alpha = float(np.clip(alpha, 0.0, 1.0))
        if initial_action is None:
            initial_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_action = np.asarray(initial_action, dtype=np.float32)

    def reset(self, action=None):
        """Reset the smoothed previous action."""
        if action is None:
            action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_action = np.asarray(action, dtype=np.float32)

    def __call__(self, points):
        """Return ``(raw_action, filtered_action)`` for one landmark frame."""
        raw_action = quest_points_to_action_7d(points)
        filtered = (1.0 - self.alpha) * self.prev_action + self.alpha * raw_action
        self.prev_action = np.clip(filtered, 0.0, 1.0).astype(np.float32)
        return raw_action, self.prev_action.copy()


def map_7d_to_mujoco_ctrl(action_7d, model):
    """Compatibility wrapper for semantic action to MuJoCo ctrl mapping."""
    return normalized_aero_action_to_ctrl(model, action_7d)


def estimate_palm_pose(landmarks):
    """Estimate Quest palm pose from 21 landmarks.

    Coordinate convention for the returned rotation matrix columns:
    x axis: pinky MCP -> index MCP lateral direction.
    y axis: wrist -> middle MCP forward direction.
    z axis: palm normal from x cross y, re-orthogonalized.
    Position: average of wrist and four MCP landmarks.
    """
    points = as_points_array(landmarks).astype(np.float64)
    wrist = points[0]
    index_mcp = points[5]
    middle_mcp = points[9]
    ring_mcp = points[13]
    pinky_mcp = points[17]
    position = np.mean(np.stack([wrist, index_mcp, middle_mcp, ring_mcp, pinky_mcp], axis=0), axis=0)

    x_axis = safe_normalize(index_mcp - pinky_mcp).astype(np.float64)
    y_hint = safe_normalize(middle_mcp - wrist).astype(np.float64)
    if float(np.linalg.norm(x_axis)) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if float(np.linalg.norm(y_hint)) < 1e-8 or abs(float(np.dot(x_axis, y_hint))) > 0.98:
        y_hint = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    z_axis = safe_normalize(np.cross(x_axis, y_hint)).astype(np.float64)
    if float(np.linalg.norm(z_axis)) < 1e-8:
        z_axis = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    y_axis = safe_normalize(np.cross(z_axis, x_axis)).astype(np.float64)
    if float(np.linalg.norm(y_axis)) < 1e-8:
        y_axis = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    return position.astype(np.float64), rotation.astype(np.float64)


class AeroHandRetargetingWrapper:
    """Wrapper for running existing formula retargeting alongside arm teleop."""

    def __init__(
        self,
        smoothing_alpha=0.25,
        disabled=False,
        initial_action=None,
        pinch_enter_distance=0.30,
        pinch_exit_distance=0.40,
    ):
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.disabled = bool(disabled)
        if initial_action is None:
            initial_action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_action = np.asarray(initial_action, dtype=np.float32)
        self.last_features = {}
        self.pinch_state = PinchHysteresis(
            enter_distance=pinch_enter_distance,
            exit_distance=pinch_exit_distance,
        )

    def reset(self, action=None):
        if action is None:
            action = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
        self.prev_action = np.asarray(action, dtype=np.float32)
        self.pinch_state.reset()

    def __call__(self, landmarks):
        if self.disabled:
            raw_action = self.prev_action.copy()
            self.last_features = {}
        else:
            raw_action = quest_points_to_action_7d(landmarks)
            self.last_features = quest_points_to_hand_features(landmarks)
            self.last_features["pinch_active"] = bool(
                self.pinch_state.update_landmarks(landmarks)
            )
        alpha = self.smoothing_alpha
        filtered = alpha * self.prev_action + (1.0 - alpha) * raw_action
        self.prev_action = np.clip(filtered, 0.0, 1.0).astype(np.float32)
        return raw_action.astype(np.float32), self.prev_action.copy()
