"""Independent Aero Hand teleoperation channel.

This module consumes only wrist-local Quest landmarks. It does not depend on
robot-arm kinematics, Quest world poses, or the robot base frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aero_quest.retargeting import (
    AeroHandRetargetingWrapper,
    apply_hand_grasp_profile,
)
from aero_quest.mujoco_control import write_normalized_aero_action_to_ctrl


SAFE_OPEN_HAND = np.array(
    [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32
)


@dataclass(frozen=True)
class AeroHandTeleopConfig:
    smoothing_alpha: float = 0.25
    disabled: bool = False
    pinch_boost: bool = False
    pinch_closed_m: float = 0.025
    pinch_open_m: float = 0.085
    pinch_boost_blend: float = 1.0
    grasp_profile: str = "none"


@dataclass(frozen=True)
class AeroHandTeleopResult:
    action: np.ndarray
    pinch_distance_m: float
    pinch_strength: float
    pinch_active: bool


def pinch_strength_from_landmarks_wrist(
    landmarks_wrist: np.ndarray,
    closed_m: float,
    open_m: float,
) -> tuple[float, float]:
    """Return metric thumb-index distance and normalized pinch strength."""
    points = np.asarray(landmarks_wrist, dtype=np.float64).reshape(21, 3)
    distance = float(np.linalg.norm(points[4] - points[8]))
    denominator = max(float(open_m) - float(closed_m), 1e-8)
    strength = float(
        np.clip((float(open_m) - distance) / denominator, 0.0, 1.0)
    )
    return distance, strength


def apply_realtime_pinch_boost(
    action: np.ndarray,
    pinch_strength: float,
    blend: float,
) -> np.ndarray:
    """Apply a lightweight thumb/index closure bias."""
    action = np.asarray(action, dtype=np.float32).reshape(7)
    strength = float(np.clip(pinch_strength, 0.0, 1.0))
    minimum = np.asarray([0.20, 0.45, 0.55, 0.75], dtype=np.float32) * strength
    target = action.copy()
    target[:4] = np.maximum(target[:4], minimum)
    blend = float(np.clip(blend, 0.0, 1.0))
    return np.clip(
        (1.0 - blend) * action + blend * target, 0.0, 1.0
    ).astype(np.float32)


class AeroHandTeleopChannel:
    """Stateful wrist-local landmarks -> Aero Hand actuator channel."""

    def __init__(self, config: AeroHandTeleopConfig | None = None):
        self.config = config or AeroHandTeleopConfig()
        self.retargeter = AeroHandRetargetingWrapper(
            self.config.smoothing_alpha,
            disabled=self.config.disabled,
            initial_action=SAFE_OPEN_HAND,
        )
        self.action = SAFE_OPEN_HAND.copy()
        self.last_result = AeroHandTeleopResult(
            action=self.action.copy(),
            pinch_distance_m=float("nan"),
            pinch_strength=0.0,
            pinch_active=False,
        )

    def process(self, landmarks_wrist: np.ndarray) -> AeroHandTeleopResult:
        """Process one Hand Channel frame without using arm/world coordinates."""
        _raw_action, action = self.retargeter(landmarks_wrist)
        distance, strength = pinch_strength_from_landmarks_wrist(
            landmarks_wrist,
            self.config.pinch_closed_m,
            self.config.pinch_open_m,
        )
        pinch_active = bool(
            self.retargeter.last_features.get("pinch_active", False)
        )
        if self.config.pinch_boost and not self.config.disabled:
            action = apply_realtime_pinch_boost(
                action, strength, self.config.pinch_boost_blend
            )
        if self.config.grasp_profile != "none" and not self.config.disabled:
            action = apply_hand_grasp_profile(
                action,
                profile=self.config.grasp_profile,
                pinch_active=pinch_active,
                pinch_strength=strength,
                blend=self.config.pinch_boost_blend,
            )
        self.action = np.asarray(action, dtype=np.float32)
        self.retargeter.prev_action = self.action.copy()
        self.last_result = AeroHandTeleopResult(
            action=self.action.copy(),
            pinch_distance_m=distance,
            pinch_strength=strength,
            pinch_active=pinch_active,
        )
        return self.last_result

    def relax_to_safe_open(self) -> np.ndarray:
        """Move the hand command toward a safe open action after stale input."""
        alpha = float(np.clip(self.config.smoothing_alpha, 0.0, 1.0))
        self.action = (
            alpha * self.action + (1.0 - alpha) * SAFE_OPEN_HAND
        ).astype(np.float32)
        self.retargeter.prev_action = self.action.copy()
        return self.action.copy()

    def apply(self, model, ctrl: np.ndarray) -> np.ndarray:
        """Write the current hand action to Aero Hand actuators only."""
        return write_normalized_aero_action_to_ctrl(
            model, self.action, ctrl=ctrl
        )
