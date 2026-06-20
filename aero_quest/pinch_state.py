"""Hysteretic pinch detection for stable retargeting task weights."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class PinchHysteresis:
    """Enter below one distance and exit above a larger distance."""

    enter_distance: float = 0.30
    exit_distance: float = 0.40
    active: bool = False

    def __post_init__(self) -> None:
        if self.enter_distance < 0:
            raise ValueError("enter_distance must be non-negative")
        if self.exit_distance <= self.enter_distance:
            raise ValueError("exit_distance must be greater than enter_distance")

    def update_distance(self, distance: float) -> bool:
        distance = float(distance)
        if not np.isfinite(distance):
            return self.active
        if self.active:
            if distance > self.exit_distance:
                self.active = False
        elif distance < self.enter_distance:
            self.active = True
        return self.active

    def update_landmarks(self, landmarks_wrist: np.ndarray) -> bool:
        from aero_quest.retargeting import palm_localize

        points = palm_localize(landmarks_wrist)
        return self.update_distance(float(np.linalg.norm(points[4] - points[8])))

    def reset(self, active: bool = False) -> None:
        self.active = bool(active)
