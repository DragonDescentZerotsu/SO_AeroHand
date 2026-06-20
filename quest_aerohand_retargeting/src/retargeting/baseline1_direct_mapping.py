from __future__ import annotations

import numpy as np


class DirectPoseMappingRetargeter:
    """Baseline 1: direct Quest landmarks/features to compact AeroHand action.

    When full 21-point wrist-local Quest landmarks are available, this baseline
    uses the existing production geometric retargeter in ``aero_quest``. The
    compact keypoint heuristic remains as a fallback for minimal mock frames.
    """

    action_names = (
        "thumb_abduction",
        "thumb_flexion_1",
        "thumb_flexion_2",
        "index_curl",
        "middle_curl",
        "ring_curl",
        "pinky_curl",
    )

    def retarget(
        self,
        keypoints: dict[str, np.ndarray],
        pinch: dict,
        landmarks_wrist: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a 7D normalized AeroHand action.

        Prefer the main repo's 21-landmark formula retargeting path. Fall back
        to compact pinch/curl proxies only when full landmarks are absent.
        """
        if landmarks_wrist is not None:
            return _retarget_full_landmarks(landmarks_wrist)
        strength = float(pinch["pinch_strength"])
        middle_curl = _tip_curl_proxy(keypoints, "middle_tip")
        ring_curl = _tip_curl_proxy(keypoints, "ring_tip")
        pinky_curl = _tip_curl_proxy(keypoints, "pinky_tip")
        action = np.array(
            [strength, 0.8 * strength, strength, strength, middle_curl, ring_curl, pinky_curl],
            dtype=np.float64,
        )
        return np.clip(action, 0.0, 1.0)


class PinchAugmentedDirectRetargeter(DirectPoseMappingRetargeter):
    """Baseline 1b: direct mapping plus explicit thumb-index pinch correction."""

    def __init__(
        self,
        thumb_abduction_min: float = 0.20,
        thumb_flexion_1_min: float = 0.45,
        thumb_flexion_2_min: float = 0.55,
        index_curl_min: float = 0.75,
        blend: float = 1.0,
    ):
        self.thumb_abduction_min = float(thumb_abduction_min)
        self.thumb_flexion_1_min = float(thumb_flexion_1_min)
        self.thumb_flexion_2_min = float(thumb_flexion_2_min)
        self.index_curl_min = float(index_curl_min)
        self.blend = float(np.clip(blend, 0.0, 1.0))

    def retarget(
        self,
        keypoints: dict[str, np.ndarray],
        pinch: dict,
        landmarks_wrist: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return Baseline 1 action with a direct pinch-strength correction.

        This does not use MuJoCo feedback yet. It is a simple feature baseline:
        when the human thumb-index distance implies a pinch, enforce minimum
        thumb/index closure action values before evaluating in MuJoCo.
        """
        base = super().retarget(keypoints, pinch, landmarks_wrist=landmarks_wrist)
        strength = float(np.clip(pinch.get("pinch_strength", 0.0), 0.0, 1.0))
        target = base.copy()
        target[0] = max(target[0], self.thumb_abduction_min * strength)
        target[1] = max(target[1], self.thumb_flexion_1_min * strength)
        target[2] = max(target[2], self.thumb_flexion_2_min * strength)
        target[3] = max(target[3], self.index_curl_min * strength)
        corrected = (1.0 - self.blend) * base + self.blend * target
        return np.clip(corrected, 0.0, 1.0)


def _retarget_full_landmarks(landmarks_wrist: np.ndarray) -> np.ndarray:
    """Run the existing AeroHand 7D geometric retargeter on 21 landmarks."""
    from aero_quest.retargeting import quest_points_to_action_7d

    action = quest_points_to_action_7d(np.asarray(landmarks_wrist, dtype=np.float64).reshape(21, 3))
    return np.asarray(action, dtype=np.float64)


def _tip_curl_proxy(keypoints: dict[str, np.ndarray], tip_name: str) -> float:
    """Estimate finger curl from fingertip forward distance as a placeholder."""
    y = float(np.asarray(keypoints[tip_name], dtype=np.float64)[1])
    return float(np.clip(1.0 - y / 0.11, 0.0, 1.0))
