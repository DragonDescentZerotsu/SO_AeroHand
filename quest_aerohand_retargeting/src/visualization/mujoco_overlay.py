from __future__ import annotations


def update_overlay(viewer, human_keypoints: dict, target_keypoints: dict, actual_keypoints: dict) -> None:
    """Update human, retarget target, and actual AeroHand overlay markers.

    TODO: Connect to MuJoCo viewer marker APIs when the real env is attached.
    """
    del viewer, human_keypoints, target_keypoints, actual_keypoints

