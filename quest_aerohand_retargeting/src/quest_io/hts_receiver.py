from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np


KEYPOINT_NAMES = (
    "thumb_tip",
    "thumb_ip",
    "thumb_mcp",
    "index_tip",
    "index_pip",
    "index_mcp",
    "middle_tip",
    "ring_tip",
    "pinky_tip",
)


@dataclass
class HandFrame:
    """Minimal Quest hand frame used by the placeholder retargeting pipeline.

    Coordinates are intentionally treated as raw Quest/HTS hand data here.
    Later modules must explicitly normalize to wrist frame or robot frame.
    """

    timestamp: float
    wrist_pose: np.ndarray
    thumb_tip: np.ndarray
    thumb_ip: np.ndarray
    thumb_mcp: np.ndarray
    index_tip: np.ndarray
    index_pip: np.ndarray
    index_mcp: np.ndarray
    middle_tip: np.ndarray
    ring_tip: np.ndarray
    pinky_tip: np.ndarray
    landmarks_wrist: np.ndarray | None = None

    def keypoints(self) -> dict[str, np.ndarray]:
        """Return named hand keypoints as 3D numpy arrays."""
        return {name: np.asarray(getattr(self, name), dtype=np.float64).reshape(3) for name in KEYPOINT_NAMES}

    def to_dict(self) -> dict:
        """Serialize the frame to JSON-compatible primitives."""
        data = {"timestamp": float(self.timestamp), "wrist_pose": np.asarray(self.wrist_pose).tolist()}
        data.update({name: np.asarray(getattr(self, name), dtype=np.float64).tolist() for name in KEYPOINT_NAMES})
        if self.landmarks_wrist is not None:
            data["landmarks_wrist"] = np.asarray(self.landmarks_wrist, dtype=np.float64).reshape(21, 3).tolist()
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "HandFrame":
        """Create a frame from a dictionary produced by :meth:`to_dict`."""
        kwargs = {"timestamp": float(data["timestamp"]), "wrist_pose": np.asarray(data["wrist_pose"], dtype=np.float64)}
        for name in KEYPOINT_NAMES:
            kwargs[name] = np.asarray(data[name], dtype=np.float64)
        if "landmarks_wrist" in data:
            kwargs["landmarks_wrist"] = np.asarray(data["landmarks_wrist"], dtype=np.float64).reshape(21, 3)
        return cls(**kwargs)


class MockHTSReceiver:
    """Generate a simple thumb-index pinch trajectory without a Quest device."""

    def __init__(self, num_frames: int = 120, dt: float = 0.02):
        self.num_frames = int(num_frames)
        self.dt = float(dt)

    def iter_frames(self) -> Iterator[HandFrame]:
        """Yield mock hand frames that close and reopen a thumb-index pinch."""
        for i in range(self.num_frames):
            phase = 0.5 - 0.5 * np.cos(2.0 * np.pi * i / max(self.num_frames - 1, 1))
            pinch_gap = 0.085 - 0.060 * phase
            index_y = 0.090 - 0.055 * phase
            wrist = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0], dtype=np.float64)
            thumb_tip = np.array([-0.5 * pinch_gap, 0.035, 0.020], dtype=np.float64)
            index_tip = np.array([0.5 * pinch_gap, index_y, 0.020], dtype=np.float64)
            keypoints = {
                "thumb_tip": thumb_tip,
                "thumb_ip": np.array([-0.035, 0.025, 0.010], dtype=np.float64),
                "thumb_mcp": np.array([-0.055, 0.000, 0.000], dtype=np.float64),
                "index_tip": index_tip,
                "index_pip": np.array([0.025, 0.060, 0.012], dtype=np.float64),
                "index_mcp": np.array([0.030, 0.020, 0.000], dtype=np.float64),
                "middle_tip": np.array([0.000, 0.105, 0.015], dtype=np.float64),
                "ring_tip": np.array([-0.025, 0.095, 0.010], dtype=np.float64),
                "pinky_tip": np.array([-0.050, 0.080, 0.005], dtype=np.float64),
            }
            yield HandFrame(
                timestamp=i * self.dt,
                wrist_pose=wrist,
                landmarks_wrist=_mock_landmarks_wrist(keypoints),
                **keypoints,
            )


def _mock_landmarks_wrist(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    """Build Quest-like 21 wrist-local landmarks from the compact mock fields."""
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    points[1] = keypoints["thumb_mcp"]
    points[2] = keypoints["thumb_ip"]
    points[3] = 0.45 * keypoints["thumb_ip"] + 0.55 * keypoints["thumb_tip"]
    points[4] = keypoints["thumb_tip"]
    points[5] = keypoints["index_mcp"]
    points[6] = keypoints["index_pip"]
    points[7] = 0.45 * keypoints["index_pip"] + 0.55 * keypoints["index_tip"]
    points[8] = keypoints["index_tip"]

    finger_specs = [
        (9, np.array([0.000, 0.025, 0.000]), keypoints["middle_tip"]),
        (13, np.array([-0.025, 0.020, 0.000]), keypoints["ring_tip"]),
        (17, np.array([-0.050, 0.015, 0.000]), keypoints["pinky_tip"]),
    ]
    for start, mcp, tip in finger_specs:
        points[start] = mcp
        points[start + 1] = 0.45 * mcp + 0.55 * tip
        points[start + 2] = 0.20 * mcp + 0.80 * tip
        points[start + 3] = tip
    return points
