"""Quest dual-channel frame abstraction.

The Quest stream is intentionally mixed-frame:

* Arm channel: wrist pose in the Quest/Unity world frame.
* Hand channel: 21 landmarks in the wrist/root-local hand frame.

Do not use ``landmarks_wrist`` as world or robot end-effector coordinates.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


LANDMARK_COUNT = 21


@dataclass
class QuestDualChannelFrame:
    hand_side: str
    recv_ts_ns: int | None
    source_ts_ns: int | None
    frame_id: int | str | None
    sequence_id: int | None

    # Arm channel: Quest/Unity world frame.
    wrist_pos_world: np.ndarray
    wrist_quat_world: np.ndarray  # xyzw

    # Hand channel: wrist/root-local frame.
    landmarks_wrist: np.ndarray

    valid: bool = True
    quality_flags: dict[str, Any] = field(default_factory=dict)


def normalize_quat_xyzw(q: Any) -> np.ndarray:
    quat = np.asarray(q, dtype=np.float64)
    if quat.shape != (4,):
        raise ValueError(f"Quaternion must have shape (4,), got {quat.shape}")
    if not np.all(np.isfinite(quat)):
        raise ValueError("Quaternion contains non-finite values")
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("Quaternion norm is near zero")
    if norm > 10.0:
        raise ValueError(f"Quaternion norm is unreasonable: {norm:.6g}")
    return quat / norm


def validate_dual_channel_frame(frame: QuestDualChannelFrame) -> bool:
    """Validate frame arrays and metadata, mutating ``valid`` and flags."""
    flags: dict[str, Any] = dict(frame.quality_flags)

    wrist_pos = np.asarray(frame.wrist_pos_world, dtype=np.float64)
    wrist_quat = np.asarray(frame.wrist_quat_world, dtype=np.float64)
    landmarks = np.asarray(frame.landmarks_wrist, dtype=np.float64)

    if wrist_pos.shape != (3,):
        flags["bad_wrist_pos_shape"] = tuple(wrist_pos.shape)
    elif not np.all(np.isfinite(wrist_pos)):
        flags["nonfinite_wrist_pos_world"] = True

    if wrist_quat.shape != (4,):
        flags["bad_wrist_quat_shape"] = tuple(wrist_quat.shape)
    elif not np.all(np.isfinite(wrist_quat)):
        flags["nonfinite_wrist_quat_world"] = True
    else:
        quat_norm = float(np.linalg.norm(wrist_quat))
        if quat_norm < 1e-8 or quat_norm > 10.0:
            flags["bad_wrist_quat_norm"] = quat_norm
        elif not 0.25 <= quat_norm <= 4.0:
            flags["suspicious_wrist_quat_norm"] = quat_norm

    if landmarks.shape != (LANDMARK_COUNT, 3):
        flags["bad_landmarks_wrist_shape"] = tuple(landmarks.shape)
    elif not np.all(np.isfinite(landmarks)):
        flags["nonfinite_landmarks_wrist"] = True

    for name in ("recv_ts_ns", "source_ts_ns", "sequence_id"):
        value = getattr(frame, name)
        if value is not None and int(value) < 0:
            flags[f"negative_{name}"] = int(value)

    frame.quality_flags = flags
    frame.valid = not _has_error_flags(flags)
    return frame.valid


def frame_to_json_dict(frame: QuestDualChannelFrame) -> dict[str, Any]:
    return {
        "hand_side": frame.hand_side,
        "recv_ts_ns": frame.recv_ts_ns,
        "source_ts_ns": frame.source_ts_ns,
        "frame_id": frame.frame_id,
        "sequence_id": frame.sequence_id,
        "wrist_pos_world": np.asarray(frame.wrist_pos_world, dtype=np.float64).tolist(),
        "wrist_quat_world": np.asarray(frame.wrist_quat_world, dtype=np.float64).tolist(),
        "landmarks_wrist": np.asarray(frame.landmarks_wrist, dtype=np.float64).tolist(),
        "valid": bool(frame.valid),
        "quality_flags": dict(frame.quality_flags),
    }


def frame_from_json_dict(d: dict[str, Any]) -> QuestDualChannelFrame:
    frame = QuestDualChannelFrame(
        hand_side=str(d["hand_side"]),
        recv_ts_ns=_optional_int(d.get("recv_ts_ns")),
        source_ts_ns=_optional_int(d.get("source_ts_ns")),
        frame_id=d.get("frame_id"),
        sequence_id=_optional_int(d.get("sequence_id")),
        wrist_pos_world=np.asarray(d["wrist_pos_world"], dtype=np.float64),
        wrist_quat_world=np.asarray(d["wrist_quat_world"], dtype=np.float64),
        landmarks_wrist=np.asarray(d["landmarks_wrist"], dtype=np.float64),
        valid=bool(d.get("valid", True)),
        quality_flags=dict(d.get("quality_flags", {})),
    )
    validate_dual_channel_frame(frame)
    if "valid" in d and not bool(d["valid"]):
        frame.valid = False
    return frame


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _has_error_flags(flags: dict[str, Any]) -> bool:
    return any(
        key.startswith(("bad_", "nonfinite_", "negative_")) or key == "conversion_error"
        for key in flags
    )
