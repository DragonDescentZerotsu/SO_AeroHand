"""Typed Quest hand stream frames and two-channel helpers.

The Quest packet is intentionally mixed-frame:

* wrist/root pose is in Q, the Quest/Unity world tracking frame.
* landmarks are in Wrist, the local wrist/root hand frame.

Do not treat ``landmarks_wrist`` as robot/world coordinates unless they have
been explicitly converted for visualization or debugging.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Optional

import numpy as np


LANDMARK_COUNT = 21
_HEADER_RE = re.compile(
    r"^\s*(?P<side>left|right)\s+(?P<kind>wrist|landmarks)"
    r"(?P<meta>.*?)\s*:\s*(?P<values>.*?)\s*$",
    re.IGNORECASE,
)
_FRAME_RE = re.compile(r"\bf\s*=\s*(\d+)", re.IGNORECASE)
_TIME_RE = re.compile(r"\bt\s*=\s*(\d+)", re.IGNORECASE)


class QuestPacketParseError(ValueError):
    """Raised when a Quest hand packet is malformed or incomplete."""


@dataclass
class QuestHandFrame:
    hand_side: str
    timestamp_ns: Optional[int]
    frame_id: Optional[int]

    # Arm Channel: Q = Quest/Unity world tracking frame.
    wrist_pos_world: np.ndarray
    wrist_quat_world: np.ndarray  # xyzw

    # Hand Channel: Wrist = local wrist/root hand frame.
    landmarks_wrist: np.ndarray

    def __post_init__(self) -> None:
        self.hand_side = str(self.hand_side).capitalize()
        if self.hand_side not in {"Left", "Right"}:
            raise ValueError(f"hand_side must be Left or Right, got {self.hand_side!r}")
        self.wrist_pos_world = np.asarray(self.wrist_pos_world, dtype=np.float64).reshape(3)
        self.wrist_quat_world = normalize_quat_xyzw(self.wrist_quat_world)
        self.landmarks_wrist = np.asarray(self.landmarks_wrist, dtype=np.float64).reshape(LANDMARK_COUNT, 3)
        if not np.all(np.isfinite(self.wrist_pos_world)):
            raise ValueError("wrist_pos_world contains non-finite values")
        if not np.all(np.isfinite(self.wrist_quat_world)):
            raise ValueError("wrist_quat_world contains non-finite values")
        if not np.all(np.isfinite(self.landmarks_wrist)):
            raise ValueError("landmarks_wrist contains non-finite values")


@dataclass
class QuestArmTarget:
    target_pos_B: np.ndarray
    target_R_B: np.ndarray | None
    delta_p_Q: np.ndarray

    @property
    def target_quat_B(self) -> np.ndarray | None:
        """Backward-compatible alias for older debug code.

        The Arm Channel now targets a rotation matrix, not a quaternion.
        """
        return self.target_R_B


@dataclass
class QuestHandCommand:
    features: dict[str, float]
    aero_action_7d: np.ndarray | None = None


def normalize_or_none(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def palm_frame_from_landmarks_wrist(landmarks_wrist: np.ndarray) -> np.ndarray:
    """Return palm frame columns [index side, finger forward, palm normal].

    The returned axes are expressed in the Quest wrist/root local frame.
    """
    points = np.asarray(landmarks_wrist, dtype=np.float64).reshape(LANDMARK_COUNT, 3)
    wrist = points[0]
    index_mcp = points[5]
    middle_mcp = points[9]
    pinky_mcp = points[17]

    x_axis = normalize_or_none(index_mcp - pinky_mcp)
    y_hint = normalize_or_none(middle_mcp - wrist)
    if x_axis is None or y_hint is None:
        raise ValueError("Cannot build palm frame from degenerate Quest landmarks")

    z_axis = normalize_or_none(np.cross(x_axis, y_hint))
    if z_axis is None:
        raise ValueError("Cannot build palm frame: index-pinky and wrist-middle axes are nearly parallel")
    y_axis = normalize_or_none(np.cross(z_axis, x_axis))
    if y_axis is None:
        raise ValueError("Cannot build palm frame: re-orthogonalized palm forward axis is degenerate")
    return np.column_stack([x_axis, y_axis, z_axis])


def palm_frame_from_quest_frame_world(frame: QuestHandFrame) -> np.ndarray:
    """Return palm frame axes expressed in the Quest/world tracking frame."""
    return quat_xyzw_to_matrix(frame.wrist_quat_world) @ palm_frame_from_landmarks_wrist(frame.landmarks_wrist)


@dataclass
class RelativeWristArmController:
    """Arm Channel mapper from Quest wrist relative pose to robot base.

    Q = Quest/Unity world tracking frame. B = robot base frame.
    ``R_BQ`` maps a vector expressed in Q into B and must be calibrated or
    configured for the workspace; it is not assumed to be the identity by the
    architecture, even though identity is a convenient debug default.
    """

    scale: float = 1.0
    R_BQ: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    deadzone: float = 0.0
    smoothing_alpha: float = 0.0
    control_orientation: bool = False
    orientation_source: str = "palm_landmarks"
    p_wrist_0_Q: np.ndarray | None = None
    R_wrist_0_Q: np.ndarray | None = None
    R_orientation_0_Q: np.ndarray | None = None
    p_ee_0_B: np.ndarray | None = None
    R_ee_0_B: np.ndarray | None = None
    previous_target_pos_B: np.ndarray | None = None

    def __post_init__(self) -> None:
        self.R_BQ = np.asarray(self.R_BQ, dtype=np.float64).reshape(3, 3)
        if self.orientation_source not in {"palm_landmarks", "wrist_pose"}:
            raise ValueError(f"orientation_source must be palm_landmarks or wrist_pose, got {self.orientation_source!r}")

    @property
    def is_calibrated(self) -> bool:
        return self.p_wrist_0_Q is not None and self.p_ee_0_B is not None

    def set_teleop_zero(
        self,
        wrist_pos_Q: np.ndarray,
        wrist_quat_Q: np.ndarray,
        ee_pos_B: np.ndarray,
        ee_R_B: np.ndarray | None = None,
        landmarks_wrist: np.ndarray | None = None,
    ) -> None:
        self.p_wrist_0_Q = np.asarray(wrist_pos_Q, dtype=np.float64).reshape(3).copy()
        self.R_wrist_0_Q = quat_xyzw_to_matrix(wrist_quat_Q)
        self.R_orientation_0_Q = (
            self._orientation_frame_Q(wrist_quat_Q, landmarks_wrist)
            if self.control_orientation
            else self.R_wrist_0_Q.copy()
        )
        self.p_ee_0_B = np.asarray(ee_pos_B, dtype=np.float64).reshape(3).copy()
        self.R_ee_0_B = np.eye(3, dtype=np.float64) if ee_R_B is None else np.asarray(ee_R_B, dtype=np.float64).reshape(3, 3).copy()
        self.previous_target_pos_B = self.p_ee_0_B.copy()

    def reset(self) -> None:
        self.p_wrist_0_Q = None
        self.R_wrist_0_Q = None
        self.R_orientation_0_Q = None
        self.p_ee_0_B = None
        self.R_ee_0_B = None
        self.previous_target_pos_B = None

    def _orientation_frame_Q(self, wrist_quat_Q: np.ndarray, landmarks_wrist: np.ndarray | None) -> np.ndarray:
        R_wrist_Q = quat_xyzw_to_matrix(wrist_quat_Q)
        if self.orientation_source == "wrist_pose":
            return R_wrist_Q
        if landmarks_wrist is None:
            raise ValueError("orientation_source=palm_landmarks requires landmarks_wrist")
        local_palm_R = palm_frame_from_landmarks_wrist(landmarks_wrist)
        return R_wrist_Q @ local_palm_R

    def compute_target(self, frame: QuestHandFrame) -> QuestArmTarget:
        if not self.is_calibrated:
            raise RuntimeError("RelativeWristArmController needs set_teleop_zero() before compute_target().")

        # Relative wrist displacement in Q. This intentionally ignores the
        # absolute Quest world origin and maps only motion since teleop zero.
        delta_p_Q = np.asarray(frame.wrist_pos_world, dtype=np.float64) - self.p_wrist_0_Q
        if float(np.linalg.norm(delta_p_Q)) < float(self.deadzone):
            delta_p_Q = np.zeros(3, dtype=np.float64)
        target_pos_B = self.p_ee_0_B + float(self.scale) * (self.R_BQ @ delta_p_Q)

        alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
        if self.previous_target_pos_B is not None and alpha > 0.0:
            target_pos_B = alpha * self.previous_target_pos_B + (1.0 - alpha) * target_pos_B
        self.previous_target_pos_B = target_pos_B.copy()

        target_R_B = None
        if self.control_orientation:
            R_orientation_t_Q = self._orientation_frame_Q(frame.wrist_quat_world, frame.landmarks_wrist)
            R_delta_Q = R_orientation_t_Q @ self.R_orientation_0_Q.T
            R_delta_B = self.R_BQ @ R_delta_Q @ self.R_BQ.T
            target_R_B = R_delta_B @ self.R_ee_0_B
        return QuestArmTarget(target_pos_B=target_pos_B, target_R_B=target_R_B, delta_p_Q=delta_p_Q)


class LandmarkHandRetargeter:
    """Hand Channel feature extractor for wrist-relative Quest landmarks."""

    def __init__(self, include_aero_action: bool = True):
        self.include_aero_action = bool(include_aero_action)

    def __call__(self, landmarks_wrist: np.ndarray) -> QuestHandCommand:
        from aero_quest.retargeting import quest_points_to_action_7d

        features = extract_hand_features(landmarks_wrist)
        aero_action_7d = quest_points_to_action_7d(landmarks_wrist) if self.include_aero_action else None
        return QuestHandCommand(features=features, aero_action_7d=aero_action_7d)


def parse_quest_hand_frames(packet: str | bytes) -> list[QuestHandFrame]:
    """Parse one text packet containing Left/Right wrist and landmark lines."""
    if isinstance(packet, bytes):
        packet = packet.decode("utf-8", errors="replace")
    partial: dict[str, dict[str, object]] = {}
    saw_header = False

    for line in str(packet).splitlines():
        if not line.strip():
            continue
        match = _HEADER_RE.match(line)
        if not match:
            raise QuestPacketParseError(f"Unrecognized Quest packet line: {line!r}")
        saw_header = True
        side = match.group("side").capitalize()
        kind = match.group("kind").lower()
        meta = match.group("meta") or ""
        values = _parse_float_values(match.group("values"))

        entry = partial.setdefault(side, {})
        frame_id = _parse_optional_int(_FRAME_RE, meta)
        timestamp_ns = _parse_optional_int(_TIME_RE, meta)
        _merge_metadata(entry, frame_id, timestamp_ns)

        if kind == "wrist":
            if len(values) != 7:
                raise QuestPacketParseError(f"{side} wrist expected 7 floats, got {len(values)}")
            entry["wrist_pos_world"] = np.asarray(values[:3], dtype=np.float64)
            entry["wrist_quat_world"] = np.asarray(values[3:], dtype=np.float64)
        elif kind == "landmarks":
            expected = LANDMARK_COUNT * 3
            if len(values) != expected:
                raise QuestPacketParseError(f"{side} landmarks expected {expected} floats, got {len(values)}")
            entry["landmarks_wrist"] = np.asarray(values, dtype=np.float64).reshape(LANDMARK_COUNT, 3)

    if not saw_header:
        raise QuestPacketParseError("Packet contained no Quest hand lines")

    frames: list[QuestHandFrame] = []
    for side, entry in partial.items():
        missing = [name for name in ("wrist_pos_world", "wrist_quat_world", "landmarks_wrist") if name not in entry]
        if missing:
            raise QuestPacketParseError(f"{side} packet missing fields: {', '.join(missing)}")
        frames.append(
            QuestHandFrame(
                hand_side=side,
                timestamp_ns=entry.get("timestamp_ns"),
                frame_id=entry.get("frame_id"),
                wrist_pos_world=entry["wrist_pos_world"],
                wrist_quat_world=entry["wrist_quat_world"],
                landmarks_wrist=entry["landmarks_wrist"],
            )
        )
    return frames


def parse_quest_hand_frame(packet: str | bytes, hand_side: str | None = None) -> QuestHandFrame:
    """Parse a text packet and return one hand frame."""
    frames = parse_quest_hand_frames(packet)
    if hand_side is None:
        if len(frames) != 1:
            raise QuestPacketParseError(f"Expected exactly one hand frame, got {len(frames)}")
        return frames[0]
    wanted = str(hand_side).capitalize()
    for frame in frames:
        if frame.hand_side == wanted:
            return frame
    raise QuestPacketParseError(f"Packet did not contain {wanted} hand")


def quest_hand_frame_from_sdk(frame) -> QuestHandFrame:
    """Convert a hand-tracking-sdk HandFrame-like object to QuestHandFrame."""
    side = getattr(frame, "side", None)
    side_value = getattr(side, "value", side)
    wrist = getattr(frame, "wrist", None)
    if wrist is None:
        raise ValueError("SDK frame has no wrist pose")
    if not all(hasattr(wrist, attr) for attr in ("x", "y", "z", "qx", "qy", "qz", "qw")):
        raise ValueError("SDK frame wrist pose must expose x/y/z/qx/qy/qz/qw")

    landmarks = getattr(frame, "landmarks", None)
    points = getattr(landmarks, "points", landmarks)
    if points is None:
        raise ValueError("SDK frame has no landmarks points")

    return QuestHandFrame(
        hand_side=str(side_value).capitalize(),
        timestamp_ns=_sdk_timestamp_ns(frame),
        frame_id=getattr(frame, "sequence_id", None),
        wrist_pos_world=np.asarray([wrist.x, wrist.y, wrist.z], dtype=np.float64),
        wrist_quat_world=np.asarray([wrist.qx, wrist.qy, wrist.qz, wrist.qw], dtype=np.float64),
        landmarks_wrist=_points_to_array(points),
    )


def convert_landmarks_wrist_to_world(
    wrist_pos_world: np.ndarray,
    wrist_quat_world: np.ndarray,
    landmarks_wrist: np.ndarray,
) -> np.ndarray:
    """Convert wrist-relative landmarks into Q/world points for debug only.

    This is useful for visualization, logging, or training data inspection.
    It is not the default Arm Channel input; arm motion should use the wrist
    world pose and relative wrist displacement.
    """
    p_wrist_Q = np.asarray(wrist_pos_world, dtype=np.float64).reshape(3)
    R_wrist_Q = quat_xyzw_to_matrix(wrist_quat_world)
    points_wrist = np.asarray(landmarks_wrist, dtype=np.float64).reshape(LANDMARK_COUNT, 3)
    return p_wrist_Q + points_wrist @ R_wrist_Q.T


def normalize_quat_xyzw(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1e-8:
        raise ValueError("Quaternion norm is near zero")
    return quat / norm


def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    x, y, z, w = normalize_quat_xyzw(quat)
    return np.asarray(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def extract_hand_features(landmarks_wrist: np.ndarray) -> dict[str, float]:
    """Extract simple hand-shape features from wrist-relative landmarks."""
    from aero_quest.retargeting import quest_points_to_hand_features

    points = np.asarray(landmarks_wrist, dtype=np.float64).reshape(LANDMARK_COUNT, 3)
    existing = quest_points_to_hand_features(points)
    features = {
        "thumb_index_pinch_distance": float(np.linalg.norm(points[4] - points[8])),
        "thumb_pinky_distance": float(np.linalg.norm(points[4] - points[20])),
        "index_curl": float(existing["index_curl"]),
        "middle_curl": float(existing["middle_curl"]),
        "ring_curl": float(existing["ring_curl"]),
        "pinky_curl": float(existing["pinky_curl"]),
        "thumb_curl": float(existing["thumb_curl"]),
        "palm_open_ratio": float(existing["palm_openness"]),
    }
    return features


def _parse_float_values(text: str) -> list[float]:
    if not text.strip():
        return []
    try:
        return [float(item.strip()) for item in text.replace(";", ",").split(",") if item.strip()]
    except ValueError as exc:
        raise QuestPacketParseError(f"Packet values must be floats: {text!r}") from exc


def _parse_optional_int(pattern: re.Pattern, text: str) -> int | None:
    match = pattern.search(text)
    return int(match.group(1)) if match else None


def _merge_metadata(entry: dict[str, object], frame_id: int | None, timestamp_ns: int | None) -> None:
    for key, value in (("frame_id", frame_id), ("timestamp_ns", timestamp_ns)):
        if value is None:
            continue
        previous = entry.get(key)
        if previous is not None and int(previous) != int(value):
            raise QuestPacketParseError(f"Mismatched {key}: {previous} vs {value}")
        entry[key] = int(value)


def _points_to_array(points) -> np.ndarray:
    rows = []
    for point in points:
        if all(hasattr(point, attr) for attr in ("x", "y", "z")):
            rows.append([float(point.x), float(point.y), float(point.z)])
        else:
            rows.append([float(point[0]), float(point[1]), float(point[2])])
    arr = np.asarray(rows, dtype=np.float64)
    if arr.shape != (LANDMARK_COUNT, 3):
        raise ValueError(f"Expected landmarks_wrist shape (21, 3), got {arr.shape}")
    return arr


def _sdk_timestamp_ns(frame) -> int | None:
    for name in ("timestamp_ns", "timestampNanos", "timestamp"):
        value = getattr(frame, name, None)
        if value is not None:
            return int(value)
    return None
