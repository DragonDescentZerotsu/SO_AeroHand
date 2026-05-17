"""Receiver wrapper for hand-tracking-sdk Quest telemetry."""

from __future__ import annotations

import time
from collections.abc import Iterator
from typing import Any

import numpy as np

from aero_quest.quest_dual_channel import QuestDualChannelFrame, validate_dual_channel_frame


class QuestTelemetryReceiver:
    """Yield validated dual-channel Quest frames from hand-tracking-sdk."""

    def __init__(
        self,
        transport: str = "tcp",
        host: str = "0.0.0.0",
        port: int = 8000,
        hand: str = "right",
        output_raw_events: bool = False,
    ):
        self.transport = transport.lower()
        self.host = host
        self.port = int(port)
        self.hand = hand.lower()
        self.output_raw_events = bool(output_raw_events)
        self._closed = False
        self._sdk = _load_sdk()

    def iter_frames(self) -> Iterator[QuestDualChannelFrame]:
        if self._sdk is None:
            raise RuntimeError(
                "hand-tracking-sdk is not installed. Install it with: "
                "python -m pip install hand-tracking-sdk"
            )

        client = self._make_client()
        hand_frame_type = getattr(self._sdk, "HandFrame", None)
        try:
            for event in client.iter_events():
                if self._closed:
                    break
                if hand_frame_type is not None and not isinstance(event, hand_frame_type):
                    continue
                if hand_frame_type is None and not hasattr(event, "wrist"):
                    continue

                try:
                    frame = self._convert_sdk_frame(event)
                except Exception as exc:  # keep one malformed frame from killing recording
                    frame = self._invalid_frame_from_event(event, str(exc))

                if not self._hand_matches(frame.hand_side):
                    continue
                yield frame
        finally:
            self._closed = True

    def close(self) -> None:
        self._closed = True

    def _make_client(self):
        transport_mode = self._transport_mode()
        stream_output = self._sdk.StreamOutput.BOTH if self.output_raw_events else self._sdk.StreamOutput.FRAMES
        config = self._sdk.HTSClientConfig(
            transport_mode=transport_mode,
            host=self.host,
            port=self.port,
            output=stream_output,
            hand_filter=self._hand_filter(),
            error_policy=self._sdk.ErrorPolicy.TOLERANT,
            include_wall_time=True,
        )
        return self._sdk.HTSClient(config)

    def _transport_mode(self):
        if self.transport in {"tcp", "tcp_server", "server"}:
            return self._sdk.TransportMode.TCP_SERVER
        if self.transport in {"tcp_client", "client"}:
            return self._sdk.TransportMode.TCP_CLIENT
        if self.transport == "udp":
            return self._sdk.TransportMode.UDP
        raise ValueError("transport must be one of: tcp, tcp_server, tcp_client, udp")

    def _hand_filter(self):
        if self.hand == "left":
            return self._sdk.HandFilter.LEFT
        if self.hand == "right":
            return self._sdk.HandFilter.RIGHT
        return self._sdk.HandFilter.BOTH

    def _hand_matches(self, hand_side: str) -> bool:
        return self.hand in {"any", "both"} or hand_side.lower() == self.hand

    def _convert_sdk_frame(self, sdk_frame: Any) -> QuestDualChannelFrame:
        side = getattr(getattr(sdk_frame, "side", None), "value", getattr(sdk_frame, "side", None))
        wrist = getattr(sdk_frame, "wrist", None)
        landmarks_obj = getattr(sdk_frame, "landmarks", None)
        points = getattr(landmarks_obj, "points", landmarks_obj)
        if wrist is None:
            raise ValueError("SDK frame has no wrist pose")
        if points is None:
            raise ValueError("SDK frame has no landmarks")

        frame = QuestDualChannelFrame(
            hand_side=_canonical_hand_side(side),
            recv_ts_ns=_first_int_attr(sdk_frame, "recv_ts_ns", "recv_time_unix_ns") or time.time_ns(),
            source_ts_ns=_first_int_attr(sdk_frame, "source_ts_ns"),
            frame_id=_frame_id(sdk_frame),
            sequence_id=_first_int_attr(sdk_frame, "sequence_id", "source_frame_seq"),
            wrist_pos_world=np.asarray([wrist.x, wrist.y, wrist.z], dtype=np.float64),
            wrist_quat_world=np.asarray([wrist.qx, wrist.qy, wrist.qz, wrist.qw], dtype=np.float64),
            landmarks_wrist=_points_to_array(points),
        )
        validate_dual_channel_frame(frame)
        return frame

    def _invalid_frame_from_event(self, event: Any, error: str) -> QuestDualChannelFrame:
        side = getattr(getattr(event, "side", None), "value", getattr(event, "side", "Unknown"))
        frame = QuestDualChannelFrame(
            hand_side=_canonical_hand_side(side),
            recv_ts_ns=time.time_ns(),
            source_ts_ns=_first_int_attr(event, "source_ts_ns"),
            frame_id=_frame_id(event),
            sequence_id=_first_int_attr(event, "sequence_id", "source_frame_seq"),
            wrist_pos_world=np.full(3, np.nan, dtype=np.float64),
            wrist_quat_world=np.full(4, np.nan, dtype=np.float64),
            landmarks_wrist=np.full((21, 3), np.nan, dtype=np.float64),
            valid=False,
            quality_flags={"conversion_error": error},
        )
        validate_dual_channel_frame(frame)
        frame.valid = False
        return frame


def _load_sdk():
    try:
        import hand_tracking_sdk
    except ImportError:
        return None
    return hand_tracking_sdk


def _canonical_hand_side(value: Any) -> str:
    text = str(value or "Unknown")
    if text.lower() == "left":
        return "Left"
    if text.lower() == "right":
        return "Right"
    return text


def _first_int_attr(obj: Any, *names: str) -> int | None:
    for name in names:
        value = getattr(obj, name, None)
        if value is not None:
            return int(value)
    return None


def _frame_id(obj: Any) -> int | str | None:
    for name in ("source_frame_seq", "frame_id"):
        value = getattr(obj, name, None)
        if value is None:
            continue
        try:
            return int(value)
        except (TypeError, ValueError):
            return str(value)
    return None


def _points_to_array(points: Any) -> np.ndarray:
    rows = []
    for point in points:
        if all(hasattr(point, attr) for attr in ("x", "y", "z")):
            rows.append([float(point.x), float(point.y), float(point.z)])
        else:
            rows.append([float(point[0]), float(point[1]), float(point[2])])
    return np.asarray(rows, dtype=np.float64)
