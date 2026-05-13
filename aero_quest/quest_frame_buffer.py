"""Latest-frame buffer for low-latency Quest control loops."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

from aero_quest.quest_dual_channel import QuestDualChannelFrame
from aero_quest.quest_receiver import QuestTelemetryReceiver


@dataclass
class LatestFrameBufferStats:
    total_frames: int = 0
    valid_frames: int = 0
    invalid_frames: int = 0
    dropped_or_overwritten_frames: int = 0
    last_frame_age_ms: float | None = None


class LatestFrameBuffer:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest: QuestDualChannelFrame | None = None
        self._latest_recv_monotonic_ns: int | None = None
        self._stats = LatestFrameBufferStats()

    def update(self, frame: QuestDualChannelFrame) -> None:
        with self._lock:
            if self._latest is not None:
                self._stats.dropped_or_overwritten_frames += 1
            self._latest = frame
            self._latest_recv_monotonic_ns = time.monotonic_ns()
            self._stats.total_frames += 1
            if frame.valid:
                self._stats.valid_frames += 1
            else:
                self._stats.invalid_frames += 1

    def get_latest(self, max_age_ms: float | None = None) -> QuestDualChannelFrame | None:
        with self._lock:
            if self._latest is None or self._latest_recv_monotonic_ns is None:
                return None
            age_ms = (time.monotonic_ns() - self._latest_recv_monotonic_ns) / 1_000_000.0
            if max_age_ms is not None and age_ms > float(max_age_ms):
                return None
            return self._latest

    def stats(self) -> dict[str, int | float | None]:
        with self._lock:
            age_ms = None
            if self._latest_recv_monotonic_ns is not None:
                age_ms = (time.monotonic_ns() - self._latest_recv_monotonic_ns) / 1_000_000.0
            return {
                "total_frames": self._stats.total_frames,
                "valid_frames": self._stats.valid_frames,
                "invalid_frames": self._stats.invalid_frames,
                "dropped_or_overwritten_frames": self._stats.dropped_or_overwritten_frames,
                "last_frame_age_ms": age_ms,
            }


class QuestReceiverThread:
    def __init__(self, buffer: LatestFrameBuffer | None = None, **receiver_kwargs) -> None:
        self.buffer = buffer or LatestFrameBuffer()
        self.receiver_kwargs = dict(receiver_kwargs)
        self.receiver: QuestTelemetryReceiver | None = None
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self.error: BaseException | None = None

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="quest-telemetry-receiver", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        if self.receiver is not None:
            self.receiver.close()
        if self._thread is not None:
            self._thread.join(timeout=timeout)

    def _run(self) -> None:
        try:
            self.receiver = QuestTelemetryReceiver(**self.receiver_kwargs)
            for frame in self.receiver.iter_frames():
                if self._stop_event.is_set():
                    break
                self.buffer.update(frame)
        except BaseException as exc:
            self.error = exc
        finally:
            if self.receiver is not None:
                self.receiver.close()
