#!/usr/bin/env python3
"""Low-latency live Quest viewer and high-quality offline replay.

Live mode stores exactly one latest valid frame. Rendering is independently
capped, so a slow Matplotlib window drops stale frames instead of building a
backlog. Recorded landmarks are raw wrist-local points, never display-smoothed.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.retargeting import get_quest_points_21
from aero_quest.skeleton_visualizer import (
    LowLatencyHandSkeletonArtists,
    RealtimeHandSkeletonArtists,
    calibration_reference_points,
    compute_hand_debug_metrics,
    make_side_by_side_debug_figure,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Low-latency live or high-quality replayed Quest hand viewer."
    )
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hand-side", choices=("left", "right", "any"), default="right")
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--replay", type=Path)
    parser.add_argument("--mode", choices=("single_panel", "side_by_side"))
    parser.add_argument(
        "--single-panel",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use one lightweight panel unless --mode overrides it",
    )
    parser.add_argument("--viewer-fps", type=float, default=10.0)
    parser.add_argument("--metrics-fps", type=float, default=30.0)
    parser.add_argument("--replay-fps", type=float, default=30.0)
    parser.add_argument("--export", type=Path, help="Replay-only GIF or MP4 output")
    parser.add_argument("--export-max-frames", type=int, default=120)
    parser.add_argument("--smooth-alpha", type=float, default=0.35)
    parser.add_argument(
        "--show-labels", action=argparse.BooleanOptionalAction, default=None
    )
    parser.add_argument(
        "--show-vectors", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-action", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--print-metrics", action="store_true")
    parser.add_argument("--save-snapshots", action="store_true")
    parser.add_argument("--snapshot-dir", type=Path, default=Path("debug"))
    parser.add_argument("--record-npz", type=Path)
    parser.add_argument(
        "--record-every",
        type=int,
        default=1,
        help="Record every Nth valid raw incoming frame",
    )
    parser.add_argument("--warning-interval", type=float, default=2.0)
    parser.add_argument("--elevation", type=float, default=24.0)
    parser.add_argument("--azimuth", type=float, default=-62.0)
    parser.add_argument("--dpi", type=int, default=180)
    return parser.parse_args()


def load_calibration(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.expanduser().open("r", encoding="utf-8") as stream:
        calibration = json.load(stream)
    if not isinstance(calibration, dict):
        raise ValueError("Calibration JSON must contain an object")
    calibration["_cache_path"] = str(path)
    return calibration


def hand_matches(frame: Any, requested: str) -> bool:
    if requested == "any":
        return True
    side = getattr(frame, "side", None)
    side = getattr(side, "value", side)
    return str(side).lower() == requested


def frame_timestamp_seconds(frame: Any, fallback: float) -> float:
    for name in ("timestamp_ns", "source_ts_ns", "recv_ts_ns", "timestampNanos"):
        value = getattr(frame, name, None)
        if value is not None:
            return float(value) / 1e9
    value = getattr(frame, "timestamp", None)
    if value is not None:
        value = float(value)
        return value / 1e9 if value > 1e12 else value
    return fallback


@dataclass
class LatestFrameBuffer:
    """Thread-safe single-slot buffer; publishing always replaces old data."""

    _lock: threading.Lock = field(default_factory=threading.Lock)
    _packet: tuple[np.ndarray, float, int] | None = None
    _version: int = 0

    def publish(self, points: np.ndarray, timestamp: float, frame_count: int) -> None:
        with self._lock:
            self._version += 1
            self._packet = (points.copy(), float(timestamp), int(frame_count))

    def latest_after(
        self, consumed_version: int
    ) -> tuple[int, tuple[np.ndarray, float, int] | None]:
        with self._lock:
            if self._version == consumed_version or self._packet is None:
                return consumed_version, None
            points, timestamp, frame_count = self._packet
            return self._version, (points.copy(), timestamp, frame_count)

    def latest(self) -> tuple[np.ndarray, float, int] | None:
        with self._lock:
            if self._packet is None:
                return None
            points, timestamp, frame_count = self._packet
            return points.copy(), timestamp, frame_count


@dataclass
class LiveState:
    latest: LatestFrameBuffer = field(default_factory=LatestFrameBuffer)
    recorded_points: list[np.ndarray] = field(default_factory=list)
    recorded_timestamps: list[float] = field(default_factory=list)
    received_count: int = 0
    invalid_count: int = 0
    receiver_error: BaseException | None = None
    stop_requested: threading.Event = field(default_factory=threading.Event)

    def publish(
        self, points: np.ndarray, timestamp: float, *, record: bool, record_every: int
    ) -> None:
        self.received_count += 1
        self.latest.publish(points, timestamp, self.received_count)
        if record and (self.received_count - 1) % record_every == 0:
            self.recorded_points.append(points.copy())
            self.recorded_timestamps.append(float(timestamp))


def start_live_receiver(args: argparse.Namespace, state: LiveState) -> threading.Thread:
    def run() -> None:
        try:
            from hand_tracking_sdk import (
                HandFrame,
                HTSClient,
                HTSClientConfig,
                StreamOutput,
                TransportMode,
            )
        except ImportError as exc:
            state.receiver_error = RuntimeError(
                "Live mode requires hand-tracking-sdk; install the quest dependency."
            )
            state.receiver_error.__cause__ = exc
            return
        try:
            client = HTSClient(
                HTSClientConfig(
                    transport_mode=TransportMode.TCP_SERVER,
                    host=args.host,
                    port=args.port,
                    output=StreamOutput.FRAMES,
                )
            )
            last_warning = 0.0
            for frame in client.iter_events():
                if state.stop_requested.is_set():
                    break
                if not isinstance(frame, HandFrame) or not hand_matches(frame, args.hand_side):
                    continue
                received_at = time.time()
                try:
                    points = get_quest_points_21(frame)
                    if not np.all(np.isfinite(points)):
                        raise ValueError("landmarks contain non-finite values")
                    state.publish(
                        points,
                        frame_timestamp_seconds(frame, received_at),
                        record=args.record_npz is not None,
                        record_every=args.record_every,
                    )
                except (TypeError, ValueError) as exc:
                    state.invalid_count += 1
                    if received_at - last_warning >= args.warning_interval:
                        print(f"Skipping invalid Quest frame ({state.invalid_count} total): {exc}")
                        last_warning = received_at
        except BaseException as exc:
            state.receiver_error = exc

    thread = threading.Thread(target=run, name="quest-latest-frame-receiver", daemon=True)
    thread.start()
    return thread


def load_replay(path: Path) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path.expanduser(), allow_pickle=False) as data:
        points = None
        for key in ("points", "landmarks", "landmarks_wrist", "P_human"):
            if key in data.files:
                points = np.asarray(data[key], dtype=np.float32)
                break
        if points is None or points.ndim != 3 or points.shape[1:] != (21, 3):
            raise ValueError(f"{path} has no [T, 21, 3] wrist-local point array")
        timestamps = None
        for key in ("timestamps", "wall_time", "timestamp"):
            if key in data.files:
                candidate = np.asarray(data[key], dtype=np.float64).reshape(-1)
                if len(candidate) == len(points):
                    timestamps = candidate
                    break
    valid = np.all(np.isfinite(points), axis=(1, 2))
    points = points[valid]
    if timestamps is None:
        timestamps = np.arange(len(valid), dtype=np.float64)
    timestamps = timestamps[valid]
    if len(points) == 0:
        raise ValueError(f"{path} contains no finite landmark frames")
    return points, timestamps


def save_recording(
    path: Path,
    state: LiveState,
    hand_side: str,
    calibration_path: Path | None,
) -> None:
    if not state.recorded_points:
        print("No recorded Quest frames; NPZ was not written.")
        return
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    points = np.asarray(state.recorded_points, dtype=np.float32)
    np.savez_compressed(
        path,
        points=points,
        landmarks=points,
        landmarks_wrist=points,
        timestamps=np.asarray(state.recorded_timestamps, dtype=np.float64),
        hand_side=np.asarray(hand_side),
        calibration_path=np.asarray(str(calibration_path) if calibration_path else ""),
    )
    print(f"Saved {len(points)} raw wrist-local frames to {path}")


def save_snapshot(fig: Any, directory: Path, index: int, dpi: int) -> Path:
    directory = directory.expanduser()
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"realtime_hand_snapshot_{index:04d}.png"
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"Saved snapshot: {path}")
    return path


def resolve_mode(args: argparse.Namespace) -> tuple[str, bool]:
    replay = args.replay is not None
    if args.mode is not None:
        mode = args.mode
    elif replay:
        mode = "side_by_side"
    else:
        mode = "single_panel" if args.single_panel else "side_by_side"
    show_labels = replay if args.show_labels is None else args.show_labels
    return mode, show_labels


def export_replay(
    points: np.ndarray,
    calibration: dict[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    count = min(len(points), args.export_max_frames)
    selected = np.linspace(0, len(points) - 1, count, dtype=int)
    canvas, canvas_ax = plt.subplots(figsize=(19, 7.2))
    canvas_ax.axis("off")
    image_artist = canvas_ax.imshow(np.zeros((10, 10, 4), dtype=np.uint8))
    holder: dict[str, Any] = {"fig": None}

    def update(index: int) -> list[Any]:
        if holder["fig"] is not None:
            plt.close(holder["fig"])
        holder["fig"] = make_side_by_side_debug_figure(
            points[int(selected[index])],
            calibration=calibration,
            show_labels=True if args.show_labels is None else args.show_labels,
            show_vectors=args.show_vectors,
            show_grid=args.grid,
            elevation=args.elevation,
            azimuth=args.azimuth,
        )
        holder["fig"].canvas.draw()
        image_artist.set_data(np.asarray(holder["fig"].canvas.buffer_rgba()))
        return [image_artist]

    animation = FuncAnimation(canvas, update, frames=count, blit=False)
    args.export.parent.mkdir(parents=True, exist_ok=True)
    if args.export.suffix.lower() == ".gif":
        animation.save(args.export, writer=PillowWriter(fps=args.replay_fps), dpi=args.dpi)
    elif args.export.suffix.lower() == ".mp4":
        animation.save(args.export, writer="ffmpeg", fps=args.replay_fps, dpi=args.dpi)
    else:
        raise ValueError("--export must end in .gif or .mp4")
    plt.close(canvas)
    if holder["fig"] is not None:
        plt.close(holder["fig"])
    print(f"Saved replay animation to {args.export}")


def main() -> int:
    args = parse_args()
    if not 0.0 <= args.smooth_alpha <= 1.0:
        raise SystemExit("--smooth-alpha must be between 0 and 1")
    if min(args.viewer_fps, args.metrics_fps, args.replay_fps) <= 0:
        raise SystemExit("viewer, metrics, and replay FPS values must be positive")
    if args.record_every < 1:
        raise SystemExit("--record-every must be at least 1")
    if args.export_max_frames < 1:
        raise SystemExit("--export-max-frames must be at least 1")
    if args.export is not None and args.replay is None:
        raise SystemExit("--export requires --replay")
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("Install visualization support: pip install -e '.[visualization]'") from exc

    mode, show_labels = resolve_mode(args)
    calibration = load_calibration(args.calibration)
    reference = calibration_reference_points(calibration)
    state = LiveState()
    replay_points = replay_timestamps = None
    if args.replay:
        replay_points, replay_timestamps = load_replay(args.replay)
        print(f"Replay mode: {len(replay_points)} frames from {args.replay}")
        if args.export is not None:
            export_replay(replay_points, calibration, args)
            return 0
    else:
        start_live_receiver(args, state)
        print(f"Live latest-frame mode on {args.host}:{args.port} ({args.hand_side})")
        print(f"Viewer capped at {args.viewer_fps:g} FPS; stale frames are dropped.")

    plt.ion()
    if mode == "single_panel":
        fig = plt.figure(figsize=(9, 8), constrained_layout=True)
        axes = [fig.add_subplot(111, projection="3d")]
        artists: Any = LowLatencyHandSkeletonArtists(
            fig, axes[0], elevation=args.elevation, azimuth=args.azimuth
        )
    else:
        fig = plt.figure(figsize=(19, 7.2), constrained_layout=True)
        axes = [fig.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
        artists = RealtimeHandSkeletonArtists(
            fig, axes, elevation=args.elevation, azimuth=args.azimuth
        )
    fig.text(
        0.5, 0.01,
        "q quit | s snapshot | l labels | v vectors | g grid | p pause | r reset view",
        ha="center", fontsize=9,
    )
    ui = {
        "quit": False, "paused": False, "labels": show_labels,
        "vectors": args.show_vectors, "grid": args.grid,
        "reset_camera": True, "snapshot": False,
    }

    def on_key(event: Any) -> None:
        key = str(event.key or "").lower()
        if key == "q":
            ui["quit"] = True
        elif key == "s":
            ui["snapshot"] = True
        elif key == "l":
            ui["labels"] = not ui["labels"]
        elif key == "v":
            ui["vectors"] = not ui["vectors"]
        elif key == "g":
            ui["grid"] = not ui["grid"]
        elif key == "p":
            ui["paused"] = not ui["paused"]
        elif key == "r":
            ui["reset_camera"] = True

    fig.canvas.mpl_connect("key_press_event", on_key)
    smoothed = None
    last_packet = None
    consumed_version = 0
    replay_index = 0
    next_view = time.monotonic()
    next_metrics = next_view
    render_times: list[float] = []
    snapshot_count = 0

    try:
        while plt.fignum_exists(fig.number) and not ui["quit"]:
            now = time.monotonic()
            if state.receiver_error is not None:
                raise RuntimeError(f"Quest receiver stopped: {state.receiver_error}")

            if args.print_metrics and not ui["paused"] and now >= next_metrics:
                metric_packet = (
                    (replay_points[replay_index], replay_timestamps[replay_index], replay_index + 1)
                    if args.replay else state.latest.latest()
                )
                if metric_packet is not None:
                    metrics = compute_hand_debug_metrics(metric_packet[0], calibration)
                    print(
                        f"frame={metric_packet[2]} "
                        f"thumb_index={metrics['thumb_index_distance']:.4f} "
                        f"thumb_middle={metrics['thumb_middle_distance']:.4f} "
                        f"pinch={metrics['pinch_state']}"
                    )
                next_metrics = now + 1.0 / args.metrics_fps

            packet = None
            if not ui["paused"] and now >= next_view:
                if args.replay:
                    packet = (
                        replay_points[replay_index],
                        float(replay_timestamps[replay_index]),
                        replay_index + 1,
                    )
                    replay_index = (replay_index + 1) % len(replay_points)
                else:
                    consumed_version, packet = state.latest.latest_after(consumed_version)
                next_view = now + 1.0 / (
                    args.replay_fps if args.replay else args.viewer_fps
                )

            if packet is not None:
                raw_points, _timestamp, frame_count = packet
                smoothed = (
                    raw_points.copy()
                    if smoothed is None
                    else args.smooth_alpha * raw_points
                    + (1.0 - args.smooth_alpha) * smoothed
                )
                last_packet = packet
                render_times.append(now)
                render_times = [value for value in render_times if now - value <= 1.0]
                if ui["reset_camera"]:
                    artists.reset_camera()
                if mode == "single_panel":
                    artists.update(
                        smoothed,
                        fps=float(len(render_times)),
                        frame_count=frame_count,
                        calibration=calibration,
                        show_labels=ui["labels"],
                        show_vectors=ui["vectors"],
                        show_grid=ui["grid"],
                    )
                else:
                    artists.update(
                        smoothed,
                        calibration=calibration,
                        reference_points=reference,
                        fps=float(len(render_times)),
                        frame_count=frame_count,
                        show_labels=ui["labels"],
                        show_vectors=ui["vectors"],
                        show_grid=ui["grid"],
                        show_action=args.show_action,
                    )
                ui["reset_camera"] = False
                fig.canvas.draw_idle()

            if ui["snapshot"]:
                if last_packet is None:
                    print("No valid frame visible; snapshot skipped.")
                else:
                    snapshot_count += 1
                    save_snapshot(fig, args.snapshot_dir, snapshot_count, args.dpi)
                ui["snapshot"] = False

            fig.canvas.flush_events()
            plt.pause(0.001)
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_requested.set()
        if args.record_npz is not None and not args.replay:
            save_recording(args.record_npz, state, args.hand_side, args.calibration)
        plt.close(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
