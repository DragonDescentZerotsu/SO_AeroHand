#!/usr/bin/env python3
"""Render Quest wrist-local, canonical, and reference hand skeletons.

Examples:
  python scripts/debug_visualize_hand_skeleton.py
  python scripts/debug_visualize_hand_skeleton.py --input data/quest_landmarks.npz \
    --frame-index 0 --output debug/alignment.png --mode side_by_side --show-labels
  python scripts/debug_visualize_hand_skeleton.py --input frames.json \
    --calibration cache/hand_calibration/right_hand_calibration.json --mode overlay
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import Any, Mapping

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.retargeting import as_points_array
from aero_quest.skeleton_visualizer import (
    calibration_reference_points,
    make_overlay_debug_figure,
    make_side_by_side_debug_figure,
    save_figure,
)


POINT_KEYS = (
    "landmarks_wrist",
    "landmarks",
    "P_human",
    "points",
    "frames",
    "hand_landmarks",
)
REFERENCE_KEYS = ("reference_points", "P_robot", "robot_points", "target_points")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize Quest 21-point hand skeleton alignment without MuJoCo."
    )
    parser.add_argument("--input", type=Path, help="Saved .npz, .pkl/.pickle, or .json data")
    parser.add_argument("--frame-index", type=int, default=0)
    parser.add_argument("--calibration", type=Path, help="Optional calibration JSON cache")
    parser.add_argument("--reference", type=Path, help="Optional reference skeleton data file")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("debug/synthetic_hand_skeleton.png"),
        help="PNG, GIF, or MP4 output path",
    )
    parser.add_argument("--mode", choices=("side_by_side", "overlay"), default="side_by_side")
    parser.add_argument("--show-labels", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--show-vectors", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--grid", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--elevation", type=float, default=24.0)
    parser.add_argument("--azimuth", type=float, default=-62.0)
    parser.add_argument("--dpi", type=int, default=180)
    parser.add_argument(
        "--max-frames",
        type=int,
        default=120,
        help="Maximum frames for GIF/MP4 output",
    )
    parser.add_argument("--fps", type=float, default=20.0)
    parser.add_argument("--show", action="store_true", help="Open an interactive window after rendering")
    return parser.parse_args()


def synthetic_hand() -> np.ndarray:
    """Return a plausible, slightly depth-curved right-hand skeleton."""
    points = np.zeros((21, 3), dtype=np.float32)
    points[0] = (0.00, 0.00, 0.00)
    points[1:5] = (
        (0.030, 0.020, -0.005),
        (0.052, 0.045, -0.010),
        (0.068, 0.070, -0.006),
        (0.078, 0.094, 0.002),
    )
    bases = ((0.038, 0.078), (0.012, 0.090), (-0.014, 0.082), (-0.038, 0.066))
    lengths = ((0.037, 0.026, 0.021), (0.043, 0.030, 0.024),
               (0.039, 0.028, 0.022), (0.032, 0.023, 0.019))
    for finger_index, ((x, y), segments) in enumerate(zip(bases, lengths)):
        start = 5 + 4 * finger_index
        points[start] = (x, y, 0.0)
        cursor = points[start].copy()
        for offset, length in enumerate(segments, start=1):
            cursor = cursor + np.array((-0.002 * finger_index, length, 0.004 * offset))
            points[start + offset] = cursor
    return points


def _find_points(value: Any, keys: tuple[str, ...] = POINT_KEYS) -> np.ndarray | None:
    if isinstance(value, np.lib.npyio.NpzFile):
        for key in keys:
            if key in value.files:
                return np.asarray(value[key])
        for key in value.files:
            found = _find_points(value[key], keys)
            if found is not None:
                return found
        return None
    if isinstance(value, Mapping):
        for key in keys:
            if key in value:
                found = _find_points(value[key], keys)
                if found is not None:
                    return found
        for nested in value.values():
            if isinstance(nested, (Mapping, list, tuple)):
                found = _find_points(nested, keys)
                if found is not None:
                    return found
        return None
    if isinstance(value, (list, tuple)) and value and isinstance(value[0], Mapping):
        frames = [_find_points(item, keys) for item in value]
        frames = [item for item in frames if item is not None]
        return np.asarray(frames) if frames else None
    try:
        array = np.asarray(value, dtype=np.float32)
    except (TypeError, ValueError):
        return None
    if array.shape == (21, 3) or (array.ndim == 3 and array.shape[1:] == (21, 3)):
        return array
    return None


def load_data(path: Path) -> Any:
    suffix = path.suffix.lower()
    if suffix == ".npz":
        return np.load(path, allow_pickle=True)
    if suffix in (".pkl", ".pickle"):
        with path.open("rb") as stream:
            return pickle.load(stream)  # Saved debug data may contain SDK-shaped objects.
    if suffix in (".json", ".jsonl"):
        with path.open("r", encoding="utf-8") as stream:
            if suffix == ".jsonl":
                return [json.loads(line) for line in stream if line.strip()]
            return json.load(stream)
    raise ValueError(f"Unsupported input format {suffix!r}; use .npz, .pkl/.pickle, or .json/.jsonl")


def load_frames(path: Path) -> tuple[np.ndarray, np.ndarray | None]:
    data = load_data(path)
    points = _find_points(data)
    reference = _find_points(data, REFERENCE_KEYS)
    if isinstance(data, np.lib.npyio.NpzFile):
        data.close()
    if points is None:
        raise ValueError(f"Could not find a 21x3 landmark array in {path}")
    points = np.asarray(points, dtype=np.float32)
    if points.shape == (21, 3):
        points = points[None, ...]
    if reference is not None:
        reference = np.asarray(reference, dtype=np.float32)
    return points, reference


def load_calibration(path: Path | None) -> dict[str, Any] | None:
    if path is None:
        return None
    with path.open("r", encoding="utf-8") as stream:
        calibration = json.load(stream)
    if not isinstance(calibration, dict):
        raise ValueError("Calibration JSON must contain an object at the top level")
    calibration["_cache_path"] = str(path)
    return calibration


def select_frame(points: np.ndarray | None, index: int) -> np.ndarray | None:
    if points is None:
        return None
    if points.shape == (21, 3):
        return as_points_array(points)
    if not -len(points) <= index < len(points):
        raise IndexError(f"frame-index {index} is outside the available range 0..{len(points) - 1}")
    return as_points_array(points[index])


def make_figure(
    points: np.ndarray,
    calibration: Mapping[str, Any] | None,
    reference: np.ndarray | None,
    args: argparse.Namespace,
) -> Any:
    kwargs = dict(
        calibration=calibration,
        reference_points=reference,
        show_labels=args.show_labels,
        show_vectors=args.show_vectors,
        show_grid=args.grid,
        elevation=args.elevation,
        azimuth=args.azimuth,
    )
    if args.mode == "overlay":
        return make_overlay_debug_figure(points, **kwargs)
    return make_side_by_side_debug_figure(points, **kwargs)


def save_animation(
    frames: np.ndarray,
    references: np.ndarray | None,
    calibration: Mapping[str, Any] | None,
    args: argparse.Namespace,
) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation, PillowWriter

    count = min(len(frames), args.max_frames)
    selected = np.linspace(0, len(frames) - 1, count, dtype=int)
    holder: dict[str, Any] = {"fig": None}

    def update(animation_index: int) -> list[Any]:
        if holder["fig"] is not None:
            plt.close(holder["fig"])
        source_index = int(selected[animation_index])
        reference = select_frame(references, source_index) if references is not None else None
        holder["fig"] = make_figure(frames[source_index], calibration, reference, args)
        # Matplotlib animation requires one persistent canvas; render each debug
        # figure to RGBA and display it on that canvas.
        holder["fig"].canvas.draw()
        image = np.asarray(holder["fig"].canvas.buffer_rgba())
        image_artist.set_data(image)
        canvas_ax.set_title(f"frame {source_index}")
        return [image_artist]

    canvas, canvas_ax = plt.subplots(figsize=(19, 7.2))
    canvas_ax.axis("off")
    image_artist = canvas_ax.imshow(np.zeros((10, 10, 4), dtype=np.uint8))
    animation = FuncAnimation(canvas, update, frames=count, blit=False)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.output.suffix.lower() == ".gif":
        animation.save(args.output, writer=PillowWriter(fps=args.fps), dpi=args.dpi)
    else:
        try:
            animation.save(args.output, writer="ffmpeg", fps=args.fps, dpi=args.dpi)
        except (RuntimeError, FileNotFoundError) as exc:
            raise RuntimeError(
                "MP4 output requires an ffmpeg executable visible to Matplotlib; "
                "use a .gif output with Pillow instead."
            ) from exc
    plt.close(canvas)
    if holder["fig"] is not None:
        plt.close(holder["fig"])


def main() -> int:
    args = parse_args()
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit("This debug tool requires Matplotlib: pip install matplotlib") from exc

    calibration = load_calibration(args.calibration)
    if args.input is None:
        frames = synthetic_hand()[None, ...]
        embedded_reference = None
        print("No --input supplied; using the synthetic wrist-local hand demo.")
    else:
        frames, embedded_reference = load_frames(args.input)
    explicit_reference = None
    if args.reference is not None:
        explicit_reference, _ = load_frames(args.reference)
    references = explicit_reference if explicit_reference is not None else embedded_reference

    suffix = args.output.suffix.lower()
    if suffix in (".gif", ".mp4"):
        save_animation(frames, references, calibration, args)
        print(f"Saved skeleton alignment animation to {args.output}")
    else:
        raw = select_frame(frames, args.frame_index)
        reference = select_frame(references, args.frame_index) if references is not None else None
        if reference is None:
            reference = calibration_reference_points(calibration)
        fig = make_figure(raw, calibration, reference, args)
        saved = save_figure(fig, args.output, dpi=args.dpi)
        if args.show:
            plt.show()
        plt.close(fig)
        print(f"Saved skeleton alignment debug image to {saved}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
