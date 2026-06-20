"""Matplotlib-only debugging for Quest-to-AeroHand skeleton alignment.

The raw panel shows the 21 landmarks exactly as received by the Hand Channel:
coordinates are local to the Quest wrist/root, not Quest world coordinates.
The canonical panel applies :func:`aero_quest.retargeting.palm_localize`: the
wrist is the origin, +X points from little MCP toward index MCP, +Y points
toward middle MCP, and lengths are divided by wrist-to-middle-MCP distance.

A correct alignment preserves finger shape and joint ordering across panels.
Only the rigid palm-frame rotation and documented uniform scaling should
change.  Unexpected mirroring, finger swaps, or non-uniform deformation are
therefore easy to spot.  This module does not import or require MuJoCo.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import numpy as np

from aero_quest.retargeting import (
    as_points_array,
    palm_localize,
    quest_points_to_action_7d,
    safe_normalize,
)


FINGERS = {
    "thumb": (0, 1, 2, 3, 4),
    "index": (0, 5, 6, 7, 8),
    "middle": (0, 9, 10, 11, 12),
    "ring": (0, 13, 14, 15, 16),
    "little": (0, 17, 18, 19, 20),
}
PALM_CHAIN = (5, 9, 13, 17)
FINGERTIPS = (4, 8, 12, 16, 20)
IMPORTANT_LABELS = {0: "wrist", 4: "thumb tip", 8: "index tip", 12: "middle tip"}
FINGER_COLORS = {
    "thumb": "#e76f51",
    "index": "#f4a261",
    "middle": "#2a9d8f",
    "ring": "#457b9d",
    "little": "#9b5de5",
}
VECTOR_SPECS = (
    (0, 4, "wrist→thumb", "#d62728"),
    (0, 8, "wrist→index", "#ff7f0e"),
    (0, 12, "wrist→middle", "#17becf"),
    (4, 8, "thumb→index", "#e377c2"),
    (4, 12, "thumb→middle", "#9467bd"),
)


def _palm_frame(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Return the origin, palm rotation, and normalization scale."""
    points = as_points_array(points).astype(np.float64)
    origin = points[0]
    x_axis = safe_normalize(points[5] - points[17]).astype(np.float64)
    y_hint = safe_normalize(points[9] - points[0]).astype(np.float64)
    if np.linalg.norm(x_axis) < 1e-8:
        x_axis = np.array([1.0, 0.0, 0.0])
    if np.linalg.norm(y_hint) < 1e-8 or abs(float(x_axis @ y_hint)) > 0.98:
        y_hint = np.array([0.0, 1.0, 0.0])
        if abs(float(x_axis @ y_hint)) > 0.98:
            y_hint = np.array([0.0, 0.0, 1.0])
    z_axis = safe_normalize(np.cross(x_axis, y_hint)).astype(np.float64)
    y_axis = safe_normalize(np.cross(z_axis, x_axis)).astype(np.float64)
    rotation = np.stack((x_axis, y_axis, z_axis), axis=1)
    scale = float(np.linalg.norm(points[9] - points[0]))
    return origin, rotation, scale if scale >= 1e-8 else 1.0


def _calibration_value(calibration: Mapping[str, Any] | None, *keys: str) -> Any:
    if not calibration:
        return None
    for key in keys:
        if key in calibration:
            return calibration[key]
    for container_key in ("hand", "calibration", "canonical", "normalization", "metrics"):
        nested = calibration.get(container_key)
        if isinstance(nested, Mapping):
            value = _calibration_value(nested, *keys)
            if value is not None:
                return value
    return None


def calibration_reference_points(calibration: Mapping[str, Any] | None) -> np.ndarray | None:
    """Extract a 21x3 reference skeleton from common calibration-cache keys."""
    value = _calibration_value(
        calibration,
        "reference_points",
        "canonical_points",
        "canonical_landmarks",
        "landmarks",
        "points",
        "mean_hand",
    )
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float32)
        if array.shape == (21, 3):
            return array
    except (TypeError, ValueError):
        pass
    return None


def calibration_normalization_scale(
    calibration: Mapping[str, Any] | None,
    default: float | None = None,
) -> float | None:
    """Return a positive cached hand scale, accepting common cache schemas."""
    value = _calibration_value(
        calibration,
        "normalization_scale",
        "palm_length",
        "hand_scale",
        "scale",
    )
    try:
        scale = float(value)
    except (TypeError, ValueError):
        return default
    return scale if np.isfinite(scale) and scale > 1e-8 else default


def canonicalize_hand_for_debug(
    points: np.ndarray,
    calibration: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Palm-localize points, optionally using a cached uniform hand scale.

    With no calibration this is exactly :func:`palm_localize`.  A cache only
    changes the denominator used for visualization; production retargeting is
    deliberately unaffected.
    """
    points = as_points_array(points)
    cached_scale = calibration_normalization_scale(calibration)
    if cached_scale is None:
        return palm_localize(points)
    origin, rotation, _ = _palm_frame(points)
    local = ((points - origin) @ rotation) / cached_scale
    return np.nan_to_num(local, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def compute_hand_debug_metrics(
    points: np.ndarray,
    calibration: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compute scale and pinch metrics without changing retargeting state."""
    points = as_points_array(points).astype(np.float64)
    palm_width = float(np.linalg.norm(points[5] - points[17]))
    palm_length = float(np.linalg.norm(points[9] - points[0]))
    normalization_scale = calibration_normalization_scale(calibration, palm_length)
    pinch_threshold = _calibration_value(
        calibration, "pinch_threshold", "thumb_index_threshold"
    )
    try:
        pinch_threshold = float(pinch_threshold)
    except (TypeError, ValueError):
        pinch_threshold = 0.35 * max(palm_width, 1e-8)
    thumb_index_distance = float(np.linalg.norm(points[4] - points[8]))
    gesture = _calibration_value(calibration, "gesture_mode", "gesture", "mode")
    cache_path = _calibration_value(calibration, "_cache_path", "cache_path", "path")
    return {
        "palm_width": palm_width,
        "palm_length": palm_length,
        "thumb_index_distance": thumb_index_distance,
        "thumb_middle_distance": float(np.linalg.norm(points[4] - points[12])),
        "normalization_scale": normalization_scale,
        "pinch_state": "PINCH" if thumb_index_distance <= pinch_threshold else "open",
        "gesture_mode": gesture if gesture is not None else "n/a",
        "calibration_cache": str(cache_path) if cache_path else "none",
    }


def plot_coordinate_frame(
    ax: Any,
    origin: np.ndarray,
    R: np.ndarray,
    scale: float = 0.25,
    label: str = "",
) -> None:
    """Draw an XYZ coordinate triad (red, green, blue) on a 3D axis."""
    origin = np.asarray(origin, dtype=float)
    R = np.asarray(R, dtype=float)
    for axis, color, name in zip(range(3), ("#d62728", "#2ca02c", "#1f77b4"), "XYZ"):
        vector = R[:, axis] * scale
        ax.quiver(*origin, *vector, color=color, linewidth=2.0, arrow_length_ratio=0.15)
        endpoint = origin + vector
        ax.text(*endpoint, f"{label} {name}".strip(), color=color, fontsize=8)
    ax.scatter(*origin, s=55, c="black", marker="x", depthshade=False)


def _set_equal_limits(ax: Any, point_sets: list[np.ndarray]) -> None:
    points = np.concatenate(point_sets, axis=0)
    center = (points.min(axis=0) + points.max(axis=0)) / 2.0
    radius = max(float(np.ptp(points, axis=0).max()) / 2.0, 1e-3) * 1.18
    ax.set_xlim(center[0] - radius, center[0] + radius)
    ax.set_ylim(center[1] - radius, center[1] + radius)
    ax.set_zlim(center[2] - radius, center[2] + radius)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:
        pass


def _draw_vectors(ax: Any, points: np.ndarray) -> None:
    for start, end, label, color in VECTOR_SPECS:
        vector = points[end] - points[start]
        ax.quiver(
            *points[start],
            *vector,
            color=color,
            linewidth=1.5,
            linestyle="--",
            arrow_length_ratio=0.10,
            alpha=0.85,
        )
        midpoint = points[start] + 0.52 * vector
        ax.text(*midpoint, label, color=color, fontsize=7)


def plot_hand_skeleton_3d(
    ax: Any,
    points: np.ndarray,
    title: str | None = None,
    show_labels: bool = True,
    show_vectors: bool = True,
    show_grid: bool = True,
    elevation: float = 24.0,
    azimuth: float = -62.0,
    alpha: float = 1.0,
    line_style: str = "-",
    label_prefix: str = "",
    set_limits: bool = True,
) -> Any:
    """Plot one color-coded 21-landmark hand skeleton on ``ax``."""
    points = as_points_array(points).astype(np.float64)
    for finger, chain in FINGERS.items():
        chain_points = points[list(chain)]
        color = FINGER_COLORS[finger]
        ax.plot(
            chain_points[:, 0],
            chain_points[:, 1],
            chain_points[:, 2],
            color=color,
            linewidth=2.5,
            linestyle=line_style,
            alpha=alpha,
            label=f"{label_prefix}{finger}",
        )
        sizes = [95 if index in FINGERTIPS else 38 for index in chain]
        ax.scatter(
            chain_points[:, 0],
            chain_points[:, 1],
            chain_points[:, 2],
            color=color,
            s=sizes,
            edgecolor="white",
            linewidth=0.7,
            depthshade=False,
            alpha=alpha,
        )
    palm = points[list(PALM_CHAIN)]
    ax.plot(*palm.T, color="#555555", linewidth=1.4, linestyle=":", alpha=alpha)
    if show_labels:
        for index, point in enumerate(points):
            text = f"{index}"
            if index in IMPORTANT_LABELS:
                text += f" {IMPORTANT_LABELS[index]}"
            ax.text(*point, text, fontsize=7, color="#222222", alpha=alpha)
    if show_vectors:
        _draw_vectors(ax, points)
    if title:
        ax.set_title(title, fontsize=11, pad=12)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.grid(show_grid, alpha=0.25)
    ax.view_init(elev=elevation, azim=azimuth)
    if set_limits:
        _set_equal_limits(ax, [points])
    return ax


def _metrics_text(metrics: Mapping[str, Any]) -> str:
    return "\n".join(
        (
            f"palm_width: {metrics['palm_width']:.4f}",
            f"palm_length: {metrics['palm_length']:.4f}",
            f"thumb↔index: {metrics['thumb_index_distance']:.4f}",
            f"thumb↔middle: {metrics['thumb_middle_distance']:.4f}",
            f"normalization_scale: {metrics['normalization_scale']:.4f}",
            f"pinch_state: {metrics['pinch_state']}",
            f"gesture: {metrics['gesture_mode']}",
            f"calibration: {metrics['calibration_cache']}",
        )
    )


def make_side_by_side_debug_figure(
    raw_points: np.ndarray,
    calibration: Mapping[str, Any] | None = None,
    reference_points: np.ndarray | None = None,
    *,
    show_labels: bool = True,
    show_vectors: bool = True,
    show_grid: bool = True,
    elevation: float = 24.0,
    azimuth: float = -62.0,
) -> Any:
    """Create raw, canonical, and reference/calibration comparison panels."""
    import matplotlib.pyplot as plt

    raw = as_points_array(raw_points)
    canonical = canonicalize_hand_for_debug(raw, calibration)
    if reference_points is None:
        reference_points = calibration_reference_points(calibration)
    reference = as_points_array(reference_points) if reference_points is not None else canonical
    third_title = (
        "Retargeting / calibration reference"
        if reference_points is not None
        else "Canonical skeleton + key vectors"
    )
    fig = plt.figure(figsize=(19, 7.2), constrained_layout=True)
    axes = [fig.add_subplot(1, 3, index + 1, projection="3d") for index in range(3)]
    common = dict(
        show_labels=show_labels,
        show_vectors=show_vectors,
        show_grid=show_grid,
        elevation=elevation,
        azimuth=azimuth,
    )
    plot_hand_skeleton_3d(axes[0], raw, "Raw Quest landmarks (Wrist frame)", **common)
    _, palm_R, raw_scale = _palm_frame(raw)
    frame_scale = max(raw_scale * 0.45, 1e-3)
    plot_coordinate_frame(axes[0], np.zeros(3), np.eye(3), frame_scale, "Wrist")
    plot_coordinate_frame(axes[0], raw[0], palm_R, frame_scale, "Palm")

    plot_hand_skeleton_3d(axes[1], canonical, "Palm-local / canonical normalized", **common)
    plot_coordinate_frame(axes[1], np.zeros(3), np.eye(3), 0.35, "Palm")

    plot_hand_skeleton_3d(axes[2], reference, third_title, **common)
    plot_coordinate_frame(axes[2], reference[0], np.eye(3), 0.35, "Reference")
    axes[0].text2D(
        0.01,
        0.99,
        _metrics_text(compute_hand_debug_metrics(raw, calibration)),
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    fig.suptitle("Quest Hand Channel → canonical hand → AeroHand reference", fontsize=14)
    return fig


def make_overlay_debug_figure(
    raw_points: np.ndarray,
    calibration: Mapping[str, Any] | None = None,
    reference_points: np.ndarray | None = None,
    *,
    show_labels: bool = True,
    show_vectors: bool = True,
    show_grid: bool = True,
    elevation: float = 24.0,
    azimuth: float = -62.0,
) -> Any:
    """Overlay geometries after expressing all of them in canonical units."""
    import matplotlib.pyplot as plt

    raw = as_points_array(raw_points)
    canonical = canonicalize_hand_for_debug(raw, calibration)
    _, raw_R, raw_scale = _palm_frame(raw)
    raw_aligned = ((raw - raw[0]) @ raw_R) / raw_scale
    if reference_points is None:
        reference_points = calibration_reference_points(calibration)
    reference = as_points_array(reference_points) if reference_points is not None else canonical

    fig = plt.figure(figsize=(10, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    plot_hand_skeleton_3d(
        ax,
        raw_aligned,
        show_labels=show_labels,
        show_vectors=False,
        show_grid=show_grid,
        elevation=elevation,
        azimuth=azimuth,
        alpha=0.38,
        label_prefix="raw aligned: ",
        set_limits=False,
    )
    plot_hand_skeleton_3d(
        ax,
        canonical,
        show_labels=False,
        show_vectors=show_vectors,
        show_grid=show_grid,
        elevation=elevation,
        azimuth=azimuth,
        alpha=0.9,
        line_style="-",
        label_prefix="canonical: ",
        set_limits=False,
    )
    if reference_points is not None:
        plot_hand_skeleton_3d(
            ax,
            reference,
            show_labels=False,
            show_vectors=False,
            show_grid=show_grid,
            elevation=elevation,
            azimuth=azimuth,
            alpha=0.65,
            line_style="--",
            label_prefix="reference: ",
            set_limits=False,
        )
    _set_equal_limits(ax, [raw_aligned, canonical, reference])
    plot_coordinate_frame(ax, np.zeros(3), np.eye(3), 0.35, "Palm")
    ax.set_title("Canonical-unit overlay (raw aligned / canonical / reference)")
    ax.text2D(
        0.01,
        0.99,
        _metrics_text(compute_hand_debug_metrics(raw, calibration)),
        transform=ax.transAxes,
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.85},
    )
    return fig


def render_realtime_debug_figure(
    fig: Any,
    axes: list[Any],
    raw_points: np.ndarray,
    *,
    calibration: Mapping[str, Any] | None = None,
    reference_points: np.ndarray | None = None,
    fps: float = 0.0,
    frame_count: int = 0,
    show_labels: bool = True,
    show_vectors: bool = True,
    show_grid: bool = True,
    elevation: float = 24.0,
    azimuth: float = -62.0,
    preserve_camera: bool = True,
    show_action: bool = True,
) -> None:
    """Redraw the three-panel real-time debug view on existing axes."""
    if len(axes) != 3:
        raise ValueError("Real-time side-by-side rendering requires exactly three axes")
    raw = as_points_array(raw_points)
    canonical = canonicalize_hand_for_debug(raw, calibration)
    if reference_points is None:
        reference_points = calibration_reference_points(calibration)
    reference = as_points_array(reference_points) if reference_points is not None else canonical

    cameras = [(getattr(ax, "elev", elevation), getattr(ax, "azim", azimuth)) for ax in axes]
    for ax in axes:
        ax.clear()
    common = dict(
        show_labels=show_labels,
        show_grid=show_grid,
        set_limits=True,
    )
    for index, ax in enumerate(axes):
        elev, azim = cameras[index] if preserve_camera else (elevation, azimuth)
        common["elevation"] = elev
        common["azimuth"] = azim
        if index == 0:
            plot_hand_skeleton_3d(
                ax, raw, "Raw Quest landmarks (Wrist frame)",
                show_vectors=False, **common
            )
        elif index == 1:
            plot_hand_skeleton_3d(
                ax, canonical, "Canonical / palm-local normalized",
                show_vectors=False, **common
            )
        else:
            plot_hand_skeleton_3d(
                ax, reference, "Retargeting reference vectors",
                show_vectors=show_vectors, **common
            )

    _, palm_R, raw_scale = _palm_frame(raw)
    plot_coordinate_frame(axes[0], raw[0], np.eye(3), max(raw_scale * 0.45, 1e-3), "Wrist")
    plot_coordinate_frame(axes[0], raw[0], palm_R, max(raw_scale * 0.45, 1e-3), "Palm")
    plot_coordinate_frame(axes[1], canonical[0], np.eye(3), 0.3, "Palm")
    plot_coordinate_frame(axes[2], reference[0], np.eye(3), 0.3, "Reference")

    metrics = compute_hand_debug_metrics(raw, calibration)
    status_lines = [
        f"FPS: {fps:5.1f}",
        f"frames: {frame_count}",
        f"palm_width: {metrics['palm_width']:.4f}",
        f"palm_length: {metrics['palm_length']:.4f}",
        f"normalization_scale: {metrics['normalization_scale']:.4f}",
        f"thumb↔index: {metrics['thumb_index_distance']:.4f}",
        f"thumb↔middle: {metrics['thumb_middle_distance']:.4f}",
        f"pinch: {metrics['pinch_state']}",
        f"calibration loaded: {'yes' if calibration else 'no'}",
    ]
    if show_action:
        action = quest_points_to_action_7d(raw)
        status_lines.append(
            "Aero action [abd,t1,t2,i,m,r,l]:\n"
            + np.array2string(action, precision=2, suppress_small=True)
        )
    axes[0].text2D(
        0.01,
        0.99,
        "\n".join(status_lines),
        transform=axes[0].transAxes,
        va="top",
        fontsize=8,
        family="monospace",
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
    )
    fig.suptitle(
        "Real-time Quest Hand Channel → canonical hand → retargeting debug",
        fontsize=14,
    )


class RealtimeHandSkeletonArtists:
    """Persistent Matplotlib artists for low-latency three-panel updates."""

    def __init__(
        self,
        fig: Any,
        axes: list[Any],
        *,
        elevation: float = 24.0,
        azimuth: float = -62.0,
    ) -> None:
        if len(axes) != 3:
            raise ValueError("Expected three 3D axes")
        self.fig = fig
        self.axes = axes
        self.elevation = elevation
        self.azimuth = azimuth
        self.lines: list[dict[str, Any]] = []
        self.scatters: list[dict[str, Any]] = []
        self.palm_lines: list[Any] = []
        self.labels: list[list[Any]] = []
        self.vector_lines: list[Any] = []
        self.vector_labels: list[Any] = []
        self.status = None
        self._limits_initialized = False
        self._build()

    def _build(self) -> None:
        titles = (
            "Raw Quest landmarks (Wrist frame)",
            "Canonical / palm-local normalized",
            "Retargeting reference vectors",
        )
        for ax, title in zip(self.axes, titles):
            ax.set_title(title, fontsize=11, pad=12)
            ax.set_xlabel("X")
            ax.set_ylabel("Y")
            ax.set_zlabel("Z")
            ax.view_init(elev=self.elevation, azim=self.azimuth)
            panel_lines = {}
            panel_scatters = {}
            for finger, chain in FINGERS.items():
                color = FINGER_COLORS[finger]
                panel_lines[finger], = ax.plot([], [], [], color=color, linewidth=2.5)
                panel_scatters[finger] = ax.scatter(
                    np.zeros(len(chain)), np.zeros(len(chain)), np.zeros(len(chain)),
                    color=color, s=[95 if i in FINGERTIPS else 38 for i in chain],
                    edgecolor="white", linewidth=0.7, depthshade=False,
                )
            palm_line, = ax.plot([], [], [], color="#555555", linewidth=1.4, linestyle=":")
            panel_labels = [ax.text(0, 0, 0, "", fontsize=7, color="#222222") for _ in range(21)]
            self.lines.append(panel_lines)
            self.scatters.append(panel_scatters)
            self.palm_lines.append(palm_line)
            self.labels.append(panel_labels)

        for start, end, label, color in VECTOR_SPECS:
            line, = self.axes[2].plot(
                [], [], [], color=color, linewidth=1.5, linestyle="--",
                marker=">", markevery=[1], markersize=5,
            )
            self.vector_lines.append(line)
            self.vector_labels.append(self.axes[2].text(0, 0, 0, label, color=color, fontsize=7))
        self.status = self.axes[0].text2D(
            0.01, 0.99, "", transform=self.axes[0].transAxes, va="top",
            fontsize=8, family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
        )
        self.fig.suptitle(
            "Real-time Quest Hand Channel → canonical hand → retargeting debug",
            fontsize=14,
        )

    def reset_camera(self) -> None:
        for ax in self.axes:
            ax.view_init(elev=self.elevation, azim=self.azimuth)

    def _update_panel(self, panel: int, points: np.ndarray, show_labels: bool) -> None:
        for finger, chain in FINGERS.items():
            chain_points = points[list(chain)]
            self.lines[panel][finger].set_data_3d(*chain_points.T)
            self.scatters[panel][finger]._offsets3d = tuple(chain_points.T)
        palm = points[list(PALM_CHAIN)]
        self.palm_lines[panel].set_data_3d(*palm.T)
        for index, (artist, point) in enumerate(zip(self.labels[panel], points)):
            artist.set_position((point[0], point[1]))
            artist.set_3d_properties(point[2])
            artist.set_text(
                f"{index} {IMPORTANT_LABELS[index]}" if index in IMPORTANT_LABELS else str(index)
            )
            artist.set_visible(show_labels)

    def _update_limits(self, raw: np.ndarray, canonical: np.ndarray, reference: np.ndarray) -> None:
        # Wrist-local dimensions are stable, so fixed limits avoid expensive
        # autoscaling and stop the camera from visually "breathing".
        if self._limits_initialized:
            return
        for ax, points in zip(self.axes, (raw, canonical, reference)):
            _set_equal_limits(ax, [points])
        self._limits_initialized = True

    def update(
        self,
        raw_points: np.ndarray,
        *,
        calibration: Mapping[str, Any] | None = None,
        reference_points: np.ndarray | None = None,
        fps: float = 0.0,
        frame_count: int = 0,
        show_labels: bool = True,
        show_vectors: bool = True,
        show_grid: bool = True,
        show_action: bool = True,
    ) -> None:
        raw = as_points_array(raw_points)
        canonical = canonicalize_hand_for_debug(raw, calibration)
        if reference_points is None:
            reference_points = calibration_reference_points(calibration)
        reference = as_points_array(reference_points) if reference_points is not None else canonical
        for panel, points in enumerate((raw, canonical, reference)):
            self._update_panel(panel, points, show_labels)
            self.axes[panel].grid(show_grid, alpha=0.25)
        self._update_limits(raw, canonical, reference)

        for line, text, spec in zip(self.vector_lines, self.vector_labels, VECTOR_SPECS):
            start, end, _label, _color = spec
            segment = reference[[start, end]]
            line.set_data_3d(*segment.T)
            midpoint = segment[0] + 0.52 * (segment[1] - segment[0])
            text.set_position((midpoint[0], midpoint[1]))
            text.set_3d_properties(midpoint[2])
            line.set_visible(show_vectors)
            text.set_visible(show_vectors)

        metrics = compute_hand_debug_metrics(raw, calibration)
        status_lines = [
            f"FPS: {fps:5.1f}",
            f"frames: {frame_count}",
            f"palm_width: {metrics['palm_width']:.4f}",
            f"palm_length: {metrics['palm_length']:.4f}",
            f"normalization_scale: {metrics['normalization_scale']:.4f}",
            f"thumb↔index: {metrics['thumb_index_distance']:.4f}",
            f"thumb↔middle: {metrics['thumb_middle_distance']:.4f}",
            f"pinch: {metrics['pinch_state']}",
            f"calibration loaded: {'yes' if calibration else 'no'}",
        ]
        if show_action:
            action = quest_points_to_action_7d(raw)
            status_lines.append(
                "Aero action [abd,t1,t2,i,m,r,l]:\n"
                + np.array2string(action, precision=2, suppress_small=True)
            )
        self.status.set_text("\n".join(status_lines))


class LowLatencyHandSkeletonArtists:
    """Persistent single-panel artists for the default low-latency live view."""

    def __init__(
        self,
        fig: Any,
        ax: Any,
        *,
        elevation: float = 24.0,
        azimuth: float = -62.0,
    ) -> None:
        self.fig = fig
        self.ax = ax
        self.elevation = elevation
        self.azimuth = azimuth
        self.lines = {}
        self.labels = []
        self.vector_lines = []
        self.vector_labels = []
        self._limits_initialized = False

        ax.set_title("Latest Quest hand skeleton (Wrist-local)", fontsize=12)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.view_init(elev=elevation, azim=azimuth)
        for finger, chain in FINGERS.items():
            color = FINGER_COLORS[finger]
            self.lines[finger], = ax.plot([], [], [], color=color, linewidth=2.5)
        key_ids = (0, *FINGERTIPS)
        self.key_markers = ax.scatter(
            np.zeros(len(key_ids)),
            np.zeros(len(key_ids)),
            np.zeros(len(key_ids)),
            s=[65, 105, 105, 105, 105, 105],
            c=["black", *(FINGER_COLORS.values())],
            edgecolor="white",
            linewidth=0.8,
            depthshade=False,
        )
        self.labels = [
            ax.text(0, 0, 0, "", fontsize=7, color="#222222") for _ in range(21)
        ]
        for _start, _end, label, color in VECTOR_SPECS:
            line, = ax.plot(
                [], [], [], color=color, linewidth=1.5, linestyle="--",
                marker=">", markevery=[1], markersize=5,
            )
            self.vector_lines.append(line)
            self.vector_labels.append(ax.text(0, 0, 0, label, color=color, fontsize=7))
        self.status = ax.text2D(
            0.01,
            0.99,
            "",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            family="monospace",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88},
        )

    def reset_camera(self) -> None:
        self.ax.view_init(elev=self.elevation, azim=self.azimuth)

    def update(
        self,
        points: np.ndarray,
        *,
        fps: float,
        frame_count: int,
        calibration: Mapping[str, Any] | None = None,
        show_labels: bool = False,
        show_vectors: bool = True,
        show_grid: bool = True,
    ) -> None:
        points = as_points_array(points)
        for finger, chain in FINGERS.items():
            self.lines[finger].set_data_3d(*points[list(chain)].T)
        key_points = points[[0, *FINGERTIPS]]
        self.key_markers._offsets3d = tuple(key_points.T)
        for index, (artist, point) in enumerate(zip(self.labels, points)):
            artist.set_position((point[0], point[1]))
            artist.set_3d_properties(point[2])
            artist.set_text(
                f"{index} {IMPORTANT_LABELS[index]}" if index in IMPORTANT_LABELS else str(index)
            )
            artist.set_visible(show_labels)
        for line, text, spec in zip(self.vector_lines, self.vector_labels, VECTOR_SPECS):
            start, end, _label, _color = spec
            segment = points[[start, end]]
            line.set_data_3d(*segment.T)
            midpoint = segment[0] + 0.52 * (segment[1] - segment[0])
            text.set_position((midpoint[0], midpoint[1]))
            text.set_3d_properties(midpoint[2])
            line.set_visible(show_vectors)
            text.set_visible(show_vectors)
        self.ax.grid(show_grid, alpha=0.25)
        if not self._limits_initialized:
            _set_equal_limits(self.ax, [points])
            self._limits_initialized = True
        metrics = compute_hand_debug_metrics(points, calibration)
        self.status.set_text(
            "\n".join(
                (
                    f"viewer FPS: {fps:4.1f}",
                    f"latest frame: {frame_count}",
                    f"thumb↔index: {metrics['thumb_index_distance']:.4f}",
                    f"thumb↔middle: {metrics['thumb_middle_distance']:.4f}",
                    f"pinch: {metrics['pinch_state']}",
                )
            )
        )


def save_figure(fig: Any, output: str | Path, dpi: int = 180) -> Path:
    """Save a figure, creating its parent directory."""
    output = Path(output).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight")
    return output
