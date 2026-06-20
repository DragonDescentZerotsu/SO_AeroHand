from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HAND_CONNECTIONS = (
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),
    (0, 9),
    (9, 10),
    (10, 11),
    (11, 12),
    (0, 13),
    (13, 14),
    (14, 15),
    (15, 16),
    (0, 17),
    (17, 18),
    (18, 19),
    (19, 20),
    (5, 9),
    (9, 13),
    (13, 17),
)


def normalize_hand_landmarks(points: np.ndarray) -> np.ndarray:
    """Center a 21-point hand skeleton at its wrist and normalize scale."""
    points = np.asarray(points, dtype=np.float64).reshape(21, 3)
    centered = points - points[0]
    scale = float(np.max(np.linalg.norm(centered, axis=1)))
    if scale < 1e-8:
        return centered
    return centered / scale


def plot_hand_skeleton(
    ax,
    points: np.ndarray,
    title: str,
    color: str,
    connections: Iterable[tuple[int, int]] = HAND_CONNECTIONS,
) -> None:
    """Plot a normalized 3D hand skeleton on one matplotlib axis."""
    points = normalize_hand_landmarks(points)
    for start, end in connections:
        segment = points[[start, end]]
        ax.plot(segment[:, 0], segment[:, 1], segment[:, 2], color=color, linewidth=1.8, alpha=0.9)
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], color=color, s=14, depthshade=False)
    ax.scatter(points[0, 0], points[0, 1], points[0, 2], color="black", s=24, depthshade=False)
    ax.set_title(title, fontsize=8, pad=6)
    _format_axis(ax)


def save_skeleton_comparison_grid(samples: list[dict], output_path: str | Path) -> None:
    """Save Quest wrist-local skeletons next to MuJoCo AeroHand skeletons."""
    if not samples:
        raise ValueError("samples must not be empty")
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = len(samples)
    fig = plt.figure(figsize=(10.0, max(3.2, 2.8 * rows)))
    for row, sample in enumerate(samples):
        quest_ax = fig.add_subplot(rows, 2, row * 2 + 1, projection="3d")
        robot_ax = fig.add_subplot(rows, 2, row * 2 + 2, projection="3d")
        frame_index = int(sample.get("frame_index", row))
        human_dist = float(sample.get("human_pinch_distance", np.nan))
        robot_dist = float(sample.get("robot_pinch_distance", np.nan))
        plot_hand_skeleton(
            quest_ax,
            sample["quest_landmarks_wrist"],
            f"Quest 3 wrist-local\nframe {frame_index} | pinch {human_dist:.3f} m",
            "#2563eb",
        )
        plot_hand_skeleton(
            robot_ax,
            sample["robot_landmarks_world"],
            f"MuJoCo AeroHand sites\nframe {frame_index} | pinch {robot_dist:.3f} m",
            "#dc2626",
        )
    fig.suptitle("Quest 3 hand skeleton vs MuJoCo AeroHand landmark skeleton", fontsize=12)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def show_human_ghost_skeleton(frames: list) -> None:
    """Compatibility hook for future interactive viewers."""
    del frames


def _format_axis(ax) -> None:
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_zlim(-1.05, 1.05)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.view_init(elev=18, azim=-70)
    try:
        ax.set_box_aspect((1, 1, 1))
    except AttributeError:  # pragma: no cover - older matplotlib.
        pass
