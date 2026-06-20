"""Configuration-driven vector tasks for Quest-to-AeroHand retargeting.

Human landmarks enter in the Quest Wrist frame. Human and robot skeletons are
canonicalized independently before vectors are compared, so no Quest-world to
robot-base transform is implied by this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from aero_quest.mujoco_landmarks import ROBOT_LANDMARK_SITE_NAMES
from aero_quest.retargeting import as_points_array, palm_localize


@dataclass(frozen=True)
class VectorTask:
    """One corresponding human/robot landmark vector."""

    name: str
    origin_human: int
    target_human: int
    origin_robot: str
    target_robot: str
    weight: float = 1.0
    pinch_weight: float | None = None

    def __post_init__(self) -> None:
        for field_name in ("origin_human", "target_human"):
            index = int(getattr(self, field_name))
            if not 0 <= index < 21:
                raise ValueError(f"{field_name} must be in [0, 20], got {index}")
        if self.origin_human == self.target_human:
            raise ValueError(f"Vector task {self.name!r} has identical human endpoints")
        if self.origin_robot not in ROBOT_LANDMARK_SITE_NAMES:
            raise ValueError(f"Unknown robot landmark site: {self.origin_robot}")
        if self.target_robot not in ROBOT_LANDMARK_SITE_NAMES:
            raise ValueError(f"Unknown robot landmark site: {self.target_robot}")
        if not np.isfinite(self.weight) or self.weight < 0:
            raise ValueError(f"Task weight must be finite and non-negative, got {self.weight}")
        if self.pinch_weight is not None and (
            not np.isfinite(self.pinch_weight) or self.pinch_weight < 0
        ):
            raise ValueError("pinch_weight must be finite and non-negative")

    @property
    def robot_indices(self) -> tuple[int, int]:
        return (
            ROBOT_LANDMARK_SITE_NAMES.index(self.origin_robot),
            ROBOT_LANDMARK_SITE_NAMES.index(self.target_robot),
        )


def load_vector_tasks(path: str | Path) -> tuple[list[VectorTask], dict[str, Any]]:
    """Load ``retargeting.vector_tasks`` and return tasks plus full config."""
    path = Path(path).expanduser()
    with path.open("r", encoding="utf-8") as stream:
        config = yaml.safe_load(stream) or {}
    retargeting = config.get("retargeting", config)
    if not isinstance(retargeting, Mapping):
        raise ValueError("retargeting config must be a mapping")
    raw_tasks = retargeting.get("vector_tasks", [])
    if not isinstance(raw_tasks, Sequence) or isinstance(raw_tasks, (str, bytes)):
        raise ValueError("retargeting.vector_tasks must be a list")
    tasks = []
    for index, raw in enumerate(raw_tasks):
        if not isinstance(raw, Mapping):
            raise ValueError(f"vector_tasks[{index}] must be a mapping")
        values = dict(raw)
        values.setdefault("name", f"vector_{index}")
        tasks.append(VectorTask(**values))
    if not tasks:
        raise ValueError(f"No vector tasks configured in {path}")
    return tasks, config


def extract_task_vectors(
    points: np.ndarray,
    tasks: Sequence[VectorTask],
    *,
    source: str,
    canonicalize: bool = True,
) -> np.ndarray:
    """Extract an ``(N, 3)`` vector array from human or robot landmarks."""
    points = as_points_array(points).astype(np.float64)
    if canonicalize:
        points = palm_localize(points).astype(np.float64)
    vectors = []
    if source == "human":
        for task in tasks:
            vectors.append(points[task.target_human] - points[task.origin_human])
    elif source == "robot":
        for task in tasks:
            origin, target = task.robot_indices
            vectors.append(points[target] - points[origin])
    else:
        raise ValueError("source must be 'human' or 'robot'")
    return np.asarray(vectors, dtype=np.float64)


def vector_matching_loss(
    human_points: np.ndarray,
    robot_points: np.ndarray,
    tasks: Sequence[VectorTask],
    *,
    pinch_active: bool = False,
    huber_delta: float = 0.08,
    canonicalize: bool = True,
) -> dict[str, Any]:
    """Return weighted robust loss between canonical human/robot vectors."""
    if huber_delta <= 0:
        raise ValueError("huber_delta must be positive")
    human_vectors = extract_task_vectors(
        human_points, tasks, source="human", canonicalize=canonicalize
    )
    robot_vectors = extract_task_vectors(
        robot_points, tasks, source="robot", canonicalize=canonicalize
    )
    errors = np.linalg.norm(robot_vectors - human_vectors, axis=1)
    huber = np.where(
        errors <= huber_delta,
        0.5 * errors * errors,
        huber_delta * (errors - 0.5 * huber_delta),
    )
    weights = np.asarray(
        [
            task.pinch_weight
            if pinch_active and task.pinch_weight is not None
            else task.weight
            for task in tasks
        ],
        dtype=np.float64,
    )
    denominator = max(float(np.sum(weights)), 1e-8)
    weighted = weights * huber
    return {
        "total": float(np.sum(weighted) / denominator),
        "per_task": {task.name: float(value) for task, value in zip(tasks, huber)},
        "errors": {task.name: float(value) for task, value in zip(tasks, errors)},
        "weights": {task.name: float(value) for task, value in zip(tasks, weights)},
    }
