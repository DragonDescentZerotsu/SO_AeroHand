"""Reusable task sampling helpers for expert trajectory generation."""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(f"MuJoCo is required for task sampling: {exc}") from exc


@dataclass(frozen=True)
class FreejointPose:
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]

    def as_json(self) -> dict[str, list[float]]:
        return {
            "pos": list(self.pos),
            "quat": list(self.quat),
        }


@dataclass(frozen=True)
class BodyPose:
    pos: tuple[float, float, float]
    quat: tuple[float, float, float, float]

    def as_json(self) -> dict[str, list[float]]:
        return {
            "pos": list(self.pos),
            "quat": list(self.quat),
        }


@dataclass(frozen=True)
class EpisodeSpec:
    """Episode-level initial state overrides and sampler metadata."""

    version: int = 1
    seed: int | None = None
    freejoint_poses: dict[str, FreejointPose] = field(default_factory=dict)
    body_poses: dict[str, BodyPose] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)

    def as_json(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "seed": self.seed,
            "freejoint_poses": {
                name: pose.as_json()
                for name, pose in sorted(self.freejoint_poses.items())
            },
            "body_poses": {
                name: pose.as_json()
                for name, pose in sorted(self.body_poses.items())
            },
            "metadata": self.metadata,
        }


@dataclass(frozen=True)
class RackBarSampleConfig:
    """Sample an object freejoint along one rack-local bar axis."""

    rack_body: str = "pipette_rack_0/pipette_rack"
    object_freejoint: str = "pipette_0_free"
    axis_local: tuple[float, float, float] = (1.0, 0.0, 0.0)
    bar_center_local: tuple[float, float, float] = (0.0, 0.0, 0.228262)
    offset_range_m: tuple[float, float] = (-0.1275, 0.1275)
    sample_rack_pose: bool = False
    rack_x_range_m: tuple[float, float] = (-0.04, 0.04)
    rack_y_range_m: tuple[float, float] = (-0.03, 0.03)
    rack_yaw_range_deg: tuple[float, float] = (-8.0, 8.0)
    beam_length_m: float = 0.255


def _normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def _quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = _normalize(np.asarray(quat, dtype=np.float64).reshape(4))
    w, x, y, z = quat
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64).reshape(3, 3)
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, matrix.reshape(9))
    if quat[0] < 0.0:
        quat *= -1.0
    return _normalize(quat)


def _yaw_matrix(yaw_rad: float) -> np.ndarray:
    c = float(np.cos(yaw_rad))
    s = float(np.sin(yaw_rad))
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _body_pose_from_model(model: mujoco.MjModel, body_name: str) -> BodyPose:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing body {body_name!r}")
    return BodyPose(
        pos=tuple(float(value) for value in model.body_pos[body_id]),
        quat=tuple(float(value) for value in model.body_quat[body_id]),
    )


def _freejoint_pose_from_qpos(model: mujoco.MjModel, qpos: np.ndarray, joint_name: str) -> FreejointPose:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Missing freejoint {joint_name!r}")
    if int(model.jnt_type[joint_id]) != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Joint {joint_name!r} is not a freejoint")
    adr = int(model.jnt_qposadr[joint_id])
    return FreejointPose(
        pos=tuple(float(value) for value in qpos[adr : adr + 3]),
        quat=tuple(float(value) for value in qpos[adr + 3 : adr + 7]),
    )


def sample_pipette_rack_bar_episode(
    model: mujoco.MjModel,
    *,
    rng: np.random.Generator,
    seed: int,
    config: RackBarSampleConfig = RackBarSampleConfig(),
) -> EpisodeSpec:
    """Sample a rack pose and slide the pipette along the rack bar.

    The offset is measured from ``bar_center_local`` projected onto
    ``axis_local``. The tuned rack/pipette local Y/Z relation and object
    orientation relative to the rack are preserved.
    """

    data = mujoco.MjData(model)
    qpos = model.qpos0.copy()
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)

    rack_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, config.rack_body)
    if rack_body_id < 0:
        raise ValueError(f"Missing rack body {config.rack_body!r}")
    base_object_pose = _freejoint_pose_from_qpos(model, qpos, config.object_freejoint)

    axis_local = _normalize(np.asarray(config.axis_local, dtype=np.float64).reshape(3))
    bar_center_local = np.asarray(config.bar_center_local, dtype=np.float64).reshape(3)
    base_rack_pos = data.xpos[rack_body_id].copy()
    base_rack_R = data.xmat[rack_body_id].reshape(3, 3).copy()
    base_rack_pose = _body_pose_from_model(model, config.rack_body)
    base_object_pos = np.asarray(base_object_pose.pos, dtype=np.float64)
    base_object_R = _quat_wxyz_to_matrix(np.asarray(base_object_pose.quat, dtype=np.float64))
    base_object_local = base_rack_R.T @ (base_object_pos - base_rack_pos)
    base_object_R_rack = base_rack_R.T @ base_object_R

    offset = float(rng.uniform(config.offset_range_m[0], config.offset_range_m[1]))
    base_offset = float(np.dot(base_object_local - bar_center_local, axis_local))
    sampled_object_local = base_object_local + (offset - base_offset) * axis_local

    rack_delta_xy = np.zeros(2, dtype=np.float64)
    rack_yaw_deg = 0.0
    sampled_rack_pos = base_rack_pos.copy()
    sampled_rack_R = base_rack_R.copy()
    if config.sample_rack_pose:
        rack_delta_xy = np.array(
            [
                float(rng.uniform(config.rack_x_range_m[0], config.rack_x_range_m[1])),
                float(rng.uniform(config.rack_y_range_m[0], config.rack_y_range_m[1])),
            ],
            dtype=np.float64,
        )
        rack_yaw_deg = float(rng.uniform(config.rack_yaw_range_deg[0], config.rack_yaw_range_deg[1]))
        sampled_rack_pos = base_rack_pos + np.array([rack_delta_xy[0], rack_delta_xy[1], 0.0], dtype=np.float64)
        sampled_rack_R = _yaw_matrix(np.deg2rad(rack_yaw_deg)) @ base_rack_R

    sampled_pos = sampled_rack_pos + sampled_rack_R @ sampled_object_local
    sampled_object_R = sampled_rack_R @ base_object_R_rack
    axis_world = _normalize(sampled_rack_R @ axis_local)
    sampled_pose = FreejointPose(
        pos=tuple(float(value) for value in sampled_pos),
        quat=tuple(float(value) for value in _matrix_to_quat_wxyz(sampled_object_R)),
    )
    rack_pose = BodyPose(
        pos=tuple(float(value) for value in sampled_rack_pos),
        quat=tuple(float(value) for value in _matrix_to_quat_wxyz(sampled_rack_R)),
    )

    return EpisodeSpec(
        seed=int(seed),
        freejoint_poses={config.object_freejoint: sampled_pose},
        body_poses={config.rack_body: rack_pose},
        metadata={
            "sampler": "pipette_rack_bar",
            "offset_reference": "rack_bar_center",
            "rack_body": config.rack_body,
            "object_freejoint": config.object_freejoint,
            "bar_center_local": [float(value) for value in bar_center_local],
            "axis_local": [float(value) for value in axis_local],
            "axis_world": [float(value) for value in axis_world],
            "offset_m": offset,
            "base_offset_m": base_offset,
            "offset_range_m": [float(value) for value in config.offset_range_m],
            "beam_length_m": float(config.beam_length_m),
            "sample_rack_pose": bool(config.sample_rack_pose),
            "rack_delta_xy_m": [float(value) for value in rack_delta_xy],
            "rack_yaw_deg": rack_yaw_deg,
            "rack_x_range_m": [float(value) for value in config.rack_x_range_m],
            "rack_y_range_m": [float(value) for value in config.rack_y_range_m],
            "rack_yaw_range_deg": [float(value) for value in config.rack_yaw_range_deg],
            "base_rack_pose": base_rack_pose.as_json(),
            "sampled_rack_pose": rack_pose.as_json(),
            "base_object_pose": base_object_pose.as_json(),
            "base_object_local": [float(value) for value in base_object_local],
            "sampled_object_local": [float(value) for value in sampled_object_local],
        },
    )


def apply_episode_spec_to_model(model: mujoco.MjModel, spec: EpisodeSpec) -> None:
    for body_name, pose in spec.body_poses.items():
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            raise ValueError(f"Missing body {body_name!r}")
        model.body_pos[body_id] = np.asarray(pose.pos, dtype=np.float64)
        model.body_quat[body_id] = _normalize(np.asarray(pose.quat, dtype=np.float64))


def apply_episode_spec_to_qpos(model: mujoco.MjModel, qpos: np.ndarray, spec: EpisodeSpec) -> None:
    for joint_name, pose in spec.freejoint_poses.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Missing freejoint {joint_name!r}")
        if int(model.jnt_type[joint_id]) != mujoco.mjtJoint.mjJNT_FREE:
            raise ValueError(f"Joint {joint_name!r} is not a freejoint")
        adr = int(model.jnt_qposadr[joint_id])
        qpos[adr : adr + 3] = np.asarray(pose.pos, dtype=np.float64)
        qpos[adr + 3 : adr + 7] = np.asarray(pose.quat, dtype=np.float64)


def write_episode_spec(path: Path, spec: EpisodeSpec) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(spec.as_json(), indent=2), encoding="utf-8")


def load_episode_spec(path: Path) -> EpisodeSpec:
    raw = json.loads(path.read_text(encoding="utf-8"))
    freejoint_poses = {
        name: FreejointPose(
            pos=tuple(float(value) for value in pose["pos"]),
            quat=tuple(float(value) for value in pose["quat"]),
        )
        for name, pose in raw.get("freejoint_poses", {}).items()
    }
    body_poses = {
        name: BodyPose(
            pos=tuple(float(value) for value in pose["pos"]),
            quat=tuple(float(value) for value in pose["quat"]),
        )
        for name, pose in raw.get("body_poses", {}).items()
    }
    return EpisodeSpec(
        version=int(raw.get("version", 1)),
        seed=raw.get("seed"),
        freejoint_poses=freejoint_poses,
        body_poses=body_poses,
        metadata=dict(raw.get("metadata", {})),
    )
