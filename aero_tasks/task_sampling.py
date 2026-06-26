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
    rack_center_xy_m: tuple[float, float] = (0.0, 0.0)
    rack_x_range_m: tuple[float, float] = (-0.36, 0.36)
    rack_y_range_m: tuple[float, float] = (-0.12, 0.24)
    rack_yaw_range_deg: tuple[float, float] = (-30.0, 30.0)
    beam_length_m: float = 0.255
    reject_robot_collision: bool = True
    robot_body_prefixes: tuple[str, ...] = ("piper_original", "piper_aerohand")
    max_pose_sample_attempts: int = 100


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


def _freejoint_qpos_adr(model: mujoco.MjModel, joint_name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Missing freejoint {joint_name!r}")
    if int(model.jnt_type[joint_id]) != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Joint {joint_name!r} is not a freejoint")
    return int(model.jnt_qposadr[joint_id])


def _freejoint_pose_from_qpos(model: mujoco.MjModel, qpos: np.ndarray, joint_name: str) -> FreejointPose:
    adr = _freejoint_qpos_adr(model, joint_name)
    return FreejointPose(
        pos=tuple(float(value) for value in qpos[adr : adr + 3]),
        quat=tuple(float(value) for value in qpos[adr + 3 : adr + 7]),
    )


def _body_matches_prefix(name: str, prefixes: tuple[str, ...]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}/") for prefix in prefixes)


def _sampled_rack_pose_collides_with_robot(
    model: mujoco.MjModel,
    *,
    qpos: np.ndarray,
    rack_body_id: int,
    rack_pos: np.ndarray,
    rack_R: np.ndarray,
    object_freejoint: str,
    object_pos: np.ndarray,
    object_R: np.ndarray,
    robot_body_prefixes: tuple[str, ...],
) -> bool:
    rack_body = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, rack_body_id) or ""
    rack_prefix = rack_body.split("/", 1)[0]
    object_prefix = object_freejoint.removesuffix("_free")
    rack_body_pos = model.body_pos[rack_body_id].copy()
    rack_body_quat = model.body_quat[rack_body_id].copy()
    qpos_test = qpos.copy()
    payload_adr = _freejoint_qpos_adr(model, object_freejoint)

    try:
        model.body_pos[rack_body_id] = rack_pos
        model.body_quat[rack_body_id] = _matrix_to_quat_wxyz(rack_R)
        qpos_test[payload_adr : payload_adr + 3] = object_pos
        qpos_test[payload_adr + 3 : payload_adr + 7] = _matrix_to_quat_wxyz(object_R)
        data = mujoco.MjData(model)
        data.qpos[:] = qpos_test
        mujoco.mj_forward(model, data)
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[contact.geom1]),
            ) or ""
            body2 = mujoco.mj_id2name(
                model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(model.geom_bodyid[contact.geom2]),
            ) or ""
            rack_or_object_1 = body1.startswith(f"{rack_prefix}/") or _body_matches_prefix(
                body1,
                (object_prefix,),
            )
            rack_or_object_2 = body2.startswith(f"{rack_prefix}/") or _body_matches_prefix(
                body2,
                (object_prefix,),
            )
            robot_1 = _body_matches_prefix(body1, robot_body_prefixes)
            robot_2 = _body_matches_prefix(body2, robot_body_prefixes)
            if (rack_or_object_1 and robot_2) or (rack_or_object_2 and robot_1):
                return True
        return False
    finally:
        model.body_pos[rack_body_id] = rack_body_pos
        model.body_quat[rack_body_id] = rack_body_quat


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

    base_offset = float(np.dot(base_object_local - bar_center_local, axis_local))
    rack_center_xy = np.asarray(config.rack_center_xy_m, dtype=np.float64).reshape(2)
    sample_attempt = 0
    last_collision = False
    max_attempts = max(1, int(config.max_pose_sample_attempts))
    for sample_attempt in range(1, max_attempts + 1):
        offset = float(rng.uniform(config.offset_range_m[0], config.offset_range_m[1]))
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
            sampled_rack_pos = np.array(
                [
                    rack_center_xy[0] + rack_delta_xy[0],
                    rack_center_xy[1] + rack_delta_xy[1],
                    base_rack_pos[2],
                ],
                dtype=np.float64,
            )
            sampled_rack_R = _yaw_matrix(np.deg2rad(rack_yaw_deg)) @ base_rack_R

        sampled_pos = sampled_rack_pos + sampled_rack_R @ sampled_object_local
        sampled_object_R = sampled_rack_R @ base_object_R_rack
        last_collision = False
        if config.sample_rack_pose and config.reject_robot_collision:
            last_collision = _sampled_rack_pose_collides_with_robot(
                model,
                qpos=qpos,
                rack_body_id=rack_body_id,
                rack_pos=sampled_rack_pos,
                rack_R=sampled_rack_R,
                object_freejoint=config.object_freejoint,
                object_pos=sampled_pos,
                object_R=sampled_object_R,
                robot_body_prefixes=config.robot_body_prefixes,
            )
        if not last_collision:
            break
    else:
        raise RuntimeError(
            "Failed to sample a rack pose without initial robot collision after "
            f"{max_attempts} attempts"
        )

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
            "rack_center_xy_m": [float(value) for value in rack_center_xy],
            "rack_delta_xy_m": [float(value) for value in rack_delta_xy],
            "rack_yaw_deg": rack_yaw_deg,
            "rack_x_range_m": [float(value) for value in config.rack_x_range_m],
            "rack_y_range_m": [float(value) for value in config.rack_y_range_m],
            "rack_yaw_range_deg": [float(value) for value in config.rack_yaw_range_deg],
            "sample_attempt": int(sample_attempt),
            "reject_robot_collision": bool(config.reject_robot_collision),
            "last_rejected_robot_collision": bool(last_collision),
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
