"""Kinematic carried-payload collision checks for task planners."""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Sequence

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(f"MuJoCo is required for payload collision checks: {exc}") from exc

from aero_tasks.motion_planning import JointGroup, PlanningModel


@dataclass(frozen=True)
class PayloadCollisionHit:
    frame_index: int
    label: str
    body1: str
    body2: str
    distance: float


@dataclass(frozen=True)
class PayloadCollisionReport:
    checked_frames: int
    hits: tuple[PayloadCollisionHit, ...]

    @property
    def ok(self) -> bool:
        return len(self.hits) == 0

    def as_json(self) -> dict[str, object]:
        return {
            "checked_frames": self.checked_frames,
            "ok": self.ok,
            "hits": [
                {
                    "frame_index": hit.frame_index,
                    "label": hit.label,
                    "body1": hit.body1,
                    "body2": hit.body2,
                    "distance": hit.distance,
                }
                for hit in self.hits
            ],
        }


@dataclass(frozen=True)
class CarriedPayloadState:
    tcp_world: np.ndarray
    tcp_R_world: np.ndarray
    payload_world: np.ndarray
    payload_R_world: np.ndarray
    payload_offset_tcp_local: np.ndarray
    payload_R_tcp: np.ndarray
    hook_world: np.ndarray
    hook_offset_tcp_local: np.ndarray

    def as_json(self) -> dict[str, object]:
        return {
            "tcp_world": self.tcp_world.tolist(),
            "payload_world": self.payload_world.tolist(),
            "payload_offset_tcp_local": self.payload_offset_tcp_local.tolist(),
            "hook_world": self.hook_world.tolist(),
            "hook_offset_tcp_local": self.hook_offset_tcp_local.tolist(),
        }


def _matches(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}/") for prefix in prefixes)


def measure_carried_payload_state(
    planner: PlanningModel,
    *,
    link_body: str,
    tcp_offset_local: np.ndarray,
    payload_body: str,
    hook_body: str,
    hook_reference_local: np.ndarray,
) -> CarriedPayloadState:
    """Measure payload root and hook poses relative to the current TCP pose.

    ``payload_body`` is usually the freejoint wrapper used for rigid sweep
    collision checks. ``hook_body`` is the body that owns
    ``hook_reference_local``; for the AutoBio pipette this is
    ``pipette_0/pipette``, not the wrapper body ``pipette_0``.
    """

    model = planner.model
    data = planner.data
    link_body_id = planner.body_id(link_body)
    payload_body_id = planner.body_id(payload_body)
    hook_body_id = planner.body_id(hook_body)
    tcp_offset_local = np.asarray(tcp_offset_local, dtype=np.float64).reshape(3)
    hook_reference_local = np.asarray(hook_reference_local, dtype=np.float64).reshape(3)

    link_pos = data.xpos[link_body_id].copy()
    tcp_R_world = data.xmat[link_body_id].reshape(3, 3).copy()
    tcp_world = link_pos + tcp_R_world @ tcp_offset_local
    payload_world = data.xpos[payload_body_id].copy()
    payload_R_world = data.xmat[payload_body_id].reshape(3, 3).copy()
    hook_body_world = data.xpos[hook_body_id].copy()
    hook_body_R_world = data.xmat[hook_body_id].reshape(3, 3).copy()
    hook_world = hook_body_world + hook_body_R_world @ hook_reference_local

    return CarriedPayloadState(
        tcp_world=tcp_world,
        tcp_R_world=tcp_R_world,
        payload_world=payload_world,
        payload_R_world=payload_R_world,
        payload_offset_tcp_local=tcp_R_world.T @ (payload_world - tcp_world),
        payload_R_tcp=tcp_R_world.T @ payload_R_world,
        hook_world=hook_world,
        hook_offset_tcp_local=tcp_R_world.T @ (hook_world - tcp_world),
    )


def check_carried_payload_path(
    planner: PlanningModel,
    *,
    group: JointGroup,
    template_qpos: np.ndarray,
    q_path: Sequence[np.ndarray],
    labels: Sequence[str],
    link_body: str,
    tcp_offset_local: np.ndarray,
    payload_freejoint: str,
    payload_prefixes: Sequence[str],
    obstacle_prefixes: Sequence[str],
    payload_offset_tcp_local: np.ndarray,
    payload_R_tcp: np.ndarray,
    carry_labels: Sequence[str],
    max_hits: int = 16,
) -> PayloadCollisionReport:
    """Check whether a payload rigidly attached to the TCP sweeps into obstacles."""

    if len(q_path) != len(labels):
        raise ValueError(f"q_path length {len(q_path)} does not match labels length {len(labels)}")

    model = planner.model
    data = planner.data
    link_body_id = planner.body_id(link_body)
    payload_joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, payload_freejoint)
    if payload_joint_id < 0:
        raise ValueError(f"Missing payload freejoint {payload_freejoint!r}")
    if int(model.jnt_type[payload_joint_id]) != mujoco.mjtJoint.mjJNT_FREE:
        raise ValueError(f"Joint {payload_freejoint!r} is not a freejoint")
    payload_qpos_adr = int(model.jnt_qposadr[payload_joint_id])
    carry_label_set = set(carry_labels)
    tcp_offset_local = np.asarray(tcp_offset_local, dtype=np.float64).reshape(3)
    payload_offset_tcp_local = np.asarray(payload_offset_tcp_local, dtype=np.float64).reshape(3)
    payload_R_tcp = np.asarray(payload_R_tcp, dtype=np.float64).reshape(3, 3)

    checked = 0
    hits: list[PayloadCollisionHit] = []
    for frame_index, (q, label) in enumerate(zip(q_path, labels, strict=True)):
        if label not in carry_label_set:
            continue
        qpos = np.asarray(template_qpos, dtype=np.float64).copy()
        planner.set_joint_group(qpos, group, np.asarray(q, dtype=np.float64))
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        link_pos = data.xpos[link_body_id].copy()
        link_R = data.xmat[link_body_id].reshape(3, 3).copy()
        tcp_pos = link_pos + link_R @ tcp_offset_local
        tcp_R = link_R
        payload_pos = tcp_pos + tcp_R @ payload_offset_tcp_local
        payload_R = tcp_R @ payload_R_tcp
        qpos[payload_qpos_adr : payload_qpos_adr + 3] = payload_pos
        qpos[payload_qpos_adr + 3 : payload_qpos_adr + 7] = _matrix_to_quat_wxyz(payload_R)
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        checked += 1

        seen_pairs: set[tuple[str, str]] = set()
        for contact_index in range(data.ncon):
            contact = data.contact[contact_index]
            body1 = (
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[contact.geom1]),
                )
                or ""
            )
            body2 = (
                mujoco.mj_id2name(
                    model,
                    mujoco.mjtObj.mjOBJ_BODY,
                    int(model.geom_bodyid[contact.geom2]),
                )
                or ""
            )
            payload_touches_obstacle = (
                _matches(body1, payload_prefixes)
                and _matches(body2, obstacle_prefixes)
            ) or (
                _matches(body2, payload_prefixes)
                and _matches(body1, obstacle_prefixes)
            )
            if not payload_touches_obstacle:
                continue
            pair = tuple(sorted((body1, body2)))
            if pair in seen_pairs:
                continue
            seen_pairs.add(pair)
            hits.append(
                PayloadCollisionHit(
                    frame_index=frame_index,
                    label=str(label),
                    body1=body1,
                    body2=body2,
                    distance=float(contact.dist),
                )
            )
            if len(hits) >= max_hits:
                return PayloadCollisionReport(checked_frames=checked, hits=tuple(hits))

    return PayloadCollisionReport(checked_frames=checked, hits=tuple(hits))


def _matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    # Avoid importing scipy here; MuJoCo can perform the conversion directly.
    quat = np.zeros(4, dtype=np.float64)
    mujoco.mju_mat2Quat(quat, np.asarray(matrix, dtype=np.float64).reshape(9))
    return quat
