"""Detachable pipette-tip attachment helpers.

The controller intentionally treats tip installation as an event layered on top
of MuJoCo contact. Collision geoms keep the socket and tip from passing through
each other; equality switching gives stable tool-tip attachment for expert
rollouts.
"""

from __future__ import annotations

from dataclasses import dataclass

import mujoco
import numpy as np


def _mj_id(model: mujoco.MjModel, obj_type: mujoco.mjtObj, name: str) -> int:
    obj_id = mujoco.mj_name2id(model, obj_type, name)
    if obj_id < 0:
        raise ValueError(f"Missing MuJoCo {obj_type.name}: {name}")
    return int(obj_id)


def _site_z_axis(data: mujoco.MjData, site_id: int) -> np.ndarray:
    return data.site_xmat[site_id].reshape(3, 3)[:, 2].copy()


@dataclass(frozen=True)
class TipAttachmentNames:
    socket_site: str = "tip_socket_site"
    tip_mount_site: str = "tip_0/tip_mount_site"
    box_weld: str = "tip_box_lock"
    pipette_weld: str = "pipette_tip_lock"
    tip_freejoint: str = "tip_0_free"
    ejector_joint: str = "pipette_ejector"


@dataclass(frozen=True)
class TipAttachmentThresholds:
    attach_distance_m: float = 0.00085
    attach_axis_angle_deg: float = 8.0
    attach_dwell_steps: int = 8
    ejector_release_qpos_m: float = -0.008
    eject_speed_m_s: float = 0.7


@dataclass
class TipAttachmentState:
    attached: bool = False
    ejected: bool = False
    attach_step: int | None = None
    eject_step: int | None = None
    aligned_steps: int = 0


class TipAttachmentController:
    """Switch welds for a single detachable pipette tip."""

    def __init__(
        self,
        model: mujoco.MjModel,
        *,
        names: TipAttachmentNames = TipAttachmentNames(),
        thresholds: TipAttachmentThresholds = TipAttachmentThresholds(),
    ) -> None:
        self.model = model
        self.names = names
        self.thresholds = thresholds
        self.state = TipAttachmentState()
        self.socket_site_id = _mj_id(model, mujoco.mjtObj.mjOBJ_SITE, names.socket_site)
        self.tip_mount_site_id = _mj_id(model, mujoco.mjtObj.mjOBJ_SITE, names.tip_mount_site)
        self.box_weld_id = _mj_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, names.box_weld)
        self.pipette_weld_id = _mj_id(model, mujoco.mjtObj.mjOBJ_EQUALITY, names.pipette_weld)
        self.tip_joint_id = _mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, names.tip_freejoint)
        self.ejector_joint_id = _mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, names.ejector_joint)
        self.tip_qpos_adr = int(model.jnt_qposadr[self.tip_joint_id])
        self.tip_qvel_adr = int(model.jnt_dofadr[self.tip_joint_id])
        self.ejector_qpos_adr = int(model.jnt_qposadr[self.ejector_joint_id])

    def reset(self, data: mujoco.MjData) -> None:
        self.state = TipAttachmentState()
        data.eq_active[self.box_weld_id] = 1
        data.eq_active[self.pipette_weld_id] = 0

    def diagnostics(self, data: mujoco.MjData) -> dict[str, float | bool | int | None]:
        socket_pos = data.site_xpos[self.socket_site_id].copy()
        mount_pos = data.site_xpos[self.tip_mount_site_id].copy()
        socket_axis = _site_z_axis(data, self.socket_site_id)
        mount_axis = _site_z_axis(data, self.tip_mount_site_id)
        axis_dot = float(np.clip(np.dot(socket_axis, mount_axis), -1.0, 1.0))
        angle_deg = float(np.rad2deg(np.arccos(axis_dot)))
        return {
            "attached": self.state.attached,
            "ejected": self.state.ejected,
            "attach_step": self.state.attach_step,
            "eject_step": self.state.eject_step,
            "aligned_steps": self.state.aligned_steps,
            "distance_m": float(np.linalg.norm(socket_pos - mount_pos)),
            "axis_angle_deg": angle_deg,
            "ejector_qpos_m": float(data.qpos[self.ejector_qpos_adr]),
            "box_weld_active": bool(data.eq_active[self.box_weld_id]),
            "pipette_weld_active": bool(data.eq_active[self.pipette_weld_id]),
        }

    def update(self, data: mujoco.MjData, *, step_index: int) -> list[str]:
        events: list[str] = []
        if not self.state.attached and not self.state.ejected:
            if self._attachment_condition(data):
                self.state.aligned_steps += 1
            else:
                self.state.aligned_steps = 0
            if self.state.aligned_steps >= self.thresholds.attach_dwell_steps:
                self._attach(data)
                self.state.attached = True
                self.state.attach_step = step_index
                events.append("attach")
        elif self.state.attached:
            ejector_qpos = float(data.qpos[self.ejector_qpos_adr])
            if ejector_qpos <= self.thresholds.ejector_release_qpos_m:
                self._eject(data)
                self.state.attached = False
                self.state.ejected = True
                self.state.eject_step = step_index
                events.append("eject")
        return events

    def _attachment_condition(self, data: mujoco.MjData) -> bool:
        socket_pos = data.site_xpos[self.socket_site_id].copy()
        mount_pos = data.site_xpos[self.tip_mount_site_id].copy()
        if float(np.linalg.norm(socket_pos - mount_pos)) > self.thresholds.attach_distance_m:
            return False
        socket_axis = _site_z_axis(data, self.socket_site_id)
        mount_axis = _site_z_axis(data, self.tip_mount_site_id)
        axis_dot = float(np.clip(np.dot(socket_axis, mount_axis), -1.0, 1.0))
        min_dot = float(np.cos(np.deg2rad(self.thresholds.attach_axis_angle_deg)))
        return axis_dot >= min_dot

    def _attach(self, data: mujoco.MjData) -> None:
        socket_pos = data.site_xpos[self.socket_site_id].copy()
        socket_quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(socket_quat, data.site_xmat[self.socket_site_id])
        data.qpos[self.tip_qpos_adr : self.tip_qpos_adr + 3] = socket_pos
        data.qpos[self.tip_qpos_adr + 3 : self.tip_qpos_adr + 7] = socket_quat
        data.qvel[self.tip_qvel_adr : self.tip_qvel_adr + 6] = 0.0
        data.eq_active[self.box_weld_id] = 0
        data.eq_active[self.pipette_weld_id] = 1
        mujoco.mj_forward(self.model, data)

    def _eject(self, data: mujoco.MjData) -> None:
        eject_axis_world = -_site_z_axis(data, self.socket_site_id)
        data.eq_active[self.pipette_weld_id] = 0
        data.eq_active[self.box_weld_id] = 0
        data.qvel[self.tip_qvel_adr : self.tip_qvel_adr + 3] = self.thresholds.eject_speed_m_s * eject_axis_world
        data.qvel[self.tip_qvel_adr + 3 : self.tip_qvel_adr + 6] = 0.0
        mujoco.mj_forward(self.model, data)
