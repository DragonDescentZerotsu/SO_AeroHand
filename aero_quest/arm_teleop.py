"""Task-space arm teleoperation utilities for Quest wrist/palm input.

This module is intentionally independent from Aero Hand finger retargeting.
The arm chain consumes a wrist/palm pose and produces arm joint targets via
relative pose mapping, task-space velocity control, and damped least-squares IK.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
import time

import numpy as np
from scipy.spatial.transform import Rotation

try:
    import mujoco
except ImportError:
    mujoco = None


def _require_mujoco() -> None:
    if mujoco is None:
        raise RuntimeError("mujoco is required. Install it with: pip install mujoco")


def clamp_norm(vec: np.ndarray, max_norm: float) -> np.ndarray:
    """Clamp a vector by Euclidean norm."""
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if max_norm <= 0.0 or norm <= max_norm or norm < 1e-12:
        return vec
    return vec * (float(max_norm) / norm)


def normalize_or_none(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray | None:
    """Return a normalized vector, or None when too small."""
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        return None
    return vec / norm


def axis_vector(axis: str) -> np.ndarray:
    """Parse '+x', '-y', etc. into a unit axis vector."""
    text = str(axis).strip().lower()
    sign = -1.0 if text.startswith("-") else 1.0
    name = text[1:] if text[:1] in {"+", "-"} else text
    mapping = {
        "x": np.array([1.0, 0.0, 0.0], dtype=np.float64),
        "y": np.array([0.0, 1.0, 0.0], dtype=np.float64),
        "z": np.array([0.0, 0.0, 1.0], dtype=np.float64),
    }
    if name not in mapping:
        raise ValueError(f"Expected axis like +x, -y, +z; got {axis!r}")
    return sign * mapping[name]


def frame_from_forward_up(forward: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    """Build a right-handed frame with columns [right, forward, up]."""
    forward_axis = normalize_or_none(forward)
    up_hint_axis = normalize_or_none(up_hint)
    if forward_axis is None:
        raise ValueError("Cannot build frame: forward vector is near zero")
    if up_hint_axis is None:
        raise ValueError("Cannot build frame: up vector is near zero")
    right_axis = normalize_or_none(np.cross(forward_axis, up_hint_axis))
    if right_axis is None:
        raise ValueError("Cannot build frame: forward and up vectors are nearly parallel")
    up_axis = normalize_or_none(np.cross(right_axis, forward_axis))
    if up_axis is None:
        raise ValueError("Cannot build frame: re-orthogonalized up vector is near zero")
    return np.stack([right_axis, forward_axis, up_axis], axis=1)


def hand_index_thumb_frame(landmarks: np.ndarray) -> np.ndarray:
    """Return hand frame columns [right, forward, up] from Quest landmarks.

    Forward is the initial index-finger direction. Up is the initial thumb
    direction, re-orthogonalized against forward.
    """
    points = np.asarray(landmarks, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Expected Quest landmarks shape (21, 3), got {points.shape}")
    index_forward = points[8] - points[5]
    if float(np.linalg.norm(index_forward)) < 1e-8:
        index_forward = points[5] - points[0]
    thumb_up = points[4] - points[2]
    if float(np.linalg.norm(thumb_up)) < 1e-8:
        thumb_up = points[4] - points[0]
    return frame_from_forward_up(index_forward, thumb_up)


def robot_control_frame(forward_axis: str = "+x", up_axis: str = "+z") -> np.ndarray:
    """Return robot control frame columns [right, forward, up]."""
    return frame_from_forward_up(axis_vector(forward_axis), axis_vector(up_axis))


def index_thumb_alignment(
    landmarks: np.ndarray,
    robot_forward_axis: str = "+x",
    robot_up_axis: str = "+z",
) -> np.ndarray:
    """Map initial hand [right, forward, up] movement into robot axes."""
    hand_frame = hand_index_thumb_frame(landmarks)
    robot_frame = robot_control_frame(robot_forward_axis, robot_up_axis)
    return robot_frame @ hand_frame.T


def orientation_error(target_R: np.ndarray, current_R: np.ndarray) -> np.ndarray:
    """Return the shortest world-frame SO(3) rotation vector to the target.

    The previous cross-product approximation scaled the error by sin(theta),
    so it incorrectly approached zero for rotations near 180 degrees.
    """
    target_R = np.asarray(target_R, dtype=np.float64).reshape(3, 3)
    current_R = np.asarray(current_R, dtype=np.float64).reshape(3, 3)
    relative_R_world = target_R @ current_R.T
    return Rotation.from_matrix(relative_R_world).as_rotvec()


def rotation_angle_from_matrix(R: np.ndarray) -> float:
    """Return the angle of a rotation matrix in radians."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_angle = 0.5 * (float(np.trace(R)) - 1.0)
    return float(np.arccos(np.clip(cos_angle, -1.0, 1.0)))


@dataclass
class WorkspaceLimiter:
    """Axis-aligned workspace clamp for target end-effector positions."""

    minimum: np.ndarray = field(default_factory=lambda: np.array([-0.4, -0.4, 0.05], dtype=np.float64))
    maximum: np.ndarray = field(default_factory=lambda: np.array([0.4, 0.4, 0.6], dtype=np.float64))

    def clamp(self, position: np.ndarray) -> np.ndarray:
        return np.clip(np.asarray(position, dtype=np.float64), self.minimum, self.maximum)


@dataclass
class RelativePoseMapper:
    """Map calibrated Quest hand motion to a robot end-effector target."""

    scale: float = 0.8
    R_align: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    R_align_rot: np.ndarray = field(default_factory=lambda: np.eye(3, dtype=np.float64))
    workspace: WorkspaceLimiter = field(default_factory=WorkspaceLimiter)
    smoothing_alpha: float = 0.0
    p_hand0: np.ndarray | None = None
    R_hand0: np.ndarray | None = None
    p_ee0: np.ndarray | None = None
    R_ee0: np.ndarray | None = None
    previous_target_position: np.ndarray | None = None
    previous_target_rotation: np.ndarray | None = None

    @property
    def is_calibrated(self) -> bool:
        return self.p_hand0 is not None and self.R_hand0 is not None and self.p_ee0 is not None and self.R_ee0 is not None

    def reset(self) -> None:
        self.p_hand0 = None
        self.R_hand0 = None
        self.p_ee0 = None
        self.R_ee0 = None
        self.previous_target_position = None
        self.previous_target_rotation = None

    def calibrate(self, hand_position: np.ndarray, hand_rotation: np.ndarray, ee_position: np.ndarray, ee_rotation: np.ndarray) -> None:
        self.p_hand0 = np.asarray(hand_position, dtype=np.float64).copy()
        self.R_hand0 = np.asarray(hand_rotation, dtype=np.float64).reshape(3, 3).copy()
        self.p_ee0 = np.asarray(ee_position, dtype=np.float64).copy()
        self.R_ee0 = np.asarray(ee_rotation, dtype=np.float64).reshape(3, 3).copy()
        self.previous_target_position = self.p_ee0.copy()
        self.previous_target_rotation = self.R_ee0.copy()

    def target_pose(self, hand_position: np.ndarray, hand_rotation: np.ndarray, control_orientation: bool = False) -> tuple[np.ndarray, np.ndarray | None]:
        if not self.is_calibrated:
            raise RuntimeError("RelativePoseMapper must be calibrated before target_pose().")
        hand_position = np.asarray(hand_position, dtype=np.float64)
        hand_rotation = np.asarray(hand_rotation, dtype=np.float64).reshape(3, 3)
        delta_p_hand = hand_position - self.p_hand0
        target_position = self.p_ee0 + float(self.scale) * (self.R_align @ delta_p_hand)
        target_position = self.workspace.clamp(target_position)

        target_rotation = None
        if control_orientation:
            delta_R_hand = self.R_hand0.T @ hand_rotation
            target_rotation = self.R_ee0 @ self.R_align_rot @ delta_R_hand

        alpha = float(np.clip(self.smoothing_alpha, 0.0, 1.0))
        if alpha > 0.0 and self.previous_target_position is not None:
            target_position = alpha * self.previous_target_position + (1.0 - alpha) * target_position
        self.previous_target_position = target_position.copy()
        if target_rotation is not None:
            self.previous_target_rotation = target_rotation.copy()
        return target_position, target_rotation


@dataclass
class VelocityTeleopConfig:
    kp_pos: float = 3.0
    kp_rot: float = 3.0
    max_linear_speed: float = 0.25
    max_angular_speed: float = 1.0
    control_orientation: bool = False


@dataclass
class VelocityCommand:
    xdot: np.ndarray
    linear: np.ndarray
    angular: np.ndarray | None
    position_error: np.ndarray
    rotation_error: np.ndarray | None


class VelocityTeleopController:
    """Convert target/current EE pose into a bounded task-space velocity."""

    def __init__(self, config: VelocityTeleopConfig | None = None):
        self.config = config or VelocityTeleopConfig()

    def compute(
        self,
        target_position: np.ndarray,
        target_rotation: np.ndarray | None,
        current_position: np.ndarray,
        current_rotation: np.ndarray,
    ) -> VelocityCommand:
        e_pos = np.asarray(target_position, dtype=np.float64) - np.asarray(current_position, dtype=np.float64)
        v_cmd = clamp_norm(float(self.config.kp_pos) * e_pos, float(self.config.max_linear_speed))
        if self.config.control_orientation and target_rotation is not None:
            e_rot = orientation_error(target_rotation, current_rotation)
            w_cmd = clamp_norm(float(self.config.kp_rot) * e_rot, float(self.config.max_angular_speed))
            return VelocityCommand(np.concatenate([v_cmd, w_cmd]), v_cmd, w_cmd, e_pos, e_rot)
        return VelocityCommand(v_cmd, v_cmd, None, e_pos, None)


def names_for_obj(model, obj_type) -> list[str]:
    _require_mujoco()
    count = {
        mujoco.mjtObj.mjOBJ_BODY: model.nbody,
        mujoco.mjtObj.mjOBJ_SITE: model.nsite,
        mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
        mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
    }[obj_type]
    return [mujoco.mj_id2name(model, obj_type, idx) or f"<unnamed_{idx}>" for idx in range(count)]


def resolve_site_or_body(model, ee_site: str | None = None, ee_body: str | None = None) -> tuple[str, int]:
    """Resolve an end-effector site/body and return ("site"|"body", id)."""
    _require_mujoco()
    if ee_site:
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, ee_site)
        if site_id < 0:
            raise ValueError(f"End-effector site not found: {ee_site}. Available sites: {names_for_obj(model, mujoco.mjtObj.mjOBJ_SITE)}")
        return "site", int(site_id)
    if ee_body:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, ee_body)
        if body_id < 0:
            raise ValueError(f"End-effector body not found: {ee_body}. Available bodies: {names_for_obj(model, mujoco.mjtObj.mjOBJ_BODY)}")
        return "body", int(body_id)
    for candidate in ("aero_wrist_site", "so101_aero_attach_site", "grasp_site", "gripperframe"):
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, candidate)
        if site_id >= 0:
            return "site", int(site_id)
    raise ValueError(
        "No end-effector site/body provided and no default site found. "
        f"Available sites: {names_for_obj(model, mujoco.mjtObj.mjOBJ_SITE)}"
    )


def select_joint_ids(model, joint_names: list[str] | None = None, joint_prefix: str | None = None) -> list[int]:
    """Select arm joint ids by explicit names or name prefix."""
    _require_mujoco()
    if joint_names:
        ids = []
        for name in joint_names:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint_id < 0:
                raise ValueError(f"Arm joint not found: {name}. Available joints: {names_for_obj(model, mujoco.mjtObj.mjOBJ_JOINT)}")
            ids.append(int(joint_id))
        return ids
    if joint_prefix:
        ids = [
            idx
            for idx in range(model.njnt)
            if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) or "").startswith(joint_prefix)
        ]
        if ids:
            return ids
    so101_defaults = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    ids = []
    for name in so101_defaults:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id >= 0:
            ids.append(int(joint_id))
    if ids:
        return ids
    raise ValueError(
        "Could not infer arm joints. Pass --arm_joint_names or --arm_joint_prefix. "
        f"Available joints: {names_for_obj(model, mujoco.mjtObj.mjOBJ_JOINT)}"
    )


def joint_qpos(model, data, joint_ids: list[int]) -> np.ndarray:
    return np.array([data.qpos[model.jnt_qposadr[joint_id]] for joint_id in joint_ids], dtype=np.float64)


def joint_qvel(model, data, joint_ids: list[int]) -> np.ndarray:
    return np.array([data.qvel[model.jnt_dofadr[joint_id]] for joint_id in joint_ids], dtype=np.float64)


def joint_ranges(model, joint_ids: list[int]) -> tuple[np.ndarray, np.ndarray]:
    lows = []
    highs = []
    for joint_id in joint_ids:
        if model.jnt_limited[joint_id]:
            lo, hi = model.jnt_range[joint_id]
        else:
            lo, hi = -np.inf, np.inf
        lows.append(float(lo))
        highs.append(float(hi))
    return np.asarray(lows, dtype=np.float64), np.asarray(highs, dtype=np.float64)


class DampedLeastSquaresIK:
    """Resolved-rate IK using MuJoCo Jacobians and selected arm joints only."""

    def __init__(
        self,
        model,
        ee_site: str | None = None,
        ee_body: str | None = None,
        joint_names: list[str] | None = None,
        joint_prefix: str | None = None,
        damping: float = 0.05,
        max_joint_speed: float = 1.5,
        smoothing_alpha: float = 0.0,
    ):
        _require_mujoco()
        self.model = model
        self.ee_kind, self.ee_id = resolve_site_or_body(model, ee_site=ee_site, ee_body=ee_body)
        self.joint_ids = select_joint_ids(model, joint_names=joint_names, joint_prefix=joint_prefix)
        self.damping = float(damping)
        self.max_joint_speed = float(max_joint_speed)
        self.smoothing_alpha = float(np.clip(smoothing_alpha, 0.0, 1.0))
        self.prev_qtarget: np.ndarray | None = None

    @property
    def joint_names(self) -> list[str]:
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, idx) or f"<joint_{idx}>" for idx in self.joint_ids]

    def ee_pose(self, data) -> tuple[np.ndarray, np.ndarray]:
        if self.ee_kind == "site":
            return data.site_xpos[self.ee_id].copy(), data.site_xmat[self.ee_id].reshape(3, 3).copy()
        return data.xpos[self.ee_id].copy(), data.xmat[self.ee_id].reshape(3, 3).copy()

    def jacobian(self, data, control_orientation: bool = False) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        if self.ee_kind == "site":
            mujoco.mj_jacSite(self.model, data, jacp, jacr, self.ee_id)
        else:
            mujoco.mj_jacBody(self.model, data, jacp, jacr, self.ee_id)
        dof_ids = [self.model.jnt_dofadr[joint_id] for joint_id in self.joint_ids]
        if control_orientation:
            return np.vstack([jacp[:, dof_ids], jacr[:, dof_ids]])
        return jacp[:, dof_ids]

    def solve(self, data, xdot_cmd: np.ndarray, dt: float, control_orientation: bool = False) -> tuple[np.ndarray, np.ndarray]:
        xdot_cmd = np.asarray(xdot_cmd, dtype=np.float64)
        J = self.jacobian(data, control_orientation=control_orientation)
        if J.shape[0] != xdot_cmd.shape[0]:
            raise ValueError(f"Jacobian/task velocity mismatch: J rows={J.shape[0]}, xdot shape={xdot_cmd.shape}")
        JJt = J @ J.T
        qdot = J.T @ np.linalg.solve(JJt + (self.damping**2) * np.eye(JJt.shape[0]), xdot_cmd)
        qdot = np.clip(qdot, -self.max_joint_speed, self.max_joint_speed)

        q_current = joint_qpos(self.model, data, self.joint_ids)
        q_target = q_current + qdot * float(dt)
        lo, hi = joint_ranges(self.model, self.joint_ids)
        q_target = np.clip(q_target, lo, hi)
        if self.smoothing_alpha > 0.0 and self.prev_qtarget is not None:
            q_target = self.smoothing_alpha * self.prev_qtarget + (1.0 - self.smoothing_alpha) * q_target
        self.prev_qtarget = q_target.copy()
        return q_target, qdot

    def apply_position_targets(self, data, q_target: np.ndarray) -> None:
        """Write selected arm joint targets to matching MuJoCo position actuators."""
        for joint_id, target in zip(self.joint_ids, np.asarray(q_target, dtype=np.float64)):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id < 0:
                for idx in range(self.model.nu):
                    if int(self.model.actuator_trnid[idx, 0]) == int(joint_id):
                        actuator_id = idx
                        break
            if actuator_id < 0:
                raise RuntimeError(
                    f"No actuator found for arm joint {joint_name}. "
                    f"Available actuators: {names_for_obj(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR)}"
                )
            lo, hi = self.model.actuator_ctrlrange[actuator_id]
            data.ctrl[actuator_id] = float(np.clip(target, lo, hi))


@dataclass
class TeleopStateMachine:
    """Small mode holder for calibration, pause, deadman, and emergency stop."""

    enabled: bool = True
    paused: bool = False
    emergency_stopped: bool = False
    calibrated: bool = False
    mode: str = "waiting_for_calibration"

    def can_control(self) -> bool:
        return self.enabled and self.calibrated and not self.paused and not self.emergency_stopped

    def set_calibrated(self, value: bool = True) -> None:
        self.calibrated = bool(value)
        self.mode = "teleop" if self.calibrated else "waiting_for_calibration"

    def toggle_pause(self) -> None:
        self.paused = not self.paused
        self.mode = "paused" if self.paused else ("teleop" if self.calibrated else "waiting_for_calibration")

    def emergency_stop(self) -> None:
        self.emergency_stopped = True
        self.mode = "estop"


class SharedAutonomyController:
    """Placeholder shared-autonomy hook; alpha=0 means full human control."""

    def __init__(self, alpha: float = 0.0):
        self.alpha = float(np.clip(alpha, 0.0, 1.0))

    def compute_action(self, obs, human_action):
        return human_action


class EpisodeLogger:
    """JSONL episode logger with fields shaped for later dataset conversion."""

    def __init__(self, record_dir: str | Path, task_name: str):
        self.record_dir = Path(record_dir)
        self.task_name = task_name
        self.file = None
        self.frame_id = 0
        self.start_time = None
        self.path: Path | None = None

    def __enter__(self):
        self.record_dir.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y%m%d_%H%M%S")
        self.path = self.record_dir / f"{self.task_name}_{stamp}.jsonl"
        self.file = self.path.open("w", encoding="utf-8")
        self.start_time = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        if self.file is not None:
            self.file.close()

    def write(self, record: dict) -> None:
        if self.file is None:
            return
        now = time.time()
        record = dict(record)
        record.setdefault("timestamp", now)
        record.setdefault("t", now - self.start_time if self.start_time is not None else 0.0)
        record.setdefault("frame_id", self.frame_id)
        self.file.write(json.dumps(record) + "\n")
        self.file.flush()
        self.frame_id += 1


def normalize(v, eps: float = 1e-8) -> np.ndarray:
    """Return a unit vector, or zeros when the vector is degenerate."""
    v = np.asarray(v, dtype=np.float64)
    norm = float(np.linalg.norm(v))
    if norm < float(eps):
        return np.zeros_like(v, dtype=np.float64)
    return v / norm


def compute_palm_pose(P: np.ndarray, previous_R: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Estimate Quest palm pose from 21 hand landmarks.

    The returned rotation matrix columns are palm-local axes expressed in the
    Quest/world frame:
    x: pinky MCP -> index MCP, y: wrist -> middle MCP, z: palm normal.
    Position is the wrist landmark ``P[0]``.
    """
    points = np.asarray(P, dtype=np.float64)
    if points.shape != (21, 3):
        raise ValueError(f"Expected Quest landmarks shape (21, 3), got {points.shape}")

    fallback_R = np.eye(3, dtype=np.float64) if previous_R is None else np.asarray(previous_R, dtype=np.float64).reshape(3, 3)
    if not np.all(np.isfinite(points)):
        return fallback_R.copy(), np.nan_to_num(points[0], nan=0.0, posinf=0.0, neginf=0.0)

    wrist = points[0]
    index_mcp = points[5]
    middle_mcp = points[9]
    pinky_mcp = points[17]

    x_axis = normalize(index_mcp - pinky_mcp)
    y_raw = normalize(middle_mcp - wrist)
    if float(np.linalg.norm(x_axis)) < 1e-8 or float(np.linalg.norm(y_raw)) < 1e-8:
        return fallback_R.copy(), wrist.copy()

    z_axis = normalize(np.cross(x_axis, y_raw))
    if float(np.linalg.norm(z_axis)) < 1e-8:
        return fallback_R.copy(), wrist.copy()

    y_axis = normalize(np.cross(z_axis, x_axis))
    if float(np.linalg.norm(y_axis)) < 1e-8:
        return fallback_R.copy(), wrist.copy()

    R_hand = np.column_stack([x_axis, y_axis, z_axis])
    if not np.all(np.isfinite(R_hand)) or float(np.linalg.det(R_hand)) <= 0.0:
        return fallback_R.copy(), wrist.copy()
    return R_hand.astype(np.float64), wrist.astype(np.float64)


def _workspace_bounds_to_arrays(workspace_bounds):
    if workspace_bounds is None:
        return None
    try:
        lo = np.asarray([workspace_bounds[axis][0] for axis in ("x", "y", "z")], dtype=np.float64)
        hi = np.asarray([workspace_bounds[axis][1] for axis in ("x", "y", "z")], dtype=np.float64)
    except (KeyError, IndexError, TypeError) as exc:
        raise ValueError("workspace_bounds must be {'x': [min, max], 'y': [min, max], 'z': [min, max]}") from exc
    if np.any(lo > hi):
        raise ValueError(f"workspace_bounds min must be <= max, got lo={lo}, hi={hi}")
    return lo, hi


def _clip_workspace(position: np.ndarray, workspace_bounds) -> np.ndarray:
    arrays = _workspace_bounds_to_arrays(workspace_bounds)
    if arrays is None:
        return np.asarray(position, dtype=np.float64)
    lo, hi = arrays
    return np.clip(np.asarray(position, dtype=np.float64), lo, hi)


def _rotation_lerp(previous_R: np.ndarray, target_R: np.ndarray, alpha: float) -> np.ndarray:
    """Small polar-projection blend for debug/teleop orientation smoothing."""
    alpha = float(np.clip(alpha, 0.0, 1.0))
    blended = alpha * np.asarray(target_R, dtype=np.float64) + (1.0 - alpha) * np.asarray(previous_R, dtype=np.float64)
    u, _s, vh = np.linalg.svd(blended)
    R = u @ vh
    if float(np.linalg.det(R)) < 0.0:
        u[:, -1] *= -1.0
        R = u @ vh
    return R


def rotation_vector_from_matrix(R: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Return axis-angle rotation vector from a 3x3 rotation matrix."""
    R = np.asarray(R, dtype=np.float64).reshape(3, 3)
    cos_theta = float(np.clip((np.trace(R) - 1.0) * 0.5, -1.0, 1.0))
    theta = float(np.arccos(cos_theta))
    if theta < eps:
        return np.zeros(3, dtype=np.float64)
    axis = np.asarray(
        [
            R[2, 1] - R[1, 2],
            R[0, 2] - R[2, 0],
            R[1, 0] - R[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= max(2.0 * float(np.sin(theta)), eps)
    return theta * axis


class MuJoCoSO101Adapter:
    """Adapter exposing the arm teleop robot interface for MuJoCo SO101 models."""

    def __init__(
        self,
        model,
        data,
        ee_site: str | None = "grasp_site",
        ee_body: str | None = None,
        joint_names: list[str] | None = None,
        joint_prefix: str | None = None,
        damping: float = 0.05,
        max_iters: int = 50,
        tolerance: float = 1e-4,
        orientation_weight: float = 0.25,
        orientation_tolerance: float = 0.08,
        require_orientation_success: bool = False,
    ):
        _require_mujoco()
        self.model = model
        self.data = data
        self.ee_kind, self.ee_id = resolve_site_or_body(model, ee_site=ee_site, ee_body=ee_body)
        self.joint_ids = select_joint_ids(model, joint_names=joint_names, joint_prefix=joint_prefix)
        self.damping = float(damping)
        self.max_iters = int(max_iters)
        self.tolerance = float(tolerance)
        self.orientation_weight = float(orientation_weight)
        self.orientation_tolerance = float(orientation_tolerance)
        self.require_orientation_success = bool(require_orientation_success)
        self.qpos_indices = np.asarray([model.jnt_qposadr[joint_id] for joint_id in self.joint_ids], dtype=np.int64)
        self.dof_indices = np.asarray([model.jnt_dofadr[joint_id] for joint_id in self.joint_ids], dtype=np.int64)
        self.lower_limits, self.upper_limits = joint_ranges(model, self.joint_ids)

    @property
    def joint_names(self) -> list[str]:
        return [mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, idx) or f"<joint_{idx}>" for idx in self.joint_ids]

    def get_ee_position(self) -> np.ndarray:
        if self.ee_kind == "site":
            return self.data.site_xpos[self.ee_id].copy()
        return self.data.xpos[self.ee_id].copy()

    def get_ee_rotation(self) -> np.ndarray:
        if self.ee_kind == "site":
            return self.data.site_xmat[self.ee_id].reshape(3, 3).copy()
        return self.data.xmat[self.ee_id].reshape(3, 3).copy()

    def get_joint_positions(self) -> np.ndarray:
        return self.data.qpos[self.qpos_indices].copy()

    def get_joint_limits(self) -> tuple[np.ndarray, np.ndarray]:
        return self.lower_limits.copy(), self.upper_limits.copy()

    def _set_qpos_for_ik(self, q: np.ndarray) -> None:
        self.data.qpos[self.qpos_indices] = np.asarray(q, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)

    def _jacobian(self, include_orientation: bool = False) -> np.ndarray:
        jacp = np.zeros((3, self.model.nv), dtype=np.float64)
        jacr = np.zeros((3, self.model.nv), dtype=np.float64)
        if self.ee_kind == "site":
            mujoco.mj_jacSite(self.model, self.data, jacp, jacr, self.ee_id)
        else:
            mujoco.mj_jacBody(self.model, self.data, jacp, jacr, self.ee_id)
        if include_orientation:
            return np.vstack([jacp[:, self.dof_indices], self.orientation_weight * jacr[:, self.dof_indices]])
        return jacp[:, self.dof_indices]

    def solve_ik(self, target_pos, target_rot=None, q_seed=None):
        """Solve damped least-squares IK.

        Position is always optimized. When ``target_rot`` is provided, a
        weighted orientation residual is added. SO101 has fewer than 6 arm DOF,
        so orientation tracking is best-effort and position remains primary.
        """
        target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
        target_rot = None if target_rot is None else np.asarray(target_rot, dtype=np.float64).reshape(3, 3)
        q_live = self.get_joint_positions()
        q = q_live.copy() if q_seed is None else np.asarray(q_seed, dtype=np.float64).reshape(len(self.joint_ids)).copy()
        q = np.clip(q, self.lower_limits, self.upper_limits)
        success = False

        try:
            for _ in range(self.max_iters):
                self._set_qpos_for_ik(q)
                pos_error = target_pos - self.get_ee_position()
                if target_rot is not None:
                    rot_error = orientation_error(target_rot, self.get_ee_rotation())
                    error = np.concatenate([pos_error, self.orientation_weight * rot_error])
                    include_orientation = True
                else:
                    rot_error = None
                    error = pos_error
                    include_orientation = False

                pos_ok = float(np.linalg.norm(pos_error)) <= self.tolerance
                rot_ok = rot_error is None or float(np.linalg.norm(rot_error)) <= self.orientation_tolerance
                if pos_ok and rot_ok:
                    success = True
                    break
                J = self._jacobian(include_orientation=include_orientation)
                JJt = J @ J.T
                dq = J.T @ np.linalg.solve(JJt + (self.damping**2) * np.eye(JJt.shape[0]), error)
                dq = np.clip(dq, -0.05, 0.05)
                q = np.clip(q + dq, self.lower_limits, self.upper_limits)

            self._set_qpos_for_ik(q)
            final_pos_ok = float(np.linalg.norm(target_pos - self.get_ee_position())) <= max(self.tolerance, 2e-3)
            if target_rot is None:
                success = success or final_pos_ok
            else:
                final_rot_error = orientation_error(target_rot, self.get_ee_rotation())
                final_rot_ok = float(np.linalg.norm(final_rot_error)) <= self.orientation_tolerance
                success = success or (final_pos_ok and (final_rot_ok or not self.require_orientation_success))
                if not final_pos_ok and not self.require_orientation_success:
                    # SO101 has only 5 arm DoF. If the weighted orientation
                    # objective prevents position convergence, fall back to
                    # position-only IK so wrist rotation never freezes reach.
                    q_position_only = q_live.copy() if q_seed is None else np.asarray(q_seed, dtype=np.float64).reshape(len(self.joint_ids)).copy()
                    q_position_only = np.clip(q_position_only, self.lower_limits, self.upper_limits)
                    for _ in range(self.max_iters):
                        self._set_qpos_for_ik(q_position_only)
                        pos_error = target_pos - self.get_ee_position()
                        if float(np.linalg.norm(pos_error)) <= self.tolerance:
                            break
                        J = self._jacobian(include_orientation=False)
                        JJt = J @ J.T
                        dq = J.T @ np.linalg.solve(JJt + (self.damping**2) * np.eye(JJt.shape[0]), pos_error)
                        dq = np.clip(dq, -0.05, 0.05)
                        q_position_only = np.clip(q_position_only + dq, self.lower_limits, self.upper_limits)
                    self._set_qpos_for_ik(q_position_only)
                    final_pos_ok = float(np.linalg.norm(target_pos - self.get_ee_position())) <= max(self.tolerance, 2e-3)
                    if final_pos_ok:
                        q = q_position_only
                        success = True
        except np.linalg.LinAlgError:
            success = False
            q = q_live.copy()
        finally:
            self._set_qpos_for_ik(q_live)

        return q.astype(np.float64), bool(success)

    def set_joint_positions(self, q_cmd) -> None:
        """Write selected arm joint position targets to matching MuJoCo actuators."""
        q_cmd = np.asarray(q_cmd, dtype=np.float64).reshape(len(self.joint_ids))
        for joint_id, target in zip(self.joint_ids, q_cmd):
            joint_name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
            actuator_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
            if actuator_id < 0:
                for idx in range(self.model.nu):
                    if int(self.model.actuator_trnid[idx, 0]) == int(joint_id):
                        actuator_id = idx
                        break
            if actuator_id < 0:
                raise RuntimeError(f"No actuator found for arm joint {joint_name}")
            lo, hi = self.model.actuator_ctrlrange[actuator_id]
            self.data.ctrl[actuator_id] = float(np.clip(target, lo, hi))


class QuestArmTeleopController:
    """Relative Quest wrist/palm pose to robot end-effector joint-position controller."""

    def __init__(
        self,
        scale=0.5,
        R_robot_from_quest=None,
        use_orientation=False,
        position_alpha=0.25,
        orientation_alpha=0.2,
        deadzone=0.005,
        max_ee_step=0.02,
        max_joint_step=0.03,
        workspace_bounds=None,
        align_to_hand_on_reset=False,
        robot_control_frame=None,
        position_mapping_mode="incremental_current_hand",
        direct_wrist_control=False,
        direct_wrist_mapping="palm_proxy",
        wrist_flex_axis="+y",
        wrist_roll_axis="+x",
        wrist_flex_gain=1.0,
        wrist_roll_gain=1.0,
    ):
        self.scale = float(scale)
        self.R_robot_from_quest = np.eye(3, dtype=np.float64) if R_robot_from_quest is None else np.asarray(R_robot_from_quest, dtype=np.float64).reshape(3, 3)
        self.align_to_hand_on_reset = bool(align_to_hand_on_reset)
        self.robot_control_frame = np.eye(3, dtype=np.float64) if robot_control_frame is None else np.asarray(robot_control_frame, dtype=np.float64).reshape(3, 3)
        self.position_mapping_mode = str(position_mapping_mode)
        if self.position_mapping_mode not in {"anchored_initial_hand", "incremental_current_hand"}:
            raise ValueError("position_mapping_mode must be 'anchored_initial_hand' or 'incremental_current_hand'")
        self.direct_wrist_control = bool(direct_wrist_control)
        self.direct_wrist_mapping = str(direct_wrist_mapping)
        if self.direct_wrist_mapping not in {"palm_proxy", "rotvec"}:
            raise ValueError("direct_wrist_mapping must be 'palm_proxy' or 'rotvec'")
        self.wrist_flex_axis = axis_vector(wrist_flex_axis)
        self.wrist_roll_axis = axis_vector(wrist_roll_axis)
        self.wrist_flex_gain = float(wrist_flex_gain)
        self.wrist_roll_gain = float(wrist_roll_gain)
        self.use_orientation = bool(use_orientation)
        self.position_alpha = float(np.clip(position_alpha, 0.0, 1.0))
        self.orientation_alpha = float(np.clip(orientation_alpha, 0.0, 1.0))
        self.deadzone = float(deadzone)
        self.max_ee_step = float(max_ee_step)
        self.max_joint_step = float(max_joint_step)
        self.workspace_bounds = workspace_bounds
        self.initialized = False
        self.stopped = False
        self.p_hand_0: np.ndarray | None = None
        self.R_hand_0: np.ndarray | None = None
        self.R_orientation_0: np.ndarray | None = None
        self.R_robot_from_orientation_quest = self.R_robot_from_quest.copy()
        self.p_ee_0: np.ndarray | None = None
        self.R_ee_0: np.ndarray | None = None
        self.prev_R_hand: np.ndarray | None = None
        self.prev_p_target: np.ndarray | None = None
        self.prev_R_target: np.ndarray | None = None
        self.prev_p_hand: np.ndarray | None = None
        self.prev_p_hand_for_mapping: np.ndarray | None = None
        self.last_q: np.ndarray | None = None
        self.last_q_cmd: np.ndarray | None = None
        self.q_anchor: np.ndarray | None = None
        self.wrist_flex_signal_0 = 0.0
        self.wrist_roll_signal_0 = 0.0
        self.last_wrist_flex_delta = 0.0
        self.last_wrist_roll_delta = 0.0

    def reset(self, P, robot, hand_rotation: np.ndarray | None = None) -> dict:
        """Anchor current Quest wrist and current robot end-effector pose."""
        R_motion, p_hand = compute_palm_pose(P, self.prev_R_hand)
        R_orientation = R_motion.copy()
        if hand_rotation is not None:
            R_orientation = np.asarray(hand_rotation, dtype=np.float64).reshape(3, 3)
        self.prev_R_hand = R_motion.copy()
        if self.align_to_hand_on_reset:
            self.R_robot_from_quest = self.robot_control_frame @ R_motion.T
            self.R_robot_from_orientation_quest = self.robot_control_frame @ R_orientation.T
        else:
            self.R_robot_from_orientation_quest = self.R_robot_from_quest.copy()
        self.p_hand_0 = p_hand.copy()
        self.R_hand_0 = R_motion.copy()
        self.R_orientation_0 = R_orientation.copy()
        self.wrist_flex_signal_0 = -float(R_orientation[2, 1])
        self.wrist_roll_signal_0 = float(R_orientation[0, 2])
        self.p_ee_0 = np.asarray(robot.get_ee_position(), dtype=np.float64).reshape(3).copy()
        self.R_ee_0 = np.asarray(robot.get_ee_rotation(), dtype=np.float64).reshape(3, 3).copy()
        q_current = np.asarray(robot.get_joint_positions(), dtype=np.float64).copy()
        self.last_q = q_current.copy()
        self.last_q_cmd = q_current.copy()
        self.q_anchor = q_current.copy()
        self.last_wrist_flex_delta = 0.0
        self.last_wrist_roll_delta = 0.0
        self.prev_p_target = self.p_ee_0.copy()
        self.prev_R_target = self.R_ee_0.copy()
        self.prev_p_hand = p_hand.copy()
        self.prev_p_hand_for_mapping = p_hand.copy()
        self.initialized = True
        self.stopped = False
        return {
            "p_hand": p_hand.copy(),
            "p_ee_target": self.prev_p_target.copy(),
            "q_target": q_current.copy(),
            "q_cmd": q_current.copy(),
            "initialized": True,
            "ik_success": True,
            "delta_p_hand": np.zeros(3, dtype=np.float64),
            "use_orientation": self.use_orientation,
        }

    def reanchor(self, P, robot, hand_rotation: np.ndarray | None = None) -> dict:
        """Set the current hand and end-effector pose as the new zero point."""
        return self.reset(P, robot, hand_rotation=hand_rotation)

    def stop(self) -> None:
        """Stop issuing new targets; the next update holds the last command."""
        self.stopped = True

    def _clip_joint_limits(self, robot, q: np.ndarray) -> np.ndarray:
        if hasattr(robot, "get_joint_limits"):
            lo, hi = robot.get_joint_limits()
            return np.clip(q, np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64))
        return q

    def update(self, P, robot, hand_rotation: np.ndarray | None = None) -> dict:
        """Process one Quest landmark frame and send a bounded joint command."""
        if not self.initialized:
            return self.reset(P, robot, hand_rotation=hand_rotation)

        R_motion, p_hand_raw = compute_palm_pose(P, self.prev_R_hand)
        R_orientation = R_motion.copy()
        if hand_rotation is not None:
            R_orientation = np.asarray(hand_rotation, dtype=np.float64).reshape(3, 3)
        self.prev_R_hand = R_motion.copy()
        if self.prev_p_hand is None:
            p_hand = p_hand_raw.copy()
        else:
            p_hand = self.position_alpha * p_hand_raw + (1.0 - self.position_alpha) * self.prev_p_hand
        self.prev_p_hand = p_hand.copy()
        q_prev = np.asarray(self.last_q_cmd if self.last_q_cmd is not None else robot.get_joint_positions(), dtype=np.float64)

        delta_p_hand = p_hand - self.p_hand_0
        if self.stopped:
            robot.set_joint_positions(q_prev)
            return self._debug(p_hand, self.prev_p_target, q_prev, q_prev, True, delta_p_hand)

        delta_hand_local = np.zeros(3, dtype=np.float64)
        mapped_ee_step = np.zeros(3, dtype=np.float64)
        if self.position_mapping_mode == "incremental_current_hand":
            delta_for_mapping = p_hand - self.prev_p_hand_for_mapping
            if float(np.linalg.norm(delta_for_mapping)) < self.deadzone:
                p_target = self.prev_p_target.copy()
            else:
                R_robot_from_current_hand = self.robot_control_frame @ R_motion.T
                delta_hand_local = R_motion.T @ delta_for_mapping
                mapped_ee_step = self.scale * (self.robot_control_frame @ delta_hand_local)
                p_target = self.prev_p_target + mapped_ee_step
                p_target = _clip_workspace(p_target, self.workspace_bounds)
                step = p_target - self.prev_p_target
                p_target = self.prev_p_target + clamp_norm(step, self.max_ee_step)
                p_target = self.position_alpha * p_target + (1.0 - self.position_alpha) * self.prev_p_target
        else:
            if float(np.linalg.norm(delta_p_hand)) < self.deadzone:
                p_target = self.prev_p_target.copy()
            else:
                mapped_total_delta = self.scale * (self.R_robot_from_quest @ delta_p_hand)
                mapped_ee_step = mapped_total_delta - (self.prev_p_target - self.p_ee_0)
                p_target = self.p_ee_0 + mapped_total_delta
                p_target = _clip_workspace(p_target, self.workspace_bounds)
                step = p_target - self.prev_p_target
                p_target = self.prev_p_target + clamp_norm(step, self.max_ee_step)
                p_target = self.position_alpha * p_target + (1.0 - self.position_alpha) * self.prev_p_target

        if self.use_orientation:
            delta_R_hand_quest = R_orientation @ self.R_orientation_0.T
            delta_R_ee_robot = self.R_robot_from_orientation_quest @ delta_R_hand_quest @ self.R_robot_from_orientation_quest.T
            R_target = delta_R_ee_robot @ self.R_ee_0
            R_target = _rotation_lerp(self.prev_R_target, R_target, self.orientation_alpha)
        else:
            R_target = self.R_ee_0.copy()

        q_target = q_prev.copy()
        ik_success = False
        delta_R_ee_robot_for_wrist = None
        try:
            ik_result = robot.solve_ik(p_target, R_target, q_seed=self.last_q)
            if isinstance(ik_result, tuple):
                q_candidate, ik_success = ik_result
            else:
                q_candidate, ik_success = ik_result, True
            q_candidate = np.asarray(q_candidate, dtype=np.float64).reshape(q_prev.shape)
            if bool(ik_success) and np.all(np.isfinite(q_candidate)):
                q_target = self._clip_joint_limits(robot, q_candidate)
        except Exception as exc:
            print(f"IK failed; holding previous arm command: {exc}")
            ik_success = False

        if self.use_orientation and self.direct_wrist_control and self.q_anchor is not None:
            delta_R_hand_quest = R_orientation @ self.R_orientation_0.T
            delta_R_ee_robot_for_wrist = self.R_robot_from_orientation_quest @ delta_R_hand_quest @ self.R_robot_from_orientation_quest.T
            joint_names = getattr(robot, "joint_names", [])
            try:
                flex_idx = list(joint_names).index("wrist_flex")
                roll_idx = list(joint_names).index("wrist_roll")
                if self.direct_wrist_mapping == "palm_proxy":
                    flex_signal = -float(R_orientation[2, 1])
                    roll_signal = float(R_orientation[0, 2])
                    flex_delta = flex_signal - self.wrist_flex_signal_0
                    roll_delta = roll_signal - self.wrist_roll_signal_0
                else:
                    rotvec_robot = rotation_vector_from_matrix(delta_R_ee_robot_for_wrist)
                    flex_delta = float(np.dot(rotvec_robot, self.wrist_flex_axis))
                    roll_delta = float(np.dot(rotvec_robot, self.wrist_roll_axis))
                flex_delta = self.orientation_alpha * flex_delta + (1.0 - self.orientation_alpha) * self.last_wrist_flex_delta
                roll_delta = self.orientation_alpha * roll_delta + (1.0 - self.orientation_alpha) * self.last_wrist_roll_delta
                q_target[flex_idx] = self.q_anchor[flex_idx] + self.wrist_flex_gain * flex_delta
                q_target[roll_idx] = self.q_anchor[roll_idx] + self.wrist_roll_gain * roll_delta
                self.last_wrist_flex_delta = float(flex_delta)
                self.last_wrist_roll_delta = float(roll_delta)
                q_target = self._clip_joint_limits(robot, q_target)
                ik_success = ik_success or np.all(np.isfinite(q_target))
            except ValueError:
                pass

        if not ik_success:
            q_cmd = q_prev.copy()
        else:
            q_delta = np.clip(q_target - q_prev, -self.max_joint_step, self.max_joint_step)
            q_cmd = self._clip_joint_limits(robot, q_prev + q_delta)

        robot.set_joint_positions(q_cmd)
        if ik_success:
            self.prev_p_target = p_target.copy()
            self.prev_R_target = R_target.copy()
        self.prev_p_hand_for_mapping = p_hand.copy()
        self.last_q = q_cmd.copy()
        self.last_q_cmd = q_cmd.copy()
        orientation_error_angle = 0.0
        if self.use_orientation:
            try:
                orientation_error_angle = rotation_angle_from_matrix(R_target @ robot.get_ee_rotation().T)
            except Exception:
                orientation_error_angle = float("nan")
        return self._debug(
            p_hand,
            p_target,
            q_target,
            q_cmd,
            ik_success,
            delta_p_hand,
            orientation_error_angle,
            delta_hand_local,
            mapped_ee_step,
        )

    def _debug(
        self,
        p_hand,
        p_ee_target,
        q_target,
        q_cmd,
        ik_success,
        delta_p_hand,
        orientation_error_angle=0.0,
        delta_hand_local=None,
        mapped_ee_step=None,
    ) -> dict:
        return {
            "p_hand": np.asarray(p_hand, dtype=np.float64).copy(),
            "p_ee_target": np.asarray(p_ee_target, dtype=np.float64).copy(),
            "q_target": np.asarray(q_target, dtype=np.float64).copy(),
            "q_cmd": np.asarray(q_cmd, dtype=np.float64).copy(),
            "initialized": bool(self.initialized),
            "ik_success": bool(ik_success),
            "delta_p_hand": np.asarray(delta_p_hand, dtype=np.float64).copy(),
            "use_orientation": bool(self.use_orientation),
            "wrist_flex_delta": float(self.last_wrist_flex_delta),
            "wrist_roll_delta": float(self.last_wrist_roll_delta),
            "orientation_error_angle": float(orientation_error_angle),
            "delta_hand_local": np.asarray(
                np.zeros(3, dtype=np.float64) if delta_hand_local is None else delta_hand_local,
                dtype=np.float64,
            ).copy(),
            "mapped_ee_step": np.asarray(
                np.zeros(3, dtype=np.float64) if mapped_ee_step is None else mapped_ee_step,
                dtype=np.float64,
            ).copy(),
        }
