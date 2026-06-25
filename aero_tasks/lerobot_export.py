"""LeRobot export helpers for MuJoCo expert trajectories."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise RuntimeError(f"MuJoCo is required for LeRobot export: {exc}") from exc


@dataclass(frozen=True)
class RenderCameraSpec:
    """Small camera recipe used for synthetic LeRobot videos."""

    name: str
    mode: str
    lookat: tuple[float, float, float] = (0.0, 0.0, 1.0)
    distance: float = 1.0
    azimuth: float = 0.0
    elevation: float = -45.0
    body: str | None = None
    target_body: str | None = None
    eye_offset_world: tuple[float, float, float] = (0.0, 0.0, 0.2)
    eye_offset_local: tuple[float, float, float] = (0.0, 0.0, 0.1)
    target_offset_local: tuple[float, float, float] = (0.08, 0.0, 0.0)
    up_axis_local: tuple[float, float, float] = (-1.0, 0.0, 0.0)


DEFAULT_HANDOFF_CAMERAS = (
    RenderCameraSpec(
        name="table_overview",
        mode="world",
        lookat=(0.0, 0.02, 0.96),
        eye_offset_world=(0.75, 0.0, 1.35),
    ),
    RenderCameraSpec(
        name="gripper_forward",
        mode="body_local_fixed_roll",
        body="piper_original/link6",
        eye_offset_local=(-0.14, -0.0, -0.06),
        target_offset_local=(0.0, 0.0, 0.2),
        up_axis_local=(-1.0, 0.0, 0.0),
    ),
    RenderCameraSpec(
        name="palm_inner",
        mode="body_local_fixed_roll",
        body="piper_aerohand/palm",
        eye_offset_local=(0.0, -0.06, 0.0),
        target_offset_local=(-0.17923, -0.02742, 0.07263),
        up_axis_local=(0.0, 0.0, 1.0),
    ),
)


def model_qpos_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for joint_id in range(model.njnt):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        qpos_adr = int(model.jnt_qposadr[joint_id])
        joint_type = int(model.jnt_type[joint_id])
        if joint_type == mujoco.mjtJoint.mjJNT_FREE:
            suffixes = ("x", "y", "z", "qw", "qx", "qy", "qz")
        elif joint_type == mujoco.mjtJoint.mjJNT_BALL:
            suffixes = ("qw", "qx", "qy", "qz")
        else:
            suffixes = ("q",)
        for offset, suffix in enumerate(suffixes):
            index = qpos_adr + offset
            if index < model.nq:
                names.append(f"{name}/{suffix}")
    if len(names) != model.nq:
        return [f"qpos_{index}" for index in range(model.nq)]
    return names


def model_ctrl_names(model: mujoco.MjModel) -> list[str]:
    names: list[str] = []
    for actuator_id in range(model.nu):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_id)
        names.append(name or f"ctrl_{actuator_id}")
    return names


def lerobot_features(
    model: mujoco.MjModel,
    camera_specs: tuple[RenderCameraSpec, ...],
    *,
    width: int,
    height: int,
) -> dict[str, dict]:
    features: dict[str, dict] = {
        "observation.state": {
            "dtype": "float32",
            "shape": (model.nq,),
            "names": model_qpos_names(model),
        },
        "action": {
            "dtype": "float32",
            "shape": (model.nu,),
            "names": model_ctrl_names(model),
        },
        "observation.stage_index": {
            "dtype": "int64",
            "shape": (1,),
            "names": ["stage_index"],
        },
    }
    for spec in camera_specs:
        features[f"observation.images.{spec.name}"] = {
            "dtype": "video",
            "shape": (3, height, width),
            "names": ["channel", "height", "width"],
        }
    return features


def stage_index_map(labels: np.ndarray) -> dict[str, int]:
    names = sorted({str(label) for label in labels.tolist()})
    return {name: index for index, name in enumerate(names)}


def sample_indices(frame_count: int, source_fps: float, target_fps: int) -> np.ndarray:
    step = max(1, int(round(float(source_fps) / float(target_fps))))
    return np.arange(0, frame_count, step, dtype=np.int64)


def _camera_from_eye_target(eye: np.ndarray, target: np.ndarray) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.fixedcamid = -1
    delta = np.asarray(eye, dtype=np.float64) - np.asarray(target, dtype=np.float64)
    distance = float(np.linalg.norm(delta))
    if distance < 1e-6:
        delta = np.array([0.0, -1.0, 0.0], dtype=np.float64)
        distance = 1.0
    horizontal = float(np.linalg.norm(delta[:2]))
    camera.lookat[:] = np.asarray(target, dtype=np.float64)
    camera.distance = distance
    camera.azimuth = float(np.degrees(np.arctan2(delta[0], -delta[1])))
    camera.elevation = float(-np.degrees(np.arctan2(delta[2], max(horizontal, 1e-9))))
    return camera


def _orthogonalize_up(forward: np.ndarray, up_hint: np.ndarray) -> np.ndarray:
    forward = np.asarray(forward, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    up = np.asarray(up_hint, dtype=np.float64)
    up = up - forward * float(np.dot(up, forward))
    norm = float(np.linalg.norm(up))
    if norm < 1e-8:
        fallback = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        if abs(float(np.dot(fallback, forward))) > 0.98:
            fallback = np.array([0.0, 1.0, 0.0], dtype=np.float64)
        up = fallback - forward * float(np.dot(fallback, forward))
        norm = float(np.linalg.norm(up))
    return up / max(norm, 1e-12)


def _set_gl_camera(scene, *, eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> None:
    """Override MuJoCo's GL camera pose while preserving renderer frustum settings."""

    forward = np.asarray(target, dtype=np.float64) - np.asarray(eye, dtype=np.float64)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    up = _orthogonalize_up(forward, up_hint)
    for gl_camera in scene.camera:
        gl_camera.pos[:] = np.asarray(eye, dtype=np.float64)
        gl_camera.forward[:] = forward
        gl_camera.up[:] = up


class MujocoTrajectoryRenderer:
    """Render named camera observations from a sequence of MuJoCo qpos frames."""

    def __init__(
        self,
        model: mujoco.MjModel,
        camera_specs: tuple[RenderCameraSpec, ...] = DEFAULT_HANDOFF_CAMERAS,
        *,
        width: int,
        height: int,
    ) -> None:
        self.model = model
        self.data = mujoco.MjData(model)
        self.camera_specs = camera_specs
        self.renderer = mujoco.Renderer(model, width=width, height=height)
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = False
        self.renderer.scene.flags[mujoco.mjtRndFlag.mjRND_REFLECTION] = False
        self._body_ids = {
            spec.body: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.body)
            for spec in camera_specs
            if spec.body is not None
        }
        self._body_ids.update(
            {
                spec.target_body: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, spec.target_body)
                for spec in camera_specs
                if spec.target_body is not None
            }
        )
        missing = [name for name, body_id in self._body_ids.items() if body_id < 0]
        if missing:
            raise ValueError(f"Missing camera body names: {missing}")

    def close(self) -> None:
        self.renderer.close()

    def _camera_for_spec(
        self,
        spec: RenderCameraSpec,
    ) -> mujoco.MjvCamera | tuple[mujoco.MjvCamera, tuple[np.ndarray, np.ndarray, np.ndarray]]:
        if spec.mode == "free":
            camera = mujoco.MjvCamera()
            camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            camera.fixedcamid = -1
            camera.lookat[:] = np.asarray(spec.lookat, dtype=np.float64)
            camera.distance = float(spec.distance)
            camera.azimuth = float(spec.azimuth)
            camera.elevation = float(spec.elevation)
            return camera
        if spec.mode == "world":
            return _camera_from_eye_target(
                np.asarray(spec.eye_offset_world, dtype=np.float64),
                np.asarray(spec.lookat, dtype=np.float64),
            )

        if spec.body is None:
            raise ValueError(f"Camera {spec.name!r} requires a body")
        body_id = self._body_ids[spec.body]
        body_pos = self.data.xpos[body_id].copy()
        body_R = self.data.xmat[body_id].reshape(3, 3).copy()

        if spec.mode == "body_world_overhead":
            target = body_pos + body_R @ np.asarray(spec.target_offset_local, dtype=np.float64)
            eye = target + np.asarray(spec.eye_offset_world, dtype=np.float64)
            return _camera_from_eye_target(eye, target)
        if spec.mode == "body_local":
            eye = body_pos + body_R @ np.asarray(spec.eye_offset_local, dtype=np.float64)
            target = body_pos + body_R @ np.asarray(spec.target_offset_local, dtype=np.float64)
            return _camera_from_eye_target(eye, target)
        if spec.mode == "body_local_fixed_roll":
            eye = body_pos + body_R @ np.asarray(spec.eye_offset_local, dtype=np.float64)
            target = body_pos + body_R @ np.asarray(spec.target_offset_local, dtype=np.float64)
            up_hint = body_R @ np.asarray(spec.up_axis_local, dtype=np.float64)
            return _camera_from_eye_target(eye, target), (eye, target, up_hint)
        if spec.mode == "body_to_body":
            if spec.target_body is None:
                raise ValueError(f"Camera {spec.name!r} requires a target_body")
            eye = body_pos + body_R @ np.asarray(spec.eye_offset_local, dtype=np.float64)
            target = self.data.xpos[self._body_ids[spec.target_body]].copy()
            return _camera_from_eye_target(eye, target)

        raise ValueError(f"Unsupported camera mode {spec.mode!r}")

    def render(self, qpos: np.ndarray) -> dict[str, np.ndarray]:
        self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        images: dict[str, np.ndarray] = {}
        for spec in self.camera_specs:
            camera = self._camera_for_spec(spec)
            fixed_roll = None
            if isinstance(camera, tuple):
                camera, fixed_roll = camera
            self.renderer.update_scene(self.data, camera=camera)
            if fixed_roll is not None:
                eye, target, up_hint = fixed_roll
                _set_gl_camera(self.renderer.scene, eye=eye, target=target, up_hint=up_hint)
            images[spec.name] = self.renderer.render().copy()
        return images

    def iter_frames(
        self,
        qpos: np.ndarray,
        indices: np.ndarray,
    ) -> Iterator[tuple[int, dict[str, np.ndarray]]]:
        for frame_index in indices:
            yield int(frame_index), self.render(qpos[int(frame_index)])


def resolve_dataset_root(output_root: Path, task_name: str, dataset_name: str) -> Path:
    return output_root.expanduser().resolve() / task_name / dataset_name
