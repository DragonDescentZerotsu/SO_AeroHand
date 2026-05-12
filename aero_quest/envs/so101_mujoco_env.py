"""Gymnasium-compatible MuJoCo environment for TheRobotStudio SO101 arm."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only when gymnasium is absent.
    gym = None

    class _Box:
        def __init__(self, low, high, shape=None, dtype=np.float32):
            self.low = np.asarray(low, dtype=dtype)
            self.high = np.asarray(high, dtype=dtype)
            self.shape = tuple(shape) if shape is not None else self.low.shape
            self.dtype = dtype

        def sample(self):
            low = np.broadcast_to(self.low, self.shape)
            high = np.broadcast_to(self.high, self.shape)
            return np.random.uniform(low, high).astype(self.dtype)

    class _Dict(dict):
        def sample(self):
            return {key: value.sample() for key, value in self.items()}

    class _Env:
        metadata: dict[str, Any] = {}

    class _Spaces:
        Box = _Box
        Dict = _Dict

    class _Gym:
        Env = _Env

    gym = _Gym()
    spaces = _Spaces()

try:
    import mujoco
except ImportError:
    mujoco = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OFFICIAL_SCENE = PROJECT_ROOT / "third_party/SO-ARM100/Simulation/SO101/scene.xml"
DEFAULT_MENAGERIE_MODEL = PROJECT_ROOT / "mujoco_menagerie/robotstudio_so101/so101.xml"


def default_so101_model_path() -> Path:
    """Return the preferred SO101 model path available in this checkout."""
    if DEFAULT_OFFICIAL_SCENE.exists():
        return DEFAULT_OFFICIAL_SCENE
    return DEFAULT_MENAGERIE_MODEL


def _require_mujoco() -> None:
    if mujoco is None:
        raise RuntimeError("mujoco is required. Install it with: pip install mujoco")


def _name_or_empty(model, obj_type, obj_id: int) -> str:
    name = mujoco.mj_id2name(model, obj_type, int(obj_id))
    return "" if name is None else str(name)


def gripper_lerobot_to_mujoco(value: float | np.ndarray, ctrlrange: np.ndarray) -> np.ndarray:
    """Convert LeRobot gripper value ``0=closed, 100=open`` to MuJoCo ctrl.

    TODO: This is a conservative linear mapping over the XML actuator range.
    The official URDF/MJCF does not fully encode the LeRobot gripper calibration,
    so real hardware or official calibration should replace this mapping later.
    """
    lo, hi = np.asarray(ctrlrange, dtype=np.float64)
    value = np.clip(np.asarray(value, dtype=np.float64), 0.0, 100.0) / 100.0
    return lo + value * (hi - lo)


def gripper_mujoco_to_lerobot(value: float | np.ndarray, ctrlrange: np.ndarray) -> np.ndarray:
    """Convert MuJoCo gripper ctrl/qpos to LeRobot ``0=closed, 100=open``."""
    lo, hi = np.asarray(ctrlrange, dtype=np.float64)
    denom = max(float(hi - lo), 1e-8)
    value = (np.asarray(value, dtype=np.float64) - lo) / denom
    return 100.0 * np.clip(value, 0.0, 1.0)


class SO101MujocoEnv(gym.Env):
    """Small SO101 MuJoCo env with normalized position-control actions."""

    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 50}

    def __init__(
        self,
        model_path: str | Path | None = None,
        control_dt: float = 0.02,
        physics_dt: float | None = None,
        render_mode: str | None = None,
        render_width: int = 640,
        render_height: int = 480,
        camera_names: list[str] | tuple[str, ...] | None = None,
        action_scale: float | np.ndarray = 1.0,
        episode_len: int = 200,
        init_qpos: list[float] | np.ndarray | dict[str, float] | None = None,
        reward_type: str = "zero",
        ee_site_name: str = "gripperframe",
        print_model_info: bool = True,
    ):
        """Create the SO101 environment.

        Actions are normalized to ``[-1, 1]`` and mapped to actuator ctrl ranges.
        Observations are a dict containing arm joint qpos/qvel, end-effector pose,
        and gripper state.
        """
        _require_mujoco()
        self.model_path = Path(model_path).expanduser() if model_path is not None else default_so101_model_path()
        if not self.model_path.is_absolute():
            self.model_path = PROJECT_ROOT / self.model_path
        if not self.model_path.exists():
            raise FileNotFoundError(f"SO101 model XML not found: {self.model_path}")

        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        if physics_dt is not None:
            self.model.opt.timestep = float(physics_dt)
        self.data = mujoco.MjData(self.model)
        self.control_dt = float(control_dt)
        self.frame_skip = max(1, int(round(self.control_dt / float(self.model.opt.timestep))))
        self.render_mode = render_mode
        self.render_width = int(render_width)
        self.render_height = int(render_height)
        self.camera_names = list(camera_names or [])
        self.action_scale = np.asarray(action_scale, dtype=np.float64)
        self.episode_len = int(episode_len)
        self.reward_type = str(reward_type)
        self.ee_site_name = str(ee_site_name)
        self._step_count = 0
        self._renderer = None

        self.joint_names = self._read_joint_names()
        self.actuator_names = self._read_actuator_names()
        self.actuator_joint_names = self._read_actuator_joint_names()
        self.control_joint_names = [name for name in self.actuator_joint_names if name]
        self.qpos_indices = np.asarray([self._joint_qpos_index(name) for name in self.control_joint_names], dtype=np.int64)
        self.qvel_indices = np.asarray([self._joint_qvel_index(name) for name in self.control_joint_names], dtype=np.int64)
        self.ctrlrange = np.asarray(self.model.actuator_ctrlrange, dtype=np.float64)
        self.ctrl_mid = 0.5 * (self.ctrlrange[:, 0] + self.ctrlrange[:, 1])
        self.ctrl_half_range = 0.5 * (self.ctrlrange[:, 1] - self.ctrlrange[:, 0])
        self.gripper_actuator_id = self._find_name_index(self.actuator_names, "gripper")
        self.gripper_joint_id = self._find_name_index(self.control_joint_names, "gripper")
        self.ee_site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, self.ee_site_name)
        self.ee_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, "gripper")

        self.init_qpos = self._build_init_qpos(init_qpos)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(self.model.nu,), dtype=np.float32)
        self.observation_space = spaces.Dict(
            {
                "qpos": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.qpos_indices),), dtype=np.float32),
                "qvel": spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.qvel_indices),), dtype=np.float32),
                "ee_pos": spaces.Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32),
                "ee_quat": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
                "gripper": spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            }
        )

        if print_model_info:
            self.print_model_info()

    def _read_joint_names(self) -> list[str]:
        return [_name_or_empty(self.model, mujoco.mjtObj.mjOBJ_JOINT, idx) for idx in range(self.model.njnt)]

    def _read_actuator_names(self) -> list[str]:
        return [_name_or_empty(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, idx) for idx in range(self.model.nu)]

    def _read_actuator_joint_names(self) -> list[str]:
        names = []
        for actuator_id in range(self.model.nu):
            trn_type = int(self.model.actuator_trntype[actuator_id])
            trn_id = int(self.model.actuator_trnid[actuator_id, 0])
            if trn_type == int(mujoco.mjtTrn.mjTRN_JOINT) and trn_id >= 0:
                names.append(_name_or_empty(self.model, mujoco.mjtObj.mjOBJ_JOINT, trn_id))
            else:
                names.append("")
        return names

    def _joint_qpos_index(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        return int(self.model.jnt_qposadr[joint_id])

    def _joint_qvel_index(self, joint_name: str) -> int:
        joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            raise ValueError(f"Joint not found: {joint_name}")
        return int(self.model.jnt_dofadr[joint_id])

    @staticmethod
    def _find_name_index(names: list[str], name: str) -> int | None:
        try:
            return names.index(name)
        except ValueError:
            return None

    def _build_init_qpos(self, init_qpos):
        qpos = self.data.qpos.copy()
        if init_qpos is None:
            for ctrl_id, joint_name in enumerate(self.control_joint_names):
                if not joint_name:
                    continue
                qpos[self._joint_qpos_index(joint_name)] = self.ctrl_mid[ctrl_id]
            return qpos
        if isinstance(init_qpos, dict):
            for joint_name, value in init_qpos.items():
                qpos[self._joint_qpos_index(joint_name)] = float(value)
            return qpos
        values = np.asarray(init_qpos, dtype=np.float64)
        if values.shape == (self.model.nq,):
            return values.copy()
        if values.shape == (len(self.qpos_indices),):
            qpos[self.qpos_indices] = values
            return qpos
        raise ValueError(f"init_qpos must have shape ({self.model.nq},) or ({len(self.qpos_indices)},), got {values.shape}")

    def print_model_info(self) -> None:
        """Print dynamically discovered joints, actuators, and qpos/qvel indices."""
        print(f"SO101 model: {self.model_path}")
        print(f"nq={self.model.nq} nv={self.model.nv} nu={self.model.nu} timestep={self.model.opt.timestep}")
        print("Joints:")
        for name in self.joint_names:
            joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
            print(
                f"  {name}: qpos[{self.model.jnt_qposadr[joint_id]}] "
                f"qvel[{self.model.jnt_dofadr[joint_id]}] range={self.model.jnt_range[joint_id]}"
            )
        print("Actuators:")
        for actuator_id, name in enumerate(self.actuator_names):
            print(
                f"  ctrl[{actuator_id}] {name}: joint={self.actuator_joint_names[actuator_id]} "
                f"ctrlrange={self.ctrlrange[actuator_id]}"
            )

    def normalized_action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        """Map normalized ``[-1, 1]`` action to actuator ctrlrange with clipping."""
        action = np.asarray(action, dtype=np.float64)
        if action.shape != (self.model.nu,):
            raise ValueError(f"Expected action shape ({self.model.nu},), got {action.shape}")
        scale = np.broadcast_to(self.action_scale, (self.model.nu,))
        action = np.clip(action, -1.0, 1.0)
        ctrl = self.ctrl_mid + action * self.ctrl_half_range * scale
        return np.clip(ctrl, self.ctrlrange[:, 0], self.ctrlrange[:, 1]).astype(np.float64)

    def _get_ee_pose(self) -> tuple[np.ndarray, np.ndarray]:
        if self.ee_site_id >= 0:
            pos = self.data.site_xpos[self.ee_site_id].copy()
            mat = self.data.site_xmat[self.ee_site_id].reshape(3, 3).copy()
        elif self.ee_body_id >= 0:
            pos = self.data.xpos[self.ee_body_id].copy()
            mat = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()
        else:
            pos = np.zeros(3, dtype=np.float64)
            mat = np.eye(3, dtype=np.float64)
        quat = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(quat, mat.reshape(-1))
        return pos, quat

    def _get_obs(self) -> dict[str, np.ndarray]:
        ee_pos, ee_quat = self._get_ee_pose()
        gripper_qpos = 0.0
        gripper_lerobot = 0.0
        if self.gripper_joint_id is not None:
            gripper_qpos = float(self.data.qpos[self.qpos_indices[self.gripper_joint_id]])
        if self.gripper_actuator_id is not None:
            gripper_lerobot = float(
                gripper_mujoco_to_lerobot(gripper_qpos, self.ctrlrange[self.gripper_actuator_id])
            )
        return {
            "qpos": self.data.qpos[self.qpos_indices].astype(np.float32).copy(),
            "qvel": self.data.qvel[self.qvel_indices].astype(np.float32).copy(),
            "ee_pos": ee_pos.astype(np.float32),
            "ee_quat": ee_quat.astype(np.float32),
            "gripper": np.asarray([gripper_qpos, gripper_lerobot], dtype=np.float32),
        }

    def _get_info(self) -> dict[str, Any]:
        return {
            "joint_names": list(self.control_joint_names),
            "actuator_names": list(self.actuator_names),
            "qpos_indices": self.qpos_indices.copy(),
            "qvel_indices": self.qvel_indices.copy(),
        }

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None):
        """Reset the simulation and return ``(obs, info)``."""
        if hasattr(super(), "reset"):
            try:
                super().reset(seed=seed)
            except TypeError:
                pass
        self._step_count = 0
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = 0.0
        self.data.ctrl[:] = np.clip(self.init_qpos[self.qpos_indices], self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        if options and "qpos" in options:
            values = np.asarray(options["qpos"], dtype=np.float64)
            if values.shape == (len(self.qpos_indices),):
                self.data.qpos[self.qpos_indices] = values
                self.data.ctrl[:] = np.clip(values, self.ctrlrange[:, 0], self.ctrlrange[:, 1])
        mujoco.mj_forward(self.model, self.data)
        return self._get_obs(), self._get_info()

    def step(self, action: np.ndarray):
        """Apply one normalized action and advance the simulation."""
        self.data.ctrl[:] = self.normalized_action_to_ctrl(action)
        for _ in range(self.frame_skip):
            mujoco.mj_step(self.model, self.data)
        self._step_count += 1
        obs = self._get_obs()
        reward = self._compute_reward(obs)
        terminated = False
        truncated = self._step_count >= self.episode_len
        info = self._get_info()
        return obs, reward, terminated, truncated, info

    def _compute_reward(self, obs: dict[str, np.ndarray]) -> float:
        if self.reward_type == "zero":
            return 0.0
        if self.reward_type == "alive":
            return 1.0
        return 0.0

    def render(self):
        """Render the current scene."""
        if self.render_mode == "human":
            return None
        if self._renderer is None:
            self._renderer = mujoco.Renderer(self.model, height=self.render_height, width=self.render_width)
        self._renderer.update_scene(self.data, camera=self.camera_names[0] if self.camera_names else None)
        return self._renderer.render()

    def close(self) -> None:
        """Release rendering resources."""
        if self._renderer is not None:
            self._renderer.close()
            self._renderer = None
