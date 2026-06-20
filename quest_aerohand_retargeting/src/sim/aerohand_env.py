from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    import mujoco
except ImportError:  # pragma: no cover - depends on optional runtime install.
    mujoco = None


ROBOT_HAND_LANDMARK_SITES = (
    "aero_wrist_lm",
    "aero_thumb_metacarpal_lm",
    "aero_thumb_proximal_lm",
    "aero_thumb_distal_lm",
    "aero_thumb_tip_lm",
    "aero_index_proximal_lm",
    "aero_index_intermediate_lm",
    "aero_index_distal_lm",
    "aero_index_tip_lm",
    "aero_middle_proximal_lm",
    "aero_middle_intermediate_lm",
    "aero_middle_distal_lm",
    "aero_middle_tip_lm",
    "aero_ring_proximal_lm",
    "aero_ring_intermediate_lm",
    "aero_ring_distal_lm",
    "aero_ring_tip_lm",
    "aero_little_proximal_lm",
    "aero_little_intermediate_lm",
    "aero_little_distal_lm",
    "aero_little_tip_lm",
)


class AeroHandSimEnv:
    """AeroHand MuJoCo wrapper API with a placeholder fallback.

    If ``model_path`` is provided, this class loads the combined arm + AeroHand
    model, applies compact 7D AeroHand actions, steps MuJoCo, and reads real
    thumb/index fingertip sites. If ``model_path`` is absent, it keeps the older
    placeholder distance model so mock demos still run without MuJoCo assets.
    """

    def __init__(
        self,
        model_path: str | None = None,
        thumb_tip_site: str = "aero_thumb_tip_site",
        index_tip_site: str = "aero_index_tip_site",
        middle_tip_site: str = "aero_middle_tip_site",
        ring_tip_site: str = "aero_ring_tip_site",
        little_tip_site: str = "aero_little_tip_site",
        settle_steps: int = 20,
        use_placeholder: bool | None = None,
    ):
        self.model_path = _resolve_model_path(model_path)
        self.thumb_tip_site = str(thumb_tip_site)
        self.index_tip_site = str(index_tip_site)
        self.middle_tip_site = str(middle_tip_site)
        self.ring_tip_site = str(ring_tip_site)
        self.little_tip_site = str(little_tip_site)
        self.settle_steps = max(1, int(settle_steps))
        self.use_placeholder = bool(use_placeholder) if use_placeholder is not None else self.model_path is None
        self.action = np.zeros(7, dtype=np.float64)
        self._contact = False
        self.model = None
        self.data = None
        self._thumb_site_id = None
        self._index_site_id = None
        self._tip_site_ids: dict[str, int] = {}
        self._hand_landmark_site_ids: list[int] = []
        self._base_ctrl = None
        if not self.use_placeholder:
            self._load_mujoco_model()

    def reset(self) -> dict:
        """Reset simulation state and return an observation dictionary."""
        self.action[:] = 0.0
        self._contact = False
        if self.model is not None and self.data is not None:
            mujoco.mj_resetData(self.model, self.data)
            self.data.ctrl[:] = self._base_ctrl
            mujoco.mj_forward(self.model, self.data)
        return self._obs()

    def step(self, action: np.ndarray) -> dict:
        """Apply one compact AeroHand action and return current sim state."""
        self.action = np.clip(np.asarray(action, dtype=np.float64).reshape(7), 0.0, 1.0)
        if self.model is not None and self.data is not None:
            from aero_quest.so101_aero_control import normalized_aero_hand_to_ctrl

            ctrl = self._base_ctrl.copy()
            self.data.ctrl[:] = normalized_aero_hand_to_ctrl(self.model, self.action, ctrl=ctrl)
            for _ in range(self.settle_steps):
                mujoco.mj_step(self.model, self.data)
            self._contact = self._read_thumb_index_contact()
            return self._obs()
        self._contact = self.robot_pinch_distance() < 0.03
        return self._obs()

    def evaluate_action(self, action: np.ndarray) -> dict:
        """Evaluate an action from the current state, then restore the state.

        This is useful for derivative-free retargeting candidates. The returned
        observation is real MuJoCo output when a model is loaded.
        """
        if self.model is None or self.data is None:
            return self.step(action)
        snapshot = self._snapshot_state()
        obs = self.step(action)
        obs = {
            "thumb_tip": np.asarray(obs["thumb_tip"], dtype=np.float64).copy(),
            "index_tip": np.asarray(obs["index_tip"], dtype=np.float64).copy(),
            "contact": bool(obs["contact"]),
            "robot_pinch_distance": float(obs["robot_pinch_distance"]),
            "fingertips": {name: pos.copy() for name, pos in self.get_fingertip_positions().items()},
            "hand_landmarks": np.asarray(obs["hand_landmarks"], dtype=np.float64).copy(),
        }
        self._restore_state(snapshot)
        return obs

    def get_thumb_tip_position(self) -> np.ndarray:
        """Return thumb-tip position in MuJoCo/world coordinates."""
        if self.data is not None and self._thumb_site_id is not None:
            return self.data.site_xpos[self._thumb_site_id].copy()
        gap = self.robot_pinch_distance()
        return np.array([-0.5 * gap, 0.0, 0.0], dtype=np.float64)

    def get_index_tip_position(self) -> np.ndarray:
        """Return index-tip position in MuJoCo/world coordinates."""
        if self.data is not None and self._index_site_id is not None:
            return self.data.site_xpos[self._index_site_id].copy()
        gap = self.robot_pinch_distance()
        return np.array([0.5 * gap, 0.0, 0.0], dtype=np.float64)

    def get_fingertip_positions(self) -> dict[str, np.ndarray]:
        """Return named thumb/index/middle/ring/little fingertip positions."""
        if self.data is not None and self._tip_site_ids:
            return {
                name: self.data.site_xpos[site_id].copy()
                for name, site_id in self._tip_site_ids.items()
            }
        gap = self.robot_pinch_distance()
        return {
            "thumb": np.array([-0.5 * gap, 0.0, 0.0], dtype=np.float64),
            "index": np.array([0.5 * gap, 0.0, 0.0], dtype=np.float64),
            "middle": np.array([0.0, 0.04, 0.0], dtype=np.float64),
            "ring": np.array([-0.02, 0.035, 0.0], dtype=np.float64),
            "little": np.array([-0.04, 0.025, 0.0], dtype=np.float64),
        }

    def get_hand_landmarks(self) -> np.ndarray:
        """Return 21 AeroHand landmark sites in Quest landmark order.

        Real MuJoCo output is in MuJoCo/world coordinates. Visualization code
        should normalize or transform it explicitly before comparing with Quest
        wrist-local landmarks.
        """
        if self.data is not None and self._hand_landmark_site_ids:
            return np.asarray(
                [self.data.site_xpos[site_id].copy() for site_id in self._hand_landmark_site_ids],
                dtype=np.float64,
            )
        fingertips = self.get_fingertip_positions()
        points = np.zeros((21, 3), dtype=np.float64)
        points[0] = np.array([0.0, 0.0, 0.0], dtype=np.float64)
        points[4] = fingertips["thumb"]
        points[8] = fingertips["index"]
        points[12] = fingertips["middle"]
        points[16] = fingertips["ring"]
        points[20] = fingertips["little"]
        for start, end in ((1, 4), (5, 8), (9, 12), (13, 16), (17, 20)):
            tip = points[end]
            for idx in range(start, end):
                alpha = (idx - start + 1) / float(end - start + 1)
                points[idx] = alpha * tip
        return points

    def get_contact_state(self) -> bool:
        """Return placeholder thumb-index contact state."""
        return bool(self._contact)

    def robot_pinch_distance(self) -> float:
        """Return robot thumb-index fingertip distance."""
        if self.data is not None:
            return float(np.linalg.norm(self.get_thumb_tip_position() - self.get_index_tip_position()))
        closure = float(np.clip(np.mean(self.action[[0, 1, 2, 3]]), 0.0, 1.0))
        return float(0.09 - 0.07 * closure)

    def render(self) -> None:
        """Render the simulation.

        TODO: Attach to ``mujoco.viewer`` once a concrete AeroHand-only or
        arm-plus-hand model path is selected for this pipeline.
        """
        return None

    def _load_mujoco_model(self) -> None:
        """Load MuJoCo model and fingertip site ids."""
        if mujoco is None:
            raise RuntimeError("mujoco is required for real AeroHandSimEnv. Install with: pip install mujoco")
        if self.model_path is None:
            raise ValueError("model_path is required unless use_placeholder=True")
        self.model = mujoco.MjModel.from_xml_path(str(self.model_path))
        self.data = mujoco.MjData(self.model)
        self._base_ctrl = _ctrl_midpoints(self.model)
        self._thumb_site_id = _site_id(self.model, self.thumb_tip_site)
        self._index_site_id = _site_id(self.model, self.index_tip_site)
        self._tip_site_ids = {
            "thumb": self._thumb_site_id,
            "index": self._index_site_id,
            "middle": _site_id(self.model, self.middle_tip_site),
            "ring": _site_id(self.model, self.ring_tip_site),
            "little": _site_id(self.model, self.little_tip_site),
        }
        self._hand_landmark_site_ids = [_site_id(self.model, name) for name in ROBOT_HAND_LANDMARK_SITES]

    def _read_thumb_index_contact(self) -> bool:
        """Return whether thumb and index tip geoms are in contact."""
        if self.model is None or self.data is None:
            return False
        thumb_names = {"th_tip"}
        index_names = {"if_tip"}
        for contact_index in range(int(self.data.ncon)):
            contact = self.data.contact[contact_index]
            geom1 = _geom_name(self.model, int(contact.geom1))
            geom2 = _geom_name(self.model, int(contact.geom2))
            if (geom1 in thumb_names and geom2 in index_names) or (geom2 in thumb_names and geom1 in index_names):
                return True
        return False

    def _obs(self) -> dict:
        return {
            "thumb_tip": self.get_thumb_tip_position(),
            "index_tip": self.get_index_tip_position(),
            "fingertips": self.get_fingertip_positions(),
            "hand_landmarks": self.get_hand_landmarks(),
            "contact": self.get_contact_state(),
            "robot_pinch_distance": self.robot_pinch_distance(),
        }

    def _snapshot_state(self) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Copy MuJoCo state arrays needed to restore after candidate eval."""
        return (
            self.data.qpos.copy(),
            self.data.qvel.copy(),
            self.data.act.copy(),
            self.data.ctrl.copy(),
        )

    def _restore_state(self, snapshot: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]) -> None:
        """Restore a snapshot created by :meth:`_snapshot_state`."""
        qpos, qvel, act, ctrl = snapshot
        self.data.qpos[:] = qpos
        self.data.qvel[:] = qvel
        if self.data.act.shape == act.shape:
            self.data.act[:] = act
        self.data.ctrl[:] = ctrl
        mujoco.mj_forward(self.model, self.data)


def _resolve_model_path(model_path: str | None) -> Path | None:
    """Resolve a model path relative to the repository root."""
    if model_path is None or str(model_path).strip().lower() in {"", "none", "null"}:
        return None
    path = Path(model_path).expanduser()
    if path.is_absolute():
        return path
    return Path(__file__).resolve().parents[3] / path


def _ctrl_midpoints(model) -> np.ndarray:
    """Return midpoint ctrl for every actuator."""
    return 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])


def _site_id(model, name: str) -> int:
    """Return site id or raise a clear error."""
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise ValueError(f"MuJoCo site not found: {name}")
    return int(site_id)


def _geom_name(model, geom_id: int) -> str:
    """Return MuJoCo geom name or an empty string."""
    if geom_id < 0:
        return ""
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
    return "" if name is None else str(name)
