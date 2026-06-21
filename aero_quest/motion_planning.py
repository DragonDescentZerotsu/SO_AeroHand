"""Lightweight MuJoCo motion-planning utilities.

The planner here is intentionally dependency-light: MuJoCo provides forward
kinematics and collision checks, SciPy solves bounded pose IK, and a compact
RRT-Connect implementation searches joint-space paths. The public interfaces
are small so a future OMPL/MoveIt backend can replace the search layer without
rewriting task scripts.
"""

from __future__ import annotations

from dataclasses import dataclass
from collections.abc import Callable, Sequence
import math
import time

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

try:
    import mujoco
except ImportError:  # pragma: no cover
    mujoco = None


def require_mujoco() -> None:
    if mujoco is None:
        raise RuntimeError("mujoco is required for motion planning")


def normalize(vec: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    vec = np.asarray(vec, dtype=np.float64)
    norm = float(np.linalg.norm(vec))
    if norm < eps:
        raise ValueError("Cannot normalize near-zero vector")
    return vec / norm


def quat_wxyz_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    return Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]]).as_matrix()


def matrix_to_quat_wxyz(matrix: np.ndarray) -> np.ndarray:
    xyzw = Rotation.from_matrix(np.asarray(matrix, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def rotation_error(target_R: np.ndarray, current_R: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(target_R @ current_R.T).as_rotvec()


def transform_inv(pos: np.ndarray, R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    inv_R = R.T
    return -(inv_R @ pos), inv_R


def transform_mul(
    a_pos: np.ndarray,
    a_R: np.ndarray,
    b_pos: np.ndarray,
    b_R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    return a_pos + a_R @ b_pos, a_R @ b_R


def relative_transform(
    parent_pos: np.ndarray,
    parent_R: np.ndarray,
    child_pos: np.ndarray,
    child_R: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    inv_pos, inv_R = transform_inv(parent_pos, parent_R)
    return transform_mul(inv_pos, inv_R, child_pos, child_R)


def frame_from_z_axis(z_axis: np.ndarray, x_hint: np.ndarray) -> np.ndarray:
    z_axis = normalize(z_axis)
    x_hint = np.asarray(x_hint, dtype=np.float64)
    x_axis = x_hint - z_axis * float(np.dot(z_axis, x_hint))
    if float(np.linalg.norm(x_axis)) < 1e-8:
        for fallback in (np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])):
            x_axis = fallback - z_axis * float(np.dot(z_axis, fallback))
            if float(np.linalg.norm(x_axis)) >= 1e-8:
                break
    x_axis = normalize(x_axis)
    y_axis = normalize(np.cross(z_axis, x_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))
    return np.stack([x_axis, y_axis, z_axis], axis=1)


def rotate_about_axis(R: np.ndarray, axis_world: np.ndarray, angle: float) -> np.ndarray:
    return Rotation.from_rotvec(normalize(axis_world) * float(angle)).as_matrix() @ R


@dataclass(frozen=True)
class JointGroup:
    names: tuple[str, ...]
    joint_ids: np.ndarray
    qpos_ids: np.ndarray
    dof_ids: np.ndarray
    lower: np.ndarray
    upper: np.ndarray

    @classmethod
    def from_names(cls, model, names: Sequence[str]) -> "JointGroup":
        require_mujoco()
        joint_ids: list[int] = []
        qpos_ids: list[int] = []
        dof_ids: list[int] = []
        lower: list[float] = []
        upper: list[float] = []
        for name in names:
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, str(name))
            if jid < 0:
                raise ValueError(f"Missing joint {name!r}")
            joint_ids.append(jid)
            qpos_ids.append(int(model.jnt_qposadr[jid]))
            dof_ids.append(int(model.jnt_dofadr[jid]))
            if bool(model.jnt_limited[jid]):
                lower.append(float(model.jnt_range[jid, 0]))
                upper.append(float(model.jnt_range[jid, 1]))
            else:
                lower.append(-math.pi)
                upper.append(math.pi)
        return cls(
            names=tuple(str(name) for name in names),
            joint_ids=np.asarray(joint_ids, dtype=np.int32),
            qpos_ids=np.asarray(qpos_ids, dtype=np.int32),
            dof_ids=np.asarray(dof_ids, dtype=np.int32),
            lower=np.asarray(lower, dtype=np.float64),
            upper=np.asarray(upper, dtype=np.float64),
        )


class PlanningModel:
    def __init__(self, model_path: str):
        require_mujoco()
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)
        self.base_qpos = self.model.qpos0.copy()
        self.base_ctrl = np.zeros(self.model.nu, dtype=np.float64)

    def body_id(self, name: str) -> int:
        bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
        if bid < 0:
            raise ValueError(f"Missing body {name!r}")
        return bid

    def joint_id(self, name: str) -> int:
        jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            raise ValueError(f"Missing joint {name!r}")
        return jid

    def set_joint_group(self, qpos: np.ndarray, group: JointGroup, q: np.ndarray) -> None:
        qpos[group.qpos_ids] = np.asarray(q, dtype=np.float64).reshape(len(group.names))

    def get_joint_group(self, qpos: np.ndarray, group: JointGroup) -> np.ndarray:
        return np.asarray(qpos[group.qpos_ids], dtype=np.float64).copy()

    def forward(self, qpos: np.ndarray | None = None):
        if qpos is not None:
            self.data.qpos[:] = qpos
        mujoco.mj_forward(self.model, self.data)
        return self.data

    def body_pose(self, body_name: str) -> tuple[np.ndarray, np.ndarray]:
        bid = self.body_id(body_name)
        self.forward()
        return self.data.xpos[bid].copy(), self.data.xmat[bid].reshape(3, 3).copy()

    def set_freejoint_pose(
        self,
        qpos: np.ndarray,
        joint_name: str,
        pos: np.ndarray,
        R: np.ndarray,
    ) -> None:
        jid = self.joint_id(joint_name)
        adr = int(self.model.jnt_qposadr[jid])
        qpos[adr : adr + 3] = np.asarray(pos, dtype=np.float64).reshape(3)
        qpos[adr + 3 : adr + 7] = matrix_to_quat_wxyz(R)

    def contact_pairs(self) -> list[tuple[str, str, float]]:
        pairs: list[tuple[str, str, float]] = []
        for index in range(self.data.ncon):
            contact = self.data.contact[index]
            body1 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[contact.geom1]),
            )
            body2 = mujoco.mj_id2name(
                self.model,
                mujoco.mjtObj.mjOBJ_BODY,
                int(self.model.geom_bodyid[contact.geom2]),
            )
            pairs.append((body1 or "", body2 or "", float(contact.dist)))
        return pairs


def _matches_prefix(name: str, prefixes: Sequence[str]) -> bool:
    return any(name == prefix or name.startswith(f"{prefix}/") for prefix in prefixes)


@dataclass(frozen=True)
class CollisionPolicy:
    robot_prefixes: tuple[str, ...]
    obstacle_prefixes: tuple[str, ...]
    object_prefixes: tuple[str, ...] = ()
    object_obstacle_prefixes: tuple[str, ...] = ()
    margin: float = 0.002

    def is_forbidden(self, body1: str, body2: str, dist: float) -> bool:
        if dist > self.margin:
            return False
        first_robot = _matches_prefix(body1, self.robot_prefixes)
        second_robot = _matches_prefix(body2, self.robot_prefixes)
        first_obstacle = _matches_prefix(body1, self.obstacle_prefixes)
        second_obstacle = _matches_prefix(body2, self.obstacle_prefixes)
        if (first_robot and second_obstacle) or (second_robot and first_obstacle):
            return True
        first_object = _matches_prefix(body1, self.object_prefixes)
        second_object = _matches_prefix(body2, self.object_prefixes)
        first_object_obstacle = _matches_prefix(body1, self.object_obstacle_prefixes)
        second_object_obstacle = _matches_prefix(body2, self.object_obstacle_prefixes)
        return (first_object and second_object_obstacle) or (second_object and first_object_obstacle)


def solve_body_pose_ik(
    planning_model: PlanningModel,
    group: JointGroup,
    body_name: str,
    target_pos: np.ndarray,
    target_R: np.ndarray,
    *,
    seed: np.ndarray,
    extra_seeds: Sequence[np.ndarray] = (),
    collision_policy: CollisionPolicy | None = None,
    fixed_qpos: np.ndarray | None = None,
    pos_weight: float = 60.0,
    rot_weight: float = 8.0,
    regularization_weight: float = 0.03,
    max_nfev: int = 250,
) -> tuple[np.ndarray, float]:
    """Solve bounded IK for one body pose and return ``(q, cost)``."""

    model = planning_model.model
    data = planning_model.data
    body_id = planning_model.body_id(body_name)
    target_pos = np.asarray(target_pos, dtype=np.float64).reshape(3)
    target_R = np.asarray(target_R, dtype=np.float64).reshape(3, 3)
    seed = np.asarray(seed, dtype=np.float64).reshape(len(group.names))
    base_qpos = planning_model.base_qpos.copy() if fixed_qpos is None else fixed_qpos.copy()
    starts = [seed.copy(), np.clip(seed, group.lower, group.upper)]
    starts.extend(np.asarray(value, dtype=np.float64).reshape(len(group.names)) for value in extra_seeds)

    def residual(q: np.ndarray) -> np.ndarray:
        qpos = base_qpos.copy()
        qpos[group.qpos_ids] = q
        data.qpos[:] = qpos
        mujoco.mj_forward(model, data)
        current_pos = data.xpos[body_id]
        current_R = data.xmat[body_id].reshape(3, 3)
        return np.concatenate(
            [
                pos_weight * (current_pos - target_pos),
                rot_weight * rotation_error(target_R, current_R),
                regularization_weight * (q - seed),
            ]
        )

    best_q: np.ndarray | None = None
    best_cost = math.inf
    for start in starts:
        start = np.clip(start, group.lower, group.upper)
        result = least_squares(
            residual,
            start,
            bounds=(group.lower, group.upper),
            max_nfev=max_nfev,
            xtol=1e-7,
            ftol=1e-7,
            gtol=1e-7,
        )
        cost = float(np.linalg.norm(residual(result.x)))
        if result.success and cost < best_cost:
            if collision_policy is not None:
                qpos = base_qpos.copy()
                qpos[group.qpos_ids] = result.x
                planning_model.forward(qpos)
                if any(collision_policy.is_forbidden(*pair) for pair in planning_model.contact_pairs()):
                    continue
            best_q = result.x.copy()
            best_cost = cost
    if best_q is None:
        raise RuntimeError(f"IK failed for {body_name} target")
    return best_q, best_cost


@dataclass(frozen=True)
class RRTConnectConfig:
    step_size: float = 0.10
    edge_resolution: float = 0.025
    max_iterations: int = 5000
    goal_bias: float = 0.12
    timeout_s: float = 8.0
    shortcut_attempts: int = 120
    seed: int = 7


@dataclass(frozen=True)
class RRTResult:
    path: np.ndarray
    iterations: int
    solve_time_s: float
    status: str


@dataclass
class _Node:
    q: np.ndarray
    parent: int


class RRTConnectPlanner:
    def __init__(
        self,
        lower: np.ndarray,
        upper: np.ndarray,
        is_state_valid: Callable[[np.ndarray], bool],
        is_edge_valid: Callable[[np.ndarray, np.ndarray], bool] | None = None,
        config: RRTConnectConfig | None = None,
    ):
        self.lower = np.asarray(lower, dtype=np.float64)
        self.upper = np.asarray(upper, dtype=np.float64)
        self.is_state_valid = is_state_valid
        self.is_edge_valid = is_edge_valid or self._default_edge_valid
        self.config = config or RRTConnectConfig()
        self.rng = np.random.default_rng(self.config.seed)

    def _default_edge_valid(self, a: np.ndarray, b: np.ndarray) -> bool:
        distance = float(np.linalg.norm(b - a))
        steps = max(2, int(math.ceil(distance / max(self.config.edge_resolution, 1e-6))) + 1)
        for alpha in np.linspace(0.0, 1.0, steps):
            if not self.is_state_valid((1.0 - alpha) * a + alpha * b):
                return False
        return True

    def _sample(self, goal: np.ndarray) -> np.ndarray:
        if self.rng.random() < self.config.goal_bias:
            return goal.copy()
        return self.rng.uniform(self.lower, self.upper)

    def _nearest(self, tree: list[_Node], q: np.ndarray) -> int:
        distances = [float(np.linalg.norm(node.q - q)) for node in tree]
        return int(np.argmin(distances))

    def _steer(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        delta = b - a
        distance = float(np.linalg.norm(delta))
        if distance <= self.config.step_size:
            return b.copy()
        return a + delta * (self.config.step_size / max(distance, 1e-12))

    def _extend(self, tree: list[_Node], q_target: np.ndarray) -> tuple[str, int]:
        nearest_id = self._nearest(tree, q_target)
        q_new = self._steer(tree[nearest_id].q, q_target)
        if not self.is_edge_valid(tree[nearest_id].q, q_new):
            return "trapped", nearest_id
        tree.append(_Node(q=q_new, parent=nearest_id))
        new_id = len(tree) - 1
        if float(np.linalg.norm(q_new - q_target)) < 1e-9:
            return "reached", new_id
        return "advanced", new_id

    def _connect(self, tree: list[_Node], q_target: np.ndarray) -> tuple[str, int]:
        status, node_id = self._extend(tree, q_target)
        while status == "advanced":
            status, node_id = self._extend(tree, q_target)
        return status, node_id

    def _trace(self, tree: list[_Node], node_id: int) -> list[np.ndarray]:
        path = []
        while node_id >= 0:
            node = tree[node_id]
            path.append(node.q)
            node_id = node.parent
        path.reverse()
        return path

    def _assemble(
        self,
        tree_a: list[_Node],
        id_a: int,
        tree_b: list[_Node],
        id_b: int,
        swapped: bool,
    ) -> np.ndarray:
        path_a = self._trace(tree_a, id_a)
        path_b = self._trace(tree_b, id_b)
        if swapped:
            path = path_b + path_a[::-1][1:]
        else:
            path = path_a + path_b[::-1][1:]
        return np.asarray(path, dtype=np.float64)

    def _shortcut(self, path: np.ndarray) -> np.ndarray:
        if len(path) <= 2 or self.config.shortcut_attempts <= 0:
            return path
        points = [p.copy() for p in path]
        for _ in range(self.config.shortcut_attempts):
            if len(points) <= 2:
                break
            i, j = sorted(self.rng.choice(len(points), size=2, replace=False).tolist())
            if j <= i + 1:
                continue
            if self.is_edge_valid(points[i], points[j]):
                points = points[: i + 1] + points[j:]
        return np.asarray(points, dtype=np.float64)

    def plan(self, start: np.ndarray, goal: np.ndarray) -> RRTResult:
        start_time = time.perf_counter()
        start = np.asarray(start, dtype=np.float64)
        goal = np.asarray(goal, dtype=np.float64)
        if not self.is_state_valid(start):
            raise RuntimeError("RRT start state is invalid")
        if not self.is_state_valid(goal):
            raise RuntimeError("RRT goal state is invalid")
        tree_start = [_Node(q=start.copy(), parent=-1)]
        tree_goal = [_Node(q=goal.copy(), parent=-1)]
        swapped = False
        for iteration in range(1, self.config.max_iterations + 1):
            if time.perf_counter() - start_time > self.config.timeout_s:
                break
            tree_a = tree_goal if swapped else tree_start
            tree_b = tree_start if swapped else tree_goal
            q_rand = self._sample(goal if not swapped else start)
            status_a, id_a = self._extend(tree_a, q_rand)
            if status_a != "trapped":
                status_b, id_b = self._connect(tree_b, tree_a[id_a].q)
                if status_b == "reached":
                    path = self._assemble(tree_a, id_a, tree_b, id_b, swapped)
                    path = self._shortcut(path)
                    return RRTResult(
                        path=path,
                        iterations=iteration,
                        solve_time_s=time.perf_counter() - start_time,
                        status="solved",
                    )
            swapped = not swapped
        raise RuntimeError("RRTConnect failed to find a path")
