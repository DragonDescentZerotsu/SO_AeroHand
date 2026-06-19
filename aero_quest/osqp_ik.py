"""Warm-started OSQP solver for constrained resolved-rate inverse kinematics."""

from __future__ import annotations

from dataclasses import dataclass
import time

import numpy as np
import osqp
from scipy import sparse


@dataclass(frozen=True)
class OSQPIKConfig:
    base_damping: float = 0.035
    accel_weight: float = 0.04
    max_joint_speed: float = 5.0
    max_joint_accel: float = 120.0
    singular_damping_threshold: float = 0.10
    singular_damping_gain: float = 0.10
    max_iter: int = 80
    eps_abs: float = 1e-4
    eps_rel: float = 1e-4


@dataclass(frozen=True)
class OSQPIKResult:
    qdot: np.ndarray
    status: str
    iterations: int
    min_singular: float
    effective_damping: float
    solve_time_s: float
    wall_time_s: float


def effective_singular_damping(
    jacobian: np.ndarray,
    base_damping: float,
    threshold: float,
    gain: float,
) -> tuple[float, float]:
    singular_values = np.linalg.svd(np.asarray(jacobian, dtype=np.float64), compute_uv=False)
    min_singular = float(singular_values[-1]) if singular_values.size else 0.0
    threshold = max(float(threshold), 0.0)
    gain = max(float(gain), 0.0)
    if threshold <= 0.0 or min_singular >= threshold:
        return float(base_damping), min_singular
    scale = 1.0 - min_singular / max(threshold, 1e-12)
    return float(base_damping) + gain * scale * scale, min_singular


def _upper_triangle_csc(matrix: np.ndarray) -> sparse.csc_matrix:
    """Return a CSC upper triangle with a fixed dense triangular pattern."""
    matrix = np.asarray(matrix, dtype=np.float64)
    n = matrix.shape[0]
    indices = np.concatenate([np.arange(column + 1, dtype=np.int32) for column in range(n)])
    indptr = np.zeros(n + 1, dtype=np.int32)
    indptr[1:] = np.cumsum(np.arange(1, n + 1, dtype=np.int32))
    data = np.concatenate([matrix[: column + 1, column] for column in range(n)])
    return sparse.csc_matrix((data, indices, indptr), shape=(n, n))


def _upper_triangle_values(matrix: np.ndarray) -> np.ndarray:
    matrix = np.asarray(matrix, dtype=np.float64)
    return np.concatenate([matrix[: column + 1, column] for column in range(matrix.shape[0])])


class OSQPVelocityIK:
    """Solve repeated velocity IK QPs using one persistent OSQP workspace."""

    def __init__(
        self,
        joint_count: int,
        task_dimension: int,
        joint_motion_weights: np.ndarray,
        task_weights: np.ndarray,
        config: OSQPIKConfig | None = None,
    ):
        self.joint_count = int(joint_count)
        self.task_dimension = int(task_dimension)
        self.joint_motion_weights = np.asarray(joint_motion_weights, dtype=np.float64).reshape(self.joint_count)
        self.task_weights = np.asarray(task_weights, dtype=np.float64).reshape(self.task_dimension)
        if np.any(self.joint_motion_weights <= 0.0):
            raise ValueError("joint_motion_weights must be positive")
        if np.any(self.task_weights <= 0.0):
            raise ValueError("task_weights must be positive")
        self.config = config or OSQPIKConfig()
        self.prev_qdot = np.zeros(self.joint_count, dtype=np.float64)
        self._solver: osqp.OSQP | None = None

    def reset(self, qdot: np.ndarray | None = None) -> None:
        self.prev_qdot = (
            np.zeros(self.joint_count, dtype=np.float64)
            if qdot is None
            else np.asarray(qdot, dtype=np.float64).reshape(self.joint_count).copy()
        )
        if self._solver is not None:
            self._solver.warm_start(x=self.prev_qdot)

    def _objective(
        self,
        jacobian: np.ndarray,
        task_velocity: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, float, float]:
        effective_damping, min_singular = effective_singular_damping(
            jacobian,
            base_damping=self.config.base_damping,
            threshold=self.config.singular_damping_threshold,
            gain=self.config.singular_damping_gain,
        )
        weighted_jacobian = self.task_weights[:, None] * jacobian
        weighted_velocity = self.task_weights * task_velocity
        regularization = np.diag((effective_damping * self.joint_motion_weights) ** 2)
        hessian = weighted_jacobian.T @ weighted_jacobian + regularization
        gradient = -(weighted_jacobian.T @ weighted_velocity)
        if self.config.accel_weight > 0.0:
            hessian += self.config.accel_weight * np.eye(self.joint_count, dtype=np.float64)
            gradient += -self.config.accel_weight * self.prev_qdot
        return hessian, gradient, effective_damping, min_singular

    def _bounds(
        self,
        q_current: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        dt: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        dt = max(float(dt), 1e-6)
        max_speed = abs(float(self.config.max_joint_speed))
        lower = np.full(self.joint_count, -max_speed, dtype=np.float64)
        upper = np.full(self.joint_count, max_speed, dtype=np.float64)
        finite_lower = np.isfinite(joint_lower)
        finite_upper = np.isfinite(joint_upper)
        lower[finite_lower] = np.maximum(
            lower[finite_lower],
            (joint_lower[finite_lower] - q_current[finite_lower]) / dt,
        )
        upper[finite_upper] = np.minimum(
            upper[finite_upper],
            (joint_upper[finite_upper] - q_current[finite_upper]) / dt,
        )

        max_accel = abs(float(self.config.max_joint_accel))
        if max_accel > 0.0:
            accel_lower = self.prev_qdot - max_accel * dt
            accel_upper = self.prev_qdot + max_accel * dt
            lower = np.maximum(lower, accel_lower)
            upper = np.minimum(upper, accel_upper)
        if np.any(lower > upper):
            raise RuntimeError(f"Infeasible qdot bounds: lower={lower}, upper={upper}")
        return lower, upper

    def solve(
        self,
        jacobian: np.ndarray,
        task_velocity: np.ndarray,
        q_current: np.ndarray,
        joint_lower: np.ndarray,
        joint_upper: np.ndarray,
        dt: float,
    ) -> OSQPIKResult:
        jacobian = np.asarray(jacobian, dtype=np.float64).reshape(self.task_dimension, self.joint_count)
        task_velocity = np.asarray(task_velocity, dtype=np.float64).reshape(self.task_dimension)
        q_current = np.asarray(q_current, dtype=np.float64).reshape(self.joint_count)
        joint_lower = np.asarray(joint_lower, dtype=np.float64).reshape(self.joint_count)
        joint_upper = np.asarray(joint_upper, dtype=np.float64).reshape(self.joint_count)

        hessian, gradient, effective_damping, min_singular = self._objective(jacobian, task_velocity)
        lower, upper = self._bounds(q_current, joint_lower, joint_upper, dt)
        p_matrix = 2.0 * 0.5 * (hessian + hessian.T)
        q_vector = 2.0 * gradient

        wall_start = time.perf_counter()
        if self._solver is None:
            self._solver = osqp.OSQP()
            self._solver.setup(
                P=_upper_triangle_csc(p_matrix),
                q=q_vector,
                A=sparse.eye(self.joint_count, format="csc"),
                l=lower,
                u=upper,
                verbose=False,
                warm_starting=True,
                polish=False,
                max_iter=self.config.max_iter,
                eps_abs=self.config.eps_abs,
                eps_rel=self.config.eps_rel,
                check_termination=10,
            )
        else:
            self._solver.update(
                Px=_upper_triangle_values(p_matrix),
                q=q_vector,
                l=lower,
                u=upper,
            )
        self._solver.warm_start(x=np.clip(self.prev_qdot, lower, upper))
        result = self._solver.solve()
        wall_time_s = time.perf_counter() - wall_start
        status = str(result.info.status)
        if result.x is None or status.lower() not in {"solved", "solved inaccurate"}:
            raise RuntimeError(f"OSQP IK failed with status={status}")
        qdot = np.asarray(result.x, dtype=np.float64)
        self.prev_qdot = qdot.copy()
        return OSQPIKResult(
            qdot=qdot,
            status=status,
            iterations=int(result.info.iter),
            min_singular=min_singular,
            effective_damping=effective_damping,
            solve_time_s=float(result.info.solve_time),
            wall_time_s=wall_time_s,
        )
