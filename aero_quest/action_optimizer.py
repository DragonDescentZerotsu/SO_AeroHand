"""Offline MuJoCo geometry optimizer for Aero Hand pseudo-label actions."""

from __future__ import annotations

import numpy as np

try:
    import mujoco
except ImportError:
    mujoco = None

try:
    from scipy.optimize import minimize
except ImportError:
    minimize = None

from aero_quest.closure_loss import closure_matching_loss
from aero_quest.mujoco_landmarks import get_robot_landmarks_21
from aero_quest.retargeting import palm_localize


class ActionOptimizer:
    """Optimize semantic 7D Aero action against MuJoCo closure features."""

    def __init__(
        self,
        model,
        data,
        apply_action_fn,
        get_landmarks_fn=None,
        w_closure=1.0,
        w_prior=1.0,
        w_smooth=0.2,
        w_bend=5.0,
        w_tip=0.5,
        w_dir=1.0,
        maxiter=30,
        settle_steps=30,
        finite_diff_eps=1e-3,
    ):
        if mujoco is None:
            raise RuntimeError("mujoco is required. Install it with: pip install mujoco")
        if minimize is None:
            raise RuntimeError("scipy is required. Install it with: pip install scipy")
        self.model = model
        self.data = data
        self.apply_action_fn = apply_action_fn
        self.get_landmarks_fn = get_landmarks_fn or get_robot_landmarks_21
        self.w_closure = float(w_closure)
        self.w_prior = float(w_prior)
        self.w_smooth = float(w_smooth)
        self.w_bend = float(w_bend)
        self.w_tip = float(w_tip)
        self.w_dir = float(w_dir)
        self.maxiter = int(maxiter)
        self.settle_steps = max(1, int(settle_steps))
        self.finite_diff_eps = float(finite_diff_eps)
        self.opt_data = mujoco.MjData(model)

    def _reset_opt_data(self) -> None:
        """Initialize the private optimization data from the shared simulation state."""
        self.opt_data.qpos[:] = self.data.qpos[:]
        self.opt_data.qvel[:] = 0.0
        self.opt_data.ctrl[:] = self.data.ctrl[:]
        if self.opt_data.act is not None and self.opt_data.act.size > 0:
            self.opt_data.act[:] = 0.0

    def _settle_opt_data(self) -> None:
        """Advance the private MuJoCo state so ctrl changes affect hand geometry."""
        for _ in range(self.settle_steps):
            mujoco.mj_step(self.model, self.opt_data)

    def loss(self, a, human_local, a_formula, a_prev=None) -> dict:
        """Evaluate optimizer loss for one candidate semantic 7D action."""
        a = np.clip(np.asarray(a, dtype=np.float64), 0.0, 1.0)
        a_formula = np.clip(np.asarray(a_formula, dtype=np.float64), 0.0, 1.0)
        self._reset_opt_data()
        self.apply_action_fn(self.model, self.opt_data, a)
        self._settle_opt_data()
        robot_world = self.get_landmarks_fn(self.model, self.opt_data)
        robot_local = palm_localize(robot_world).astype(np.float64)
        closure = closure_matching_loss(
            human_local,
            robot_local,
            w_bend=self.w_bend,
            w_tip=self.w_tip,
            w_dir=self.w_dir,
        )
        prior = float(np.mean((a - a_formula) ** 2))
        if a_prev is None:
            smooth = 0.0
        else:
            a_prev = np.clip(np.asarray(a_prev, dtype=np.float64), 0.0, 1.0)
            smooth = float(np.mean((a - a_prev) ** 2))
        total = float(self.w_closure * closure["total"] + self.w_prior * prior + self.w_smooth * smooth)
        return {
            "total": float(total if np.isfinite(total) else 0.0),
            "closure": float(closure["total"]),
            "bend": float(closure["bend"]),
            "tip": float(closure["tip"]),
            "direction": float(closure["direction"]),
            "prior": float(prior if np.isfinite(prior) else 0.0),
            "smooth": float(smooth if np.isfinite(smooth) else 0.0),
        }

    def optimize(self, human_landmarks, a_formula, a_prev=None) -> dict:
        """Optimize action for one human 21-landmark frame."""
        human_local = palm_localize(human_landmarks).astype(np.float64)
        a_formula = np.clip(np.asarray(a_formula, dtype=np.float64), 0.0, 1.0)
        loss_before = self.loss(a_formula, human_local, a_formula, a_prev)

        def objective(x):
            return self.loss(x, human_local, a_formula, a_prev)["total"]

        result = minimize(
            objective,
            x0=a_formula,
            method="L-BFGS-B",
            bounds=[(0.0, 1.0)] * 7,
            options={"maxiter": self.maxiter, "eps": self.finite_diff_eps},
        )
        success = bool(result.success)
        if success:
            a_opt = np.clip(result.x.astype(np.float64), 0.0, 1.0)
        else:
            a_opt = a_formula.copy()
        loss_after = self.loss(a_opt, human_local, a_formula, a_prev)
        delta = a_opt - a_formula
        return {
            "success": success,
            "a_opt": a_opt.astype(np.float32),
            "a_formula": a_formula.astype(np.float32),
            "delta": delta.astype(np.float32),
            "loss_before": loss_before,
            "loss_after": loss_after,
            "message": str(result.message),
            "nfev": int(getattr(result, "nfev", 0)),
        }
