from __future__ import annotations

import numpy as np

from .baseline2_optimized_retargeting import MuJoCoVectorOptimizedRetargeter, human_pairwise_shape, robot_pairwise_shape


class MuJoCoPinchAwareRetargeter(MuJoCoVectorOptimizedRetargeter):
    """Baseline 3: MuJoCo vector retargeting with pinch-aware loss.

    Baseline 2 optimizes overall five-fingertip shape. That can improve hand
    shape while still missing the thumb-index pinch. This retargeter keeps the
    same derivative-free MuJoCo candidate evaluation, then adds:

    ``lambda_pinch * (robot_pinch_distance - human_pinch_distance)^2``

    and, when the human hand is pinching, an extra close-distance term that
    directly encourages AeroHand thumb/index closure.
    """

    def __init__(
        self,
        sim_env,
        smooth_weight: float = 0.05,
        limit_weight: float = 0.01,
        lambda_pinch: float = 25.0,
        lambda_close: float = 10.0,
        pinch_strength_threshold: float = 0.55,
        contact_distance_m: float = 0.03,
        num_candidates: int = 40,
        sample_radius: float = 0.24,
        local_radius: float = 0.10,
        seed: int = 11,
    ):
        super().__init__(
            sim_env,
            smooth_weight=smooth_weight,
            limit_weight=limit_weight,
            num_candidates=num_candidates,
            sample_radius=sample_radius,
            local_radius=local_radius,
            seed=seed,
        )
        self.lambda_pinch = float(lambda_pinch)
        self.lambda_close = float(lambda_close)
        self.pinch_strength_threshold = float(pinch_strength_threshold)
        self.contact_distance_m = float(contact_distance_m)

    def retarget(
        self,
        keypoints: dict[str, np.ndarray],
        pinch: dict,
        landmarks_wrist: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the lowest-loss sampled pinch-aware action for one frame."""
        seed_action = self.seed_retargeter.retarget(keypoints, pinch, landmarks_wrist=landmarks_wrist)
        target_shape = human_pairwise_shape(keypoints)
        human_pinch_distance = float(pinch["pinch_distance"])
        pinch_strength = float(pinch["pinch_strength"])
        candidates = self._candidate_actions(seed_action, pinch_strength)
        best_action = seed_action
        best_terms = self.objective_terms(seed_action, target_shape, human_pinch_distance, pinch_strength)
        best_loss = sum(best_terms.values())
        for action in candidates:
            terms = self.objective_terms(action, target_shape, human_pinch_distance, pinch_strength)
            loss = sum(terms.values())
            if loss < best_loss:
                best_loss = loss
                best_action = action
                best_terms = terms
        self.previous_action = best_action.copy()
        self.last_terms = best_terms
        return best_action

    def objective_terms(
        self,
        action: np.ndarray,
        target_shape: np.ndarray,
        human_pinch_distance: float,
        pinch_strength: float,
    ) -> dict[str, float]:
        """Evaluate vector, pinch, close, smooth, and limit losses."""
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(7), 0.0, 1.0)
        obs = self.sim_env.evaluate_action(action)
        robot_shape = robot_pairwise_shape(obs["fingertips"])
        robot_pinch_distance = float(obs["robot_pinch_distance"])
        vector_loss = float(np.mean((robot_shape - target_shape) ** 2))
        pinch_loss = float((robot_pinch_distance - float(human_pinch_distance)) ** 2)
        close_gate = max(0.0, (float(pinch_strength) - self.pinch_strength_threshold) / max(1.0 - self.pinch_strength_threshold, 1e-8))
        close_loss = float(close_gate * robot_pinch_distance**2)
        previous = action if self.previous_action is None else self.previous_action
        smooth_loss = float(np.linalg.norm(action - previous) ** 2)
        limit_loss = float(np.mean((np.maximum(0.05 - action, 0.0) + np.maximum(action - 0.95, 0.0)) ** 2))
        return {
            "L_vector": vector_loss,
            "L_pinch": self.lambda_pinch * pinch_loss,
            "L_close": self.lambda_close * close_loss,
            "L_smooth": self.smooth_weight * smooth_loss,
            "L_limit": self.limit_weight * limit_loss,
        }

    def _candidate_actions(self, seed_action: np.ndarray, pinch_strength: float = 0.0) -> list[np.ndarray]:
        """Sample candidates plus explicit thumb-index pinch candidates."""
        seed_action = np.clip(np.asarray(seed_action, dtype=np.float64).reshape(7), 0.0, 1.0)
        strength = float(np.clip(pinch_strength, 0.0, 1.0))
        previous = seed_action if self.previous_action is None else self.previous_action
        thumb_index_bias = np.array([0.20, 0.45, 0.55, 0.80, 0.0, 0.0, 0.0], dtype=np.float64) * strength
        strong_pinch = seed_action.copy()
        strong_pinch[:4] = np.maximum(strong_pinch[:4], np.array([0.35, 0.70, 0.85, 0.95]) * strength)
        full_pinch = seed_action.copy()
        full_pinch[:4] = np.maximum(full_pinch[:4], np.array([0.55, 0.90, 1.00, 1.00]) * strength)
        candidates = [
            seed_action,
            previous,
            np.clip(0.5 * seed_action + 0.5 * previous, 0.0, 1.0),
            np.clip(seed_action + thumb_index_bias, 0.0, 1.0),
            np.clip(0.5 * previous + 0.5 * strong_pinch, 0.0, 1.0),
            np.clip(strong_pinch, 0.0, 1.0),
            np.clip(full_pinch, 0.0, 1.0),
        ]
        centers = [seed_action, previous, strong_pinch, full_pinch]
        while len(candidates) < self.num_candidates:
            center = centers[len(candidates) % len(centers)]
            radius = self.local_radius if center is previous else self.sample_radius
            noise = self.rng.normal(0.0, radius, size=7)
            candidates.append(np.clip(center + noise, 0.0, 1.0))
        return candidates


class ContactAwareRetargeter(MuJoCoPinchAwareRetargeter):
    """Backward-compatible name for Baseline 3."""
