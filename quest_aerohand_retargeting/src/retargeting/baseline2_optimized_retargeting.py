from __future__ import annotations

import numpy as np

from .baseline1_direct_mapping import DirectPoseMappingRetargeter


class OptimizedVectorRetargeter:
    """Baseline 2: placeholder for optimized fingertip-vector retargeting."""

    def __init__(self, smooth_weight: float = 0.05, limit_weight: float = 0.01):
        self.smooth_weight = float(smooth_weight)
        self.limit_weight = float(limit_weight)
        self.previous_action = np.zeros(7, dtype=np.float64)

    def objective_terms(self, action: np.ndarray, target_vectors: dict[str, np.ndarray]) -> dict[str, float]:
        """Return placeholder objective terms for ``L_vector + L_smooth + L_limit``."""
        action = np.asarray(action, dtype=np.float64).reshape(7)
        vector_loss = float(sum(np.linalg.norm(v) for v in target_vectors.values()) / max(len(target_vectors), 1))
        smooth_loss = float(np.linalg.norm(action - self.previous_action) ** 2)
        limit_loss = float(np.sum(np.maximum(action - 1.0, 0.0) ** 2 + np.maximum(-action, 0.0) ** 2))
        return {
            "L_vector": vector_loss,
            "L_smooth": self.smooth_weight * smooth_loss,
            "L_limit": self.limit_weight * limit_loss,
        }

    def retarget(
        self,
        keypoints: dict[str, np.ndarray],
        pinch: dict,
        landmarks_wrist: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return a deterministic placeholder action and update smooth state.

        TODO: Replace with an optimizer over AeroHand controls once fingertip
        site names and differentiable/finite-difference simulation hooks are set.
        """
        del landmarks_wrist
        target_vectors = {
            "thumb_index": np.asarray(keypoints["index_tip"]) - np.asarray(keypoints["thumb_tip"]),
            "index_middle": np.asarray(keypoints["middle_tip"]) - np.asarray(keypoints["index_tip"]),
        }
        strength = float(pinch["pinch_strength"])
        raw = np.array([0.5 * strength, 0.6 * strength, 0.7 * strength, strength, 0.2, 0.2, 0.2])
        self.objective_terms(raw, target_vectors)
        action = np.clip(0.75 * self.previous_action + 0.25 * raw, 0.0, 1.0)
        self.previous_action = action.copy()
        return action


class MuJoCoVectorOptimizedRetargeter:
    """Baseline 2: optimize 7D action against MuJoCo fingertip vectors.

    This is a lightweight derivative-free optimizer. It samples candidate
    actions around the direct-mapping seed and previous action, evaluates each
    candidate through ``AeroHandSimEnv``, and minimizes:

    ``L_vector + L_smooth + L_limit``.
    """

    tip_names = ("thumb", "index", "middle", "ring", "little")
    pair_indices = tuple((i, j) for i in range(5) for j in range(i + 1, 5))

    def __init__(
        self,
        sim_env,
        smooth_weight: float = 0.05,
        limit_weight: float = 0.01,
        num_candidates: int = 28,
        sample_radius: float = 0.22,
        local_radius: float = 0.10,
        seed: int = 7,
    ):
        self.sim_env = sim_env
        self.smooth_weight = float(smooth_weight)
        self.limit_weight = float(limit_weight)
        self.num_candidates = max(4, int(num_candidates))
        self.sample_radius = float(sample_radius)
        self.local_radius = float(local_radius)
        self.rng = np.random.default_rng(int(seed))
        self.seed_retargeter = DirectPoseMappingRetargeter()
        self.previous_action: np.ndarray | None = None
        self.last_terms: dict[str, float] = {}

    def retarget(
        self,
        keypoints: dict[str, np.ndarray],
        pinch: dict,
        landmarks_wrist: np.ndarray | None = None,
    ) -> np.ndarray:
        """Return the best sampled action for one hand frame."""
        seed_action = self.seed_retargeter.retarget(keypoints, pinch, landmarks_wrist=landmarks_wrist)
        target_shape = human_pairwise_shape(keypoints)
        candidates = self._candidate_actions(seed_action)
        best_action = seed_action
        best_terms = self.objective_terms(seed_action, target_shape)
        best_loss = sum(best_terms.values())
        for action in candidates:
            terms = self.objective_terms(action, target_shape)
            loss = sum(terms.values())
            if loss < best_loss:
                best_loss = loss
                best_action = action
                best_terms = terms
        self.previous_action = best_action.copy()
        self.last_terms = best_terms
        return best_action

    def objective_terms(self, action: np.ndarray, target_shape: np.ndarray) -> dict[str, float]:
        """Evaluate ``L_vector + L_smooth + L_limit`` for an action."""
        action = np.clip(np.asarray(action, dtype=np.float64).reshape(7), 0.0, 1.0)
        obs = self.sim_env.evaluate_action(action)
        robot_shape = robot_pairwise_shape(obs["fingertips"])
        vector_loss = float(np.mean((robot_shape - target_shape) ** 2))
        previous = action if self.previous_action is None else self.previous_action
        smooth_loss = float(np.linalg.norm(action - previous) ** 2)
        limit_loss = float(np.mean((np.maximum(0.05 - action, 0.0) + np.maximum(action - 0.95, 0.0)) ** 2))
        return {
            "L_vector": vector_loss,
            "L_smooth": self.smooth_weight * smooth_loss,
            "L_limit": self.limit_weight * limit_loss,
        }

    def _candidate_actions(self, seed_action: np.ndarray) -> list[np.ndarray]:
        """Sample bounded candidates around direct seed and previous action."""
        seed_action = np.clip(np.asarray(seed_action, dtype=np.float64).reshape(7), 0.0, 1.0)
        previous = seed_action if self.previous_action is None else self.previous_action
        candidates = [
            seed_action,
            previous,
            np.clip(0.5 * seed_action + 0.5 * previous, 0.0, 1.0),
        ]
        centers = [seed_action, previous]
        while len(candidates) < self.num_candidates:
            center = centers[len(candidates) % len(centers)]
            radius = self.local_radius if center is previous else self.sample_radius
            noise = self.rng.normal(0.0, radius, size=7)
            candidates.append(np.clip(center + noise, 0.0, 1.0))
        return candidates


def human_pairwise_shape(keypoints: dict[str, np.ndarray]) -> np.ndarray:
    """Return normalized human fingertip pairwise distances."""
    points = np.asarray(
        [
            keypoints["thumb_tip"],
            keypoints["index_tip"],
            keypoints["middle_tip"],
            keypoints["ring_tip"],
            keypoints["pinky_tip"],
        ],
        dtype=np.float64,
    )
    return _normalized_pairwise_distances(points)


def robot_pairwise_shape(fingertips: dict[str, np.ndarray]) -> np.ndarray:
    """Return normalized robot fingertip pairwise distances."""
    points = np.asarray([fingertips[name] for name in MuJoCoVectorOptimizedRetargeter.tip_names], dtype=np.float64)
    return _normalized_pairwise_distances(points)


def _normalized_pairwise_distances(points: np.ndarray) -> np.ndarray:
    """Return pairwise distances normalized by their mean."""
    points = np.asarray(points, dtype=np.float64).reshape(5, 3)
    distances = np.asarray(
        [np.linalg.norm(points[i] - points[j]) for i, j in MuJoCoVectorOptimizedRetargeter.pair_indices],
        dtype=np.float64,
    )
    scale = max(float(np.mean(distances)), 1e-8)
    return distances / scale
