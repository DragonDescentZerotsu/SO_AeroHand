from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from preprocessing.pinch_features import extract_pinch_features
from quest_io.data_loader import load_quest_dual_channel_jsonl
from retargeting.baseline1_direct_mapping import DirectPoseMappingRetargeter, PinchAugmentedDirectRetargeter
from retargeting.baseline2_optimized_retargeting import MuJoCoVectorOptimizedRetargeter
from retargeting.baseline3_contact_aware_retargeting import MuJoCoPinchAwareRetargeter
from sim.aerohand_env import AeroHandSimEnv
from utils.config import load_config
from visualization.human_skeleton_viewer import save_skeleton_comparison_grid


def build_retargeter(method: str, cfg: dict, sim_env: AeroHandSimEnv):
    """Create the selected retargeter."""
    if method == "baseline1":
        return DirectPoseMappingRetargeter()
    if method == "baseline1b":
        return PinchAugmentedDirectRetargeter(**dict(cfg.get("retargeting", {}).get("baseline1b", {})))
    if method == "baseline2":
        params = dict(cfg.get("retargeting", {}).get("baseline2", {}))
        return MuJoCoVectorOptimizedRetargeter(
            sim_env,
            smooth_weight=cfg["retargeting"]["smooth_weight"],
            limit_weight=cfg["retargeting"]["limit_weight"],
            **params,
        )
    if method == "baseline3":
        params = dict(cfg.get("retargeting", {}).get("baseline3", {}))
        return MuJoCoPinchAwareRetargeter(
            sim_env,
            smooth_weight=cfg["retargeting"]["smooth_weight"],
            limit_weight=cfg["retargeting"]["limit_weight"],
            **params,
        )
    raise ValueError(f"Unsupported method: {method}")


def choose_frame_indices(total_frames: int, num_frames: int, explicit: str | None) -> set[int]:
    """Return selected frame indices."""
    if total_frames <= 0:
        return set()
    if explicit:
        indices = {int(value.strip()) for value in explicit.split(",") if value.strip()}
        return {idx for idx in indices if 0 <= idx < total_frames}
    count = max(1, min(int(num_frames), total_frames))
    return {int(round(value)) for value in np.linspace(0, total_frames - 1, count)}


def collect_skeleton_samples(
    frames: list,
    frame_indices: set[int],
    retargeter,
    sim_env: AeroHandSimEnv,
    cfg: dict,
) -> list[dict]:
    """Replay frames and collect Quest/MuJoCo hand skeleton samples."""
    prep_cfg = cfg["preprocessing"]
    samples = []
    sim_env.reset()
    max_index = max(frame_indices)
    for index, frame in enumerate(frames[: max_index + 1]):
        keypoints = frame.keypoints()
        pinch = extract_pinch_features(
            keypoints,
            closed_m=prep_cfg.get("pinch_closed_m", 0.025),
            open_m=prep_cfg.get("pinch_open_m", 0.085),
        )
        action = retargeter.retarget(keypoints, pinch, landmarks_wrist=frame.landmarks_wrist)
        obs = sim_env.step(action)
        if index not in frame_indices:
            continue
        if frame.landmarks_wrist is None:
            raise ValueError(f"Frame {index} has no Quest 21-point landmarks_wrist")
        samples.append(
            {
                "frame_index": index,
                "quest_landmarks_wrist": np.asarray(frame.landmarks_wrist, dtype=np.float64).reshape(21, 3),
                "robot_landmarks_world": np.asarray(obs["hand_landmarks"], dtype=np.float64).reshape(21, 3),
                "human_pinch_distance": float(pinch["pinch_distance"]),
                "robot_pinch_distance": float(obs["robot_pinch_distance"]),
            }
        )
    return samples


def main() -> None:
    """Save Quest 3 and MuJoCo AeroHand skeleton comparison plots."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/teleop_episodes/baseline1_test.jsonl")
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs/default.yaml"))
    parser.add_argument("--out", default=str(PROJECT_DIR / "data/processed/skeleton_compare/baseline1b_skeletons.png"))
    parser.add_argument("--method", choices=["baseline1", "baseline1b", "baseline2", "baseline3"], default="baseline1b")
    parser.add_argument("--num-frames", type=int, default=6)
    parser.add_argument("--frame-indices", default=None, help="Comma-separated frame indices, e.g. 0,500,1000.")
    parser.add_argument("--include-invalid", action="store_true")
    args = parser.parse_args()

    cfg = load_config(args.config)
    frames = load_quest_dual_channel_jsonl(args.input, valid_only=not args.include_invalid)
    if not frames:
        raise SystemExit(f"No usable frames loaded from {args.input}")
    frame_indices = choose_frame_indices(len(frames), args.num_frames, args.frame_indices)
    if not frame_indices:
        raise SystemExit("No valid frame indices selected")

    sim_env = AeroHandSimEnv(**cfg["sim"])
    retargeter = build_retargeter(args.method, cfg, sim_env)
    samples = collect_skeleton_samples(frames, frame_indices, retargeter, sim_env, cfg)
    save_skeleton_comparison_grid(samples, args.out)
    print(
        {
            "method": args.method,
            "frames_loaded": len(frames),
            "frames_plotted": sorted(frame_indices),
            "output": str(Path(args.out)),
        }
    )


if __name__ == "__main__":
    main()
