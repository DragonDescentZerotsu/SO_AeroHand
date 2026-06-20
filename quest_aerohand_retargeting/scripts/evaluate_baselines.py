from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from evaluation.metrics import summarize_metrics
from preprocessing.coordinate_transform import normalize_to_wrist_frame, scale_normalize
from preprocessing.filtering import LowPassFilter
from preprocessing.pinch_features import extract_pinch_features
from quest_io.data_logger import JsonlHandDataLogger
from quest_io.hts_receiver import MockHTSReceiver
from residual.sim_residual_correction import SimResidualCorrector
from retargeting.baseline1_direct_mapping import DirectPoseMappingRetargeter
from retargeting.baseline2_optimized_retargeting import OptimizedVectorRetargeter
from retargeting.baseline3_contact_aware_retargeting import ContactAwareRetargeter
from sim.aerohand_env import AeroHandSimEnv
from utils.config import load_config


def run_method(name: str, retargeter, frames: list, cfg: dict, residual=None) -> tuple[list[dict], dict]:
    """Run one baseline over frames and return raw records plus summary metrics."""
    prep_cfg = cfg["preprocessing"]
    sim_env = AeroHandSimEnv(**cfg["sim"])
    sim_env.reset()
    filt = LowPassFilter(alpha=prep_cfg.get("low_pass_alpha", 0.25))
    records = []
    for frame in frames:
        keypoints = normalize_to_wrist_frame(frame.keypoints(), frame.wrist_pose)
        keypoints = scale_normalize(keypoints, reference_m=prep_cfg.get("scale_reference_m", 0.10))
        keypoints = filt.apply(keypoints)
        pinch = extract_pinch_features(
            keypoints,
            closed_m=prep_cfg.get("pinch_closed_m", 0.025),
            open_m=prep_cfg.get("pinch_open_m", 0.085),
        )
        action = retargeter.retarget(keypoints, pinch, landmarks_wrist=frame.landmarks_wrist)
        obs = sim_env.step(action)
        if residual is not None:
            delta = residual.predict({"pinch": pinch, "sim": obs}, action)
            action = np.clip(action + delta, 0.0, 1.0)
            obs = sim_env.step(action)
        records.append(
            {
                "method": name,
                "timestamp": frame.timestamp,
                "human_pinch_distance": float(pinch["pinch_distance"]),
                "robot_pinch_distance": float(obs["robot_pinch_distance"]),
                "contact": bool(obs["contact"]),
                "action": np.asarray(action, dtype=float).tolist(),
                "latency_s": 0.0,
            }
        )
    metrics = summarize_metrics(records, cfg["evaluation"].get("pinch_success_threshold_m", 0.03))
    return records, metrics


def main() -> None:
    """Run the mock pinch trajectory through all placeholder methods."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs/default.yaml"))
    args = parser.parse_args()
    cfg = load_config(args.config)
    demo_cfg = cfg["demo"]
    output_dir = Path(demo_cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    frames = list(MockHTSReceiver(num_frames=demo_cfg["num_frames"], dt=demo_cfg["dt"]).iter_frames())
    JsonlHandDataLogger(output_dir / "mock_quest_hand.jsonl").write_frames(frames)

    methods = {
        "baseline1_direct": DirectPoseMappingRetargeter(),
        "baseline2_optimized_placeholder": OptimizedVectorRetargeter(
            smooth_weight=cfg["retargeting"]["smooth_weight"],
            limit_weight=cfg["retargeting"]["limit_weight"],
        ),
        "baseline3_contact_aware_placeholder": ContactAwareRetargeter(
            lambda_pinch=cfg["retargeting"]["lambda_pinch"],
            smooth_weight=cfg["retargeting"]["smooth_weight"],
            limit_weight=cfg["retargeting"]["limit_weight"],
        ),
    }

    all_metrics = {}
    all_records = []
    for name, method in methods.items():
        records, metrics = run_method(name, method, frames, cfg)
        all_records.extend(records)
        all_metrics[name] = metrics

    proposed_records, proposed_metrics = run_method(
        "proposed_zero_residual",
        ContactAwareRetargeter(lambda_pinch=cfg["retargeting"]["lambda_pinch"]),
        frames,
        cfg,
        residual=SimResidualCorrector(),
    )
    all_records.extend(proposed_records)
    all_metrics["proposed_zero_residual"] = proposed_metrics

    (output_dir / "mock_baseline_records.json").write_text(json.dumps(all_records, indent=2), encoding="utf-8")
    (output_dir / "mock_metrics.json").write_text(json.dumps(all_metrics, indent=2), encoding="utf-8")
    print(json.dumps(all_metrics, indent=2))


if __name__ == "__main__":
    main()
