from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_DIR / "src"))

from evaluation.metrics import summarize_metrics
from evaluation.plot_results import plot_action_curves, plot_pinch_distances
from preprocessing.pinch_features import extract_pinch_features
from quest_io.data_loader import load_quest_dual_channel_jsonl
from retargeting.baseline1_direct_mapping import DirectPoseMappingRetargeter, PinchAugmentedDirectRetargeter
from retargeting.baseline2_optimized_retargeting import MuJoCoVectorOptimizedRetargeter
from retargeting.baseline3_contact_aware_retargeting import MuJoCoPinchAwareRetargeter
from sim.aerohand_env import AeroHandSimEnv
from utils.config import load_config


def run_recorded_method(method_name: str, retargeter, frames: list, cfg: dict) -> tuple[list[dict], dict]:
    """Evaluate one retargeter on recorded Quest dual-channel frames."""
    prep_cfg = cfg["preprocessing"]
    sim_env = AeroHandSimEnv(**cfg["sim"])
    sim_env.reset()
    records = []
    for frame in frames:
        keypoints = frame.keypoints()
        pinch = extract_pinch_features(
            keypoints,
            closed_m=prep_cfg.get("pinch_closed_m", 0.025),
            open_m=prep_cfg.get("pinch_open_m", 0.085),
        )
        action = retargeter.retarget(keypoints, pinch, landmarks_wrist=frame.landmarks_wrist)
        obs = sim_env.step(action)
        records.append(_make_record(method_name, frame, pinch, action, obs, retargeter))
    metrics = summarize_metrics(records, cfg["evaluation"].get("pinch_success_threshold_m", 0.03))
    return records, metrics


def run_baseline1(frames: list, cfg: dict) -> tuple[list[dict], dict]:
    """Evaluate pure geometric Baseline 1 on recorded frames."""
    return run_recorded_method("baseline1_direct_recorded", DirectPoseMappingRetargeter(), frames, cfg)


def run_baseline1b(frames: list, cfg: dict) -> tuple[list[dict], dict]:
    """Evaluate Baseline 1b pinch-augmented direct mapping on recorded frames."""
    params = dict(cfg.get("retargeting", {}).get("baseline1b", {}))
    retargeter = PinchAugmentedDirectRetargeter(**params)
    return run_recorded_method("baseline1b_pinch_augmented_recorded", retargeter, frames, cfg)


def run_baseline2(frames: list, cfg: dict) -> tuple[list[dict], dict]:
    """Evaluate Baseline 2 MuJoCo vector-optimized retargeting."""
    sim_env = AeroHandSimEnv(**cfg["sim"])
    params = dict(cfg.get("retargeting", {}).get("baseline2", {}))
    retargeter = MuJoCoVectorOptimizedRetargeter(
        sim_env,
        smooth_weight=cfg["retargeting"]["smooth_weight"],
        limit_weight=cfg["retargeting"]["limit_weight"],
        **params,
    )
    return run_recorded_method_with_env("baseline2_vector_optimized_recorded", retargeter, frames, cfg, sim_env)


def run_baseline3(frames: list, cfg: dict) -> tuple[list[dict], dict]:
    """Evaluate Baseline 3 MuJoCo pinch-aware retargeting."""
    sim_env = AeroHandSimEnv(**cfg["sim"])
    params = dict(cfg.get("retargeting", {}).get("baseline3", {}))
    retargeter = MuJoCoPinchAwareRetargeter(
        sim_env,
        smooth_weight=cfg["retargeting"]["smooth_weight"],
        limit_weight=cfg["retargeting"]["limit_weight"],
        **params,
    )
    return run_recorded_method_with_env("baseline3_pinch_aware_recorded", retargeter, frames, cfg, sim_env)


def run_recorded_method_with_env(method_name: str, retargeter, frames: list, cfg: dict, sim_env: AeroHandSimEnv) -> tuple[list[dict], dict]:
    """Evaluate one retargeter with a caller-owned sim env."""
    prep_cfg = cfg["preprocessing"]
    sim_env.reset()
    records = []
    for frame in frames:
        keypoints = frame.keypoints()
        pinch = extract_pinch_features(
            keypoints,
            closed_m=prep_cfg.get("pinch_closed_m", 0.025),
            open_m=prep_cfg.get("pinch_open_m", 0.085),
        )
        action = retargeter.retarget(keypoints, pinch, landmarks_wrist=frame.landmarks_wrist)
        obs = sim_env.step(action)
        records.append(_make_record(method_name, frame, pinch, action, obs, retargeter))
    metrics = summarize_metrics(records, cfg["evaluation"].get("pinch_success_threshold_m", 0.03))
    return records, metrics


def _make_record(method_name: str, frame, pinch: dict, action: np.ndarray, obs: dict, retargeter) -> dict:
    """Build one evaluation record, including optimizer terms when available."""
    record = {
        "method": method_name,
        "timestamp": float(frame.timestamp),
        "human_pinch_distance": float(pinch["pinch_distance"]),
        "human_pinch_strength": float(pinch["pinch_strength"]),
        "robot_pinch_distance": float(obs["robot_pinch_distance"]),
        "contact": bool(obs["contact"]),
        "action": np.asarray(action, dtype=float).tolist(),
        "latency_s": 0.0,
    }
    if getattr(retargeter, "last_terms", None):
        record["objective_terms"] = {key: float(value) for key, value in retargeter.last_terms.items()}
    return record


def write_action_csv(records: list[dict], output_path: Path, action_names: tuple[str, ...]) -> None:
    """Write per-frame Baseline 1 action and pinch data to CSV."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "timestamp",
                "human_pinch_distance",
                "human_pinch_strength",
                "robot_pinch_distance",
                "contact",
                *action_names,
            ]
        )
        for record in records:
            writer.writerow(
                [
                    record["timestamp"],
                    record["human_pinch_distance"],
                    record["human_pinch_strength"],
                    record["robot_pinch_distance"],
                    int(record["contact"]),
                    *record["action"],
                ]
            )


def main() -> None:
    """Run Baseline 1 on a recorded Quest dual-channel JSONL file."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/teleop_episodes/baseline1_test.jsonl")
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs/default.yaml"))
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "data/processed/baseline1_recorded"))
    parser.add_argument("--include-invalid", action="store_true")
    parser.add_argument("--method", choices=["baseline1", "baseline1b", "both", "baseline2", "baseline3", "all"], default="both")
    parser.add_argument("--max-frames", type=int, default=None, help="Optional cap for slow MuJoCo optimization experiments.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = load_quest_dual_channel_jsonl(args.input, valid_only=not args.include_invalid)
    if args.max_frames is not None:
        frames = frames[: max(0, int(args.max_frames))]
    if not frames:
        raise SystemExit(f"No usable frames loaded from {args.input}")

    action_names = DirectPoseMappingRetargeter.action_names
    runs = []
    if args.method in {"baseline1", "both", "all"}:
        runs.append(("baseline1", *run_baseline1(frames, cfg)))
    if args.method in {"baseline1b", "both", "all"}:
        runs.append(("baseline1b", *run_baseline1b(frames, cfg)))
    if args.method in {"baseline2", "all"}:
        runs.append(("baseline2", *run_baseline2(frames, cfg)))
    if args.method in {"baseline3", "all"}:
        runs.append(("baseline3", *run_baseline3(frames, cfg)))

    summary = {"frames": len(frames), "metrics": {}, "out_dir": str(out_dir)}
    for prefix, records, metrics in runs:
        (out_dir / f"{prefix}_records.json").write_text(json.dumps(records, indent=2), encoding="utf-8")
        (out_dir / f"{prefix}_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
        write_action_csv(records, out_dir / f"{prefix}_actions.csv", action_names)
        plot_pinch_distances(records, str(out_dir / f"{prefix}_pinch_distances.png"))
        plot_action_curves(records, str(out_dir / f"{prefix}_action_curves.png"), action_names)
        summary["metrics"][prefix] = metrics

    if "baseline1" in summary["metrics"] and "baseline1b" in summary["metrics"]:
        comparison = {
            key: summary["metrics"]["baseline1b"][key] - summary["metrics"]["baseline1"][key]
            for key in summary["metrics"]["baseline1"]
            if key in summary["metrics"]["baseline1b"]
        }
        (out_dir / "baseline1_vs_1b_delta.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        summary["baseline1b_minus_baseline1"] = comparison
    if "baseline1" in summary["metrics"] and "baseline2" in summary["metrics"]:
        comparison = {
            key: summary["metrics"]["baseline2"][key] - summary["metrics"]["baseline1"][key]
            for key in summary["metrics"]["baseline1"]
            if key in summary["metrics"]["baseline2"]
        }
        (out_dir / "baseline1_vs_2_delta.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        summary["baseline2_minus_baseline1"] = comparison
    if "baseline2" in summary["metrics"] and "baseline3" in summary["metrics"]:
        comparison = {
            key: summary["metrics"]["baseline3"][key] - summary["metrics"]["baseline2"][key]
            for key in summary["metrics"]["baseline2"]
            if key in summary["metrics"]["baseline3"]
        }
        (out_dir / "baseline2_vs_3_delta.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        summary["baseline3_minus_baseline2"] = comparison
    if "baseline1b" in summary["metrics"] and "baseline3" in summary["metrics"]:
        comparison = {
            key: summary["metrics"]["baseline3"][key] - summary["metrics"]["baseline1b"][key]
            for key in summary["metrics"]["baseline1b"]
            if key in summary["metrics"]["baseline3"]
        }
        (out_dir / "baseline1b_vs_3_delta.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
        summary["baseline3_minus_baseline1b"] = comparison

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
