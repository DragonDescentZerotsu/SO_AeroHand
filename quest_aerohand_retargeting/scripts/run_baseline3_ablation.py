from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).resolve().parents[1]
REPO_ROOT = PROJECT_DIR.parent
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(PROJECT_DIR / "src"))
sys.path.insert(0, str(PROJECT_DIR / "scripts"))

from evaluate_recorded_baseline1 import run_baseline3
from quest_io.data_loader import load_quest_dual_channel_jsonl
from utils.config import load_config


VARIANTS = {
    "A_default": {"lambda_pinch": 25.0, "lambda_close": 10.0},
    "B_closure": {"lambda_pinch": 35.0, "lambda_close": 15.0},
    "C_natural": {"lambda_pinch": 20.0, "lambda_close": 5.0},
}


def main() -> None:
    """Run small Baseline 3 weight ablations on recorded Quest data."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/teleop_episodes/baseline1_test.jsonl")
    parser.add_argument("--config", default=str(PROJECT_DIR / "configs/default.yaml"))
    parser.add_argument("--max-frames", type=int, default=300)
    parser.add_argument("--out", default=str(PROJECT_DIR / "data/processed/baseline3_ablation.json"))
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    frames = load_quest_dual_channel_jsonl(args.input)
    frames = frames[: max(0, int(args.max_frames))]
    if not frames:
        raise SystemExit(f"No usable frames loaded from {args.input}")

    summary = {"frames": len(frames), "variants": {}}
    for name, params in VARIANTS.items():
        cfg = copy.deepcopy(base_cfg)
        cfg["retargeting"]["baseline3"].update(params)
        _, metrics = run_baseline3(frames, cfg)
        summary["variants"][name] = {
            "lambda_pinch": params["lambda_pinch"],
            "lambda_close": params["lambda_close"],
            "metrics": metrics,
        }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
