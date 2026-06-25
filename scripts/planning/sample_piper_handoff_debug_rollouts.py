"""Sample Piper pipette handoff debug rollouts, keeping successes and failures.

This script is for inspection, not training export. It samples episode specs,
runs the single-episode planner with ``--allow-failed-grasp`` and
``--record-video``, then keeps the raw MuJoCo trajectory, summary and video for
every rollout that reaches dynamics playback. Attempts that fail before rollout
are logged and skipped.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.task_sampling import (  # noqa: E402
    RackBarSampleConfig,
    sample_pipette_rack_bar_episode,
    write_episode_spec,
)

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {exc}") from exc


DEFAULT_MODEL = PROJECT_ROOT / "models/piper_aero_hand/scenes/Piper_dual_pipette_rack_table.xml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/debug_rollouts/piper_pipette_handoff"
DEFAULT_PLANNER = PROJECT_ROOT / "scripts/planning/plan_piper_gripper_pipette_handoff.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--planner-script", default=str(DEFAULT_PLANNER))
    parser.add_argument("--num-rollouts", type=int, default=6)
    parser.add_argument("--seed-start", type=int, default=101)
    parser.add_argument("--max-attempts", type=int, default=60)
    parser.add_argument("--rack-bar-offset-min-m", type=float, default=-0.1275)
    parser.add_argument("--rack-bar-offset-max-m", type=float, default=0.1275)
    parser.add_argument(
        "--fixed-rack-pose",
        action="store_true",
        help="Keep rack pose fixed and only slide pipette along the rack bar.",
    )
    parser.add_argument("--rack-x-min-m", type=float, default=-0.04)
    parser.add_argument("--rack-x-max-m", type=float, default=0.04)
    parser.add_argument("--rack-y-min-m", type=float, default=-0.03)
    parser.add_argument("--rack-y-max-m", type=float, default=0.03)
    parser.add_argument("--rack-yaw-min-deg", type=float, default=-8.0)
    parser.add_argument("--rack-yaw-max-deg", type=float, default=8.0)
    parser.add_argument("--video-stride", type=int, default=5)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_summary(path: Path) -> dict[str, object] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    args = parse_args()
    model_path = Path(args.model).resolve()
    planner_script = Path(args.planner_script).resolve()
    output_root = Path(args.output_root).resolve()
    run_name = args.run_name or time.strftime("debug_random_%Y%m%d_%H%M%S")
    run_dir = output_root / run_name
    if run_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"{run_dir} already exists; pass --overwrite")
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    rng = np.random.default_rng(args.seed_start)
    sampler_config = RackBarSampleConfig(
        offset_range_m=(args.rack_bar_offset_min_m, args.rack_bar_offset_max_m),
        sample_rack_pose=not args.fixed_rack_pose,
        rack_x_range_m=(args.rack_x_min_m, args.rack_x_max_m),
        rack_y_range_m=(args.rack_y_min_m, args.rack_y_max_m),
        rack_yaw_range_deg=(args.rack_yaw_min_deg, args.rack_yaw_max_deg),
    )

    rollouts: list[dict[str, object]] = []
    skipped_attempts: list[dict[str, object]] = []
    attempt_index = 0
    while len(rollouts) < args.num_rollouts and attempt_index < args.max_attempts:
        seed = args.seed_start + attempt_index
        rollout_index = len(rollouts)
        attempt_dir = run_dir / f"attempt_{attempt_index:06d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        spec = sample_pipette_rack_bar_episode(
            model,
            rng=rng,
            seed=seed,
            config=sampler_config,
        )
        spec_path = attempt_dir / "episode_spec.json"
        write_episode_spec(spec_path, spec)

        cmd = [
            sys.executable,
            str(planner_script),
            "--model",
            str(model_path),
            "--output-dir",
            str(attempt_dir),
            "--seed",
            str(seed),
            "--episode-spec",
            str(spec_path),
            "--allow-failed-grasp",
            "--record-video",
            "--video-stride",
            str(args.video_stride),
        ]
        print(
            f"[attempt {attempt_index:06d}] seed={seed} "
            f"offset={spec.metadata.get('offset_m')} -> rollout candidate {rollout_index:06d}",
            flush=True,
        )
        started = time.perf_counter()
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
        )
        elapsed_s = time.perf_counter() - started
        log_path = attempt_dir / "planner.log"
        log_path.write_text(result.stdout, encoding="utf-8")

        npz_path = attempt_dir / "piper_gripper_pipette_handoff_expert.npz"
        summary_path = attempt_dir / "summary.json"
        video_path = attempt_dir / "piper_gripper_pipette_handoff_expert.mp4"
        summary = load_summary(summary_path)
        has_rollout = npz_path.exists() and summary is not None
        if not has_rollout:
            skipped_attempts.append(
                {
                    "attempt_index": attempt_index,
                    "seed": seed,
                    "returncode": result.returncode,
                    "elapsed_s": elapsed_s,
                    "attempt_dir": str(attempt_dir),
                    "episode_spec": spec.as_json(),
                    "log": str(log_path),
                    "reason": "planner_failed_before_rollout",
                }
            )
            print(f"[attempt {attempt_index:06d}] skipped before rollout", flush=True)
            attempt_index += 1
            continue

        dynamics = dict(summary.get("dynamics", {}))
        task_success = bool(summary.get("task_success", False))
        record = {
            "rollout_index": rollout_index,
            "attempt_index": attempt_index,
            "seed": seed,
            "returncode": result.returncode,
            "elapsed_s": elapsed_s,
            "task_success": task_success,
            "dynamic_handoff_success": bool(dynamics.get("dynamic_handoff_success", False)),
            "hook_handoff_reached": bool(dynamics.get("hook_handoff_reached", False)),
            "release_survived": bool(dynamics.get("release_survived", False)),
            "palm_grasp_stable": bool(dynamics.get("palm_grasp_stable", False)),
            "offset_m": spec.metadata.get("offset_m"),
            "rack_delta_xy_m": spec.metadata.get("rack_delta_xy_m"),
            "rack_yaw_deg": spec.metadata.get("rack_yaw_deg"),
            "attempt_dir": str(attempt_dir),
            "trajectory": str(npz_path),
            "summary": str(summary_path),
            "video": str(video_path) if video_path.exists() else None,
            "episode_spec": spec.as_json(),
        }
        rollouts.append(record)
        print(
            f"[rollout {rollout_index:06d}] success={task_success} "
            f"dynamic={record['dynamic_handoff_success']} "
            f"hook={record['hook_handoff_reached']} "
            f"release={record['release_survived']} "
            f"palm={record['palm_grasp_stable']}",
            flush=True,
        )
        attempt_index += 1

    manifest = {
        "model": str(model_path),
        "planner_script": str(planner_script),
        "run_dir": str(run_dir),
        "num_requested_rollouts": args.num_rollouts,
        "num_rollouts": len(rollouts),
        "attempts_completed": attempt_index,
        "skipped_attempts_count": len(skipped_attempts),
        "sampler": {
            "rack_bar_offset_range_m": [
                args.rack_bar_offset_min_m,
                args.rack_bar_offset_max_m,
            ],
            "sample_rack_pose": not args.fixed_rack_pose,
            "rack_x_range_m": [args.rack_x_min_m, args.rack_x_max_m],
            "rack_y_range_m": [args.rack_y_min_m, args.rack_y_max_m],
            "rack_yaw_range_deg": [args.rack_yaw_min_deg, args.rack_yaw_max_deg],
        },
        "rollouts": rollouts,
        "skipped_attempts": skipped_attempts,
    }
    manifest_path = run_dir / "debug_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote debug manifest: {manifest_path}")
    if len(rollouts) < args.num_rollouts:
        raise RuntimeError(
            f"Only collected {len(rollouts)} rollouts after {attempt_index} attempts; "
            f"requested {args.num_rollouts}."
        )


if __name__ == "__main__":
    main()
