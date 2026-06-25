"""Generate Piper pipette handoff expert episodes in LeRobot format.

This is intentionally a thin batch/export layer around the single-episode
planner. Future tasks should keep the same shape: task planner produces MuJoCo
qpos/ctrl/summary, this layer renders cameras and writes LeRobot episodes.
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

from aero_tasks.lerobot_export import (  # noqa: E402
    DEFAULT_HANDOFF_CAMERAS,
    MujocoTrajectoryRenderer,
    lerobot_features,
    resolve_dataset_root,
    sample_indices,
    stage_index_map,
)

try:
    import mujoco
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {exc}") from exc

try:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing LeRobot dataset writer: {exc}") from exc


DEFAULT_MODEL = PROJECT_ROOT / "models/piper_aero_hand/scenes/Piper_dual_pipette_rack_table.xml"
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "outputs/lerobot"
DEFAULT_TASK_NAME = "piper_pipette_handoff"
DEFAULT_TASK_PROMPT = (
    "Use the original Piper gripper to pick a pipette from the rack, hand it to the Aero Hand palm, "
    "and close four non-thumb fingers to hold the pipette."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--task-name", default=DEFAULT_TASK_NAME)
    parser.add_argument("--dataset-name", default=None)
    parser.add_argument("--repo-id", default="aero_quest/piper_pipette_handoff")
    parser.add_argument("--task-prompt", default=DEFAULT_TASK_PROMPT)
    parser.add_argument("--num-episodes", type=int, default=2)
    parser.add_argument("--seed-start", type=int, default=11)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=320)
    parser.add_argument("--height", type=int, default=240)
    parser.add_argument("--planner-script", default=str(PROJECT_ROOT / "scripts/planning/plan_piper_gripper_pipette_handoff.py"))
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-failed-raw", action="store_true")
    parser.add_argument("--skip-render", action="store_true", help="Write state/action only, for fast export smoke tests.")
    return parser.parse_args()


def run_single_episode_planner(
    *,
    python_exe: str,
    planner_script: Path,
    model_path: Path,
    raw_episode_dir: Path,
    seed: int,
) -> tuple[Path, Path, float]:
    raw_episode_dir.mkdir(parents=True, exist_ok=True)
    cmd = [
        python_exe,
        str(planner_script),
        "--model",
        str(model_path),
        "--output-dir",
        str(raw_episode_dir),
        "--seed",
        str(seed),
    ]
    start = time.perf_counter()
    subprocess.run(cmd, cwd=PROJECT_ROOT, check=True)
    elapsed_s = time.perf_counter() - start
    return (
        raw_episode_dir / "piper_gripper_pipette_handoff_expert.npz",
        raw_episode_dir / "summary.json",
        elapsed_s,
    )


def make_dataset(
    *,
    dataset_root: Path,
    repo_id: str,
    model: mujoco.MjModel,
    fps: int,
    width: int,
    height: int,
    use_videos: bool,
) -> LeRobotDataset:
    features = lerobot_features(
        model,
        DEFAULT_HANDOFF_CAMERAS if use_videos else (),
        width=width,
        height=height,
    )
    return LeRobotDataset.create(
        repo_id=repo_id,
        root=dataset_root,
        fps=fps,
        features=features,
        robot_type="dual_piper_original_gripper_and_piper_aerohand_mujoco",
        use_videos=use_videos,
        image_writer_threads=4 if use_videos else 0,
        vcodec="h264",
    )


def export_episode(
    *,
    dataset: LeRobotDataset,
    model: mujoco.MjModel,
    npz_path: Path,
    task_prompt: str,
    fps: int,
    width: int,
    height: int,
    use_videos: bool,
) -> dict[str, object]:
    raw = np.load(npz_path, allow_pickle=False)
    qpos = np.asarray(raw["qpos"], dtype=np.float64)
    ctrl = np.asarray(raw["ctrl"], dtype=np.float64)
    labels = np.asarray(raw["labels"])
    source_fps = 1.0 / float(model.opt.timestep)
    indices = sample_indices(qpos.shape[0], source_fps, fps)
    label_to_index = stage_index_map(labels)

    renderer = None
    if use_videos:
        model.vis.global_.offwidth = max(model.vis.global_.offwidth, width)
        model.vis.global_.offheight = max(model.vis.global_.offheight, height)
        renderer = MujocoTrajectoryRenderer(
            model,
            DEFAULT_HANDOFF_CAMERAS,
            width=width,
            height=height,
        )

    try:
        for frame_index in indices:
            frame_index = int(frame_index)
            frame = {
                "observation.state": qpos[frame_index].astype(np.float32),
                "action": ctrl[frame_index].astype(np.float32),
                "observation.stage_index": np.asarray(
                    [label_to_index[str(labels[frame_index])]],
                    dtype=np.int64,
                ),
                "task": task_prompt,
            }
            if renderer is not None:
                images = renderer.render(qpos[frame_index])
                for camera_name, image in images.items():
                    frame[f"observation.images.{camera_name}"] = image
            dataset.add_frame(frame)
    finally:
        if renderer is not None:
            renderer.close()

    dataset.save_episode(parallel_encoding=False)
    return {
        "raw_frames": int(qpos.shape[0]),
        "exported_frames": int(indices.shape[0]),
        "source_fps": source_fps,
        "export_fps": int(fps),
        "stage_index": label_to_index,
    }


def main() -> None:
    args = parse_args()
    dataset_name = args.dataset_name or time.strftime("v0_%Y%m%d_%H%M%S")
    output_root = Path(args.output_root)
    dataset_root = resolve_dataset_root(output_root, args.task_name, dataset_name)
    if dataset_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Dataset root already exists, pass --overwrite to replace it: {dataset_root}")
        shutil.rmtree(dataset_root)
    dataset_root.parent.mkdir(parents=True, exist_ok=True)
    raw_root = dataset_root / "raw"

    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = (PROJECT_ROOT / model_path).resolve()
    model = mujoco.MjModel.from_xml_path(str(model_path))
    use_videos = not bool(args.skip_render)
    dataset = make_dataset(
        dataset_root=dataset_root,
        repo_id=args.repo_id,
        model=model,
        fps=args.fps,
        width=args.width,
        height=args.height,
        use_videos=use_videos,
    )
    raw_root.mkdir(parents=True, exist_ok=True)

    manifest: dict[str, object] = {
        "task_name": args.task_name,
        "dataset_name": dataset_name,
        "repo_id": args.repo_id,
        "task_prompt": args.task_prompt,
        "model": str(model_path),
        "fps": args.fps,
        "width": args.width,
        "height": args.height,
        "use_videos": use_videos,
        "camera_names": [camera.name for camera in DEFAULT_HANDOFF_CAMERAS] if use_videos else [],
        "episodes": [],
    }

    try:
        for episode_index in range(args.num_episodes):
            seed = int(args.seed_start + episode_index)
            raw_episode_dir = raw_root / f"episode_{episode_index:06d}"
            print(f"[episode {episode_index:06d}] planning with seed={seed}")
            try:
                npz_path, summary_path, planning_elapsed_s = run_single_episode_planner(
                    python_exe=sys.executable,
                    planner_script=Path(args.planner_script).expanduser().resolve(),
                    model_path=model_path,
                    raw_episode_dir=raw_episode_dir,
                    seed=seed,
                )
                with summary_path.open("r", encoding="utf-8") as f:
                    summary = json.load(f)
                if not summary.get("task_success", False):
                    raise RuntimeError(f"Planner exported unsuccessful episode: {summary_path}")
                print(f"[episode {episode_index:06d}] exporting LeRobot frames")
                export_info = export_episode(
                    dataset=dataset,
                    model=model,
                    npz_path=npz_path,
                    task_prompt=args.task_prompt,
                    fps=args.fps,
                    width=args.width,
                    height=args.height,
                    use_videos=use_videos,
                )
                episode_record = {
                    "episode_index": episode_index,
                    "seed": seed,
                    "raw_dir": str(raw_episode_dir),
                    "trajectory": str(npz_path),
                    "summary": str(summary_path),
                    "planning_elapsed_s": planning_elapsed_s,
                    "task_success": bool(summary.get("task_success", False)),
                    "dynamics": summary.get("dynamics", {}),
                    **export_info,
                }
                manifest["episodes"].append(episode_record)
                (dataset_root / "generation_manifest.json").write_text(
                    json.dumps(manifest, indent=2),
                    encoding="utf-8",
                )
            except Exception:
                if not args.keep_failed_raw and raw_episode_dir.exists():
                    shutil.rmtree(raw_episode_dir)
                raise
    finally:
        dataset.finalize()
        dataset.stop_image_writer()

    # Re-open the resulting dataset once; this catches missing parquet footers or video metadata early.
    loaded = LeRobotDataset(args.repo_id, root=dataset_root, download_videos=False)
    manifest["total_episodes"] = int(loaded.num_episodes)
    manifest["total_frames"] = int(loaded.num_frames)
    (dataset_root / "generation_manifest.json").write_text(
        json.dumps(manifest, indent=2),
        encoding="utf-8",
    )
    print(f"Wrote LeRobot dataset: {dataset_root}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
