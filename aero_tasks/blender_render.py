"""Blender rendering orchestration for MuJoCo expert trajectories."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np

from aero_tasks.lerobot_export import DEFAULT_HANDOFF_CAMERAS, RenderCameraSpec, sample_indices


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKER = PROJECT_ROOT / "scripts/blender/render_trajectory_worker.py"


@dataclass(frozen=True)
class BlenderRenderConfig:
    trajectory: Path
    model: Path | None = None
    wet_state: Path | None = None
    out_dir: Path = PROJECT_ROOT / "outputs/debug_rollouts/blender_render"
    output_name: str = "blender_render.mp4"
    camera: str = "table_overview"
    fps: int = 20
    width: int = 1280
    height: int = 720
    max_frames: int | None = 120
    stride: int | None = None
    engine: str = "BLENDER_EEVEE_NEXT"
    samples: int = 64
    blender: str = "blender"
    save_blend: bool = True
    visible_groups: tuple[int, ...] = (0, 1, 2)


def resolve_project_path(path: Path | str, *, base: Path = PROJECT_ROOT) -> Path:
    resolved = Path(path).expanduser()
    return resolved if resolved.is_absolute() else (base / resolved).resolve()


def resolve_model_path(raw_path: str | Path, trajectory_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return (trajectory_path.parent / path).resolve()


def load_qpos_trajectory(trajectory_path: Path, model_override: Path | None = None) -> tuple[np.ndarray, Path]:
    trajectory_path = resolve_project_path(trajectory_path)
    with np.load(trajectory_path, allow_pickle=False) as archive:
        if "qpos" not in archive:
            raise ValueError(f"Trajectory is missing required 'qpos': {trajectory_path}")
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[0] == 0:
            raise ValueError(f"Expected qpos with shape [frames, nq], got {qpos.shape}")
        if model_override is None:
            if "model" not in archive:
                raise ValueError("Trajectory has no model field; pass --model")
            raw_model = str(np.asarray(archive["model"]).item())
        else:
            raw_model = str(model_override)
    model_path = resolve_model_path(raw_model, trajectory_path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")
    return qpos, model_path


def selected_frame_indices(frame_count: int, *, source_fps: float, fps: int, stride: int | None, max_frames: int | None) -> np.ndarray:
    if stride is None:
        indices = sample_indices(frame_count, source_fps, fps)
    else:
        indices = np.arange(0, frame_count, max(1, int(stride)), dtype=np.int64)
    if max_frames is not None:
        indices = indices[: max(1, int(max_frames))]
    if indices.size == 0:
        indices = np.array([0], dtype=np.int64)
    return indices


def camera_specs_json(camera_specs: tuple[RenderCameraSpec, ...] = DEFAULT_HANDOFF_CAMERAS) -> list[dict[str, object]]:
    return [asdict(spec) for spec in camera_specs]


def write_render_manifest(
    config: BlenderRenderConfig,
    *,
    model_path: Path,
    frame_indices: np.ndarray,
    camera_specs: tuple[RenderCameraSpec, ...] = DEFAULT_HANDOFF_CAMERAS,
) -> Path:
    out_dir = resolve_project_path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    wet_state = None if config.wet_state is None else str(resolve_project_path(config.wet_state))
    manifest = {
        "trajectory": str(resolve_project_path(config.trajectory)),
        "model": str(model_path),
        "wet_state": wet_state,
        "output_video": str(out_dir / config.output_name),
        "output_blend": str(out_dir / "scene.blend"),
        "camera": config.camera,
        "fps": int(config.fps),
        "width": int(config.width),
        "height": int(config.height),
        "engine": config.engine,
        "samples": int(config.samples),
        "frame_indices": frame_indices.astype(int).tolist(),
        "camera_specs": camera_specs_json(camera_specs),
        "visible_groups": [int(group) for group in config.visible_groups],
    }
    manifest_path = out_dir / "blender_render_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return manifest_path


def build_blender_command(config: BlenderRenderConfig, manifest_path: Path, *, worker: Path = DEFAULT_WORKER) -> list[str]:
    if shutil.which(config.blender) is None and importlib.util.find_spec("bpy") is not None:
        return [
            sys.executable,
            str(worker),
            "--",
            "--manifest",
            str(manifest_path),
            "--save-blend" if config.save_blend else "--no-save-blend",
        ]
    return [
        config.blender,
        "--background",
        "--python",
        str(worker),
        "--",
        "--manifest",
        str(manifest_path),
        "--save-blend" if config.save_blend else "--no-save-blend",
    ]


def write_command_script(command: list[str], out_dir: Path) -> Path:
    script_path = out_dir / "render_command.sh"
    quoted = " ".join("'" + part.replace("'", "'\"'\"'") + "'" for part in command)
    script_path.write_text("#!/usr/bin/env bash\nset -euo pipefail\n" + quoted + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def prepare_blender_render(
    config: BlenderRenderConfig,
    *,
    camera_specs: tuple[RenderCameraSpec, ...] = DEFAULT_HANDOFF_CAMERAS,
) -> tuple[Path, list[str]]:
    qpos, model_path = load_qpos_trajectory(config.trajectory, config.model)
    try:
        import mujoco

        source_fps = 1.0 / float(mujoco.MjModel.from_xml_path(str(model_path)).opt.timestep)
    except Exception:
        source_fps = 500.0
    frame_indices = selected_frame_indices(
        qpos.shape[0],
        source_fps=source_fps,
        fps=config.fps,
        stride=config.stride,
        max_frames=config.max_frames,
    )
    manifest_path = write_render_manifest(config, model_path=model_path, frame_indices=frame_indices, camera_specs=camera_specs)
    command = build_blender_command(config, manifest_path)
    write_command_script(command, resolve_project_path(config.out_dir))
    return manifest_path, command


def run_blender_render(
    config: BlenderRenderConfig,
    *,
    dry_run: bool = False,
    camera_specs: tuple[RenderCameraSpec, ...] = DEFAULT_HANDOFF_CAMERAS,
) -> Path:
    manifest_path, command = prepare_blender_render(config, camera_specs=camera_specs)
    out_dir = resolve_project_path(config.out_dir)
    if dry_run:
        return manifest_path
    if command[0] == config.blender and shutil.which(config.blender) is None:
        raise FileNotFoundError(
            f"Blender executable {config.blender!r} is not available. "
            f"Prepared manifest and command script under {out_dir}."
        )
    subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return out_dir / config.output_name
