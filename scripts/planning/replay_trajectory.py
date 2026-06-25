"""Interactively replay an exported MuJoCo trajectory in the native viewer."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import sys
import threading
import time

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.task_sampling import apply_episode_spec_to_model, load_episode_spec  # noqa: E402

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:  # pragma: no cover
    raise SystemExit(f"Missing dependency: {exc}") from exc


DEFAULT_TRAJECTORY = (
    PROJECT_ROOT
    / "outputs/piper_gripper_pipette_handoff/piper_gripper_pipette_handoff_expert.npz"
)


@dataclass(frozen=True)
class Trajectory:
    path: Path
    model_path: Path
    qpos: np.ndarray
    labels: np.ndarray | None


@dataclass(frozen=True)
class HandoffMarkers:
    target_site_id: int
    target_body_id: int
    pipette_body_id: int
    target_axis_local: np.ndarray
    hook_reference_local: np.ndarray
    surface_offset_m: float
    vertical_offset_m: float
    approach_axis_world: np.ndarray


class PlaybackState:
    def __init__(self, frame_count: int, *, speed: float, loop: bool) -> None:
        self.frame_count = frame_count
        self.frame = 0
        self.speed = speed
        self.loop = loop
        self.paused = False
        self.restart_requested = False
        self.step_delta = 0
        self.lock = threading.Lock()

    def handle_key(self, keycode: int) -> None:
        with self.lock:
            if keycode == ord(" "):
                self.paused = not self.paused
            elif keycode in (ord("R"), ord("r")):
                self.restart_requested = True
            elif keycode in (ord("L"), ord("l")):
                self.loop = not self.loop
            elif keycode == 262:  # GLFW_KEY_RIGHT
                self.paused = True
                self.step_delta += 1
            elif keycode == 263:  # GLFW_KEY_LEFT
                self.paused = True
                self.step_delta -= 1
            elif keycode == 265:  # GLFW_KEY_UP
                self.speed = min(8.0, self.speed * 1.25)
            elif keycode == 264:  # GLFW_KEY_DOWN
                self.speed = max(0.05, self.speed / 1.25)


def resolve_model_path(raw_path: str, trajectory_path: Path) -> Path:
    path = Path(raw_path).expanduser()
    if path.is_absolute():
        return path.resolve()
    project_candidate = (PROJECT_ROOT / path).resolve()
    if project_candidate.exists():
        return project_candidate
    return (trajectory_path.parent / path).resolve()


def load_trajectory(path: Path, model_override: str | None = None) -> Trajectory:
    path = path.expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Trajectory does not exist: {path}")

    with np.load(path, allow_pickle=False) as archive:
        if "qpos" not in archive:
            raise ValueError(f"Trajectory is missing required 'qpos' array: {path}")
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
        if qpos.ndim != 2 or qpos.shape[0] == 0:
            raise ValueError(f"Expected non-empty qpos array shaped [frames, nq], got {qpos.shape}")
        labels = np.asarray(archive["labels"]).astype(str) if "labels" in archive else None
        if labels is not None and labels.shape != (qpos.shape[0],):
            raise ValueError(
                f"labels shape {labels.shape} does not match qpos frame count {qpos.shape[0]}"
            )
        if model_override is None:
            if "model" not in archive:
                raise ValueError("Trajectory has no 'model' field; pass --model explicitly")
            raw_model_path = str(np.asarray(archive["model"]).item())
        else:
            raw_model_path = model_override

    model_path = resolve_model_path(raw_model_path, path)
    if not model_path.exists():
        raise FileNotFoundError(f"MuJoCo model does not exist: {model_path}")
    return Trajectory(path=path, model_path=model_path, qpos=qpos, labels=labels)


def apply_frame(model: mujoco.MjModel, data: mujoco.MjData, qpos: np.ndarray) -> None:
    if qpos.shape != (model.nq,):
        raise ValueError(f"Trajectory qpos width is {qpos.size}, but model.nq is {model.nq}")
    data.qpos[:] = qpos
    data.qvel[:] = 0.0
    data.qacc[:] = 0.0
    mujoco.mj_forward(model, data)


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-10:
        raise ValueError("Cannot normalize near-zero vector")
    return vector / norm


def load_handoff_markers(
    model: mujoco.MjModel,
    summary_path: Path | None,
) -> HandoffMarkers | None:
    if summary_path is None or not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    required = (
        "handoff_target_site",
        "handoff_target_body",
        "pipette_hook_reference_local",
        "handoff_target_surface_offset_m",
        "handoff_approach_axis_world",
    )
    if not all(key in summary for key in required):
        return None

    target_site_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_SITE,
        summary["handoff_target_site"],
    )
    target_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        summary["handoff_target_body"],
    )
    pipette_body_id = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_BODY,
        "pipette_0/pipette",
    )
    if min(target_site_id, target_body_id, pipette_body_id) < 0:
        return None
    return HandoffMarkers(
        target_site_id=target_site_id,
        target_body_id=target_body_id,
        pipette_body_id=pipette_body_id,
        target_axis_local=np.array([0.0, 0.0, 1.0], dtype=np.float64),
        hook_reference_local=np.asarray(
            summary["pipette_hook_reference_local"],
            dtype=np.float64,
        ),
        surface_offset_m=float(summary["handoff_target_surface_offset_m"]),
        vertical_offset_m=float(summary.get("handoff_target_vertical_offset_m", 0.0)),
        approach_axis_world=normalize(
            np.asarray(summary["handoff_approach_axis_world"], dtype=np.float64)
        ),
    )


def set_sphere(geom, position: np.ndarray, radius: float, rgba: tuple[float, ...]) -> None:
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.array([radius, radius, radius], dtype=np.float64),
        np.asarray(position, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )


def set_line(
    geom,
    start: np.ndarray,
    end: np.ndarray,
    width: float,
    rgba: tuple[float, ...],
) -> None:
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.ones(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        geom,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.asarray(start, dtype=np.float64),
        np.asarray(end, dtype=np.float64),
    )


def update_handoff_markers(
    viewer,
    data: mujoco.MjData,
    markers: HandoffMarkers,
) -> None:
    target_center = data.site_xpos[markers.target_site_id].copy()
    target_body_R = data.xmat[markers.target_body_id].reshape(3, 3)
    target_axis = normalize(target_body_R @ markers.target_axis_local)
    world_up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    top_direction = normalize(world_up - target_axis * float(np.dot(world_up, target_axis)))
    target_point = (
        target_center
        + markers.surface_offset_m * top_direction
        + np.array([0.0, 0.0, markers.vertical_offset_m], dtype=np.float64)
    )

    pipette_R = data.xmat[markers.pipette_body_id].reshape(3, 3)
    hook_point = (
        data.xpos[markers.pipette_body_id]
        + pipette_R @ markers.hook_reference_local
    )
    axis_half_length = 0.035
    approach_start = target_point - 0.05 * markers.approach_axis_world

    with viewer.lock():
        scene = viewer.user_scn
        set_sphere(scene.geoms[0], target_center, 0.004, (1.0, 0.15, 0.1, 0.9))
        set_sphere(scene.geoms[1], target_point, 0.005, (1.0, 0.85, 0.05, 0.95))
        set_sphere(scene.geoms[2], hook_point, 0.004, (0.1, 1.0, 0.2, 0.95))
        set_line(
            scene.geoms[3],
            target_center - axis_half_length * target_axis,
            target_center + axis_half_length * target_axis,
            4.0,
            (0.1, 0.35, 1.0, 0.9),
        )
        set_line(
            scene.geoms[4],
            approach_start,
            target_point,
            4.0,
            (0.05, 0.9, 0.9, 0.9),
        )
        set_line(
            scene.geoms[5],
            hook_point,
            target_point,
            2.0,
            (1.0, 0.1, 0.8, 0.8),
        )
        scene.ngeom = 6


def replay(
    trajectory: Trajectory,
    *,
    trajectory_fps: float | None,
    speed: float,
    loop: bool,
    summary_path: Path | None,
) -> None:
    model = mujoco.MjModel.from_xml_path(str(trajectory.model_path))
    episode_spec_path = trajectory.path.parent / "episode_spec.json"
    if episode_spec_path.exists():
        apply_episode_spec_to_model(model, load_episode_spec(episode_spec_path))
    if trajectory.qpos.shape[1] != model.nq:
        raise ValueError(
            f"Trajectory qpos width is {trajectory.qpos.shape[1]}, but model.nq is {model.nq}"
        )
    data = mujoco.MjData(model)
    markers = load_handoff_markers(model, summary_path)
    if trajectory_fps is None:
        trajectory_fps = 1.0 / float(model.opt.timestep)
    state = PlaybackState(len(trajectory.qpos), speed=speed, loop=loop)
    apply_frame(model, data, trajectory.qpos[0])

    print(f"Trajectory: {trajectory.path}")
    print(f"Model: {trajectory.model_path}")
    print(f"Frames: {len(trajectory.qpos)}, source FPS: {trajectory_fps:g}")
    print("Controls: Space pause | Left/Right step | Up/Down speed | R restart | L loop")
    if markers is not None:
        print(
            "Markers: red=finger center | yellow=top hook target | green=hook reference | "
            "blue=finger axis | cyan=insert direction | magenta=target error"
        )

    with mujoco.viewer.launch_passive(model, data, key_callback=state.handle_key) as viewer:
        last_wall_time = time.monotonic()
        frame_position = 0.0
        last_reported_label: str | None = None

        while viewer.is_running():
            now = time.monotonic()
            elapsed = min(now - last_wall_time, 0.25)
            last_wall_time = now

            with state.lock:
                if state.restart_requested:
                    state.frame = 0
                    frame_position = 0.0
                    state.restart_requested = False

                if state.step_delta:
                    state.frame = int(np.clip(state.frame + state.step_delta, 0, state.frame_count - 1))
                    frame_position = float(state.frame)
                    state.step_delta = 0

                if not state.paused:
                    frame_position += elapsed * trajectory_fps * state.speed
                    if frame_position >= state.frame_count:
                        if state.loop:
                            frame_position %= state.frame_count
                        else:
                            frame_position = float(state.frame_count - 1)
                            state.paused = True
                    state.frame = min(int(frame_position), state.frame_count - 1)

                frame = state.frame
                paused = state.paused
                playback_speed = state.speed
                looping = state.loop

            apply_frame(model, data, trajectory.qpos[frame])
            if markers is not None:
                update_handoff_markers(viewer, data, markers)
            viewer.sync()

            label = trajectory.labels[frame] if trajectory.labels is not None else None
            if label != last_reported_label:
                print(
                    f"frame={frame}/{state.frame_count - 1} label={label or '-'} "
                    f"speed={playback_speed:.2f}x paused={paused} loop={looping}"
                )
                last_reported_label = label

            time.sleep(1.0 / 120.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trajectory", nargs="?", default=str(DEFAULT_TRAJECTORY))
    parser.add_argument("--model", help="Override the model path stored in the trajectory")
    parser.add_argument(
        "--trajectory-fps",
        type=float,
        help="Override trajectory FPS (default: derive from the model timestep)",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Initial playback speed")
    parser.add_argument("--no-loop", action="store_true", help="Pause on the final frame")
    parser.add_argument(
        "--summary",
        help="Planning summary JSON used for handoff markers (default: trajectory directory/summary.json)",
    )
    parser.add_argument("--no-markers", action="store_true", help="Disable handoff markers")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.trajectory_fps is not None and args.trajectory_fps <= 0.0:
        raise SystemExit("--trajectory-fps must be positive")
    if args.speed <= 0.0:
        raise SystemExit("--speed must be positive")
    trajectory = load_trajectory(Path(args.trajectory), args.model)
    summary_path = None
    if not args.no_markers:
        summary_path = (
            Path(args.summary).expanduser().resolve()
            if args.summary
            else trajectory.path.parent / "summary.json"
        )
    replay(
        trajectory,
        trajectory_fps=args.trajectory_fps,
        speed=args.speed,
        loop=not args.no_loop,
        summary_path=summary_path,
    )


if __name__ == "__main__":
    main()
