#!/usr/bin/env python3
"""Render a MuJoCo qpos trajectory with Blender."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.blender_render import BlenderRenderConfig, prepare_blender_render, run_blender_render  # noqa: E402


DEFAULT_TRAJECTORY = PROJECT_ROOT / "outputs/piper_gripper_pipette_handoff/piper_gripper_pipette_handoff_expert.npz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trajectory", type=Path, default=DEFAULT_TRAJECTORY)
    parser.add_argument("--model", type=Path, default=None)
    parser.add_argument("--wet-state", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, default=PROJECT_ROOT / "outputs/debug_rollouts/blender_handoff")
    parser.add_argument("--output-name", default="blender_render.mp4")
    parser.add_argument("--camera", default="table_overview")
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--width", type=int, default=1280)
    parser.add_argument("--height", type=int, default=720)
    parser.add_argument("--max-frames", type=int, default=120)
    parser.add_argument("--stride", type=int, default=None)
    parser.add_argument("--engine", default="BLENDER_EEVEE_NEXT")
    parser.add_argument("--samples", type=int, default=64)
    parser.add_argument("--blender", default="blender")
    parser.add_argument("--dry-run", action="store_true", help="Only write manifest and render_command.sh.")
    parser.add_argument("--no-save-blend", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BlenderRenderConfig(
        trajectory=args.trajectory,
        model=args.model,
        wet_state=args.wet_state,
        out_dir=args.out_dir,
        output_name=args.output_name,
        camera=args.camera,
        fps=args.fps,
        width=args.width,
        height=args.height,
        max_frames=args.max_frames,
        stride=args.stride,
        engine=args.engine,
        samples=args.samples,
        blender=args.blender,
        save_blend=not args.no_save_blend,
    )
    if args.dry_run:
        manifest, command = prepare_blender_render(config)
        print(f"Wrote Blender manifest: {manifest}")
        print(f"Wrote command script: {Path(args.out_dir) / 'render_command.sh'}")
        print("Command:")
        print(" ".join(command))
        return
    try:
        output = run_blender_render(config)
    except FileNotFoundError as exc:
        manifest, _ = prepare_blender_render(config)
        print(str(exc))
        print(f"Blender manifest: {manifest}")
        print(f"Command script: {Path(args.out_dir) / 'render_command.sh'}")
        raise SystemExit(2) from exc
    print(f"Wrote Blender video: {output}")


if __name__ == "__main__":
    main()
