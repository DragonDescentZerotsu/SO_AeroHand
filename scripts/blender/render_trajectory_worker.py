#!/usr/bin/env python3
"""Blender-side worker for rendering MuJoCo qpos trajectories."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.blender_liquid import BlenderLiquidOverlay, WetStateSeries  # noqa: E402
from aero_tasks.blender_scene import (  # noqa: E402
    add_default_lighting,
    animate_camera,
    animate_mujoco_geoms,
    configure_render,
    create_blender_scene_from_mujoco,
)


def parse_worker_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--save-blend", dest="save_blend", action="store_true", default=True)
    parser.add_argument("--no-save-blend", dest="save_blend", action="store_false")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = []
    return parser.parse_args(argv)


def choose_camera(camera_specs: list[dict[str, object]], name: str) -> dict[str, object]:
    for spec in camera_specs:
        if spec["name"] == name:
            return spec
    available = ", ".join(str(spec["name"]) for spec in camera_specs)
    raise ValueError(f"Unknown camera {name!r}; available: {available}")


def main() -> None:
    import bpy
    import mujoco

    args = parse_worker_args()
    manifest_path = args.manifest.expanduser().resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    bpy.ops.wm.read_factory_settings(use_empty=True)
    model = mujoco.MjModel.from_xml_path(manifest["model"])
    with np.load(manifest["trajectory"], allow_pickle=False) as archive:
        qpos = np.asarray(archive["qpos"], dtype=np.float64)
    frame_indices = np.asarray(manifest["frame_indices"], dtype=np.int64)
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(f"qpos shape {qpos.shape} does not match model.nq={model.nq}")

    geoms = create_blender_scene_from_mujoco(model)
    animate_mujoco_geoms(model, qpos, frame_indices, geoms)
    camera_spec = choose_camera(manifest["camera_specs"], manifest["camera"])
    animate_camera(model, qpos, frame_indices, camera_spec)

    if manifest.get("wet_state"):
        wet_state = WetStateSeries.load(Path(manifest["wet_state"]))
        BlenderLiquidOverlay().animate(wet_state, frame_indices)

    add_default_lighting()
    configure_render(
        width=int(manifest["width"]),
        height=int(manifest["height"]),
        fps=int(manifest["fps"]),
        engine=str(manifest["engine"]),
        samples=int(manifest["samples"]),
        output_path=str(Path(manifest["output_video"])),
    )

    if args.save_blend:
        bpy.ops.wm.save_as_mainfile(filepath=str(Path(manifest["output_blend"])), check_existing=False, compress=True)
    bpy.ops.render.render(animation=True)


if __name__ == "__main__":
    main()
