#!/usr/bin/env python3
"""Render a small Blender liquid refraction comparison scene."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.blender_scene import make_glass_material, make_material  # noqa: E402


DEFAULT_OUT_DIR = PROJECT_ROOT / "outputs/debug_rollouts/blender_liquid_refraction_compare"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=544)
    parser.add_argument("--fps", type=int, default=20)
    parser.add_argument("--frames", type=int, default=80)
    parser.add_argument("--cycles-samples", type=int, default=64)
    parser.add_argument("--engine", choices=("all", "eevee", "cycles"), default="all")
    argv = sys.argv
    if "--" in argv:
        argv = argv[argv.index("--") + 1 :]
    else:
        argv = argv[1:]
    return parser.parse_args(argv)


def enable_cycles_if_needed(engine: str) -> None:
    if engine != "CYCLES":
        return
    try:
        import addon_utils

        addon_utils.enable("cycles", default_set=True, persistent=True)
    except Exception:
        pass


def can_set_engine(engine: str) -> bool:
    import bpy

    enable_cycles_if_needed(engine)
    previous = bpy.context.scene.render.engine
    try:
        bpy.context.scene.render.engine = engine
        return True
    except Exception:
        return False
    finally:
        try:
            bpy.context.scene.render.engine = previous
        except Exception:
            pass


def look_at(obj: object, eye: tuple[float, float, float], target: tuple[float, float, float]) -> None:
    from mathutils import Vector

    obj.location = eye
    direction = Vector(target) - Vector(eye)
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def liquid_material(name: str, rgba: tuple[float, float, float, float], *, ior: float = 1.333) -> object:
    return make_glass_material(name, np.asarray(rgba, dtype=np.float64), ior=ior)


def opaque_material(name: str, rgba: tuple[float, float, float, float]) -> object:
    return make_material(name, np.asarray(rgba, dtype=np.float64), roughness=0.35)


def add_box(name: str, location: tuple[float, float, float], scale: tuple[float, float, float], material: object) -> object:
    import bpy

    bpy.ops.mesh.primitive_cube_add(size=1.0, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    obj.data.materials.append(material)
    return obj


def add_cylinder(name: str, radius: float, depth: float, location: tuple[float, float, float], material: object, *, vertices: int = 96) -> object:
    import bpy

    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=location)
    obj = bpy.context.object
    obj.name = name
    obj.data.materials.append(material)
    return obj


def add_tapered_column(
    name: str,
    bottom_radius: float,
    top_radius: float,
    height: float,
    location: tuple[float, float, float],
    material: object,
    *,
    segments: int = 96,
) -> object:
    import bpy

    vertices = []
    faces = []
    z0 = -0.5 * height
    z1 = 0.5 * height
    for idx in range(segments):
        theta = 2.0 * np.pi * idx / segments
        c = float(np.cos(theta))
        s = float(np.sin(theta))
        vertices.append((bottom_radius * c, bottom_radius * s, z0))
        vertices.append((top_radius * c, top_radius * s, z1))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, z0))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, z1))
    for idx in range(segments):
        nxt = (idx + 1) % segments
        faces.append((2 * idx, 2 * nxt, 2 * nxt + 1, 2 * idx + 1))
        faces.append((bottom_center, 2 * nxt, 2 * idx))
        faces.append((top_center, 2 * idx + 1, 2 * nxt + 1))
    mesh = bpy.data.meshes.new(f"{name}_mesh")
    mesh.from_pydata(vertices, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)
    obj.location = location
    obj.data.materials.append(material)
    return obj


def add_refraction_scene() -> dict[str, object]:
    import bpy

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = -0.35
    scene.view_settings.gamma = 1.0
    if scene.world is None:
        scene.world = bpy.data.worlds.new("refraction_world")
    scene.world.color = (0.08, 0.08, 0.08)
    if hasattr(scene.render, "film_transparent"):
        scene.render.film_transparent = False

    yellow = opaque_material("stripe_yellow", (0.85, 0.62, 0.05, 1.0))
    cyan = opaque_material("stripe_cyan", (0.02, 0.58, 0.72, 1.0))
    magenta = opaque_material("stripe_magenta", (0.70, 0.08, 0.62, 1.0))
    black = opaque_material("stripe_black", (0.02, 0.02, 0.02, 1.0))
    red = opaque_material("stripe_red", (0.75, 0.12, 0.06, 1.0))
    green = opaque_material("stripe_green", (0.03, 0.52, 0.18, 1.0))
    blue = opaque_material("stripe_blue", (0.05, 0.14, 0.72, 1.0))
    glass = liquid_material("clear_tube_glass", (0.82, 0.94, 1.0, 1.0), ior=1.46)
    liquid = liquid_material("current_liquid_style", (0.34, 0.72, 1.0, 1.0), ior=1.333)
    tip_liquid = liquid_material("tip_liquid_style", (0.34, 0.72, 1.0, 1.0), ior=1.333)

    # A high-contrast target directly behind the liquid, so refraction is visible.
    add_box("dark_refraction_backdrop", (0.0, 0.0475, 0.030), (0.210, 0.001, 0.128), opaque_material("backdrop", (0.11, 0.11, 0.12, 1.0)))
    stripe_mats = [black, yellow, blue, red, cyan, black, green, magenta, blue, yellow, black, red, cyan, green, black, magenta, yellow, blue, black, red]
    x0 = -0.100
    for index, material in enumerate(stripe_mats):
        add_box(
            f"back_stripe_{index:02d}",
            (x0 + 0.0105 * index, 0.045, 0.030),
            (0.0095, 0.0012, 0.118),
            material,
        )
    for index in range(10):
        add_box(
            f"horizontal_rule_{index:02d}",
            (0.0, 0.0435, -0.020 + 0.011 * index),
            (0.205, 0.0012, 0.0014),
            black if index % 2 == 0 else blue,
        )

    add_box("matte_table", (0.0, 0.0, -0.035), (0.22, 0.16, 0.003), opaque_material("table", (0.34, 0.34, 0.32, 1.0)))
    add_cylinder("transparent_tube_wall", 0.015, 0.070, (-0.032, 0.0, 0.008), glass)
    liquid_obj = add_cylinder("bulk_liquid", 0.0115, 0.038, (-0.032, 0.0, -0.008), liquid)
    tip_obj = add_tapered_column(
        "tip_liquid_column",
        bottom_radius=0.0010,
        top_radius=0.0040,
        height=0.064,
        location=(0.036, -0.001, 0.005),
        material=tip_liquid,
    )
    tip_obj.rotation_euler[1] = np.deg2rad(-8.0)

    bpy.ops.object.light_add(type="AREA", location=(0.0, -0.12, 0.12))
    light = bpy.context.object
    light.name = "large_softbox"
    light.data.energy = 60.0
    light.data.size = 0.12
    bpy.ops.object.camera_add()
    camera = bpy.context.object
    look_at(camera, (0.0, -0.145, 0.030), (0.0, 0.010, 0.022))
    scene.camera = camera

    return {
        "liquid": liquid_obj,
        "tip_liquid": tip_obj,
    }


def configure_engine(engine: str, width: int, height: int, samples: int) -> str:
    import bpy

    scene = bpy.context.scene
    enable_cycles_if_needed(engine)
    scene.render.engine = engine
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.fps = 20
    if engine == "CYCLES":
        scene.cycles.samples = int(samples)
        scene.cycles.use_denoising = True
        scene.cycles.device = "CPU"
    elif engine.startswith("BLENDER_EEVEE"):
        eevee = getattr(scene, "eevee", None)
        if eevee is not None:
            for attr, value in (
                ("use_raytracing", True),
                ("ray_tracing_method", "SCREEN"),
                ("taa_render_samples", int(samples)),
                ("taa_samples", int(samples)),
            ):
                if hasattr(eevee, attr):
                    setattr(eevee, attr, value)
            ray_options = getattr(eevee, "ray_tracing_options", None)
            if ray_options is not None:
                for attr, value in (
                    ("screen_trace_quality", 1.0),
                    ("screen_trace_thickness", 0.2),
                    ("resolution_scale", "1"),
                    ("use_denoise", True),
                ):
                    if hasattr(ray_options, attr):
                        try:
                            setattr(ray_options, attr, value)
                        except Exception:
                            pass
    return engine


def render_engine(engine: str, args: argparse.Namespace, out_dir: Path) -> dict[str, object]:
    import bpy

    objects = add_refraction_scene()
    engine_id = configure_engine(engine, args.width, args.height, args.cycles_samples)
    engine_name = "cycles" if engine_id == "CYCLES" else "eevee"
    frames_dir = out_dir / f"{engine_name}_frames"
    frames_dir.mkdir(parents=True, exist_ok=True)
    for old_frame in frames_dir.glob("*.png"):
        old_frame.unlink()
    frame_paths: list[Path] = []
    for frame in range(int(args.frames)):
        t = frame / max(1, int(args.frames) - 1)
        angle = 2.0 * np.pi * t
        objects["liquid"].rotation_euler[2] = 0.18 * np.sin(angle)
        objects["tip_liquid"].location.x = 0.040 - 0.016 * np.sin(angle)
        bpy.context.scene.frame_set(frame)
        frame_path = frames_dir / f"frame_{frame:04d}.png"
        bpy.context.scene.render.filepath = str(frame_path)
        bpy.ops.render.render(write_still=True)
        frame_paths.append(frame_path)

    video_path = out_dir / f"{engine_name}_refraction.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-framerate",
            str(int(args.fps)),
            "-i",
            str(frames_dir / "frame_%04d.png"),
            "-pix_fmt",
            "yuv420p",
            "-c:v",
            "libx264",
            "-crf",
            "18",
            str(video_path),
        ],
        check=True,
    )
    still_path = out_dir / f"{engine_name}_refraction_still.png"
    still_path.write_bytes(frame_paths[min(len(frame_paths) // 2, len(frame_paths) - 1)].read_bytes())
    return {"engine": engine_id, "video": str(video_path), "still": str(still_path)}


def main() -> None:
    import bpy

    args = parse_args()
    out_dir = args.out_dir.expanduser()
    if not out_dir.is_absolute():
        out_dir = PROJECT_ROOT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    enum_engines = {item.identifier for item in bpy.context.scene.render.bl_rna.properties["engine"].enum_items}
    requested = []
    if args.engine in ("all", "eevee"):
        requested.append("BLENDER_EEVEE_NEXT" if "BLENDER_EEVEE_NEXT" in enum_engines else "BLENDER_EEVEE")
    if args.engine in ("all", "cycles"):
        requested.append("CYCLES")

    results: list[dict[str, object]] = []
    unavailable: list[str] = []
    for engine in requested:
        if not can_set_engine(engine):
            unavailable.append(engine)
            continue
        results.append(render_engine(engine, args, out_dir))

    summary = {
        "enum_engines": sorted(enum_engines),
        "settable_engines": [engine for engine in requested if engine not in unavailable],
        "requested_engines": requested,
        "unavailable_engines": unavailable,
        "results": results,
        "note": "This scene uses the project liquid material over a stripe target to make transparent refraction easier to inspect.",
    }
    summary_path = out_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
