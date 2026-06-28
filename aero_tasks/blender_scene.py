"""MuJoCo-to-Blender scene helpers.

This module is importable without Blender. Functions that touch Blender import
``bpy`` lazily and are meant to run from Blender's Python interpreter.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
import re
from typing import Any

import numpy as np


def safe_name(name: str, fallback: str) -> str:
    text = name or fallback
    text = re.sub(r"[^A-Za-z0-9_.:/-]+", "_", text)
    return text[:180]


def srgb_to_linear_channel(value: float) -> float:
    value = float(value)
    if value <= 0.04045:
        return max(0.0, value / 12.92)
    return ((value + 0.055) / 1.055) ** 2.4


def srgb_to_linear_rgba(rgba: np.ndarray) -> tuple[float, float, float, float]:
    return (
        srgb_to_linear_channel(float(rgba[0])),
        srgb_to_linear_channel(float(rgba[1])),
        srgb_to_linear_channel(float(rgba[2])),
        float(rgba[3]),
    )


def unit_box_mesh() -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    vertices = [
        (-1, -1, -1),
        (1, -1, -1),
        (1, 1, -1),
        (-1, 1, -1),
        (-1, -1, 1),
        (1, -1, 1),
        (1, 1, 1),
        (-1, 1, 1),
    ]
    faces = [(0, 1, 2, 3), (4, 7, 6, 5), (0, 4, 5, 1), (1, 5, 6, 2), (2, 6, 7, 3), (3, 7, 4, 0)]
    return vertices, faces


def unit_cylinder_mesh(segments: int = 32, *, caps: bool = True) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    vertices: list[tuple[float, float, float]] = []
    for z in (-1.0, 1.0):
        for index in range(segments):
            angle = 2.0 * math.pi * index / segments
            vertices.append((math.cos(angle), math.sin(angle), z))
    faces: list[tuple[int, ...]] = []
    if caps:
        faces.append(tuple(range(segments - 1, -1, -1)))
        faces.append(tuple(range(segments, 2 * segments)))
    for index in range(segments):
        nxt = (index + 1) % segments
        faces.append((index, nxt, segments + nxt, segments + index))
    return vertices, faces


def unit_disk_mesh(segments: int = 64) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    vertices = [(0.0, 0.0, 0.0)]
    for index in range(segments):
        angle = 2.0 * math.pi * index / segments
        vertices.append((math.cos(angle), math.sin(angle), 0.0))
    faces = [(0, index, 1 + (index % segments)) for index in range(1, segments + 1)]
    return vertices, faces


def unit_sphere_mesh(segments: int = 24, rings: int = 12) -> tuple[list[tuple[float, float, float]], list[tuple[int, ...]]]:
    vertices: list[tuple[float, float, float]] = [(0.0, 0.0, 1.0)]
    for ring in range(1, rings):
        phi = math.pi * ring / rings
        z = math.cos(phi)
        radius = math.sin(phi)
        for index in range(segments):
            theta = 2.0 * math.pi * index / segments
            vertices.append((radius * math.cos(theta), radius * math.sin(theta), z))
    vertices.append((0.0, 0.0, -1.0))
    south = len(vertices) - 1
    faces: list[tuple[int, ...]] = []
    first = 1
    for index in range(segments):
        faces.append((0, first + index, first + ((index + 1) % segments)))
    for ring in range(rings - 2):
        start = 1 + ring * segments
        nxt_start = start + segments
        for index in range(segments):
            faces.append((start + index, start + ((index + 1) % segments), nxt_start + ((index + 1) % segments), nxt_start + index))
    last = 1 + (rings - 2) * segments
    for index in range(segments):
        faces.append((last + ((index + 1) % segments), last + index, south))
    return vertices, faces


def make_mesh(name: str, vertices: list[tuple[float, float, float]], faces: list[tuple[int, ...]]) -> Any:
    import bpy

    mesh = bpy.data.meshes.new(name)
    mesh.from_pydata(vertices, [], faces)
    mesh.validate()
    mesh.update()
    mesh.materials.append(None)
    return mesh


def mujoco_mesh_data(model: Any, mesh_id: int) -> tuple[list[tuple[float, float, float]], list[tuple[int, int, int]]]:
    vert_start = int(model.mesh_vertadr[mesh_id])
    vert_end = vert_start + int(model.mesh_vertnum[mesh_id])
    face_start = int(model.mesh_faceadr[mesh_id])
    face_end = face_start + int(model.mesh_facenum[mesh_id])
    vertices = [tuple(float(v) for v in row) for row in model.mesh_vert[vert_start:vert_end]]
    faces = [tuple(int(v) for v in row) for row in model.mesh_face[face_start:face_end]]
    return vertices, faces


def make_material(name: str, rgba: np.ndarray, *, roughness: float = 0.45, metallic: float = 0.0) -> Any:
    import bpy

    rgba = np.asarray(rgba, dtype=np.float64).reshape(4)
    material = bpy.data.materials.new(safe_name(name, "material"))
    material.diffuse_color = tuple(float(v) for v in rgba)
    material.use_nodes = True
    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = srgb_to_linear_rgba(rgba)
        bsdf.inputs["Alpha"].default_value = float(rgba[3])
        bsdf.inputs["Roughness"].default_value = float(roughness)
        bsdf.inputs["Metallic"].default_value = float(metallic)
    if rgba[3] < 0.999:
        material.blend_method = "BLEND"
        if hasattr(material, "surface_render_method"):
            material.surface_render_method = "BLENDED"
        material.use_screen_refraction = True
        material.show_transparent_back = True
    return material


@dataclass
class BlenderGeom:
    geom_id: int
    obj: Any
    scale: np.ndarray


def geom_rgba(model: Any, geom_id: int) -> np.ndarray:
    geom_rgba_value = np.asarray(model.geom_rgba[geom_id], dtype=np.float64).copy()
    if geom_rgba_value[3] <= 1e-6:
        return geom_rgba_value
    if int(model.geom_matid[geom_id]) >= 0:
        rgba = np.asarray(model.mat_rgba[int(model.geom_matid[geom_id])], dtype=np.float64).copy()
    else:
        rgba = geom_rgba_value
    if rgba[3] <= 1e-6:
        rgba = np.array([0.75, 0.75, 0.75, 1.0], dtype=np.float64)
    return rgba


def mesh_for_geom(model: Any, geom_id: int, cache: dict[str, Any], *, transparent: bool = False) -> tuple[Any, np.ndarray] | None:
    import mujoco

    geom_type = int(model.geom_type[geom_id])
    size = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
        key = "plane"
        cache.setdefault(key, make_mesh("primitive_plane", *unit_box_mesh()))
        return cache[key], np.array([max(size[0], 1.0), max(size[1], 1.0), 0.002])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        key = "sphere"
        cache.setdefault(key, make_mesh("primitive_sphere", *unit_sphere_mesh()))
        return cache[key], np.array([size[0], size[0], size[0]])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_ELLIPSOID):
        key = "sphere"
        cache.setdefault(key, make_mesh("primitive_sphere", *unit_sphere_mesh()))
        return cache[key], np.maximum(size[:3], 1e-6)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        key = "cylinder"
        cache.setdefault(key, make_mesh("primitive_cylinder", *unit_cylinder_mesh()))
        return cache[key], np.array([size[0], size[0], max(size[1] + size[0], 1e-6)])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        key = "open_cylinder" if transparent else "cylinder"
        cache.setdefault(key, make_mesh(f"primitive_{key}", *unit_cylinder_mesh(caps=not transparent)))
        return cache[key], np.array([size[0], size[0], max(size[1], 1e-6)])
    if geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
        key = "box"
        cache.setdefault(key, make_mesh("primitive_box", *unit_box_mesh()))
        return cache[key], np.maximum(size[:3], 1e-6)
    if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
        mesh_id = int(model.geom_dataid[geom_id])
        key = f"mesh_{mesh_id}"
        if key not in cache:
            cache[key] = make_mesh(safe_name(f"mesh_{mesh_id}", key), *mujoco_mesh_data(model, mesh_id))
        return cache[key], np.ones(3)
    return None


def create_blender_scene_from_mujoco(model: Any, *, visible_groups: set[int] | None = None) -> list[BlenderGeom]:
    import bpy

    collection = bpy.data.collections.new("MuJoCo")
    bpy.context.scene.collection.children.link(collection)
    mesh_cache: dict[str, Any] = {}
    material_cache: dict[tuple[float, float, float, float], Any] = {}
    geoms: list[BlenderGeom] = []
    for geom_id in range(model.ngeom):
        if visible_groups is not None and int(model.geom_group[geom_id]) not in visible_groups:
            continue
        rgba = geom_rgba(model, geom_id)
        if rgba[3] <= 1e-5:
            continue
        mesh_scale = mesh_for_geom(model, geom_id, mesh_cache, transparent=bool(rgba[3] < 0.5))
        if mesh_scale is None:
            continue
        mesh, scale = mesh_scale
        name = safe_name(model.geom(geom_id).name, f"geom_{geom_id:04d}")
        obj = bpy.data.objects.new(name, mesh)
        collection.objects.link(obj)
        key = tuple(float(v) for v in rgba)
        if key not in material_cache:
            material_cache[key] = make_material(f"mat_{len(material_cache):03d}", rgba)
        obj.data.materials[0] = material_cache[key]
        geoms.append(BlenderGeom(geom_id=geom_id, obj=obj, scale=np.asarray(scale, dtype=np.float64)))
    return geoms


def set_object_pose(obj: Any, pos: np.ndarray, xmat: np.ndarray, scale: np.ndarray) -> None:
    from mathutils import Matrix

    rot = Matrix(np.asarray(xmat, dtype=np.float64).reshape(3, 3).tolist())
    obj.location = tuple(float(v) for v in np.asarray(pos, dtype=np.float64).reshape(3))
    obj.rotation_mode = "QUATERNION"
    obj.rotation_quaternion = rot.to_quaternion()
    obj.scale = tuple(float(v) for v in np.asarray(scale, dtype=np.float64).reshape(3))


def animate_mujoco_geoms(model: Any, qpos: np.ndarray, frame_indices: np.ndarray, geoms: list[BlenderGeom]) -> Any:
    import bpy
    import mujoco

    data = mujoco.MjData(model)
    bpy.context.scene.frame_start = 0
    bpy.context.scene.frame_end = max(0, len(frame_indices) - 1)
    for out_frame, source_frame in enumerate(frame_indices.astype(int).tolist()):
        bpy.context.scene.frame_set(out_frame)
        data.qpos[:] = qpos[source_frame]
        mujoco.mj_forward(model, data)
        for geom in geoms:
            set_object_pose(geom.obj, data.geom_xpos[geom.geom_id], data.geom_xmat[geom.geom_id], geom.scale)
            geom.obj.keyframe_insert(data_path="location", frame=out_frame)
            geom.obj.keyframe_insert(data_path="rotation_quaternion", frame=out_frame)
            geom.obj.keyframe_insert(data_path="scale", frame=out_frame)
    return data


def camera_eye_target_up(model: Any, data: Any, spec: dict[str, Any]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    import mujoco

    mode = spec["mode"]
    if mode == "world":
        return (
            np.asarray(spec["eye_offset_world"], dtype=np.float64),
            np.asarray(spec["lookat"], dtype=np.float64),
            np.array([0.0, 0.0, 1.0], dtype=np.float64),
        )
    body_name = spec.get("body")
    if body_name is None:
        raise ValueError(f"Camera mode {mode!r} requires a body")
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing camera body {body_name!r}")
    body_pos = data.xpos[body_id].copy()
    body_R = data.xmat[body_id].reshape(3, 3).copy()
    target = body_pos + body_R @ np.asarray(spec["target_offset_local"], dtype=np.float64)
    eye = body_pos + body_R @ np.asarray(spec["eye_offset_local"], dtype=np.float64)
    up = body_R @ np.asarray(spec["up_axis_local"], dtype=np.float64)
    if mode == "body_world_overhead":
        eye = target + np.asarray(spec["eye_offset_world"], dtype=np.float64)
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    if mode == "body_to_body":
        target_name = spec.get("target_body")
        target_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_name)
        if target_id < 0:
            raise ValueError(f"Missing target body {target_name!r}")
        target = data.xpos[target_id].copy()
    return eye, target, up


def set_camera_pose(camera_obj: Any, eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray) -> None:
    from mathutils import Matrix

    eye = np.asarray(eye, dtype=np.float64).reshape(3)
    target = np.asarray(target, dtype=np.float64).reshape(3)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    up = np.asarray(up_hint, dtype=np.float64).reshape(3)
    up = up - forward * float(np.dot(up, forward))
    if float(np.linalg.norm(up)) < 1e-8:
        up = np.array([0.0, 0.0, 1.0], dtype=np.float64)
        up = up - forward * float(np.dot(up, forward))
    up /= max(float(np.linalg.norm(up)), 1e-12)
    right = np.cross(forward, up)
    right /= max(float(np.linalg.norm(right)), 1e-12)
    up = np.cross(right, forward)
    rotation = np.column_stack([right, up, -forward])
    camera_obj.location = tuple(float(v) for v in eye)
    camera_obj.rotation_mode = "QUATERNION"
    camera_obj.rotation_quaternion = Matrix(rotation.tolist()).to_quaternion()


def animate_camera(model: Any, qpos: np.ndarray, frame_indices: np.ndarray, spec: dict[str, Any], *, fovy_deg: float = 45.0) -> Any:
    import bpy
    import mujoco

    camera_data = bpy.data.cameras.new(safe_name(str(spec["name"]), "camera"))
    camera_data.lens_unit = "FOV"
    camera_data.angle = math.radians(float(fovy_deg))
    camera_obj = bpy.data.objects.new(camera_data.name, camera_data)
    bpy.context.scene.collection.objects.link(camera_obj)
    bpy.context.scene.camera = camera_obj
    data = mujoco.MjData(model)
    for out_frame, source_frame in enumerate(frame_indices.astype(int).tolist()):
        bpy.context.scene.frame_set(out_frame)
        data.qpos[:] = qpos[source_frame]
        mujoco.mj_forward(model, data)
        eye, target, up = camera_eye_target_up(model, data, spec)
        set_camera_pose(camera_obj, eye, target, up)
        camera_obj.keyframe_insert(data_path="location", frame=out_frame)
        camera_obj.keyframe_insert(data_path="rotation_quaternion", frame=out_frame)
    return camera_obj


def configure_render(*, width: int, height: int, fps: int, engine: str, samples: int, output_path: str) -> dict[str, str]:
    import bpy

    scene = bpy.context.scene
    scene.render.resolution_x = int(width)
    scene.render.resolution_y = int(height)
    scene.render.fps = int(fps)
    try:
        scene.render.engine = engine
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE" if "BLENDER_EEVEE" in {item.identifier for item in scene.render.bl_rna.properties["engine"].enum_items} else "CYCLES"
    if scene.render.engine == "CYCLES":
        scene.cycles.samples = int(samples)
    scene.render.film_transparent = False
    output = Path(output_path)
    try:
        scene.render.image_settings.file_format = "FFMPEG"
        scene.render.ffmpeg.format = "MPEG4"
        scene.render.ffmpeg.codec = "H264"
        scene.render.ffmpeg.constant_rate_factor = "MEDIUM"
        scene.render.filepath = str(output)
        mode = {"mode": "ffmpeg", "output_video": str(output)}
    except TypeError:
        frames_dir = output.with_suffix("")
        frames_dir.mkdir(parents=True, exist_ok=True)
        for old_frame in frames_dir.glob("*.png"):
            old_frame.unlink()
        scene.render.image_settings.file_format = "PNG"
        scene.render.filepath = str(frames_dir / "frame_")
        mode = {"mode": "png_sequence", "frames_dir": str(frames_dir), "output_video": str(output)}
    world = scene.world or bpy.data.worlds.new("World")
    scene.world = world
    world.color = (0.02, 0.02, 0.025)
    return mode


def add_default_lighting() -> None:
    import bpy

    light_data = bpy.data.lights.new("key_area", type="AREA")
    light_data.energy = 600.0
    light_data.size = 4.0
    light = bpy.data.objects.new("key_area", light_data)
    bpy.context.scene.collection.objects.link(light)
    light.location = (0.3, -0.6, 1.8)
    fill_data = bpy.data.lights.new("fill_area", type="AREA")
    fill_data.energy = 120.0
    fill_data.size = 3.0
    fill = bpy.data.objects.new("fill_area", fill_data)
    bpy.context.scene.collection.objects.link(fill)
    fill.location = (-0.8, 0.4, 1.1)
