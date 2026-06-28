"""Blender liquid overlays driven by wet-state JSONL logs."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import numpy as np

from aero_tasks.blender_scene import make_glass_material, make_mesh, safe_name, unit_disk_mesh


TIP_LIQUID_HEIGHT_M = 0.0325
TIP_LIQUID_BOTTOM_RADIUS_M = 0.00012
TIP_LIQUID_TOP_RADIUS_M = 0.00098
LIQUID_SURFACE_RENDER_OFFSET_M = 0.00005


@dataclass(frozen=True)
class WetStateSeries:
    records: list[dict[str, Any]]
    by_frame_index: dict[int, dict[str, Any]]

    @classmethod
    def load(cls, path: Path) -> "WetStateSeries":
        records: list[dict[str, Any]] = []
        by_frame: dict[int, dict[str, Any]] = {}
        with Path(path).open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                record = json.loads(line)
                records.append(record)
                if "frame_index" in record:
                    by_frame[int(record["frame_index"])] = record
        return cls(records=records, by_frame_index=by_frame)

    def record_for(self, source_frame: int, out_frame: int) -> dict[str, Any] | None:
        if source_frame in self.by_frame_index:
            return self.by_frame_index[source_frame]
        if out_frame in self.by_frame_index:
            return self.by_frame_index[out_frame]
        if 0 <= out_frame < len(self.records):
            return self.records[out_frame]
        if 0 <= source_frame < len(self.records):
            return self.records[source_frame]
        return None


def normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.array([0.0, 0.0, 1.0], dtype=np.float64)
    return vector / norm


def frame_from_normal(normal: np.ndarray) -> np.ndarray:
    z_axis = normalize(normal)
    helper = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    if abs(float(np.dot(helper, z_axis))) > 0.95:
        helper = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    x_axis = normalize(np.cross(helper, z_axis))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.column_stack([x_axis, y_axis, z_axis])


def frustum_mesh(
    name: str,
    *,
    lower_radius_m: float,
    upper_radius_m: float,
    height_m: float,
    sides: int = 24,
    caps: bool = False,
) -> Any:
    vertices: list[tuple[float, float, float]] = []
    z0 = -0.5 * float(height_m)
    z1 = 0.5 * float(height_m)
    for z, radius in ((z0, lower_radius_m), (z1, upper_radius_m)):
        for index in range(sides):
            theta = 2.0 * np.pi * index / sides
            vertices.append((float(radius * np.cos(theta)), float(radius * np.sin(theta)), z))
    bottom_center = len(vertices)
    vertices.append((0.0, 0.0, z0))
    top_center = len(vertices)
    vertices.append((0.0, 0.0, z1))

    faces: list[tuple[int, int, int]] = []
    for index in range(sides):
        nxt = (index + 1) % sides
        faces.append((index, nxt, sides + nxt))
        faces.append((index, sides + nxt, sides + index))
        if caps:
            faces.append((bottom_center, nxt, index))
            faces.append((top_center, sides + index, sides + nxt))
    return make_mesh(name, vertices, faces)


def tip_liquid_geometry(record: dict[str, Any]) -> dict[str, float | int]:
    raw = record.get("tip_liquid_geometry")
    geometry = raw if isinstance(raw, dict) else {}
    return {
        "height_m": float(geometry.get("height_m", TIP_LIQUID_HEIGHT_M)),
        "bottom_radius_m": float(geometry.get("bottom_radius_m", TIP_LIQUID_BOTTOM_RADIUS_M)),
        "top_radius_m": float(geometry.get("top_radius_m", TIP_LIQUID_TOP_RADIUS_M)),
    }


def tip_radius_at_height(height_m: float, geometry: dict[str, float | int]) -> float:
    height = max(float(geometry["height_m"]), 1e-6)
    bottom_radius = max(float(geometry["bottom_radius_m"]), 1e-7)
    top_radius = max(float(geometry["top_radius_m"]), bottom_radius)
    alpha = max(0.0, min(1.0, float(height_m) / height))
    return bottom_radius + (top_radius - bottom_radius) * alpha


def tip_volume_weight(height_m: float, geometry: dict[str, float | int]) -> float:
    bottom_radius = max(float(geometry["bottom_radius_m"]), 1e-7)
    top_radius = tip_radius_at_height(height_m, geometry)
    return max(0.0, float(height_m)) * (bottom_radius**2 + bottom_radius * top_radius + top_radius**2)


def tip_fill_height(fill_fraction: float, geometry: dict[str, float | int]) -> float:
    fill = max(0.0, min(1.0, float(fill_fraction)))
    if fill <= 1e-9:
        return 0.0
    height = max(float(geometry["height_m"]), 1e-6)
    target = fill * tip_volume_weight(height, geometry)
    lo = 0.0
    hi = height
    for _ in range(32):
        mid = 0.5 * (lo + hi)
        if tip_volume_weight(mid, geometry) < target:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def liquid_top_in_tip(fill_fraction: float, geometry: dict[str, float | int]) -> tuple[float, float] | None:
    top_z = tip_fill_height(fill_fraction, geometry)
    if top_z <= 1e-9:
        return None
    return top_z, tip_radius_at_height(top_z, geometry)


def surface_specs(record: dict[str, Any]) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = []
    if "surface_center_world" in record:
        specs.append(
            {
                "name": "surface",
                "center_world": record["surface_center_world"],
                "normal_world": record.get("surface_normal_world", [0.0, 0.0, 1.0]),
                "frame_world": record.get("surface_frame_world"),
                "half_width_m": record.get("surface_half_width", record.get("surface_half_width_m", 0.01)),
                "half_height_m": record.get("surface_half_height", record.get("surface_half_height_m", 0.01)),
                "color": record.get("liquid_color", [0.35, 0.75, 1.0, 0.45]),
            }
        )
    for key in ("source_hit", "target_hit"):
        hit = record.get(key)
        if not isinstance(hit, dict):
            continue
        surface = hit.get("surface")
        if not isinstance(surface, dict) or surface.get("center_world") is None:
            continue
        specs.append(
            {
                "name": key.removesuffix("_hit"),
                "center_world": surface["center_world"],
                "normal_world": surface.get("normal_world", [0.0, 0.0, 1.0]),
                "frame_world": surface.get("frame_world"),
                "half_width_m": surface.get("half_width_m") or 0.004,
                "half_height_m": surface.get("half_height_m") or 0.004,
                "color": record.get(key.removesuffix("_hit"), {}).get("liquid_color", [0.35, 0.75, 1.0, 0.42]),
                "bulk_mesh_world": surface.get("bulk_mesh_world"),
            }
        )
    return specs


def tip_liquid_spec(record: dict[str, Any]) -> dict[str, Any] | None:
    tip = record.get("tip")
    tip_site = record.get("tip_site_world")
    if tip_site is None and isinstance(record.get("source_hit"), dict):
        # detection logs store tip geometry under each hit, but not always the raw site.
        tip_site = record.get("source_hit", {}).get("tip_site_world")
    if not isinstance(tip, dict) or tip_site is None:
        return None
    volume = float(tip.get("volume_ul", 0.0))
    capacity = max(float(tip.get("capacity_ul", 1.0)), 1e-9)
    if volume <= 1e-6:
        return None
    return {
        "tip_site_world": tip_site,
        "tip_axis_world": record.get("tip_axis_world", [0.0, 0.0, 1.0]),
        "fill_fraction": max(0.0, min(1.0, volume / capacity)),
        "geometry": tip_liquid_geometry(record),
        "color": tip.get("liquid_color", [0.35, 0.75, 1.0, 0.45]),
    }


class BlenderLiquidOverlay:
    """Small Blender liquid overlay animated from wet-state records."""

    def __init__(self) -> None:
        import bpy

        collection = bpy.data.collections.new("LiquidOverlay")
        bpy.context.scene.collection.children.link(collection)
        self.collection = collection
        self.surface_mesh = make_mesh("liquid_surface_disk", *unit_disk_mesh())
        self.surface_objects: dict[str, Any] = {}
        self.tip_liquid_objects: list[Any] = []
        self.bulk_objects: list[Any] = []
        self.materials: dict[tuple[float, float, float, float], Any] = {}

    def material(self, rgba: list[float] | tuple[float, ...]) -> Any:
        rgba_tuple = tuple(float(v) for v in rgba)
        if len(rgba_tuple) == 3:
            rgba_tuple = (*rgba_tuple, 0.45)
        if rgba_tuple not in self.materials:
            mat = make_glass_material(f"liquid_{len(self.materials):03d}", np.asarray(rgba_tuple), ior=1.333)
            self.materials[rgba_tuple] = mat
        return self.materials[rgba_tuple]

    def ensure_surface_object(self, name: str, color: list[float] | tuple[float, ...]) -> Any:
        import bpy

        if name not in self.surface_objects:
            obj = bpy.data.objects.new(safe_name(f"liquid_surface_{name}", name), self.surface_mesh)
            self.collection.objects.link(obj)
            obj.data.materials[0] = self.material(color)
            self.surface_objects[name] = obj
        return self.surface_objects[name]

    def hide_object(self, obj: Any, frame: int) -> None:
        obj.hide_viewport = True
        obj.hide_render = True
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        obj.keyframe_insert(data_path="hide_render", frame=frame)

    def show_object(self, obj: Any, frame: int) -> None:
        obj.hide_viewport = False
        obj.hide_render = False
        obj.keyframe_insert(data_path="hide_viewport", frame=frame)
        obj.keyframe_insert(data_path="hide_render", frame=frame)

    def set_bulk_mesh(self, name: str, spec: dict[str, Any], frame: int) -> None:
        import bpy

        bulk = spec.get("bulk_mesh_world")
        if not isinstance(bulk, dict):
            return
        vertices = bulk.get("vertices")
        faces = bulk.get("faces")
        if not vertices or not faces:
            return
        mesh = make_mesh(
            safe_name(f"liquid_bulk_mesh_{name}_{frame:04d}", "liquid_bulk_mesh"),
            [tuple(float(v) for v in vertex) for vertex in vertices],
            [tuple(int(i) for i in face) for face in faces],
        )
        obj = bpy.data.objects.new(safe_name(f"liquid_bulk_{name}_{frame:04d}", "liquid_bulk"), mesh)
        self.collection.objects.link(obj)
        obj.data.materials[0] = self.material(spec.get("color", [0.35, 0.75, 1.0, 0.42]))
        self.bulk_objects.append(obj)
        if frame > 0:
            self.hide_object(obj, frame - 1)
        self.show_object(obj, frame)
        self.hide_object(obj, frame + 1)

    def set_surface(self, obj: Any, spec: dict[str, Any], frame: int) -> None:
        from mathutils import Matrix

        normal = normalize(np.asarray(spec["normal_world"], dtype=np.float64))
        center = np.asarray(spec["center_world"], dtype=np.float64).reshape(3) + normal * LIQUID_SURFACE_RENDER_OFFSET_M
        if spec.get("frame_world") is not None:
            basis = np.asarray(spec["frame_world"], dtype=np.float64).reshape(3, 3)
        else:
            basis = frame_from_normal(np.asarray(spec["normal_world"], dtype=np.float64))
        obj.location = tuple(float(v) for v in center)
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = Matrix(basis.tolist()).to_quaternion()
        obj.scale = (float(spec["half_width_m"]), float(spec["half_height_m"]), 1.0)
        self.show_object(obj, frame)
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        obj.keyframe_insert(data_path="scale", frame=frame)

    def hide_tip_objects(self, frame: int) -> None:
        for obj in self.tip_liquid_objects:
            self.hide_object(obj, frame)

    def set_tip_liquid(self, spec: dict[str, Any], frame: int) -> None:
        from mathutils import Matrix
        import bpy

        site = np.asarray(spec["tip_site_world"], dtype=np.float64).reshape(3)
        axis = normalize(np.asarray(spec.get("tip_axis_world", [0.0, 0.0, 1.0]), dtype=np.float64))
        basis_quat = Matrix(frame_from_normal(axis).tolist()).to_quaternion()
        geometry = spec.get("geometry")
        tip_geometry = geometry if isinstance(geometry, dict) else tip_liquid_geometry({})
        top = liquid_top_in_tip(float(spec["fill_fraction"]), tip_geometry)
        if top is None:
            return
        fill_height, top_radius = top
        mesh = frustum_mesh(
            f"liquid_tip_mesh_{frame:04d}",
            lower_radius_m=max(float(tip_geometry["bottom_radius_m"]), 1e-7),
            upper_radius_m=top_radius,
            height_m=fill_height,
            caps=True,
        )
        obj = bpy.data.objects.new(f"liquid_tip_{frame:04d}", mesh)
        self.collection.objects.link(obj)
        obj.data.materials[0] = self.material(spec["color"])
        obj.location = tuple(float(v) for v in site + axis * (0.5 * fill_height))
        obj.rotation_mode = "QUATERNION"
        obj.rotation_quaternion = basis_quat
        self.tip_liquid_objects.append(obj)
        if frame > 0:
            self.hide_object(obj, frame - 1)
        self.show_object(obj, frame)
        obj.keyframe_insert(data_path="location", frame=frame)
        obj.keyframe_insert(data_path="rotation_quaternion", frame=frame)
        self.hide_object(obj, frame + 1)

    def animate(self, wet_state: WetStateSeries, frame_indices: np.ndarray) -> None:
        import bpy

        for out_frame, source_frame in enumerate(frame_indices.astype(int).tolist()):
            bpy.context.scene.frame_set(out_frame)
            record = wet_state.record_for(source_frame, out_frame)
            if record is None:
                continue
            current_names: set[str] = set()
            for spec in surface_specs(record):
                name = str(spec["name"])
                obj = self.ensure_surface_object(name, spec["color"])
                self.set_surface(obj, spec, out_frame)
                self.set_bulk_mesh(name, spec, out_frame)
                current_names.add(name)
            for name, obj in self.surface_objects.items():
                if name not in current_names:
                    self.hide_object(obj, out_frame)
            tip_spec = tip_liquid_spec(record)
            if tip_spec is None:
                self.hide_tip_objects(out_frame)
            else:
                self.set_tip_liquid(tip_spec, out_frame)
