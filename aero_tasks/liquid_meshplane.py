"""Optional meshplane-backed liquid geometry for real container meshes."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys
from typing import Any

import numpy as np

from aero_tasks.liquid import ContainerGeometry, LiquidSurface, M3_TO_UL, UL_TO_M3, _normalize


def _load_autobio_meshplane(autobio_root: Path | None = None) -> tuple[type[Any], type[Any]]:
    if autobio_root is not None:
        root = Path(autobio_root).expanduser()
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
    try:
        from liquid import ContainerDefinition  # type: ignore
        from meshplane import MeshPlane  # type: ignore
    except ImportError as exc:  # pragma: no cover - depends on optional AutoBio checkout
        raise ImportError(
            "MeshPlaneGeometry requires AutoBio's liquid.py and meshplane module. "
            "Pass autobio_root pointing at the AutoBio/autobio directory or add it to PYTHONPATH."
        ) from exc
    return ContainerDefinition, MeshPlane


@dataclass(frozen=True, kw_only=True)
class MeshPlaneGeometry(ContainerGeometry):
    """Container geometry whose liquid surface is solved from an interior mesh.

    This backend mirrors AutoBio's meshplane approach: for a requested liquid
    volume, solve the clipping plane distance whose enclosed interior volume
    matches that volume. It is intentionally optional so the core liquid module
    remains lightweight for simple wells and reservoirs.
    """

    definition: Any
    meshplane: Any
    previous_distance_m: float | None = None

    @classmethod
    def from_trimesh(
        cls,
        mesh: Any,
        *,
        autobio_root: Path | None = None,
        split_top: bool = True,
        split_bottom: bool = False,
        opening: str = "top",
    ) -> "MeshPlaneGeometry":
        ContainerDefinition, MeshPlane = _load_autobio_meshplane(autobio_root)
        definition = ContainerDefinition.from_object_mesh(
            mesh,
            split_top=split_top,
            split_bottom=split_bottom,
            opening=opening,
        )
        return cls(definition=definition, meshplane=MeshPlane(definition.interior))

    @property
    def capacity_ul(self) -> float:
        return float(getattr(self.definition.interior, "volume")) * M3_TO_UL

    def volume_to_height_m(self, volume_ul: float) -> float:
        surface = self.surface(volume_ul)
        return surface.height_m

    def surface(
        self,
        volume_ul: float,
        *,
        gravity_world: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=np.float64),
        acceleration_world: np.ndarray = np.zeros(3, dtype=np.float64),
        container_pos_world: np.ndarray | None = None,
        container_rot_world: np.ndarray | None = None,
    ) -> LiquidSurface:
        rot = np.eye(3, dtype=np.float64) if container_rot_world is None else np.asarray(container_rot_world, dtype=np.float64).reshape(3, 3)
        pos = np.zeros(3, dtype=np.float64) if container_pos_world is None else np.asarray(container_pos_world, dtype=np.float64).reshape(3)

        effective_up_world = -(np.asarray(gravity_world, dtype=np.float64).reshape(3) - np.asarray(acceleration_world, dtype=np.float64).reshape(3))
        normal_world = _normalize(effective_up_world)
        normal_local = rot.T @ normal_world
        self.meshplane.set_plane_normal(*normal_local)

        if self.previous_distance_m is None:
            low, high, _ = self.meshplane.get_plane_distance_range()
            object.__setattr__(self, "previous_distance_m", 0.5 * (float(low) + float(high)))
        distance = float(self.meshplane.solve_plane_distance(max(0.0, float(volume_ul)) * UL_TO_M3, self.previous_distance_m))
        object.__setattr__(self, "previous_distance_m", distance)
        result = self.meshplane.calculate_plane(distance)

        center_world = pos + rot @ np.asarray(result.center, dtype=np.float64)
        frame_world = rot @ np.asarray(result.frame, dtype=np.float64)
        return LiquidSurface(
            height_m=float(np.asarray(result.center, dtype=np.float64)[2]),
            normal_world=tuple(normal_world.tolist()),
            center_world=tuple(center_world.tolist()),
            frame_world=tuple(tuple(float(v) for v in row) for row in frame_world.tolist()),
            half_width_m=float(result.half_width),
            half_height_m=float(result.half_height),
            distance_m=distance,
        )
