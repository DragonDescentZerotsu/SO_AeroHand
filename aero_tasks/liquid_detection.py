"""Tip/container geometric tests for semantic liquid transfer."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from aero_tasks.liquid import ContainerState, LiquidSurface, _normalize


@dataclass(frozen=True)
class CircularContainerRegion:
    """Simple cylindrical acceptance region for source tubes, wells, reservoirs."""

    name: str
    center_world: tuple[float, float, float]
    radius_m: float
    bottom_z_m: float = 0.0
    top_z_m: float = 0.05
    rotation_world: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    @property
    def center(self) -> np.ndarray:
        return np.asarray(self.center_world, dtype=np.float64)

    @property
    def rotation(self) -> np.ndarray:
        return np.asarray(self.rotation_world, dtype=np.float64).reshape(3, 3)

    def local_point(self, point_world: np.ndarray) -> np.ndarray:
        return self.rotation.T @ (np.asarray(point_world, dtype=np.float64).reshape(3) - self.center)

    def contains_lateral(self, point_world: np.ndarray) -> tuple[bool, float, float]:
        local = self.local_point(point_world)
        radial = float(np.linalg.norm(local[:2]))
        in_height = float(self.bottom_z_m) <= float(local[2]) <= float(self.top_z_m)
        return bool(radial <= float(self.radius_m) and in_height), radial, float(local[2])


@dataclass(frozen=True)
class TipContainerHit:
    container_name: str
    tip_in_container: bool
    tip_in_liquid: bool
    signed_depth_m: float
    radial_m: float
    local_z_m: float
    surface: LiquidSurface

    def as_json(self) -> dict[str, object]:
        return {
            "container_name": self.container_name,
            "tip_in_container": self.tip_in_container,
            "tip_in_liquid": self.tip_in_liquid,
            "signed_depth_m": float(self.signed_depth_m),
            "radial_m": float(self.radial_m),
            "local_z_m": float(self.local_z_m),
            "surface": self.surface.as_json(),
        }


def detect_tip_in_circular_container(
    tip_site_world: np.ndarray,
    *,
    container: ContainerState,
    region: CircularContainerRegion,
    gravity_world: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=np.float64),
    acceleration_world: np.ndarray = np.zeros(3, dtype=np.float64),
) -> TipContainerHit:
    """Classify whether a tip site is inside a container and below its surface."""

    surface = container.surface(
        gravity_world=gravity_world,
        acceleration_world=acceleration_world,
        container_pos_world=region.center,
        container_rot_world=region.rotation,
    )
    tip = np.asarray(tip_site_world, dtype=np.float64).reshape(3)
    tip_in_container, radial_m, local_z_m = region.contains_lateral(tip)
    if surface.center_world is None:
        surface_point = region.center + region.rotation @ np.array([0.0, 0.0, surface.height_m], dtype=np.float64)
    else:
        surface_point = np.asarray(surface.center_world, dtype=np.float64)
    normal = _normalize(np.asarray(surface.normal_world, dtype=np.float64))
    signed_depth = float(np.dot(surface_point - tip, normal))
    return TipContainerHit(
        container_name=container.name,
        tip_in_container=tip_in_container,
        tip_in_liquid=bool(tip_in_container and signed_depth > 0.0 and container.volume_ul > 0.0),
        signed_depth_m=signed_depth,
        radial_m=radial_m,
        local_z_m=local_z_m,
        surface=surface,
    )
