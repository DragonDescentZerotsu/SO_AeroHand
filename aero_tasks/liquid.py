"""Semantic liquid transfer helpers for wet-lab task simulation.

This module intentionally models liquid as a bookkeeping layer plus simple
geometry, not as CFD. MuJoCo remains responsible for contact and tool motion;
this layer tracks sample identity, volume, contamination, and visual proxies.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np


UL_TO_M3 = 1e-9
M3_TO_UL = 1e9
EPS_UL = 1e-9

TipCleanState = Literal["clean", "used", "contaminated"]
LiquidEventKind = Literal["aspirate", "air_aspirate", "dispense", "spill", "touch_forbidden_surface"]


def _clamp(value: float, lo: float, hi: float) -> float:
    return min(max(float(value), float(lo)), float(hi))


def _normalize(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64).reshape(3)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        raise ValueError("Cannot normalize a near-zero vector")
    return vector / norm


def mixture_id(first: str | None, second: str | None) -> str | None:
    """Return a stable sample id for combining two liquid identities."""

    if first in (None, ""):
        return second
    if second in (None, ""):
        return first
    if first == second:
        return first
    parts: set[str] = set()
    for value in (first, second):
        if value.startswith("mix(") and value.endswith(")"):
            parts.update(part for part in value[4:-1].split("+") if part)
        else:
            parts.add(value)
    return "mix(" + "+".join(sorted(parts)) + ")"


def mix_color(first: tuple[float, float, float, float], second: tuple[float, float, float, float], alpha: float) -> tuple[float, float, float, float]:
    alpha = _clamp(alpha, 0.0, 1.0)
    a = np.asarray(first, dtype=np.float64)
    b = np.asarray(second, dtype=np.float64)
    return tuple((1.0 - alpha) * a + alpha * b)


@dataclass(frozen=True)
class LiquidSurface:
    """A simple free-surface proxy for a container."""

    height_m: float
    normal_world: tuple[float, float, float]
    center_world: tuple[float, float, float] | None = None
    frame_world: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None = None
    half_width_m: float | None = None
    half_height_m: float | None = None
    distance_m: float | None = None

    def as_json(self) -> dict[str, object]:
        return {
            "height_m": float(self.height_m),
            "normal_world": list(self.normal_world),
            "center_world": None if self.center_world is None else list(self.center_world),
            "frame_world": None if self.frame_world is None else [list(row) for row in self.frame_world],
            "half_width_m": self.half_width_m,
            "half_height_m": self.half_height_m,
            "distance_m": self.distance_m,
        }


@dataclass(frozen=True)
class FrustumSegment:
    """One local-frame tapered segment for pipette liquid visualization."""

    lower_radius_m: float
    upper_radius_m: float
    height_m: float

    def volume_weight(self) -> float:
        if self.lower_radius_m < 0.0 or self.upper_radius_m < 0.0 or self.height_m <= 0.0:
            raise ValueError("Frustum radii must be nonnegative and height must be positive")
        return float(self.height_m) * (
            float(self.lower_radius_m) ** 2
            + float(self.lower_radius_m) * float(self.upper_radius_m)
            + float(self.upper_radius_m) ** 2
        )

    def partial_height_fraction_for_volume_fraction(self, volume_fraction: float) -> float:
        """Invert the truncated-cone volume curve on [0, 1]."""

        target = _clamp(volume_fraction, 0.0, 1.0) * self.volume_weight()
        if target <= 0.0:
            return 0.0
        full = self.volume_weight()
        if target >= full:
            return 1.0
        lo = 0.0
        hi = 1.0
        for _ in range(32):
            mid = 0.5 * (lo + hi)
            radius_at_mid = float(self.lower_radius_m) + mid * (float(self.upper_radius_m) - float(self.lower_radius_m))
            partial = float(self.height_m) * mid * (
                float(self.lower_radius_m) ** 2
                + float(self.lower_radius_m) * radius_at_mid
                + radius_at_mid**2
            )
            if partial < target:
                lo = mid
            else:
                hi = mid
        return 0.5 * (lo + hi)


@dataclass(frozen=True)
class ContainerGeometry:
    """Base class for monotonic volume-to-height approximations."""

    bottom_z_m: float = 0.0

    def volume_to_height_m(self, volume_ul: float) -> float:
        raise NotImplementedError

    def surface(
        self,
        volume_ul: float,
        *,
        gravity_world: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=np.float64),
        acceleration_world: np.ndarray = np.zeros(3, dtype=np.float64),
        container_pos_world: np.ndarray | None = None,
        container_rot_world: np.ndarray | None = None,
    ) -> LiquidSurface:
        effective_down = np.asarray(gravity_world, dtype=np.float64).reshape(3) - np.asarray(acceleration_world, dtype=np.float64).reshape(3)
        normal_world = -_normalize(effective_down)
        height_m = self.volume_to_height_m(volume_ul)
        center_world = None
        frame_world = None
        if container_pos_world is not None:
            pos = np.asarray(container_pos_world, dtype=np.float64).reshape(3)
            rot = np.eye(3, dtype=np.float64) if container_rot_world is None else np.asarray(container_rot_world, dtype=np.float64).reshape(3, 3)
            center_world = tuple((pos + rot @ np.array([0.0, 0.0, height_m], dtype=np.float64)).tolist())
            frame_world = tuple(tuple(float(v) for v in row) for row in rot.tolist())
        return LiquidSurface(
            height_m=height_m,
            normal_world=tuple(normal_world.tolist()),
            center_world=center_world,
            frame_world=frame_world,
        )


@dataclass(frozen=True)
class ConstantAreaGeometry(ContainerGeometry):
    """Reservoir/well approximation with constant horizontal area."""

    area_m2: float = 1.0

    def volume_to_height_m(self, volume_ul: float) -> float:
        if self.area_m2 <= 0.0:
            raise ValueError("area_m2 must be positive")
        return float(self.bottom_z_m + max(0.0, float(volume_ul)) * UL_TO_M3 / self.area_m2)


@dataclass(frozen=True)
class CylindricalGeometry(ConstantAreaGeometry):
    radius_m: float = 0.001

    def __post_init__(self) -> None:
        if self.radius_m <= 0.0:
            raise ValueError("radius_m must be positive")
        object.__setattr__(self, "area_m2", float(np.pi * self.radius_m * self.radius_m))


@dataclass(frozen=True)
class ConicalCylindricalGeometry(ContainerGeometry):
    """Piecewise centrifuge-tube approximation: cone bottom plus cylinder body."""

    cone_height_m: float = 0.012
    cylinder_radius_m: float = 0.014

    def volume_to_height_m(self, volume_ul: float) -> float:
        if self.cone_height_m <= 0.0 or self.cylinder_radius_m <= 0.0:
            raise ValueError("cone_height_m and cylinder_radius_m must be positive")
        volume_m3 = max(0.0, float(volume_ul)) * UL_TO_M3
        radius = float(self.cylinder_radius_m)
        cone_height = float(self.cone_height_m)
        cone_volume = np.pi * radius * radius * cone_height / 3.0
        if volume_m3 <= cone_volume:
            # Cone volume below height h: V = pi * R^2 / (3H^2) * h^3.
            h = (3.0 * cone_height * cone_height * volume_m3 / (np.pi * radius * radius)) ** (1.0 / 3.0)
            return float(self.bottom_z_m + h)
        cylinder_height = (volume_m3 - cone_volume) / (np.pi * radius * radius)
        return float(self.bottom_z_m + cone_height + cylinder_height)


@dataclass
class ContainerState:
    """Hidden semantic liquid state for a source, reservoir, or well."""

    name: str
    geometry: ContainerGeometry
    volume_ul: float = 0.0
    capacity_ul: float = 0.0
    sample_id: str | None = None
    liquid_color: tuple[float, float, float, float] = (0.1, 0.35, 1.0, 0.55)
    contaminated_by: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.volume_ul = _clamp(self.volume_ul, 0.0, self.capacity_ul if self.capacity_ul > 0.0 else float("inf"))
        if self.volume_ul <= EPS_UL:
            self.sample_id = None

    def remove(self, requested_ul: float) -> float:
        transferred = min(max(0.0, float(requested_ul)), self.volume_ul)
        self.volume_ul -= transferred
        if self.volume_ul <= EPS_UL:
            self.volume_ul = 0.0
            self.sample_id = None
        return transferred

    def add(
        self,
        volume_ul: float,
        *,
        sample_id: str | None,
        color: tuple[float, float, float, float] | None = None,
    ) -> float:
        room = max(0.0, float(self.capacity_ul) - float(self.volume_ul)) if self.capacity_ul > 0.0 else float(volume_ul)
        transferred = min(max(0.0, float(volume_ul)), room)
        if transferred <= EPS_UL:
            return 0.0
        old_volume = float(self.volume_ul)
        self.volume_ul += transferred
        old_sample = self.sample_id
        self.sample_id = mixture_id(self.sample_id, sample_id)
        if old_sample not in (None, sample_id) and sample_id is not None:
            self.contaminated_by.add(sample_id)
        if color is not None:
            blend = transferred / max(old_volume + transferred, EPS_UL)
            self.liquid_color = mix_color(self.liquid_color, color, blend)
        return transferred

    def surface(
        self,
        *,
        gravity_world: np.ndarray = np.array([0.0, 0.0, -9.81], dtype=np.float64),
        acceleration_world: np.ndarray = np.zeros(3, dtype=np.float64),
        container_pos_world: np.ndarray | None = None,
        container_rot_world: np.ndarray | None = None,
    ) -> LiquidSurface:
        """Return a quasi-static free surface for larger open containers."""

        return self.geometry.surface(
            self.volume_ul,
            gravity_world=gravity_world,
            acceleration_world=acceleration_world,
            container_pos_world=container_pos_world,
            container_rot_world=container_rot_world,
        )

    def as_json(self) -> dict[str, object]:
        return {
            "name": self.name,
            "volume_ul": float(self.volume_ul),
            "capacity_ul": float(self.capacity_ul),
            "sample_id": self.sample_id,
            "liquid_color": list(self.liquid_color),
            "contaminated_by": sorted(self.contaminated_by),
        }


@dataclass
class PipetteTipState:
    """Hidden semantic state for a detachable pipette tip."""

    capacity_ul: float
    volume_ul: float = 0.0
    sample_id: str | None = None
    liquid_color: tuple[float, float, float, float] = (0.1, 0.35, 1.0, 0.55)
    attached: bool = True
    clean_state: TipCleanState = "clean"
    air_aspirated_ul: float = 0.0
    contaminated_by: set[str] = field(default_factory=set)

    def __post_init__(self) -> None:
        self.volume_ul = _clamp(self.volume_ul, 0.0, self.capacity_ul)
        self.air_aspirated_ul = _clamp(self.air_aspirated_ul, 0.0, self.capacity_ul - self.volume_ul)
        if self.volume_ul <= EPS_UL:
            self.sample_id = None

    def room_ul(self) -> float:
        return max(0.0, float(self.capacity_ul) - float(self.volume_ul) - float(self.air_aspirated_ul))

    def add_liquid(
        self,
        volume_ul: float,
        *,
        sample_id: str | None,
        color: tuple[float, float, float, float],
    ) -> float:
        transferred = min(max(0.0, float(volume_ul)), self.room_ul())
        if transferred <= EPS_UL:
            return 0.0
        old_volume = float(self.volume_ul)
        old_sample = self.sample_id
        self.volume_ul += transferred
        self.sample_id = mixture_id(self.sample_id, sample_id)
        if old_sample not in (None, sample_id) and sample_id is not None:
            self.contaminated_by.add(sample_id)
            self.clean_state = "contaminated"
        elif self.clean_state == "clean":
            self.clean_state = "used"
        blend = transferred / max(old_volume + transferred, EPS_UL)
        self.liquid_color = mix_color(self.liquid_color, color, blend)
        return transferred

    def remove_liquid(self, requested_ul: float) -> float:
        transferred = min(max(0.0, float(requested_ul)), float(self.volume_ul))
        self.volume_ul -= transferred
        if self.volume_ul <= EPS_UL:
            self.volume_ul = 0.0
            self.sample_id = None
        return transferred

    def release_air(self, requested_ul: float) -> float:
        released = min(max(0.0, float(requested_ul)), float(self.air_aspirated_ul))
        self.air_aspirated_ul -= released
        return released

    def mark_forbidden_touch(self, surface_id: str) -> None:
        self.clean_state = "contaminated"
        self.contaminated_by.add(surface_id)

    def liquid_column_segments(self, segment_count: int = 20) -> list[float]:
        """Return per-segment alpha values for a local-frame tip liquid column."""

        if segment_count <= 0:
            raise ValueError("segment_count must be positive")
        return [
            fraction * self.liquid_color[3]
            for fraction in self.liquid_column_segment_fractions([1.0] * segment_count)
        ]

    def liquid_column_segment_fractions(self, segment_capacity_weights: Sequence[float]) -> list[float]:
        """Return bottom-to-top fill fractions for arbitrary tip visual segments.

        ``segment_capacity_weights`` should be proportional to each segment's
        physical volume, for example ``radius**2 * height`` for cylinder
        proxies. This keeps the visible liquid height consistent with volume in
        a tapered tip instead of assuming equal volume per unit length.
        """

        weights = np.asarray(segment_capacity_weights, dtype=np.float64).reshape(-1)
        if weights.size == 0:
            raise ValueError("segment_capacity_weights must not be empty")
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("segment_capacity_weights must be finite and positive")

        segment_capacities_ul = weights / float(np.sum(weights)) * float(self.capacity_ul)
        remaining_ul = _clamp(self.volume_ul, 0.0, self.capacity_ul)
        fractions: list[float] = []
        for capacity_ul in segment_capacities_ul:
            fraction = _clamp(remaining_ul / max(float(capacity_ul), EPS_UL), 0.0, 1.0)
            fractions.append(fraction)
            remaining_ul -= fraction * float(capacity_ul)
        return fractions

    def liquid_column_frustum_height_fractions(self, segments: Sequence[FrustumSegment]) -> list[float]:
        """Return bottom-to-top height fractions for tapered visual segments."""

        if len(segments) == 0:
            raise ValueError("segments must not be empty")
        weights = np.asarray([segment.volume_weight() for segment in segments], dtype=np.float64)
        if np.any(~np.isfinite(weights)) or np.any(weights <= 0.0):
            raise ValueError("segments must have positive finite volume")

        segment_capacities_ul = weights / float(np.sum(weights)) * float(self.capacity_ul)
        remaining_ul = _clamp(self.volume_ul, 0.0, self.capacity_ul)
        fractions: list[float] = []
        for segment, capacity_ul in zip(segments, segment_capacities_ul, strict=True):
            volume_fraction = _clamp(remaining_ul / max(float(capacity_ul), EPS_UL), 0.0, 1.0)
            height_fraction = segment.partial_height_fraction_for_volume_fraction(volume_fraction)
            fractions.append(height_fraction)
            remaining_ul -= volume_fraction * float(capacity_ul)
        return fractions

    def as_json(self) -> dict[str, object]:
        return {
            "attached": bool(self.attached),
            "clean_state": self.clean_state,
            "volume_ul": float(self.volume_ul),
            "capacity_ul": float(self.capacity_ul),
            "sample_id": self.sample_id,
            "liquid_color": list(self.liquid_color),
            "air_aspirated_ul": float(self.air_aspirated_ul),
            "contaminated_by": sorted(self.contaminated_by),
        }


@dataclass(frozen=True)
class PlungerModel:
    """Map a slide-joint button/plunger position to displaced volume."""

    qpos_rest_m: float = 0.0
    qpos_pressed_m: float = -0.008
    stroke_volume_ul: float = 200.0

    def depression(self, qpos_m: float) -> float:
        span = float(self.qpos_pressed_m) - float(self.qpos_rest_m)
        if abs(span) < 1e-12:
            raise ValueError("Plunger qpos range must be nonzero")
        return _clamp((float(qpos_m) - float(self.qpos_rest_m)) / span, 0.0, 1.0)


@dataclass(frozen=True)
class LiquidEvent:
    kind: LiquidEventKind
    volume_ul: float
    source: str | None = None
    target: str | None = None
    sample_id: str | None = None
    note: str = ""

    def as_json(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "volume_ul": float(self.volume_ul),
            "source": self.source,
            "target": self.target,
            "sample_id": self.sample_id,
            "note": self.note,
        }


@dataclass
class PipetteLiquidController:
    """Stateful aspirate/dispense controller driven by plunger travel."""

    tip: PipetteTipState
    plunger: PlungerModel
    previous_depression: float = 0.0
    events: list[LiquidEvent] = field(default_factory=list)

    @classmethod
    def from_initial_qpos(
        cls,
        *,
        tip: PipetteTipState,
        plunger: PlungerModel,
        qpos_m: float,
    ) -> "PipetteLiquidController":
        return cls(tip=tip, plunger=plunger, previous_depression=plunger.depression(qpos_m))

    def update(
        self,
        qpos_m: float,
        *,
        source: ContainerState | None = None,
        target: ContainerState | None = None,
        tip_in_liquid: bool = False,
        tip_in_target: bool = False,
    ) -> list[LiquidEvent]:
        """Advance liquid state from a new plunger qpos.

        Releasing the plunger (depression decreases) aspirates if the tip is in
        a source liquid; pressing it (depression increases) dispenses into a
        target if available, otherwise spills.
        """

        if not self.tip.attached:
            self.previous_depression = self.plunger.depression(qpos_m)
            return []

        depression = self.plunger.depression(qpos_m)
        delta = depression - self.previous_depression
        self.previous_depression = depression
        requested_ul = abs(delta) * float(self.plunger.stroke_volume_ul)
        if requested_ul <= EPS_UL:
            return []
        if delta < 0.0:
            return self._aspirate(requested_ul, source=source, tip_in_liquid=tip_in_liquid)
        return self._dispense(requested_ul, target=target, tip_in_target=tip_in_target)

    def _aspirate(self, requested_ul: float, *, source: ContainerState | None, tip_in_liquid: bool) -> list[LiquidEvent]:
        if source is None or not tip_in_liquid:
            air = min(requested_ul, self.tip.room_ul())
            self.tip.air_aspirated_ul += air
            event = LiquidEvent("air_aspirate", air, note="tip not in source liquid")
            self.events.append(event)
            return [event]

        source_sample_id = source.sample_id
        source_color = source.liquid_color
        removed = source.remove(min(requested_ul, self.tip.room_ul()))
        added = self.tip.add_liquid(removed, sample_id=source_sample_id, color=source_color)
        event = LiquidEvent("aspirate", added, source=source.name, target="tip", sample_id=self.tip.sample_id)
        self.events.append(event)
        return [event]

    def _dispense(self, requested_ul: float, *, target: ContainerState | None, tip_in_target: bool) -> list[LiquidEvent]:
        sample_id = self.tip.sample_id
        color = self.tip.liquid_color
        removed = self.tip.remove_liquid(requested_ul)
        released_air = self.tip.release_air(max(0.0, requested_ul - removed))
        if target is None or not tip_in_target:
            note = "tip not in target"
            if released_air > EPS_UL:
                note += f"; {released_air:.6g}uL air released"
            event = LiquidEvent("spill", removed, source="tip", sample_id=sample_id, note=note)
            self.events.append(event)
            return [event]

        added = target.add(removed, sample_id=sample_id, color=color)
        if added < removed:
            spill = removed - added
            event = LiquidEvent("dispense", added, source="tip", target=target.name, sample_id=sample_id, note=f"{spill:.6g}uL overflow spilled")
        else:
            note = f"{released_air:.6g}uL air released" if released_air > EPS_UL else ""
            event = LiquidEvent("dispense", added, source="tip", target=target.name, sample_id=sample_id, note=note)
        self.events.append(event)
        return [event]

    def touch_forbidden_surface(self, surface_id: str) -> LiquidEvent:
        self.tip.mark_forbidden_touch(surface_id)
        event = LiquidEvent("touch_forbidden_surface", 0.0, source="tip", target=surface_id, note="tip marked contaminated")
        self.events.append(event)
        return event

    def as_json(self) -> dict[str, object]:
        return {
            "tip": self.tip.as_json(),
            "previous_depression": float(self.previous_depression),
            "events": [event.as_json() for event in self.events],
        }


@dataclass
class WetLabLiquidState:
    """Small registry tying containers and one pipette tip controller together."""

    containers: dict[str, ContainerState]
    pipette: PipetteLiquidController
    frame_index: int = 0

    def container(self, name: str | None) -> ContainerState | None:
        if name is None:
            return None
        try:
            return self.containers[name]
        except KeyError as exc:
            raise KeyError(f"Unknown liquid container {name!r}") from exc

    def update_pipette_from_plunger(
        self,
        qpos_m: float,
        *,
        source_name: str | None = None,
        target_name: str | None = None,
        tip_in_liquid: bool = False,
        tip_in_target: bool = False,
    ) -> list[LiquidEvent]:
        events = self.pipette.update(
            qpos_m,
            source=self.container(source_name),
            target=self.container(target_name),
            tip_in_liquid=tip_in_liquid,
            tip_in_target=tip_in_target,
        )
        self.frame_index += 1
        return events

    def touch_forbidden_surface(self, surface_id: str) -> LiquidEvent:
        return self.pipette.touch_forbidden_surface(surface_id)

    def as_json(self) -> dict[str, object]:
        return {
            "frame_index": int(self.frame_index),
            "containers": {
                name: container.as_json()
                for name, container in sorted(self.containers.items())
            },
            "pipette": self.pipette.as_json(),
        }
