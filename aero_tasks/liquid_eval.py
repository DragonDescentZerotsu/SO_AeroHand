"""Hidden-state biological correctness checks for liquid-transfer tasks."""

from __future__ import annotations

from dataclasses import dataclass

from aero_tasks.liquid import ContainerState, PipetteTipState


@dataclass(frozen=True)
class WellVolumeExpectation:
    container_name: str
    sample_id: str | None
    min_volume_ul: float
    max_volume_ul: float


@dataclass(frozen=True)
class BiologicalCorrectnessScore:
    well_ok: bool
    sample_ok: bool
    tip_hygiene_ok: bool
    no_contamination: bool
    volume_ok: bool

    @property
    def score(self) -> float:
        return float(
            self.well_ok
            and self.sample_ok
            and self.tip_hygiene_ok
            and self.no_contamination
            and self.volume_ok
        )

    def as_json(self) -> dict[str, object]:
        return {
            "well_ok": bool(self.well_ok),
            "sample_ok": bool(self.sample_ok),
            "tip_hygiene_ok": bool(self.tip_hygiene_ok),
            "no_contamination": bool(self.no_contamination),
            "volume_ok": bool(self.volume_ok),
            "score": float(self.score),
        }


def evaluate_bcs(
    containers: dict[str, ContainerState],
    *,
    expectations: list[WellVolumeExpectation],
    tip: PipetteTipState | None = None,
    allow_used_clean_tip: bool = True,
) -> BiologicalCorrectnessScore:
    """Evaluate the first BioDexBench BCS definition from hidden state."""

    well_ok = all(expectation.container_name in containers for expectation in expectations)

    sample_checks: list[bool] = []
    volume_checks: list[bool] = []
    contamination_checks: list[bool] = []
    for expectation in expectations:
        container = containers.get(expectation.container_name)
        if container is None:
            sample_checks.append(False)
            volume_checks.append(False)
            contamination_checks.append(False)
            continue
        sample_checks.append(container.sample_id == expectation.sample_id)
        volume_checks.append(float(expectation.min_volume_ul) <= float(container.volume_ul) <= float(expectation.max_volume_ul))
        contamination_checks.append(len(container.contaminated_by) == 0)

    if tip is None:
        tip_hygiene_ok = True
        tip_uncontaminated = True
    else:
        allowed_states = {"clean", "used"} if allow_used_clean_tip else {"clean"}
        tip_hygiene_ok = tip.clean_state in allowed_states
        tip_uncontaminated = len(tip.contaminated_by) == 0 and tip.clean_state != "contaminated"

    return BiologicalCorrectnessScore(
        well_ok=well_ok,
        sample_ok=all(sample_checks),
        tip_hygiene_ok=tip_hygiene_ok,
        no_contamination=all(contamination_checks) and tip_uncontaminated,
        volume_ok=all(volume_checks),
    )
