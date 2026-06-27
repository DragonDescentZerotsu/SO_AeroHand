import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.liquid import (
    ConstantAreaGeometry,
    ContainerState,
    PipetteLiquidController,
    PipetteTipState,
    PlungerModel,
    FrustumSegment,
    WetLabLiquidState,
    mixture_id,
)


def test_plunger_release_aspirates_and_lowers_source_surface():
    source = ContainerState(
        name="tube_A",
        geometry=ConstantAreaGeometry(area_m2=1e-4),
        volume_ul=100.0,
        capacity_ul=1000.0,
        sample_id="A",
    )
    tip = PipetteTipState(capacity_ul=200.0)
    plunger = PlungerModel(qpos_rest_m=0.0, qpos_pressed_m=-0.008, stroke_volume_ul=200.0)
    controller = PipetteLiquidController.from_initial_qpos(
        tip=tip,
        plunger=plunger,
        qpos_m=-0.004,
    )

    before_height = source.surface().height_m
    events = controller.update(0.0, source=source, tip_in_liquid=True)

    assert events[0].kind == "aspirate"
    assert np.isclose(events[0].volume_ul, 100.0)
    assert np.isclose(tip.volume_ul, 100.0)
    assert tip.sample_id == "A"
    assert np.isclose(source.volume_ul, 0.0)
    assert source.surface().height_m < before_height


def test_plunger_press_dispenses_to_target_and_updates_mixture():
    target = ContainerState(
        name="well_B1",
        geometry=ConstantAreaGeometry(area_m2=2e-5),
        volume_ul=50.0,
        capacity_ul=300.0,
        sample_id="B",
    )
    tip = PipetteTipState(capacity_ul=200.0, volume_ul=80.0, sample_id="A")
    plunger = PlungerModel(qpos_rest_m=0.0, qpos_pressed_m=-0.008, stroke_volume_ul=200.0)
    controller = PipetteLiquidController.from_initial_qpos(
        tip=tip,
        plunger=plunger,
        qpos_m=0.0,
    )

    events = controller.update(-0.002, target=target, tip_in_target=True)

    assert events[0].kind == "dispense"
    assert np.isclose(events[0].volume_ul, 50.0)
    assert np.isclose(tip.volume_ul, 30.0)
    assert np.isclose(target.volume_ul, 100.0)
    assert target.sample_id == "mix(A+B)"
    assert "A" in target.contaminated_by


def test_air_aspirate_and_spill_are_logged_without_container_transfer():
    tip = PipetteTipState(capacity_ul=100.0, volume_ul=20.0, sample_id="A")
    plunger = PlungerModel(qpos_rest_m=0.0, qpos_pressed_m=-0.01, stroke_volume_ul=100.0)
    controller = PipetteLiquidController.from_initial_qpos(
        tip=tip,
        plunger=plunger,
        qpos_m=-0.005,
    )

    air_events = controller.update(0.0, tip_in_liquid=False)
    assert np.isclose(tip.air_aspirated_ul, 50.0)
    spill_events = controller.update(-0.003, tip_in_target=False)

    assert air_events[0].kind == "air_aspirate"
    assert spill_events[0].kind == "spill"
    assert np.isclose(spill_events[0].volume_ul, 20.0)
    assert np.isclose(tip.volume_ul, 0.0)
    assert np.isclose(tip.air_aspirated_ul, 40.0)


def test_air_aspirate_occupies_tip_capacity():
    source = ContainerState(
        name="source",
        geometry=ConstantAreaGeometry(area_m2=1e-4),
        volume_ul=100.0,
        capacity_ul=500.0,
        sample_id="S",
    )
    tip = PipetteTipState(capacity_ul=100.0)
    plunger = PlungerModel(qpos_rest_m=0.0, qpos_pressed_m=-0.01, stroke_volume_ul=100.0)
    controller = PipetteLiquidController.from_initial_qpos(
        tip=tip,
        plunger=plunger,
        qpos_m=-0.01,
    )

    controller.update(-0.005, tip_in_liquid=False)
    controller.update(0.0, source=source, tip_in_liquid=True)

    assert np.isclose(tip.air_aspirated_ul, 50.0)
    assert np.isclose(tip.volume_ul, 50.0)
    assert np.isclose(source.volume_ul, 50.0)


def test_container_surface_normal_tracks_effective_acceleration():
    container = ContainerState(
        name="reservoir",
        geometry=ConstantAreaGeometry(area_m2=1e-4),
        volume_ul=100.0,
        capacity_ul=1000.0,
        sample_id="A",
    )

    surface = container.surface(
        gravity_world=np.array([0.0, 0.0, -9.81]),
        acceleration_world=np.array([1.0, 0.0, 0.0]),
    )

    normal = np.asarray(surface.normal_world)
    assert normal[0] > 0.0
    assert normal[2] > 0.0
    assert np.isclose(np.linalg.norm(normal), 1.0)


def test_tip_liquid_column_segments_are_local_fill_proxy():
    tip = PipetteTipState(capacity_ul=200.0, volume_ul=55.0)

    alphas = tip.liquid_column_segments(segment_count=4)

    assert len(alphas) == 4
    assert alphas[0] > 0.0
    assert 0.0 < alphas[1] < alphas[0]
    assert alphas[2] == 0.0
    assert alphas[3] == 0.0


def test_tip_liquid_column_segment_fractions_respect_segment_capacity():
    tip = PipetteTipState(capacity_ul=100.0, volume_ul=50.0)

    fractions = tip.liquid_column_segment_fractions([1.0, 3.0])

    assert fractions[0] == 1.0
    assert np.isclose(fractions[1], 1.0 / 3.0)


def test_tip_frustum_height_fraction_inverts_cone_volume():
    tip = PipetteTipState(capacity_ul=100.0, volume_ul=12.5)

    fractions = tip.liquid_column_frustum_height_fractions(
        [FrustumSegment(lower_radius_m=0.0, upper_radius_m=2.0, height_m=1.0)]
    )

    assert np.isclose(fractions[0], 0.5)


def test_tip_frustum_height_fraction_fills_segments_by_volume():
    tip = PipetteTipState(capacity_ul=100.0, volume_ul=50.0)

    fractions = tip.liquid_column_frustum_height_fractions(
        [
            FrustumSegment(lower_radius_m=1.0, upper_radius_m=1.0, height_m=1.0),
            FrustumSegment(lower_radius_m=1.0, upper_radius_m=1.0, height_m=3.0),
        ]
    )

    assert fractions[0] == 1.0
    assert np.isclose(fractions[1], 1.0 / 3.0)


def test_mixture_id_is_stable():
    assert mixture_id("B", "A") == "mix(A+B)"
    assert mixture_id("mix(A+B)", "C") == "mix(A+B+C)"


def test_wet_lab_registry_updates_by_container_name():
    source = ContainerState(
        name="source",
        geometry=ConstantAreaGeometry(area_m2=1e-4),
        volume_ul=200.0,
        capacity_ul=500.0,
        sample_id="S",
    )
    target = ContainerState(
        name="target",
        geometry=ConstantAreaGeometry(area_m2=1e-4),
        volume_ul=0.0,
        capacity_ul=500.0,
    )
    tip = PipetteTipState(capacity_ul=100.0)
    plunger = PlungerModel(qpos_rest_m=0.0, qpos_pressed_m=-0.01, stroke_volume_ul=100.0)
    state = WetLabLiquidState(
        containers={"source": source, "target": target},
        pipette=PipetteLiquidController.from_initial_qpos(tip=tip, plunger=plunger, qpos_m=-0.01),
    )

    state.update_pipette_from_plunger(0.0, source_name="source", tip_in_liquid=True)
    state.update_pipette_from_plunger(-0.005, target_name="target", tip_in_target=True)

    assert np.isclose(state.containers["source"].volume_ul, 100.0)
    assert np.isclose(state.containers["target"].volume_ul, 50.0)
    assert np.isclose(state.pipette.tip.volume_ul, 50.0)
    assert state.as_json()["frame_index"] == 2
