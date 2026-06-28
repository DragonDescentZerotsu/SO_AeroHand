import sys
from pathlib import Path

import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.liquid import ConstantAreaGeometry, ContainerState, PipetteTipState
from aero_tasks.liquid_detection import CircularContainerRegion, detect_tip_in_circular_container
from aero_tasks.liquid_eval import WellVolumeExpectation, evaluate_bcs


def test_circular_container_detection_tracks_surface_depth():
    container = ContainerState(
        name="source",
        geometry=ConstantAreaGeometry(area_m2=1e-4),
        volume_ul=1000.0,
        capacity_ul=2000.0,
        sample_id="A",
    )
    region = CircularContainerRegion(
        name="source",
        center_world=(0.1, -0.2, 0.0),
        radius_m=0.02,
        top_z_m=0.05,
    )

    below_surface = np.array([0.1, -0.2, 0.006], dtype=np.float64)
    above_surface = np.array([0.1, -0.2, 0.020], dtype=np.float64)
    outside_radius = np.array([0.13, -0.2, 0.006], dtype=np.float64)

    hit = detect_tip_in_circular_container(below_surface, container=container, region=region)
    miss_above = detect_tip_in_circular_container(above_surface, container=container, region=region)
    miss_outside = detect_tip_in_circular_container(outside_radius, container=container, region=region)

    assert hit.tip_in_container
    assert hit.tip_in_liquid
    assert hit.signed_depth_m > 0.0
    assert not miss_above.tip_in_liquid
    assert not miss_outside.tip_in_container
    assert not miss_outside.tip_in_liquid


def test_bcs_evaluator_scores_expected_hidden_state():
    containers = {
        "A1": ContainerState(
            name="A1",
            geometry=ConstantAreaGeometry(area_m2=1e-4),
            volume_ul=49.0,
            capacity_ul=200.0,
            sample_id="sample_A",
        )
    }
    tip = PipetteTipState(capacity_ul=200.0, clean_state="used")

    score = evaluate_bcs(
        containers,
        expectations=[
            WellVolumeExpectation(
                container_name="A1",
                sample_id="sample_A",
                min_volume_ul=45.0,
                max_volume_ul=55.0,
            )
        ],
        tip=tip,
    )

    assert score.score == 1.0
    assert score.as_json()["volume_ok"]


def test_bcs_evaluator_fails_wrong_sample_or_volume():
    containers = {
        "A1": ContainerState(
            name="A1",
            geometry=ConstantAreaGeometry(area_m2=1e-4),
            volume_ul=20.0,
            capacity_ul=200.0,
            sample_id="sample_B",
        )
    }

    score = evaluate_bcs(
        containers,
        expectations=[
            WellVolumeExpectation(
                container_name="A1",
                sample_id="sample_A",
                min_volume_ul=45.0,
                max_volume_ul=55.0,
            )
        ],
    )

    assert score.score == 0.0
    assert not score.sample_ok
    assert not score.volume_ok


def test_meshplane_geometry_smoke_when_autobio_checkout_exists():
    autobio_root = Path("/data/tianang/projects/AutoBio/autobio")
    tube_mesh = autobio_root / "assets/container/centrifuge_1500ul_no_lid_vis/visual.obj"
    if not tube_mesh.exists():
        pytest.skip("AutoBio centrifuge tube asset is not available")

    trimesh = pytest.importorskip("trimesh")
    from aero_tasks.liquid_meshplane import MeshPlaneGeometry

    mesh = trimesh.load(tube_mesh)
    mesh.apply_scale(0.001)
    geometry = MeshPlaneGeometry.from_trimesh(mesh, autobio_root=autobio_root)
    surface = geometry.surface(900.0)

    assert surface.center_world is not None
    assert surface.half_width_m is not None
    assert surface.half_width_m > 0.0
    assert surface.distance_m is not None
