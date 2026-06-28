from __future__ import annotations

import json
from pathlib import Path
import sys

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_tasks.blender_liquid import WetStateSeries, surface_specs, tip_liquid_spec
from aero_tasks.blender_render import BlenderRenderConfig, prepare_blender_render, selected_frame_indices


def test_selected_frame_indices_limits_frames() -> None:
    indices = selected_frame_indices(1000, source_fps=500.0, fps=20, stride=None, max_frames=5)
    assert indices.tolist() == [0, 25, 50, 75, 100]


def test_prepare_blender_render_writes_manifest(tmp_path: Path) -> None:
    model_path = tmp_path / "scene.xml"
    model_path.write_text("<mujoco model='empty'><worldbody/></mujoco>\n", encoding="utf-8")
    trajectory_path = tmp_path / "trajectory.npz"
    np.savez_compressed(trajectory_path, qpos=np.zeros((10, 0), dtype=np.float64), model=str(model_path))
    config = BlenderRenderConfig(
        trajectory=trajectory_path,
        out_dir=tmp_path / "render",
        max_frames=3,
        fps=20,
    )
    manifest_path, command = prepare_blender_render(config)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["model"] == str(model_path)
    assert manifest["frame_indices"] == [0]
    assert (tmp_path / "render" / "render_command.sh").exists()
    assert command[:3] == ["blender", "--background", "--python"]


def test_wet_state_series_and_overlay_specs(tmp_path: Path) -> None:
    path = tmp_path / "wet_state.jsonl"
    record = {
        "frame_index": 7,
        "tip": {"volume_ul": 50.0, "capacity_ul": 200.0, "liquid_color": [0.7, 0.9, 1.0, 0.4]},
        "tip_site_world": [0.0, 0.0, 0.02],
        "source_hit": {
            "surface": {
                "center_world": [0.0, 0.0, 0.01],
                "normal_world": [0.0, 0.0, 1.0],
                "half_width_m": 0.004,
                "half_height_m": 0.004,
            }
        },
    }
    path.write_text(json.dumps(record) + "\n", encoding="utf-8")
    series = WetStateSeries.load(path)
    loaded = series.record_for(7, 0)
    assert loaded is not None
    assert surface_specs(loaded)[0]["name"] == "source"
    tip = tip_liquid_spec(loaded)
    assert tip is not None
    assert tip["fill_fraction"] == 0.25
