import numpy as np

from aero_quest.skeleton_visualizer import (
    LowLatencyHandSkeletonArtists,
    calibration_normalization_scale,
    calibration_reference_points,
    canonicalize_hand_for_debug,
    compute_hand_debug_metrics,
)


def make_hand():
    points = np.zeros((21, 3), dtype=np.float32)
    points[5] = (1.0, 1.0, 0.0)
    points[9] = (0.0, 2.0, 0.0)
    points[17] = (-1.0, 1.0, 0.0)
    points[4] = (0.8, 2.0, 0.0)
    points[8] = (0.5, 3.0, 0.0)
    points[12] = (0.0, 3.2, 0.0)
    return points


def test_compute_hand_debug_metrics_uses_wrist_local_geometry():
    metrics = compute_hand_debug_metrics(make_hand())
    assert metrics["palm_width"] == 2.0
    assert metrics["palm_length"] == 2.0
    assert np.isclose(metrics["thumb_index_distance"], np.sqrt(1.09))
    assert metrics["normalization_scale"] == 2.0


def test_calibration_reference_points_accepts_nested_schema():
    points = make_hand()
    calibration = {"canonical": {"canonical_landmarks": points.tolist()}}
    assert np.allclose(calibration_reference_points(calibration), points)


def test_calibration_scale_and_metadata_are_reported():
    metrics = compute_hand_debug_metrics(
        make_hand(),
        {"normalization": {"normalization_scale": 0.123}, "_cache_path": "cache/test.json"},
    )
    assert metrics["normalization_scale"] == 0.123
    assert metrics["calibration_cache"] == "cache/test.json"


def test_calibration_scale_changes_debug_canonical_size_only():
    points = make_hand()
    default = canonicalize_hand_for_debug(points)
    calibrated = canonicalize_hand_for_debug(
        points, {"normalization": {"normalization_scale": 4.0}}
    )
    assert np.allclose(calibrated, default * 0.5)
    assert calibration_normalization_scale({"scale": -1.0}) is None


def test_low_latency_artist_reuses_created_lines():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    artists = LowLatencyHandSkeletonArtists(fig, ax)
    line_ids = {name: id(line) for name, line in artists.lines.items()}
    artists.update(make_hand(), fps=10.0, frame_count=1)
    artists.update(make_hand() * 1.01, fps=10.0, frame_count=2)
    assert line_ids == {name: id(line) for name, line in artists.lines.items()}
    plt.close(fig)
