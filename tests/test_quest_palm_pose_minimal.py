import argparse
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.retargeting import as_points_array, estimate_palm_pose


def rotation_matrix_z(theta: float) -> np.ndarray:
    c = float(np.cos(theta))
    s = float(np.sin(theta))
    return np.asarray([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def make_fake_landmarks(translation: np.ndarray, yaw: float, curl: float = 0.2) -> np.ndarray:
    """Make deterministic Quest-like landmarks with known palm axes."""
    curl = float(np.clip(curl, 0.0, 1.0))
    points = np.zeros((21, 3), dtype=np.float64)
    points[0] = [0.0, 0.0, 0.0]

    finger_ids = [(5, 6, 7, 8), (9, 10, 11, 12), (13, 14, 15, 16), (17, 18, 19, 20)]
    mcp_x = [-0.045, -0.015, 0.015, 0.045]
    y_values = [0.045, 0.075, 0.103, 0.128]
    for x, ids in zip(mcp_x, finger_ids):
        for k, landmark_id in enumerate(ids):
            bend = curl * max(0, k - 1)
            points[landmark_id] = [x, y_values[k] - 0.012 * curl * k, -0.020 * bend]

    points[1] = [-0.055, 0.020, -0.004]
    points[2] = [-0.075, 0.040, -0.008]
    points[3] = [-0.092, 0.062 - 0.012 * curl, -0.014 - 0.014 * curl]
    points[4] = [-0.108, 0.084 - 0.026 * curl, -0.020 - 0.026 * curl]

    rotation = rotation_matrix_z(yaw)
    transformed = points @ rotation.T + np.asarray(translation, dtype=np.float64)
    return transformed.astype(np.float32)


def load_landmark_frames(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "P_human" in data:
            landmarks = data["P_human"]
        elif "landmarks" in data:
            landmarks = data["landmarks"]
        else:
            raise SystemExit(f"{path} must contain key 'P_human' or 'landmarks'. Keys: {list(data.keys())}")
    landmarks = np.asarray(landmarks, dtype=np.float32)
    if landmarks.ndim == 2:
        landmarks = landmarks[None, ...]
    if landmarks.ndim != 3 or landmarks.shape[1:] != (21, 3):
        raise SystemExit(f"Expected landmarks shape (T, 21, 3), got {landmarks.shape}")
    return landmarks


def axis_diagnostics(points: np.ndarray, rotation: np.ndarray) -> dict:
    points = as_points_array(points).astype(np.float64)
    index_mcp = points[5]
    middle_mcp = points[9]
    pinky_mcp = points[17]
    wrist = points[0]
    lateral = pinky_mcp - index_mcp
    forward = middle_mcp - wrist
    lateral /= max(float(np.linalg.norm(lateral)), 1e-12)
    forward /= max(float(np.linalg.norm(forward)), 1e-12)
    return {
        "x_dot_index_to_pinky": float(np.dot(rotation[:, 0], lateral)),
        "y_dot_wrist_to_middle": float(np.dot(rotation[:, 1], forward)),
        "x_dot_y": float(np.dot(rotation[:, 0], rotation[:, 1])),
        "x_norm": float(np.linalg.norm(rotation[:, 0])),
        "y_norm": float(np.linalg.norm(rotation[:, 1])),
        "z_norm": float(np.linalg.norm(rotation[:, 2])),
        "det": float(np.linalg.det(rotation)),
        "orth_error": float(np.linalg.norm(rotation.T @ rotation - np.eye(3))),
    }


def print_pose(label: str, points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    position, rotation = estimate_palm_pose(points)
    diag = axis_diagnostics(points, rotation)
    print(f"{label}:")
    print(f"  landmarks_shape={points.shape} has_nan={bool(np.isnan(points).any())}")
    print(f"  wrist={np.array2string(points[0], precision=6, suppress_small=True)}")
    print(f"  palm_position={np.array2string(position, precision=6, suppress_small=True)}")
    print(f"  palm_rotation=")
    for row in rotation:
        print(f"    {np.array2string(row, precision=6, suppress_small=True)}")
    print(
        "  diagnostics "
        f"det={diag['det']:.6f} orth_error={diag['orth_error']:.9f} "
        f"x_dot_index_to_pinky={diag['x_dot_index_to_pinky']:.6f} "
        f"y_dot_wrist_to_middle={diag['y_dot_wrist_to_middle']:.6f} "
        f"x_dot_y={diag['x_dot_y']:.6f} "
        f"axis_norms=({diag['x_norm']:.6f},{diag['y_norm']:.6f},{diag['z_norm']:.6f})"
    )
    return position, rotation


def parse_args():
    parser = argparse.ArgumentParser(description="Minimal Quest 21-landmark wrist/palm pose estimation test.")
    parser.add_argument("--input", default=None, help="Optional NPZ with P_human or landmarks array.")
    parser.add_argument("--frame", type=int, default=0, help="Frame index for --input.")
    parser.add_argument("--curl", type=float, default=0.2)
    parser.add_argument("--yaw_deg", type=float, default=35.0)
    parser.add_argument("--translation", nargs=3, type=float, default=[0.12, -0.04, 0.25])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.input:
        path = Path(args.input).expanduser()
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        frames = load_landmark_frames(path)
        frame_id = int(np.clip(args.frame, 0, len(frames) - 1))
        print(f"input={path} total_frames={len(frames)} selected_frame={frame_id}")
        print_pose(f"recorded_frame[{frame_id}]", frames[frame_id])
        return

    translation = np.asarray(args.translation, dtype=np.float64)
    yaw = np.deg2rad(float(args.yaw_deg))
    base = make_fake_landmarks(np.zeros(3, dtype=np.float64), yaw=0.0, curl=args.curl)
    moved = make_fake_landmarks(translation, yaw=yaw, curl=args.curl)
    base_position, base_rotation = print_pose("fake_base", base)
    moved_position, moved_rotation = print_pose("fake_translated_rotated", moved)

    print("relative:")
    print(f"  expected_translation_approx={np.array2string(translation, precision=6, suppress_small=True)}")
    print(f"  measured_position_delta={np.array2string(moved_position - base_position, precision=6, suppress_small=True)}")
    print(f"  relative_rotation_det={float(np.linalg.det(base_rotation.T @ moved_rotation)):.6f}")
    print(f"  finite={bool(np.all(np.isfinite(moved_position)) and np.all(np.isfinite(moved_rotation)))}")


if __name__ == "__main__":
    main()
