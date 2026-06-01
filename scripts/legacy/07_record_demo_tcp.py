import importlib.util
import json
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np
from hand_tracking_sdk import (
    HandFrame,
    HTSClient,
    HTSClientConfig,
    StreamOutput,
    TransportMode,
)

try:
    from scripts.retarget import HandRetargeter
except ImportError:
    from retarget import HandRetargeter


PROJECT_ROOT = Path.home() / "Projects/aero_quest_sim"
MODEL_PATH = PROJECT_ROOT / "mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml"
FAKE_CONTROL_PATH = PROJECT_ROOT / "scripts/legacy/03_fake_7d_control.py"
EPISODE_ID = "demo_quest_aero_sim_001"
OUTPUT_PATH = PROJECT_ROOT / "data" / f"{EPISODE_ID}.jsonl"


def load_fake_control_helpers():
    spec = importlib.util.spec_from_file_location("fake_7d_control", FAKE_CONTROL_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.map_7d_to_mujoco_ctrl, module.print_actuator_info


def is_right_hand_frame(frame):
    side = getattr(frame, "side", None)
    side_value = getattr(side, "value", side)
    return str(side_value).lower() == "right"


def point_to_xyz(point):
    if all(hasattr(point, attr) for attr in ("x", "y", "z")):
        return [float(point.x), float(point.y), float(point.z)]
    if isinstance(point, (list, tuple)) and len(point) >= 3:
        return [float(point[0]), float(point[1]), float(point[2])]
    raise ValueError(f"Unsupported landmark point format: {point!r}")


def extract_landmarks_from_frame(frame):
    landmarks = getattr(frame, "landmarks", None)
    points = getattr(landmarks, "points", landmarks)
    if points is None:
        raise ValueError("HandFrame has no landmarks points")
    arr = np.asarray([point_to_xyz(point) for point in points], dtype=np.float32)
    if arr.shape != (21, 3):
        raise ValueError(f"Expected 21 landmarks, got shape {arr.shape}")
    return arr


def pose_to_dict(pose):
    if pose is None:
        return {"x": 0.0, "y": 0.0, "z": 0.0, "qx": 0.0, "qy": 0.0, "qz": 0.0, "qw": 1.0}
    return {
        "x": float(getattr(pose, "x", 0.0)),
        "y": float(getattr(pose, "y", 0.0)),
        "z": float(getattr(pose, "z", 0.0)),
        "qx": float(getattr(pose, "qx", 0.0)),
        "qy": float(getattr(pose, "qy", 0.0)),
        "qz": float(getattr(pose, "qz", 0.0)),
        "qw": float(getattr(pose, "qw", 1.0)),
    }


def make_record(frame_id, start_time, frame, landmarks, action_7d, ctrl, data):
    wall_time = time.time()
    return {
        "episode_id": EPISODE_ID,
        "frame_id": int(frame_id),
        "wall_time": float(wall_time),
        "t": float(wall_time - start_time),
        "quest": {
            "side": "Right",
            "sequence_id": int(getattr(frame, "sequence_id", -1)),
            "wrist": pose_to_dict(getattr(frame, "wrist", None)),
            "landmarks": landmarks.astype(float).tolist(),
        },
        "action": {
            "aero_action_7d": action_7d.astype(float).tolist(),
            "mujoco_ctrl": ctrl.astype(float).tolist(),
        },
        "sim": {
            "qpos": data.qpos.astype(float).tolist(),
            "qvel": data.qvel.astype(float).tolist(),
            "ctrl": data.ctrl.astype(float).tolist(),
        },
    }


def main():
    print("Before running:")
    print("  1. Confirm: adb reverse tcp:8000 tcp:8000")
    print("  2. Quest HTS: TCP, 127.0.0.1, port 8000")
    print("  3. Open your right hand and hold still for the first ~8 frames")
    print(f"Recording to: {OUTPUT_PATH}")

    map_7d_to_mujoco_ctrl, print_actuator_info = load_fake_control_helpers()
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    print(f"Loading model: {MODEL_PATH}")
    print_actuator_info(model)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    retargeter = HandRetargeter()
    client = HTSClient(
        HTSClientConfig(
            transport_mode=TransportMode.TCP_SERVER,
            host="0.0.0.0",
            port=8000,
            output=StreamOutput.FRAMES,
        )
    )

    print("Waiting for Quest TCP connection on port 8000...")
    frame_id = 0
    start_time = time.time()
    with OUTPUT_PATH.open("w", encoding="utf-8") as f:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            for frame in client.iter_events():
                if not viewer.is_running():
                    break

                if not isinstance(frame, HandFrame):
                    continue
                if not is_right_hand_frame(frame):
                    continue

                landmarks = extract_landmarks_from_frame(frame)
                action_7d = retargeter(landmarks)
                ctrl = map_7d_to_mujoco_ctrl(action_7d, model)
                data.ctrl[:] = ctrl
                mujoco.mj_step(model, data)
                viewer.sync()

                record = make_record(frame_id, start_time, frame, landmarks, action_7d, ctrl, data)
                f.write(json.dumps(record) + "\n")
                f.flush()

                frame_id += 1
                if frame_id % 50 == 0:
                    action_text = np.array2string(action_7d, precision=3, suppress_small=True)
                    print(f"recorded_frames={frame_id} action_7d={action_text}")

    print(f"Finished recording {frame_id} frames to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
