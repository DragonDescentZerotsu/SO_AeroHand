import argparse
import queue
import sys
import threading
import time
from pathlib import Path

import numpy as np

try:
    import mujoco
    import mujoco.viewer
except ImportError as exc:
    raise SystemExit("Missing runtime dependency. Install with: pip install mujoco numpy") from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.arm_teleop import (
    DampedLeastSquaresIK,
    VelocityTeleopConfig,
    VelocityTeleopController,
    joint_qpos,
)
from aero_quest.quest_hand_frame import RelativeWristArmController, quest_hand_frame_from_sdk


DEFAULT_MODEL = PROJECT_ROOT / "mujoco_menagerie/robotstudio_so101/scene.xml"
DEFAULT_ARM_JOINTS = "shoulder_pan,shoulder_lift,elbow_flex,wrist_flex,wrist_roll"

# Same verified Arm Channel axis map as the target-ball stage:
# Quest/Unity Q: +X right, +Y up, +Z forward.
# SO101/MuJoCo debug B: +X forward, +Y left, +Z up.
DEFAULT_R_BQ = np.asarray(
    [
        [0.0, 0.0, 1.0],
        [-1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def parse_vec3(values, name):
    if isinstance(values, (list, tuple, np.ndarray)):
        text = " ".join(str(value) for value in values)
    else:
        text = str(values)
    parsed = [float(v) for v in text.replace(",", " ").split()]
    if len(parsed) != 3:
        raise argparse.ArgumentTypeError(f"{name} expected 3 floats, got {text!r}")
    return np.asarray(parsed, dtype=np.float64)


def parse_matrix(text):
    values = [float(v) for v in str(text).replace(",", " ").split()]
    if len(values) != 9:
        raise argparse.ArgumentTypeError("--R_BQ expects 9 floats, row-major")
    return np.asarray(values, dtype=np.float64).reshape(3, 3)


def resolve_model(path_text):
    path = Path(path_text).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"MuJoCo model XML not found: {path}")
    return path


def ctrl_midpoints(model):
    return 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])


def hand_matches(quest_frame, hand):
    return hand == "any" or quest_frame.hand_side.lower() == hand


def start_quest_receiver(args, frame_queue):
    try:
        from hand_tracking_sdk import HandFrame, HTSClient, HTSClientConfig, StreamOutput, TransportMode
    except ImportError as exc:
        raise SystemExit("Quest TCP streaming requires: pip install hand-tracking-sdk") from exc

    def run():
        client = HTSClient(
            HTSClientConfig(
                transport_mode=TransportMode.TCP_SERVER,
                host=args.host,
                port=args.port,
                output=StreamOutput.FRAMES,
            )
        )
        for frame in client.iter_events():
            if not isinstance(frame, HandFrame):
                continue
            try:
                quest_frame = quest_hand_frame_from_sdk(frame)
            except ValueError as exc:
                print(f"Skipping invalid Quest SDK frame: {exc}")
                continue
            if hand_matches(quest_frame, args.hand):
                frame_queue.put((time.time(), quest_frame))

    thread = threading.Thread(target=run, name="quest-arm-channel-so101-ik-receiver", daemon=True)
    thread.start()
    return thread


def drain_latest(frame_queue):
    latest = None
    count = 0
    while True:
        try:
            latest = frame_queue.get_nowait()
            count += 1
        except queue.Empty:
            return latest, count


def make_key_callback(arm_channel, latest_frame_ref, ik, data, state):
    def on_key(keycode):
        try:
            key = chr(keycode).lower()
        except (TypeError, ValueError):
            return
        if key == "r":
            frame = latest_frame_ref["frame"]
            if frame is None:
                print("Re-zero requested, but no Quest hand frame has arrived yet.")
                return
            ee_pos_B, ee_R_B = ik.ee_pose(data)
            arm_channel.set_teleop_zero(
                frame.wrist_pos_world,
                frame.wrist_quat_world,
                ee_pos_B,
                ee_R_B,
                landmarks_wrist=frame.landmarks_wrist,
            )
            state["target_pos_B"] = ee_pos_B.copy()
            state["target_R_B"] = ee_R_B.copy()
            print("Re-zeroed: current Quest wrist maps to current SO101 end-effector pose.")
        elif key == "p":
            state["paused"] = not state["paused"]
            print(f"paused={state['paused']}")

    return on_key


def draw_target_sphere(viewer, pos_B, radius):
    scn = viewer.user_scn
    if scn.ngeom >= scn.maxgeom:
        return
    geom = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        geom,
        mujoco.mjtGeom.mjGEOM_SPHERE,
        np.asarray([radius, 0.0, 0.0], dtype=np.float64),
        np.asarray(pos_B, dtype=np.float64),
        np.eye(3, dtype=np.float64).reshape(-1),
        np.asarray([1.0, 0.08, 0.04, 0.75], dtype=np.float32),
    )
    scn.ngeom += 1


def parse_args():
    parser = argparse.ArgumentParser(description="Step 2: Quest Arm Channel wrist motion -> SO101 IK -> MuJoCo arm follows.")
    parser.add_argument("--model", "--model-path", dest="model", default=str(DEFAULT_MODEL))
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--hand", choices=["right", "left", "any"], default="right")
    parser.add_argument("--ee-site", default="so101_aero_attach_site")
    parser.add_argument("--ee-body", default=None)
    parser.add_argument("--arm-joint-names", default=DEFAULT_ARM_JOINTS)
    parser.add_argument("--arm-joint-prefix", default=None)
    parser.add_argument("--scale", type=float, default=0.6)
    parser.add_argument("--R_BQ", type=parse_matrix, default=None)
    parser.add_argument("--workspace-min", nargs=3, default=["0.05", "-0.35", "0.03"])
    parser.add_argument("--workspace-max", nargs=3, default=["0.55", "0.35", "0.60"])
    parser.add_argument("--deadzone", type=float, default=0.005)
    parser.add_argument("--target-smoothing-alpha", type=float, default=0.10)
    parser.add_argument("--kp-pos", type=float, default=5.0)
    parser.add_argument("--kp-rot", type=float, default=3.0)
    parser.add_argument("--max-linear-speed", type=float, default=0.18)
    parser.add_argument("--max-angular-speed", type=float, default=1.0)
    parser.add_argument("--ik-damping", type=float, default=0.05)
    parser.add_argument("--max-joint-speed", type=float, default=1.2)
    parser.add_argument("--joint-target-smoothing-alpha", type=float, default=0.0)
    parser.add_argument("--control-orientation", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--orientation-source", choices=["palm_landmarks", "wrist_pose"], default="palm_landmarks")
    parser.add_argument("--timeout", type=float, default=0.30)
    parser.add_argument("--debug-interval", type=float, default=0.25)
    parser.add_argument("--target-radius", type=float, default=0.025)
    parser.add_argument("--disable-gravity", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    workspace_min = parse_vec3(args.workspace_min, "--workspace-min")
    workspace_max = parse_vec3(args.workspace_max, "--workspace-max")
    R_BQ = DEFAULT_R_BQ.copy() if args.R_BQ is None else np.asarray(args.R_BQ, dtype=np.float64).reshape(3, 3)
    model_path = resolve_model(args.model)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    if args.disable_gravity:
        model.opt.gravity[:] = 0.0
    data = mujoco.MjData(model)
    data.ctrl[:] = ctrl_midpoints(model)
    mujoco.mj_forward(model, data)

    joint_names = [name.strip() for name in args.arm_joint_names.split(",") if name.strip()] if args.arm_joint_names else None
    ik = DampedLeastSquaresIK(
        model,
        ee_site=args.ee_site,
        ee_body=args.ee_body,
        joint_names=joint_names,
        joint_prefix=args.arm_joint_prefix,
        damping=args.ik_damping,
        max_joint_speed=args.max_joint_speed,
        smoothing_alpha=args.joint_target_smoothing_alpha,
    )
    vel_controller = VelocityTeleopController(
        VelocityTeleopConfig(
            kp_pos=args.kp_pos,
            kp_rot=args.kp_rot,
            max_linear_speed=args.max_linear_speed,
            max_angular_speed=args.max_angular_speed,
            control_orientation=args.control_orientation,
        )
    )
    arm_channel = RelativeWristArmController(
        scale=args.scale,
        R_BQ=R_BQ,
        deadzone=args.deadzone,
        smoothing_alpha=args.target_smoothing_alpha,
        control_orientation=args.control_orientation,
        orientation_source=args.orientation_source,
    )

    ee_pos_B, ee_R_B = ik.ee_pose(data)
    print("Before running:")
    print(f"  1. Confirm: adb reverse tcp:{args.port} tcp:{args.port}")
    print(f"  2. Quest HTS: TCP, localhost, port {args.port}")
    print("  3. This stage uses Arm Channel only: wrist relative motion -> SO101 IK.")
    print("  4. Aero Hand retargeting is disabled/not used.")
    print("  5. Press R to re-zero, P to pause/resume.")
    print(f"model={model_path}")
    print(f"ee_site={args.ee_site} arm_joints={ik.joint_names}")
    print(f"ee_start_B={np.array2string(ee_pos_B, precision=5)}")
    print(f"workspace_min={np.array2string(workspace_min, precision=3)} workspace_max={np.array2string(workspace_max, precision=3)}")
    print(f"scale={args.scale} R_BQ=\n{np.array2string(R_BQ, precision=4, suppress_small=True)}")
    print(
        f"control_orientation={args.control_orientation} orientation_source={args.orientation_source} "
        f"kp_rot={args.kp_rot} max_angular_speed={args.max_angular_speed}"
    )
    print("Expected default mapping: hand up -> +Z, hand forward -> +X, hand left -> +Y.")

    if args.dry_run:
        print(f"dry_run=true q_arm={np.array2string(joint_qpos(model, data, ik.joint_ids), precision=6)}")
        return

    frame_queue = queue.Queue()
    start_quest_receiver(args, frame_queue)
    print(f"Waiting for Quest TCP connection on {args.host}:{args.port}...")

    latest_frame_ref = {"frame": None}
    state = {
        "paused": False,
        "target_pos_B": ee_pos_B.copy(),
        "target_R_B": ee_R_B.copy(),
        "last_qtarget": joint_qpos(model, data, ik.joint_ids),
    }
    last_frame_time = 0.0
    last_debug_time = 0.0
    last_frame_id = None
    last_delta_p_Q = np.zeros(3, dtype=np.float64)
    last_velocity = np.zeros(3, dtype=np.float64)
    last_error = np.zeros(3, dtype=np.float64)
    last_rot_error = np.zeros(3, dtype=np.float64)

    with mujoco.viewer.launch_passive(
        model,
        data,
        key_callback=make_key_callback(arm_channel, latest_frame_ref, ik, data, state),
    ) as viewer:
        while viewer.is_running():
            latest, drained = drain_latest(frame_queue)
            got_new_frame = latest is not None
            if latest is not None:
                frame_time, quest_frame = latest
                latest_frame_ref["frame"] = quest_frame
                last_frame_time = frame_time
                last_frame_id = quest_frame.frame_id
                if not arm_channel.is_calibrated:
                    ee_pos_B, ee_R_B = ik.ee_pose(data)
                    arm_channel.set_teleop_zero(
                        quest_frame.wrist_pos_world,
                        quest_frame.wrist_quat_world,
                        ee_pos_B,
                        ee_R_B,
                        landmarks_wrist=quest_frame.landmarks_wrist,
                    )
                    state["target_pos_B"] = ee_pos_B.copy()
                    state["target_R_B"] = ee_R_B.copy()
                    print("Teleop zero set: current Quest wrist maps to current SO101 end-effector pose.")
                target = arm_channel.compute_target(quest_frame)
                unclipped_target_B = target.target_pos_B
                state["target_pos_B"] = np.clip(unclipped_target_B, workspace_min, workspace_max)
                if target.target_R_B is not None:
                    state["target_R_B"] = target.target_R_B.copy()
                last_delta_p_Q = target.delta_p_Q

            now = time.time()
            stale = last_frame_time == 0.0 or now - last_frame_time > args.timeout
            ee_pos_B, ee_R_B = ik.ee_pose(data)
            if arm_channel.is_calibrated and not stale and not state["paused"]:
                target_R_B = state["target_R_B"] if args.control_orientation else None
                cmd = vel_controller.compute(state["target_pos_B"], target_R_B, ee_pos_B, ee_R_B)
                last_velocity = cmd.xdot
                last_error = cmd.position_error
                last_rot_error = np.zeros(3, dtype=np.float64) if cmd.rotation_error is None else cmd.rotation_error
                qtarget, _qdot = ik.solve(
                    data,
                    last_velocity,
                    dt=float(model.opt.timestep),
                    control_orientation=args.control_orientation,
                )
                state["last_qtarget"] = qtarget.copy()
                ik.apply_position_targets(data, qtarget)
            else:
                last_velocity = np.zeros(3, dtype=np.float64)
                last_error = state["target_pos_B"] - ee_pos_B
                last_rot_error = np.zeros(3, dtype=np.float64)
                ik.apply_position_targets(data, state["last_qtarget"])

            viewer.user_scn.ngeom = 0
            draw_target_sphere(viewer, state["target_pos_B"], args.target_radius)
            mujoco.mj_step(model, data)
            viewer.sync()

            if now - last_debug_time >= args.debug_interval:
                print(
                    "debug "
                    f"frame_id={last_frame_id} new={got_new_frame} drained={drained} stale={stale} paused={state['paused']} "
                    f"delta_p_Q={np.array2string(last_delta_p_Q, precision=4, suppress_small=True)} "
                    f"target_B={np.array2string(state['target_pos_B'], precision=4, suppress_small=True)} "
                    f"ee_B={np.array2string(ee_pos_B, precision=4, suppress_small=True)} "
                    f"err={np.array2string(last_error, precision=4, suppress_small=True)} "
                    f"rot_err={np.array2string(last_rot_error, precision=4, suppress_small=True)} "
                    f"xdot={np.array2string(last_velocity, precision=4, suppress_small=True)}"
                )
                last_debug_time = now

            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
