# Aero Quest Sim

Meta Quest hand tracking to MuJoCo teleoperation for an SO101 arm with an Aero Hand.

The current main path is a two-channel controller:

```text
Quest wrist position                 -> SO101 shoulder/elbow position IK
Quest palm direction                 -> SO101 wrist_flex + wrist_roll
Quest wrist-local hand landmarks       -> Aero Hand 7D finger action
```

## Quick Start

Clone with the MuJoCo model submodules and install the Python package:

```bash
git clone git@github.com:DragonDescentZerotsu/SO_AeroHand.git
cd aero_quest_sim
conda activate aero_sim
python -m pip install -e ".[dev,quest]"
```

For an existing checkout, initialize the submodules with:

```bash
git submodule update --init --recursive
```

Then run from the project root:

```bash
adb devices
adb reverse tcp:8000 tcp:8000
```

Quest Hand Tracking Streamer settings:

```text
Protocol: TCP
IP/Host: localhost or 127.0.0.1
Port: 8000
Hand: Right
```

Start full SO101 + Aero Hand teleoperation:

```bash
python scripts/quest_so101_aero_nullspace_ik_teleop.py
```

Useful keys in the MuJoCo viewer:

```text
R  re-zero current Quest hand pose to current SO101 end-effector pose
P  pause/resume arm motion
```

The full teleop script uses:

```text
model:   models/so101_aero_hand/SO101_aerohand.xml
arm EE:  so101_aero_attach_site
hand:    right hand by default
```

## Main Concepts

Quest packets are mixed-frame:

```text
wrist_pos_world, wrist_quat_world  in Q, the Quest/Unity world tracking frame
landmarks_wrist                    in Wrist, the local Quest hand root frame
```

The arm and hand channels intentionally consume different data:

```text
Arm Channel:
  wrist position controls target end-effector position
  shoulder_pan, shoulder_lift, elbow_flex mainly position the Aero Hand
  wrist_flex and wrist_roll follow palm direction, while also affecting hand position

Hand Channel:
  wrist-local 21 hand landmarks control Aero Hand finger motion
```

The default Quest-to-robot axis map is:

```text
Quest/Unity Q: +X right, +Y up, +Z forward
Robot base B:  +X forward, +Y left, +Z up

R_BQ =
[[ 0, 0, 1],
 [-1, 0, 0],
 [ 0, 1, 0]]
```

More detail, including formulas and frame definitions, is in:

```text
docs/quest_dual_channel_pipeline.md
```

## Control Formula Summary

Position:

```text
delta_p_Q  = p_wrist_t_Q - p_wrist_0_Q
p_target_B = p_ee_0_B + scale * R_BQ @ delta_p_Q
```

Palm direction:

```text
R_palm_wrist = frame from wrist/index/middle/pinky landmarks
R_palm_Q     = R_wrist_Q @ R_palm_wrist
R_delta_Q    = R_palm_t_Q @ R_palm_0_Q.T
R_delta_B    = R_BQ @ R_delta_Q @ R_BQ.T
R_target_B   = R_delta_B @ R_ee_0_B
```

The SO101 arm uses three-joint position velocity IK plus separate palm-direction `wrist_flex` and `wrist_roll` targets. The Aero Hand uses the existing 7D formula retargeting:

```text
[thumb_abduction, thumb_flexion_1, thumb_flexion_2,
 index_curl, middle_curl, ring_curl, little_curl]
```

## Quest Telemetry Layer

This repo includes a local Quest telemetry layer that treats `external/hand-tracking-streamer` as an upstream data source and keeps project-specific logging, quality checks, buffering, and replay tools in `aero_quest/` and `scripts/`.

Install and fetch the dependencies:

```bash
git clone https://github.com/wengmister/hand-tracking-streamer.git external/hand-tracking-streamer
python -m pip install hand-tracking-sdk
```

Record, analyze, and replay Quest dual-channel data:

```bash
adb devices
adb reverse tcp:8000 tcp:8000
python scripts/record_quest_dual_channel.py --transport tcp --host 0.0.0.0 --port 8000 --out logs/test.jsonl --duration 30
python scripts/analyze_quest_latency.py --log logs/test.jsonl
python scripts/replay_quest_dual_channel.py --log logs/test.jsonl --realtime
```

The logged frame keeps the arm channel (`wrist_pos_world`, `wrist_quat_world`) separate from the hand channel (`landmarks_wrist`). `landmarks_wrist` are wrist-local landmarks, not world or robot coordinates. More detail is in `docs/quest_telemetry_layer.md`.

## Useful Scripts

Full current teleop:

```bash
python scripts/quest_so101_aero_nullspace_ik_teleop.py
```

Arm channel only, same SO101 IK control mode:

```bash
python scripts/quest_arm_channel_so101_ik.py
```

Target-ball stage for checking Quest-to-robot translation axes:

```bash
python scripts/quest_arm_channel_target_ball.py
```

Aero Hand only:

```bash
python scripts/quest_tcp_aero_teleop.py --alpha 0.25
```

Debug incoming Quest channels without controlling the robot:

```bash
python scripts/debug_quest_dual_channel.py
```

Inspect the combined model:

```bash
python scripts/so101_aero_viewer.py
```

## Build And Check

Refresh the combined SO101 + Aero Hand MJCF:

```bash
python scripts/build_so101_aero_scene.py
```

Run core checks:

```bash
pytest tests/test_quest_hand_frame.py tests/test_so101_aero_model.py
python scripts/quest_so101_aero_nullspace_ik_teleop.py --dry-run
```

## Repository Layout

```text
aero_quest/       Python package for retargeting, buffering, quality checks, and control
scripts/          Teleoperation, recording, replay, model generation, and diagnostics
models/           Project-owned MuJoCo scenes generated from third-party assets
docs/             Pipeline notes and tutorials
tests/            Offline and MuJoCo smoke tests
mujoco_menagerie/ Third-party MuJoCo assets, tracked as a git submodule
third_party/      Other third-party robot assets, tracked as git submodules
```

Runtime logs and local upstream checkouts under `logs/` and `external/` are intentionally ignored.

## License

No project license has been selected yet. Third-party assets remain under their
own licenses; see `THIRD_PARTY_NOTICES.md` and the license files inside each
submodule before redistributing models or derived assets.

## Legacy Aero Hand Only Path

The older command:

```bash
python scripts/legacy/06_quest_to_mujoco_tcp.py
```

is kept as a compatibility wrapper for Aero Hand-only teleoperation. It does not control the SO101 arm. For current full robot teleoperation, use:

```bash
python scripts/quest_so101_aero_nullspace_ik_teleop.py
```

## Troubleshooting

If Quest does not connect:

```bash
adb devices
adb reverse --list
```

If MuJoCo viewer fails with `ERROR: could not initialize GLFW`, run from a desktop terminal with a working display.

If translation axes are wrong, tune `R_BQ` or first run the target-ball stage. If palm orientation appears reversed, inspect the palm frame in `aero_quest/quest_hand_frame.py`.
