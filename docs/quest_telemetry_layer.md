# Quest Telemetry Layer

This stage adds a local Python telemetry layer around `wengmister/hand-tracking-streamer` and the official `hand-tracking-sdk`. It records, validates, analyzes, buffers, and replays Quest 3 hand telemetry without changing the Unity/Quest app or connecting the data to SO101 IK or Aero Hand retargeting yet.

`hand-tracking-streamer` lives under `external/` because it is an upstream data source. The local checkout is intentionally ignored by git; clone it with the command below after checking out this repository. Local adapter code lives in `aero_quest/` and local tools live in `scripts/`, so upstream updates can be pulled without mixing them with project-specific robot-control code.

We do not modify the Unity/Quest app in this stage. The goal is to first prove that Quest wrist pose and hand landmarks arrive at stable rates with clear logging, latency checks, and replay. Once that is reliable, the robot-control layer can consume the same `QuestDualChannelFrame` objects.

## Dual-Channel Definition

Arm channel:

```text
wrist_pos_world
wrist_quat_world
```

These are the Quest wrist/root pose in the Quest/Unity world frame. Later they will drive SO101 relative-position mapping, wrist orientation, and IK.

Hand channel:

```text
landmarks_wrist
```

These are 21 hand landmarks in the wrist/root-local hand frame. Later they will drive Aero Hand 7D finger retargeting.

Important: `landmarks_wrist` are wrist-local. They are not world coordinates, not robot coordinates, and should not directly control the robot end-effector position.

## TCP vs UDP Testing Plan

Start with TCP because it is easier to debug through `adb reverse` and should avoid packet loss while validating the pipeline. Then compare UDP recordings for lower overhead and possible jitter/loss tradeoffs:

```bash
python scripts/record_quest_dual_channel.py --transport tcp --out logs/tcp_test.jsonl --duration 30
python scripts/record_quest_dual_channel.py --transport udp --out logs/udp_test.jsonl --duration 30
python scripts/analyze_quest_latency.py --log logs/tcp_test.jsonl
python scripts/analyze_quest_latency.py --log logs/udp_test.jsonl
```

Compare average FPS, interval standard deviation, p95/p99 frame interval, dropped-frame estimates, and invalid landmark counts.

## Recommended First Test

From the project root:

```bash
adb devices
adb reverse tcp:8000 tcp:8000
python scripts/record_quest_dual_channel.py --transport tcp --host 0.0.0.0 --port 8000 --out logs/test.jsonl
python scripts/analyze_quest_latency.py --log logs/test.jsonl
python scripts/replay_quest_dual_channel.py --log logs/test.jsonl --realtime
```

Quest Hand Tracking Streamer should be configured for TCP on port `8000`. With `adb reverse`, the headset can target `localhost` or `127.0.0.1`.

## Components

`aero_quest/quest_dual_channel.py` defines `QuestDualChannelFrame`, validation, quaternion normalization, and JSON-safe serialization.

`aero_quest/quest_receiver.py` wraps `hand-tracking-sdk` and converts SDK `HandFrame` objects into dual-channel frames.

`aero_quest/quest_frame_buffer.py` provides a thread-safe latest-frame buffer for low-latency control loops. Old frames are overwritten instead of queued for real-time robot control.

`aero_quest/quest_logger.py` writes one dual-channel frame per JSONL line.

`aero_quest/quest_data_quality.py` computes FPS, jitter, dropped-frame estimates, timestamp ordering, wrist jumps, quaternion norms, and landmark-shape health.

`aero_quest/quest_replay.py` replays JSONL logs with recorded timing or as fast as possible.

## Dependency

Install the SDK with:

```bash
python -m pip install hand-tracking-sdk
```

SDK-dependent imports are isolated inside `aero_quest/quest_receiver.py`, so recorded logs can still be analyzed and replayed without MuJoCo, SO101 models, or Aero Hand retargeting.
