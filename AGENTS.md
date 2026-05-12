# Aero Quest Sim Agent Notes

This project connects Meta Quest hand tracking to MuJoCo teleoperation for an SO101 arm with an Aero Hand. The current milestone is a clear two-channel data flow, not perfect robot control.

## Quest Data Entry

Quest hand tracking enters through the Hand Tracking Streamer app over TCP, usually with:

```text
adb reverse tcp:8000 tcp:8000
Quest HTS -> TCP localhost:8000 -> hand-tracking-sdk -> Python scripts
```

The main receivers are in:

- `scripts/04_receive_quest_tcp.py`: minimal TCP receiver smoke test.
- `scripts/quest_tcp_aero_teleop.py`: Aero Hand-only Quest landmark retargeting.
- `scripts/quest_arm_channel_so101_aero_full_teleop.py`: current SO101 arm plus Aero Hand teleop.
- `scripts/quest_arm_channel_so101_ik.py`: arm-only version of the current Arm Channel control.
- `scripts/debug_quest_dual_channel.py`: lightweight parser/channel debug script.

Shared typed parsing and frame conversion lives in `aero_quest/quest_hand_frame.py`.

## Two-Channel Architecture

The packet is mixed-frame:

- Arm Channel: `wrist_pos_world` and `wrist_quat_world` from `_hand.GetRootPose(out Pose rootPose)`. These are in Q, the Quest/Unity world tracking frame.
- Hand Channel: `landmarks_wrist` from `_hand.GetJointPosesFromWrist(out ReadOnlyHandJointPoses joints)`. These are in Wrist, the local wrist/root hand frame.

Do not assume the wrist pose and landmarks share a coordinate frame.

## Coordinate Frames

- `Q`: Quest/Unity world tracking frame. It is a local tracking frame, not the robot base.
- `Wrist`: Quest wrist/root local hand frame.
- `B`: robot base frame.

Use explicit names such as `wrist_pos_Q`, `wrist_pos_world`, `landmarks_wrist`, `R_BQ`, and `target_pos_B`.

## Code Map

- Receiving: `scripts/04_receive_quest_tcp.py`, full teleop in `scripts/quest_arm_channel_so101_aero_full_teleop.py`.
- Parsing and typed frame model: `aero_quest/quest_hand_frame.py`.
- Arm control and SO101 IK helpers: `aero_quest/arm_teleop.py`.
- Aero Hand retargeting: `aero_quest/retargeting.py`.
- SO101 + Aero action application: `aero_quest/so101_aero_control.py`.
- MuJoCo landmark helpers: `aero_quest/mujoco_landmarks.py`.
- Simulation entry points: `scripts/quest_arm_channel_so101_aero_full_teleop.py`, `scripts/quest_arm_channel_so101_ik.py`, `scripts/quest_arm_channel_target_ball.py`.

## Safety Rules

- Do not treat the Quest world origin as the robot base origin.
- Do not treat wrist-relative landmarks as world landmarks.
- Keep the Arm Channel and Hand Channel separate.
- Prefer relative wrist/root motion for robot arm control.
- Make `R_BQ` configurable; do not bake in identity as a robot assumption.
- Add explicit frame names to variables and comments.
- Avoid large rewrites unless the local code structure requires it.
- Add small tests or debug scripts when changing parsing or coordinate transforms.
- Only call `convert_landmarks_wrist_to_world` for visualization, debugging, training data inspection, or legacy code that explicitly needs world points.
