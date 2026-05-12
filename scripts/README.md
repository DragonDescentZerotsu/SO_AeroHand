# Scripts

Run commands from the project root:

```bash
conda activate aero_sim
cd ~/Projects/aero_quest_sim
```

## Current Main Scripts

### Full SO101 + Aero Hand Teleop

```bash
adb reverse tcp:8000 tcp:8000
python scripts/quest_arm_channel_so101_aero_full_teleop.py
```

Two-channel control:

```text
wrist position       -> SO101 shoulder/elbow position IK
palm direction       -> SO101 wrist_flex + wrist_roll
wrist-local landmarks -> Aero Hand 7D finger action
```

### Arm Channel Only

```bash
python scripts/quest_arm_channel_so101_ik.py
```

Uses the same SO101 arm control mode as full teleop, without Aero Hand retargeting.

### Translation Axis Target Ball

```bash
python scripts/quest_arm_channel_target_ball.py
```

Use this to verify `R_BQ` and Quest translation directions before controlling the arm.

### Incoming Quest Debug

```bash
python scripts/debug_quest_dual_channel.py
```

Prints wrist/root data, wrist-local landmarks, arm target preview, and hand features.

## Aero Hand Only

```bash
python scripts/quest_tcp_aero_teleop.py --alpha 0.25
```

Compatibility wrapper:

```bash
python scripts/06_quest_to_mujoco_tcp.py
```

These control only `mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml`.

## Model Utilities

Build the combined SO101 + Aero Hand scene:

```bash
python scripts/build_so101_aero_scene.py
```

View the combined model:

```bash
python scripts/so101_aero_viewer.py
```

Check the SO101/Aero attachment:

```bash
python scripts/check_so101_aero_alignment.py
```

Inspect a MuJoCo model:

```bash
python scripts/inspect_mujoco_model.py
```

## Tests And Diagnostics

```bash
python tests/test_arm_ik_minimal.py
python tests/test_arm_joint_control_minimal.py
python tests/test_aero_hand_retargeting_minimal.py
python tests/test_aero_landmark_reachability.py
```

## Legacy Numbered Scripts

The numbered scripts are retained for the original Aero Hand-only workflow:

```text
01_check_mujoco.py
02_print_actuators.py
03_fake_7d_control.py
04_receive_quest.py
04_receive_quest_tcp.py
06_quest_to_mujoco_tcp.py
07_record_demo_tcp.py
```

For current full robot teleoperation, prefer `quest_arm_channel_so101_aero_full_teleop.py`.
