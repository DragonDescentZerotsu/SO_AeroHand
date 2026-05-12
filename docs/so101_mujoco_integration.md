# SO101 MuJoCo Integration

This project now includes a standalone SO101 MuJoCo environment. It is separate
from the Quest hand tracking and Aero Hand retargeting code.

## Assets

The preferred official resource path is:

```text
third_party/SO-ARM100/Simulation/SO101
```

It should contain:

```text
assets/
scene.xml
so101_new_calib.xml
so101_new_calib.urdf
so101_old_calib.xml
so101_old_calib.urdf
joints_properties.xml
```

The default config uses:

```text
third_party/SO-ARM100/Simulation/SO101/scene.xml
```

That scene includes `so101_new_calib.xml`. The old calibration files are kept
as reference, but are not used by default.

If the third-party checkout is missing, clone it from the project root:

```bash
git clone --depth 1 https://github.com/TheRobotStudio/SO-ARM100.git third_party/SO-ARM100
```

There is also a derived MuJoCo Menagerie model at:

```text
mujoco_menagerie/robotstudio_so101/so101.xml
```

The environment falls back to that file only if the official scene is absent.

## Dependencies

Install:

```bash
pip install mujoco gymnasium pyyaml
```

`gymnasium` is recommended for training integrations. The local smoke tests can
run with the lightweight fallback, but real training code should install it.

## Environment

The env lives in:

```text
aero_quest/envs/so101_mujoco_env.py
```

It exposes a Gymnasium-style API:

```python
obs, info = env.reset()
obs, reward, terminated, truncated, info = env.step(action)
```

Actions are normalized:

```text
action in [-1, 1]^nu
```

They are mapped to MuJoCo actuator `ctrlrange` dynamically. Joint names,
actuator names, qpos indices, and qvel indices are read from the loaded model.

Observation is a dict:

```text
qpos      arm/gripper controlled joint positions
qvel      arm/gripper controlled joint velocities
ee_pos    end-effector site/body position
ee_quat   end-effector orientation quaternion
gripper   [MuJoCo gripper qpos, LeRobot-style gripper percent]
```

## Config

The default config is:

```text
configs/env/so101_mujoco.yaml
```

Important fields:

```text
model_path
use_new_calib
control_dt
physics_dt
render_width
render_height
camera_names
action_scale
episode_len
init_qpos
reward_type
ee_site_name
```

This repository does not currently have a full Hydra or training config system,
so the SO101 scripts load this YAML directly.

## Viewer

Run:

```bash
python scripts/so101_viewer.py
```

With random actions:

```bash
python scripts/so101_viewer.py --random-actions
```

Override model:

```bash
python scripts/so101_viewer.py \
  --model-path third_party/SO-ARM100/Simulation/SO101/scene.xml
```

## Random Rollout

Run:

```bash
python scripts/so101_random_rollout.py \
  --steps 500 \
  --output data/so101_random_rollout.npz
```

This saves observations, actions, rewards, qpos, end-effector positions, gripper
state, joint names, and actuator names.

## Smoke Test

Run:

```bash
python tests/test_so101_mujoco_env.py
```

The test checks that the XML exists, MuJoCo loads, reset/step work, shapes stay
stable, and 100 random steps do not produce NaN.

## Gripper Mapping

LeRobot describes the gripper as:

```text
0   = fully closed
100 = fully open
```

The current MJCF/URDF does not fully encode that exact hardware mapping. The
helpers:

```python
gripper_lerobot_to_mujoco()
gripper_mujoco_to_lerobot()
```

use a conservative linear mapping over the XML actuator range for now.

TODO: replace this with real hardware or official calibration data before using
gripper values as precise real-robot commands.

## Training Integration

There is no existing `train.py` or Hydra training entrypoint in this repository.
For a future trainer, instantiate:

```python
from aero_quest.envs import SO101MujocoEnv

env = SO101MujocoEnv(model_path="third_party/SO-ARM100/Simulation/SO101/scene.xml")
```

Then use `env.action_space`, `env.observation_space`, `reset`, and `step` as in
any Gymnasium environment.

## Future Combination

The standalone SO101 env is independent of Quest hand tracking and Aero Hand
retargeting. A first-pass combined SO101 arm + Aero Hand MJCF is available at:

```text
mujoco_menagerie/so101_aero_hand/scene.xml
```

Refresh it from the source XML files with:

```bash
python scripts/build_so101_aero_scene.py
```

Open it:

```bash
python scripts/so101_aero_viewer.py
```

Quest teleop for the combined model:

```bash
adb reverse tcp:8000 tcp:8000
python scripts/quest_arm_channel_so101_aero_full_teleop.py
```

The current teleop path is split into:

```text
SO101 Arm Channel:  Quest wrist/palm pose -> position + orientation IK
Aero Hand Channel:  Quest wrist-local landmarks -> 7D normalized hand action
```

The SO101 original moving-jaw gripper is removed before attaching the Aero Hand.
The arm end-effector control site is:

```text
so101_aero_attach_site
```

This site is closer to the gripper-removed wrist attachment than the stock
`gripperframe`, which is near the old jaw tip. The generated combined model also
keeps Aero Hand landmark sites for retargeting and diagnostics.

The attachment site pose in the SO101 gripper body is:

```text
pos="-0.0079 -0.000218121 -0.035"
```

Tune the attached palm pose visually if the physical mounting changes.
