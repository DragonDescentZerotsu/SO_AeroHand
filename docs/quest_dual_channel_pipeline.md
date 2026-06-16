# Quest Dual-Channel Pipeline

This document describes the current SO101 + Aero Hand teleoperation path:

```text
scripts/quest_so101_aero_nullspace_ik_teleop.py
```

The design goal is to keep the arm and hand data separate until they are applied to the combined MuJoCo model.

## Data Source

The Quest app streams hand tracking over TCP:

```text
Quest Hand Tracking Streamer
-> adb reverse tcp:8000 tcp:8000
-> hand-tracking-sdk
-> Python receiver
```

The typed runtime representation lives in:

```text
aero_quest/quest_hand_frame.py
```

## Packet Data

Each `QuestHandFrame` contains:

```python
hand_side: str
timestamp_ns: int | None
frame_id: int | None
wrist_pos_world: np.ndarray   # shape (3,)
wrist_quat_world: np.ndarray  # shape (4,), xyzw
landmarks_wrist: np.ndarray   # shape (21, 3)
```

The data is mixed-frame:

```text
wrist_pos_world, wrist_quat_world  -> Q frame
landmarks_wrist                    -> Wrist frame
```

`Q` is the Quest/Unity world tracking frame. `Wrist` is the local hand root frame. `B` is the robot base frame.

## Coordinate Frames

### Quest World: Q

For the current Quest/Unity streamer:

```text
+X right
+Y up
+Z forward
```

### Robot Base: B

The SO101 debug base frame used by the project is:

```text
+X forward
+Y left
+Z up
```

The default vector map from Quest world into robot base is:

```python
R_BQ = np.array([
    [0.0,  0.0, 1.0],
    [-1.0, 0.0, 0.0],
    [0.0,  1.0, 0.0],
])
```

This gives:

```text
Quest +Z forward -> robot +X forward
Quest +X right   -> robot -Y right
Quest +Y up      -> robot +Z up
```

### Wrist Frame

`landmarks_wrist` are not world points. They are hand joint points expressed relative to the Quest wrist/root frame.

Use:

```python
landmarks_Q = convert_landmarks_wrist_to_world(
    wrist_pos_world,
    wrist_quat_world,
    landmarks_wrist,
)
```

only for visualization, debugging, or data inspection.

## Dual Channels

### Arm Channel

Inputs:

```text
wrist_pos_world
wrist_quat_world
landmarks_wrist for palm direction
```

Output:

```text
p_target_B
R_target_B
```

The SO101 arm controls the site:

```text
so101_aero_attach_site
```

This is the wrist-roll/attachment point after removing the stock SO101 gripper, not the Aero Hand fingertip grasp site.

### Hand Channel

Input:

```text
landmarks_wrist
```

Output:

```text
7D normalized Aero Hand action in [0, 1]
```

Order:

```text
0 thumb_abduction
1 thumb_flexion_1
2 thumb_flexion_2
3 index_curl
4 middle_curl
5 ring_curl
6 little_curl
```

The combined model maps this 7D action onto the Aero Hand actuators with the same semantic mapping used by the Aero Hand-only scripts.

## Arm Position Formula

At teleop zero:

```text
p_wrist_0_Q = current Quest wrist position
p_ee_0_B    = current SO101 end-effector position
R_ee_0_B    = current SO101 end-effector orientation
```

At time `t`:

```text
delta_p_Q  = p_wrist_t_Q - p_wrist_0_Q
p_target_B = p_ee_0_B + scale_pos * R_BQ @ delta_p_Q
```

The target is clipped to the configured axis-aligned workspace.

## Palm Direction

Palm direction is built from landmarks in the Wrist frame, then transformed into Quest world using the Quest wrist rotation.

Landmark indices:

```text
0  wrist
5  index MCP
9  middle MCP
17 pinky MCP
```

Palm-local axes:

```text
x_palm = normalize(index_mcp - pinky_mcp)
y_hint = normalize(middle_mcp - wrist)
z_palm = normalize(x_palm cross y_hint)
y_palm = normalize(z_palm cross x_palm)
```

Matrix:

```text
R_palm_wrist = [x_palm, y_palm, z_palm]
R_palm_Q     = R_wrist_Q @ R_palm_wrist
```

This step is important: it guarantees that palm direction is expressed in the real Quest/world tracking frame before mapping into robot base.

## Arm Orientation Formula

At teleop zero:

```text
R_palm_0_Q = R_wrist_0_Q @ R_palm_wrist_0
R_ee_0_B   = current SO101 end-effector orientation
```

At time `t`:

```text
R_palm_t_Q = R_wrist_t_Q @ R_palm_wrist_t
R_delta_Q  = R_palm_t_Q @ R_palm_0_Q.T
R_delta_B  = R_BQ @ R_delta_Q @ R_BQ.T
R_target_B = R_delta_B @ R_ee_0_B
```

The interpretation is:

```text
the operator palm rotated this much relative to calibration
-> the robot end-effector rotates the same amount relative to calibration
```

You can fall back to wrist-root orientation instead of palm landmarks:

```bash
python scripts/quest_so101_aero_nullspace_ik_teleop.py --orientation-source wrist_pose
```

## IK

The task-space controller computes:

```text
linear velocity  = kp_pos * (p_target_B - p_ee_B)
angular velocity = kp_rot * orientation_error(R_target_B, R_ee_B)
```

Both are clamped:

```text
max_linear_speed
max_angular_speed
```

`DampedLeastSquaresIK` then solves with a 6D Jacobian when orientation is enabled:

```text
xdot = [vx, vy, vz, wx, wy, wz]
qdot = J.T @ inv(J @ J.T + damping^2 I) @ xdot
q_target = q_current + qdot * dt
```

The controlled SO101 joints are:

```text
shoulder_pan
shoulder_lift
elbow_flex
wrist_flex
wrist_roll
```

## Runtime Commands

Current full teleop:

```bash
adb reverse tcp:8000 tcp:8000
python scripts/quest_so101_aero_nullspace_ik_teleop.py
```

Arm-only controller with the same arm math:

```bash
python scripts/quest_arm_channel_so101_ik.py
```

Debug the incoming two-channel packet:

```bash
python scripts/debug_quest_dual_channel.py
```

Check only translation axis mapping with the red target ball:

```bash
python scripts/quest_arm_channel_target_ball.py
```

## Common Mistakes

- Treating `landmarks_wrist` as Quest/world coordinates.
- Driving arm position from fingertip landmarks instead of wrist position.
- Forgetting that quaternions in this code are `xyzw`.
- Treating Quest world origin as the robot base origin.
- Using `grasp_site` as the SO101 arm end-effector when the task is the gripper-removed wrist attachment point.
- Assuming `R_BQ = identity` outside quick debugging.

## Validation

Useful checks:

```bash
python tests/test_quest_hand_frame.py
python scripts/quest_so101_aero_nullspace_ik_teleop.py --dry-run
python tests/test_arm_ik_minimal.py --model models/so101_aero_hand/SO101_aerohand.xml --ee_site so101_aero_attach_site
```
