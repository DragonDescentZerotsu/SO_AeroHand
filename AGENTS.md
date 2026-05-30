# Aero Quest Sim 智能体说明

本项目将 Meta Quest 手部追踪连接到 MuJoCo 遥操作，用于控制带有 Aero Hand 的 SO101 机械臂。当前里程碑是建立清晰的双通道数据流，而不是实现完美的机器人控制。

## Quest 数据入口

Quest 手部追踪数据通过 Hand Tracking Streamer 应用经 TCP 进入，通常流程如下：

```text
adb reverse tcp:8000 tcp:8000
Quest HTS -> TCP localhost:8000 -> hand-tracking-sdk -> Python scripts
```

主要接收端位于：

- `scripts/04_receive_quest_tcp.py`：最小 TCP 接收器冒烟测试。
- `scripts/quest_tcp_aero_teleop.py`：仅针对 Aero Hand 的 Quest landmark 重定向。
- `scripts/quest_arm_channel_so101_aero_full_teleop.py`：当前 SO101 机械臂加 Aero Hand 遥操作。
- `scripts/quest_arm_channel_so101_ik.py`：当前 Arm Channel 控制的仅机械臂版本。
- `scripts/debug_quest_dual_channel.py`：轻量级解析器/通道调试脚本。

共享的类型化解析和坐标帧转换位于 `aero_quest/quest_hand_frame.py`。

## 双通道架构

数据包是混合坐标帧的：

- Arm Channel：来自 `_hand.GetRootPose(out Pose rootPose)` 的 `wrist_pos_world` 和 `wrist_quat_world`。它们位于 Q，即 Quest/Unity 世界追踪坐标帧。
- Hand Channel：来自 `_hand.GetJointPosesFromWrist(out ReadOnlyHandJointPoses joints)` 的 `landmarks_wrist`。它们位于 Wrist，即局部腕部/手部根坐标帧。

不要假设腕部位姿和 landmarks 共享同一个坐标帧。

## 坐标帧

- `Q`：Quest/Unity 世界追踪坐标帧。它是一个局部追踪坐标帧，不是机器人基座。
- `Wrist`：Quest 腕部/根部局部手部坐标帧。
- `B`：机器人基座坐标帧。

使用显式命名，例如 `wrist_pos_Q`、`wrist_pos_world`、`landmarks_wrist`、`R_BQ` 和 `target_pos_B`。

## 代码地图

- 接收：`scripts/04_receive_quest_tcp.py`，完整遥操作在 `scripts/quest_arm_channel_so101_aero_full_teleop.py`。
- 解析和类型化坐标帧模型：`aero_quest/quest_hand_frame.py`。
- 机械臂控制和 SO101 IK 辅助：`aero_quest/arm_teleop.py`。
- Aero Hand 重定向：`aero_quest/retargeting.py`。
- SO101 + Aero 动作应用：`aero_quest/so101_aero_control.py`。
- MuJoCo landmark 辅助：`aero_quest/mujoco_landmarks.py`。
- 仿真入口点：`scripts/quest_arm_channel_so101_aero_full_teleop.py`、`scripts/quest_arm_channel_so101_ik.py`、`scripts/quest_arm_channel_target_ball.py`。

## 安全规则

- 不要把 Quest 世界原点当作机器人基座原点。
- 不要把相对于腕部的 landmarks 当作世界坐标 landmarks。
- 保持 Arm Channel 和 Hand Channel 分离。
- 机器人机械臂控制优先使用相对腕部/根部运动。
- 让 `R_BQ` 可配置；不要把恒等矩阵硬编码为机器人假设。
- 给变量和注释添加显式坐标帧名称。
- 除非本地代码结构确实需要，否则避免大规模重写。
- 修改解析或坐标变换时，添加小型测试或调试脚本。
- 仅在可视化、调试、训练数据检查，或明确需要世界坐标点的遗留代码中调用 `convert_landmarks_wrist_to_world`。
