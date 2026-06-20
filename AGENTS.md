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
- `scripts/teleop/quest_tcp_aero_teleop.py`：仅针对 Aero Hand 的 Quest landmark 重定向。
- `scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py`：当前 SO101 机械臂加 Aero Hand 遥操作。
- `scripts/teleop/quest_piper_aero_ik_teleop.py`：当前 Piper 6DoF 机械臂加 Aero Hand 遥操作，默认使用 full-pose IK。
- `scripts/teleop/quest_arm_channel_so101_ik.py`：当前 Arm Channel 控制的仅机械臂版本。
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

- 接收：`scripts/04_receive_quest_tcp.py`，完整遥操作在 `scripts/teleop/`。
- 解析和类型化坐标帧模型：`aero_quest/quest_hand_frame.py`。
- 机械臂控制和 SO101 IK 辅助：`aero_quest/arm_teleop.py`。
- 约束速度 IK：`aero_quest/osqp_ik.py`。
- Aero Hand 实时控制通道：`aero_quest/aero_hand_teleop.py`。
- Aero Hand 重定向：`aero_quest/retargeting.py`。
- SO101 + Aero 动作应用：`aero_quest/so101_aero_control.py`。
- MuJoCo landmark 辅助：`aero_quest/mujoco_landmarks.py`。
- 仿真入口点：`scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py`、`scripts/teleop/quest_piper_aero_ik_teleop.py`、`scripts/teleop/quest_arm_channel_so101_ik.py`、`scripts/teleop/quest_arm_channel_target_ball.py`。

## 遥操作入口选择

- `scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py`：SO101 + Aero Hand。SO101 只有 5 个 arm DoF，所以默认 `--ik-mode position_nullspace`，先保证末端位置，再用剩余自由度尽量跟随姿态。
- `scripts/teleop/quest_piper_aero_ik_teleop.py`：Piper + Aero Hand。Piper 有 6 个 arm DoF，所以默认 `--ik-mode full_pose`，位置和姿态作为同一个 6D 任务求解。
- `scripts/teleop/quest_arm_channel_so101_ik.py`：仅机械臂 Arm Channel 调试，不做 Aero Hand 手指重定向。
- `scripts/teleop/quest_arm_channel_target_ball.py`：只移动 MuJoCo target ball，用来验证 Quest 到机器人坐标轴映射，不适合作为真实机械臂控制器。

## MuJoCo 模型和场景生成

基础机器人模型和任务场景分开管理：

- `models/so101_aero_hand/SO101_aerohand.xml`：基础 SO101 机械臂 + Aero Hand 组合模型。这个文件只描述机器人本体和手，不要把 pipette、rack、桌面任务物体等直接塞进这里。
- `scripts/scenes/build_so101_aero_scene.py`：只负责生成基础机器人模型 `SO101_aerohand.xml`。
- `models/piper_aero_hand/Piper_aerohand.xml`：基础 AgileX Piper 机械臂 + Aero Hand 组合模型。这里保留 Piper 的 `link6/joint6` wrist roll，删除原平行夹爪 `link7/link8`，把 Aero palm 的安装轴对齐到 `link6` 的 `+Z` 末端轴。
- `scripts/scenes/build_piper_aero_scene.py`：生成基础 Piper + Aero Hand 模型。
- `configs/scenes/*.yaml`：任务场景 recipe。这里描述基础模型、要放入的物体、物体初始位姿、是否添加 `freejoint`，以及之后训练用的随机化/任务字段。
- `aero_quest/scene_builder.py`：通用场景组合器。它读取 recipe，把外部 MJCF 物体通过 MuJoCo `<model>` / `<attach>` 组合到基础机器人场景中，并按输出目录重写 mesh/model 路径。
- `scripts/scenes/build_scene_from_config.py`：从 `configs/scenes/*.yaml` 生成具体任务场景。
- `models/so101_aero_hand/scenes/*.xml`：生成后的任务场景，例如 `SO101_aerohand_pipette.xml`。

当前 pipette 示例：

```bash
python scripts/scenes/build_so101_aero_scene.py
python scripts/scenes/build_piper_aero_scene.py
python scripts/scenes/build_scene_from_config.py --config configs/scenes/pipette_grasp.yaml
python -m mujoco.viewer --mjcf=models/so101_aero_hand/scenes/SO101_aerohand_pipette.xml
```

桌面 pipette 抓取示例：

```bash
python scripts/scenes/build_scene_from_config.py --config configs/scenes/pipette_table_grasp.yaml
python -m mujoco.viewer --mjcf=models/so101_aero_hand/scenes/SO101_aerohand_pipette_table.xml
```

新增任务场景时，优先新增 `configs/scenes/<task>.yaml`，不要复制粘贴基础机器人 XML。需要训练 policy 时，MJCF 负责拓扑和碰撞，episode reset 时再由 Python 环境根据 YAML 中的 `randomize` 字段随机化 arm qpos、object freejoint pose 和 target object。可抓取物体必须放在带 `freejoint` 的 wrapper body 下，否则只是固定场景物体，无法被抓起来。

当前 pipette 场景 recipe 引用 `external/AutoBio/autobio/model/object/*.gen.xml`。`external/` 是本地忽略目录，所以新机器需要先运行：

```bash
git clone https://github.com/autobio-bench/AutoBio.git external/AutoBio
```

如果需要让场景随仓库自动获取，应把 AutoBio 改为 git submodule，或把所需 MJCF 与 mesh assets 复制到项目自有资源目录。

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
