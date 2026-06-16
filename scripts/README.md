# Scripts 脚本说明

请从项目根目录运行以下命令：

```bash
conda activate aero_sim
cd /data/tianang/projects/SO_AeroHand
```

## 当前主要脚本

### SO101 + Aero Hand 完整遥操作

```bash
adb reverse tcp:8000 tcp:8000
python scripts/quest_so101_aero_nullspace_ik_teleop.py
```

当前教学/默认控制器的数据流：

```text
腕部位置              -> 5 关节 SO101 位置优先 IK
掌心/腕部方向         -> nullspace 姿态修正
头部位姿              -> 稳定的 Quest 参考坐标帧
腕部局部 landmarks    -> Aero Hand 7D 手指动作
```

注意这里的 “arm” 是项目里的机器人控制命名，指 SO101 机械臂，不是 Quest 端提供了人体手臂追踪。Quest 原始数据仍然主要是 head tracking 和 hand tracking；本项目把 Quest 手部 root/wrist pose 派生为机器人机械臂控制输入。

这里的 “Quest 双通道数据” 是项目里的控制数据划分，不是 Quest 官方术语。两个通道都来自 Quest hand tracking：

- 机械臂控制通道，代码里有时写作 Arm Channel：使用 Quest 手部 root/wrist pose，也就是 `wrist_pos_world` 和 `wrist_quat_world`；它们位于 Quest/Unity 世界追踪坐标帧 `Q`，用于控制 SO101 机械臂。
- 手部控制通道，代码里有时写作 Hand Channel：使用 Quest 手部 `landmarks_wrist`；它们位于腕部/手部根局部坐标帧 `Wrist`，用于控制 Aero Hand 手指。

`head pose` 不属于这两个控制通道。它是参考信息，通常用于稳定 Quest 参考帧、做相对运动或辅助校准。

这也是为什么说 Quest 数据是混合坐标帧：

- 不要把 Quest 世界原点当作机器人基座 `B`，也不要把 `landmarks_wrist` 当作世界坐标点。

### 调试与标定

```bash
python scripts/04_receive_quest_tcp.py
python scripts/debug_quest_dual_channel.py
python scripts/quest_arm_channel_so101_ik.py
python scripts/quest_arm_channel_target_ball.py
```

这些脚本用于验证 TCP 接收、查看 head/hand 帧、检查 Quest 双通道数据、运行仅机械臂控制，或在控制完整机器人前确认平移轴方向。

其中带 `arm_channel` 的脚本名表示“使用 Quest hand root/wrist pose 驱动机器人机械臂控制通道”：

- `quest_arm_channel_so101_ik.py`：只控制 SO101 机械臂，不做 Aero Hand 手指重定向。
- `quest_arm_channel_target_ball.py`：用目标球可视化机械臂控制通道的目标位置，用于检查 `R_BQ` 和平移轴方向。

### Quest 日志工具

```bash
python scripts/record_quest_dual_channel.py --transport tcp --out logs/test.jsonl
python scripts/record_quest_landmarks_npz.py --output data/quest_landmarks.npz --num-frames 3000
python scripts/analyze_quest_latency.py --log logs/test.jsonl
python scripts/replay_quest_dual_channel.py --log logs/test.jsonl --realtime
python scripts/test_latest_frame_buffer.py
```

常用用途：

- `record_quest_dual_channel.py`：记录 Quest 双通道数据。
- `record_quest_landmarks_npz.py`：记录单手 21 个 wrist-local landmarks 到压缩 `.npz`，用于手部重定向数据采集；不记录机械臂控制通道使用的 hand root/wrist 世界位姿。
- `analyze_quest_latency.py`：分析日志中的帧率和延迟。
- `replay_quest_dual_channel.py`：按日志回放 Quest 数据，支持实时回放。
- `test_latest_frame_buffer.py`：测试 latest-frame 缓冲逻辑。

### 仅 Aero Hand

```bash
python scripts/quest_tcp_aero_teleop.py --alpha 0.25
```

该脚本只控制 `mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml` 中的 Aero Hand，不控制 SO101 机械臂。

## 模型工具

构建组合后的 SO101 + Aero Hand 场景：

```bash
python scripts/build_so101_aero_scene.py
```

输出基础机器人模型：

```text
models/so101_aero_hand/SO101_aerohand.xml
```

从配置生成带任务物体的场景，例如带一个动态 pipette 的抓取测试场景：

```bash
python scripts/build_scene_from_config.py --config configs/scenes/pipette_grasp.yaml
```

输出：

```text
models/so101_aero_hand/scenes/SO101_aerohand_pipette.xml
```

查看组合模型：

```bash
python scripts/so101_aero_viewer.py
```

检查 SO101 与 Aero Hand 的连接和对齐：

```bash
python scripts/check_so101_aero_alignment.py
```

检查任意 MuJoCo 模型：

```bash
python scripts/inspect_mujoco_model.py
```

## 测试与诊断

```bash
python tests/test_arm_ik_minimal.py
python tests/test_arm_joint_control_minimal.py
python tests/test_aero_hand_retargeting_minimal.py
python tests/test_aero_landmark_reachability.py
```

这里 `test_arm_*` 中的 `arm` 同样指 SO101 机器人机械臂控制和 IK，不是 Quest 人体手臂数据。

## Legacy 脚本

较早的实验脚本和兼容包装器位于 `scripts/legacy/`：

```text
scripts/legacy/01_check_mujoco.py
scripts/legacy/02_print_actuators.py
scripts/legacy/03_fake_7d_control.py
scripts/legacy/04_receive_quest.py
scripts/legacy/06_quest_to_mujoco_tcp.py
scripts/legacy/07_record_demo_tcp.py
scripts/legacy/quest_so101_aero_split_wrist_teleop.py
scripts/legacy/quest_so101_aero_weighted_ik_teleop.py
scripts/legacy/quest_tcp_arm_ik_teleop_minimal.py
scripts/legacy/quest_tcp_so101_teleop.py
```

当前完整机器人遥操作请优先使用 `quest_so101_aero_nullspace_ik_teleop.py`。

Legacy 文件名里的 `arm` 也沿用同一含义：由 Quest hand root/wrist pose 派生出来的机器人机械臂控制输入。
