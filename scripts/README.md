# Scripts 脚本说明

请从项目根目录运行以下命令：

```bash
conda activate aero_sim
cd /home/shane/Projects/aero_quest_sim
```

## 当前主要脚本

### SO101 + Aero Hand 完整遥操作

```bash
adb reverse tcp:8000 tcp:8000
python scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py
```

当前教学/默认控制器的数据流：

```text
腕部位置              -> 5 关节 SO101 位置优先 IK
掌心/腕部方向         -> nullspace 姿态修正
头部位姿              -> 稳定的 Quest 参考坐标帧
腕部局部 landmarks    -> Aero Hand 7D 手指动作
```

### Piper + Aero Hand 完整遥操作

```bash
adb reverse tcp:8000 tcp:8000
python scripts/teleop/quest_piper_aero_ik_teleop.py
```

Piper 有 6 个 arm DoF，默认使用 `--ik-mode full_pose`，把末端位置和姿态一起作为 6D task-space IK 求解。SO101 入口默认保留 `--ik-mode position_nullspace`，因为 5DoF 臂无法稳定满足任意 6D 姿态。

### Piper 6DoF 纯机械臂遥操作

```bash
adb reverse tcp:8000 tcp:8000
python scripts/teleop/quest_arm_channel_piper_ik.py
```

该入口只读取 Arm Channel，不运行 Aero Hand 手指重定向，默认使用 OSQP 约束 IK。

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
python scripts/teleop/quest_arm_channel_so101_ik.py
python scripts/teleop/quest_arm_channel_target_ball.py
```

这些脚本用于验证 TCP 接收、查看 head/hand 帧、检查 Quest 双通道数据、运行仅机械臂控制，或在控制完整机器人前确认平移轴方向。

其中带 `arm_channel` 的脚本名表示“使用 Quest hand root/wrist pose 驱动机器人机械臂控制通道”：

- `quest_arm_channel_so101_ik.py`：只控制 SO101 机械臂，不做 Aero Hand 手指重定向。
- `quest_arm_channel_target_ball.py`：用目标球可视化机械臂控制通道的目标位置，用于检查 `R_BQ` 和平移轴方向。

这两个调试脚本都不是 Piper + Aero Hand 的推荐入口；Piper 请使用 `scripts/teleop/quest_piper_aero_ik_teleop.py`。

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

### 手部骨架对齐可视化

无需连接 Quest 或安装 MuJoCo，即可生成合成手的三面板调试图：

```bash
pip install -e '.[visualization]'
python scripts/debug_visualize_hand_skeleton.py
```

读取录制的 wrist-local landmarks 并比较原始、掌心局部归一化和参考骨架：

```bash
python scripts/debug_visualize_hand_skeleton.py \
  --input data/quest_landmarks.npz \
  --frame-index 0 \
  --calibration cache/hand_calibration/right_hand_calibration.json \
  --output debug/hand_skeleton_alignment.png \
  --mode side_by_side \
  --show-labels
```

左图保留 Quest Hand Channel 的腕部局部坐标；中图使用与重定向相同的
`palm_localize()` 掌心坐标和尺度；右图显示 calibration/reference 骨架（若无则显示
canonical 骨架和关键向量）。手部重定向必须使用 wrist-local landmarks，因为 Quest
root pose 属于世界追踪帧，是独立的机械臂控制通道。正确对齐时，三图中的手指拓扑与
弯曲形状应保持一致，只发生预期的刚体旋转和统一尺度归一化。坐标轴颜色为 X 红、
Y 绿、Z 蓝。使用 `--mode overlay` 可在 canonical 单位下检查重合；多帧输入输出为
`.gif` 或 `.mp4` 时可生成短动画（GIF 需要 Pillow，MP4 需要 ffmpeg）。

实时 Quest 调试采用单槽 latest-frame buffer：接收线程始终覆盖旧帧，窗口渲染较慢时
直接丢弃过期帧，不会形成等待队列。默认使用单面板、无 landmark 标签和最高 10 FPS：

```bash
adb reverse tcp:8000 tcp:8000
python scripts/debug_realtime_hand_skeleton.py \
  --hand-side right \
  --viewer-fps 10 \
  --record-npz debug/quest_hand_session.npz
```

窗口快捷键：`q` 退出、`s` 截图、`l` landmark 标签、`v` 关键向量、`g` 网格、
`p` 暂停、`r` 重置视角。按 `s` 会保存高分辨率 PNG。`--record-every N` 可每 N 个
有效输入帧录制一次；NPZ 保存原始 `points[T,21,3]`、timestamps、hand_side 和
calibration path，显示平滑不会修改录制数据。

无需 Quest 即可进行高质量三面板回放。Replay 默认打开完整 landmark 标签：

```bash
python scripts/debug_realtime_hand_skeleton.py \
  --replay debug/quest_hand_session.npz \
  --mode side_by_side \
  --show-labels \
  --replay-fps 30
```

离线回放还可导出 GIF；MP4 需要系统安装 ffmpeg：

```bash
python scripts/debug_realtime_hand_skeleton.py \
  --replay debug/quest_hand_session.npz \
  --export debug/quest_hand_replay.gif \
  --export-max-frames 120
```

实时 viewer 和 teleop 都以 TCP server 监听同一端口时不能同时启动；该脚本是独立
诊断入口。若要在 teleop 运行期间同步观察，需要后续增加由唯一 Quest 接收端向控制器
与 viewer 分发 latest frame 的本地 IPC 层。

### 仅 Aero Hand

```bash
python scripts/teleop/quest_tcp_aero_teleop.py --alpha 0.25
```

该脚本只控制 `mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml` 中的 Aero Hand，不控制 SO101 机械臂。

## 模型工具

构建组合后的 SO101 + Aero Hand 场景：

```bash
python scripts/scenes/build_so101_aero_scene.py
```

输出基础机器人模型：

```text
models/so101_aero_hand/SO101_aerohand.xml
```

构建 AgileX Piper + Aero Hand 场景：

```bash
python scripts/scenes/build_piper_aero_scene.py
```

输出基础机器人模型：

```text
models/piper_aero_hand/Piper_aerohand.xml
```

从配置生成带任务物体的场景，例如带一个动态 pipette 的抓取测试场景：

```bash
python scripts/scenes/build_scene_from_config.py --config configs/scenes/pipette_grasp.yaml
```

输出：

```text
models/so101_aero_hand/scenes/SO101_aerohand_pipette.xml
```

生成桌面上的 pipette 抓取场景：

```bash
python scripts/scenes/build_scene_from_config.py --config configs/scenes/pipette_table_grasp.yaml
```

输出：

```text
models/so101_aero_hand/scenes/SO101_aerohand_pipette_table.xml
```

查看组合模型：

```bash
python scripts/so101_aero_viewer.py
```

查看 Piper + Aero Hand 模型：

```bash
python -m mujoco.viewer --mjcf=models/piper_aero_hand/Piper_aerohand.xml
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

当前完整机器人遥操作请优先使用 `scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py` 或 `scripts/teleop/quest_piper_aero_ik_teleop.py`。

Legacy 文件名里的 `arm` 也沿用同一含义：由 Quest hand root/wrist pose 派生出来的机器人机械臂控制输入。
