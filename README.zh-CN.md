# Aero Quest Sim

将 Meta Quest 手部追踪连接到 MuJoCo 遥操作，用于控制带有 Aero Hand 的 SO101 机械臂。

当前主流程是一个双通道控制器：

```text
Quest 腕部位置                    -> SO101 肩/肘位置 IK
Quest 手掌方向                    -> SO101 wrist_flex + wrist_roll
Quest 腕部局部手部 landmarks       -> Aero Hand 7D 手指动作
```

## 快速开始

克隆仓库时带上 MuJoCo 模型子模块，并安装 Python 包：

```bash
git clone git@github.com:DragonDescentZerotsu/SO_AeroHand.git
cd aero_quest_sim
conda activate aero_sim
python -m pip install -e ".[dev,quest]"
```

对于已有的本地仓库，初始化子模块：

```bash
git submodule update --init --recursive
```

然后从项目根目录运行：

```bash
adb devices
adb reverse tcp:8000 tcp:8000
```

Quest Hand Tracking Streamer 设置：

```text
Protocol: TCP
IP/Host: localhost or 127.0.0.1
Port: 8000
Hand: Right
```

启动完整的 SO101 + Aero Hand 遥操作：

```bash
python scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py
```

启动完整的 Piper + Aero Hand 遥操作：

```bash
python scripts/teleop/quest_piper_aero_ik_teleop.py
```

### 仿真在 remote，Quest 接在本地 Mac

如果 MuJoCo 仿真运行在 remote 主机，而 Quest 通过 USB 接在本地 Mac 上，不能只在 remote 上运行 `adb reverse`。`adb reverse` 必须在连接 Quest 的本地 Mac 上运行；remote 上的 Python 只负责监听 TCP。

推荐的数据链路是：

```text
Quest HTS localhost:8000
  -> adb reverse
  -> Mac localhost:18000
  -> ssh -L
  -> remote localhost:8000
  -> Python/MuJoCo
```

Mac 上安装 Android platform tools：

```bash
brew install android-platform-tools
```

Mac 上打开 SSH 本地端口转发。这里用 `18000` 作为 Mac 端口，是为了避开 Mac 上可能已经被 VS Code 等程序占用的 `8000`：

```bash
ssh -N -L 18000:127.0.0.1:8000 <user>@<remote-host>
```

Mac 上确认 SSH 正在监听：

```bash
lsof -nP -iTCP:18000 -sTCP:LISTEN
```

Mac 上连接 Quest，并设置 `adb reverse`：

```bash
adb devices
adb -s 2G97C5ZHCV042T reverse tcp:8000 tcp:18000
adb -s 2G97C5ZHCV042T reverse --list
```

成功时应看到类似：

```text
UsbFfs tcp:8000 tcp:18000
```

这表示 Quest 设备里的 `localhost:8000` 会被转发到 Mac 的 `localhost:18000`。再由上面的 SSH tunnel 转发到 remote 的 `localhost:8000`。

Quest Hand Tracking Streamer 仍然设置为：

```text
Protocol: TCP
IP/Host: localhost or 127.0.0.1
Port: 8000
Hand: Right
```

remote 上先用轻量调试脚本确认能收到帧：

```bash
python scripts/debug_quest_dual_channel.py --host 127.0.0.1 --port 8000 --hand any
```

确认有 `frame_id` 和腕部/landmark 数据持续输出后，再启动完整遥操作：

```bash
python scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py --host 127.0.0.1 --port 8000 --hand any
```

MuJoCo viewer 中常用按键：

```text
R  将当前 Quest 手部位姿重新置零到当前 SO101 末端执行器位姿
P  暂停/恢复机械臂运动
```

完整遥操作脚本使用：

```text
model:   models/so101_aero_hand/SO101_aerohand.xml
arm EE:  aero_wrist_site
hand:    默认右手
```

## 主要概念

Quest 数据包包含混合坐标帧：

```text
wrist_pos_world, wrist_quat_world  位于 Q，即 Quest/Unity 世界追踪坐标帧
landmarks_wrist                    位于 Wrist，即 Quest 手部局部根坐标帧
```

机械臂通道和手部通道有意消费不同的数据：

```text
Arm Channel:
  腕部位置控制目标末端执行器位置
  shoulder_pan、shoulder_lift、elbow_flex 主要用于定位 Aero Hand
  wrist_flex 和 wrist_roll 跟随手掌方向，同时也会影响手部位置

Hand Channel:
  腕部局部的 21 个手部 landmarks 控制 Aero Hand 手指运动
```

默认的 Quest 到机器人坐标轴映射是：

```text
Quest/Unity Q: +X 向右，+Y 向上，+Z 向前
Robot base B:  +X 向前，+Y 向左，+Z 向上

R_BQ =
[[ 0, 0, 1],
 [-1, 0, 0],
 [ 0, 1, 0]]
```

更多细节，包括公式和坐标帧定义，见：

```text
docs/quest_dual_channel_pipeline.md
```

## 控制公式摘要

位置：

```text
delta_p_Q  = p_wrist_t_Q - p_wrist_0_Q
p_target_B = p_ee_0_B + scale * R_BQ @ delta_p_Q
```

手掌方向：

```text
R_palm_wrist = frame from wrist/index/middle/pinky landmarks
R_palm_Q     = R_wrist_Q @ R_palm_wrist
R_delta_Q    = R_palm_t_Q @ R_palm_0_Q.T
R_delta_B    = R_BQ @ R_delta_Q @ R_BQ.T
R_target_B   = R_delta_B @ R_ee_0_B
```

SO101 机械臂使用三关节位置速度 IK，并额外设置基于手掌方向的 `wrist_flex` 和 `wrist_roll` 目标。Aero Hand 使用现有的 7D 公式重定向：

```text
[thumb_abduction, thumb_flexion_1, thumb_flexion_2,
 index_curl, middle_curl, ring_curl, little_curl]
```

## Quest 遥测层

本仓库包含一个本地 Quest 遥测层，将 `external/hand-tracking-streamer` 视为上游数据源，并把项目专用的日志、质量检查、缓冲和回放工具放在 `aero_quest/` 与 `scripts/` 中。

安装并获取依赖：

```bash
git clone https://github.com/wengmister/hand-tracking-streamer.git external/hand-tracking-streamer
python -m pip install hand-tracking-sdk
```

记录、分析和回放 Quest 双通道数据：

```bash
adb devices
adb reverse tcp:8000 tcp:8000
python scripts/record_quest_dual_channel.py --transport tcp --host 0.0.0.0 --port 8000 --out logs/test.jsonl --duration 30
python scripts/analyze_quest_latency.py --log logs/test.jsonl
python scripts/replay_quest_dual_channel.py --log logs/test.jsonl --realtime
```

日志帧会将机械臂通道（`wrist_pos_world`、`wrist_quat_world`）和手部通道（`landmarks_wrist`）分开保存。`landmarks_wrist` 是腕部局部 landmarks，不是世界坐标或机器人坐标。更多细节见 `docs/quest_telemetry_layer.md`。

## 常用脚本

当前完整遥操作：

```bash
python scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py
```

Piper + Aero Hand 6DoF 完整遥操作：

```bash
python scripts/teleop/quest_piper_aero_ik_teleop.py
```

Piper 入口默认使用 `--ik-mode full_pose`，因为 Piper 有 6 个 arm DoF，可以把位置和姿态一起作为 6D task-space IK 求解。SO101 入口默认使用 `--ik-mode position_nullspace`，因为 SO101 只有 5 个 arm DoF，优先保证位置，再用剩余自由度尽量跟随姿态。

仅 Arm Channel，使用相同的 SO101 IK 控制模式：

```bash
python scripts/teleop/quest_arm_channel_so101_ik.py
```

用于检查 Quest 到机器人平移坐标轴的 target-ball 阶段：

```bash
python scripts/teleop/quest_arm_channel_target_ball.py
```

仅 Aero Hand：

```bash
python scripts/teleop/quest_tcp_aero_teleop.py --alpha 0.25
```

只调试传入的 Quest 通道，不控制机器人：

```bash
python scripts/debug_quest_dual_channel.py
```

检查组合模型：

```bash
python scripts/so101_aero_viewer.py
```

## 构建和检查

刷新组合后的 SO101 + Aero Hand MJCF：

```bash
python scripts/scenes/build_so101_aero_scene.py
```

运行核心检查：

```bash
pytest tests/test_quest_hand_frame.py tests/test_so101_aero_model.py
python scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py --dry-run
python scripts/teleop/quest_piper_aero_ik_teleop.py --dry-run
```

## 仓库布局

```text
aero_quest/       用于重定向、缓冲、质量检查和控制的 Python 包
scripts/          遥操作、记录、回放、模型生成和诊断脚本
models/           从第三方资产生成的项目自有 MuJoCo 场景
docs/             流水线说明和教程
tests/            离线测试和 MuJoCo 冒烟测试
mujoco_menagerie/ 第三方 MuJoCo 资产，以 git 子模块跟踪
third_party/      其他第三方机器人资产，以 git 子模块跟踪
```

`logs/` 和 `external/` 下的运行时日志与本地上游仓库会被有意忽略。

## 许可证

本项目尚未选择项目许可证。第三方资产仍遵循其各自许可证；在重新分发模型或派生资产前，请查看 `THIRD_PARTY_NOTICES.md` 以及各子模块内的许可证文件。

## 旧版仅 Aero Hand 路径

旧命令：

```bash
python scripts/legacy/06_quest_to_mujoco_tcp.py
```

保留为仅 Aero Hand 遥操作的兼容包装器。它不会控制 SO101 机械臂。当前完整机器人遥操作请使用：

```bash
python scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py
python scripts/teleop/quest_piper_aero_ik_teleop.py
```

## 故障排查

如果 Quest 无法连接：

```bash
adb devices
adb reverse --list
```

如果 `adb reverse` 报错 `more than one device/emulator`，说明 Mac 同时看到了多个 Android/Quest 设备。使用 `adb devices` 找到 Quest 的 serial，并显式指定：

```bash
adb -s <quest-serial> reverse tcp:8000 tcp:18000
adb -s <quest-serial> reverse --list
```

如果 Quest HTS 已显示 connected，但 remote 终端一直打印 `frame_id=None new=False stale=True`，说明 remote Python 没有收到有效 Quest 帧。优先检查 Mac 本地端口是否被占用：

```bash
lsof -nP -iTCP:8000 -sTCP:LISTEN
lsof -nP -iTCP:18000 -sTCP:LISTEN
```

若 `127.0.0.1:8000` 被 `Code Helper` 或其他程序占用，请使用上面的 remote 流程：Mac 端 SSH 监听 `18000`，并设置 `adb reverse tcp:8000 tcp:18000`。Quest HTS 仍然填 `localhost:8000`。

如果 MuJoCo viewer 报错 `ERROR: could not initialize GLFW`，请从带可用显示环境的桌面终端运行。

如果平移坐标轴不正确，请调整 `R_BQ`，或先运行 target-ball 阶段。如果手掌朝向看起来反了，请检查 `aero_quest/quest_hand_frame.py` 中的手掌坐标帧。
