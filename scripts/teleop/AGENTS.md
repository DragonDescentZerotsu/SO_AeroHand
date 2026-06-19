# Teleop 脚本说明

本目录只放 Quest/MuJoCo 遥操作入口。所有命令默认从项目根目录运行：

```bash
cd /data/tianang/projects/SO_AeroHand
conda activate aero_sim
```

Quest Hand Tracking Streamer 使用 TCP 时，常规链路是：

```text
Quest HTS -> TCP localhost:8000 -> hand-tracking-sdk -> scripts/teleop/*
```

如果 Quest 通过 USB 接在同一台机器上，先运行：

```bash
adb reverse tcp:8000 tcp:8000
```

## Quest 在本地 Mac、MuJoCo 在远端

`adb reverse` 必须运行在实际通过 USB 连接 Quest 的 Mac 上，不能只在远端 workstation 上运行。推荐链路：

```text
Quest HTS localhost:8000
  -> adb reverse
  -> Mac localhost:18000
  -> ssh -L
  -> remote localhost:8000
  -> Python/MuJoCo
```

Mac 安装 Android platform tools，并打开 SSH 本地端口转发：

```bash
brew install android-platform-tools
ssh -N -L 18000:127.0.0.1:8000 <user>@<remote-host>
lsof -nP -iTCP:18000 -sTCP:LISTEN
```

Mac 找到 Quest serial，并把设备内的 `localhost:8000` 转到 Mac 的 SSH tunnel：

```bash
adb devices
adb -s <quest-serial> reverse tcp:8000 tcp:18000
adb -s <quest-serial> reverse --list
```

Quest HTS 仍配置为 TCP、`localhost`、端口 `8000`。远端先验证输入，再启动完整遥操作：

```bash
python scripts/debug_quest_dual_channel.py --host 127.0.0.1 --port 8000 --hand any
python scripts/teleop/quest_piper_aero_ik_teleop.py --host 127.0.0.1 --port 8000 --hand any
```

SO101 使用相同链路，只需把最后一个入口替换为 `quest_so101_aero_ik_teleop.py`。

## 推荐入口

### `quest_aero_arm_ik_teleop.py`

共享遥操作引擎，不是面向某一种机器人的推荐命令入口。它集中实现：

- Quest TCP 接收、双通道坐标帧处理和 head reference。
- MuJoCo viewer、相机、重力补偿和 actuator 控制。
- DLS position/nullspace、full-pose 和 OSQP full-pose IK。
- Aero Hand wrist-local landmark 重定向。

SO101 和 Piper 入口只覆盖模型、关节列表、初始姿态、IK 模式及控制参数，然后调用这里的 `main()`。日常运行应选择下面对应机器人的入口。

### `quest_piper_aero_ik_teleop.py`

Piper + Aero Hand 的当前推荐完整遥操作入口。

用途：

- Arm Channel：Quest wrist/root pose 控制 Piper 6DoF 机械臂末端位置和姿态。
- Hand Channel：Quest wrist-local landmarks 控制 Aero Hand 手指。
- 默认使用 `models/piper_aero_hand/Piper_aerohand.xml`。
- 默认 IK 模式是 `osqp_full_pose`，即 OSQP QP IK。
- 默认使用 `--reference-frame head`，把 Quest wrist pose 转到稳定的 head reference 里再控制机械臂。

常用命令：

```bash
python scripts/teleop/quest_piper_aero_ik_teleop.py \
  --host 127.0.0.1 \
  --port 8000 \
  --hand any \
  --scale 1.0 \
  --kp-pos 12 \
  --max-linear-speed 0.65
```

干跑检查：

```bash
python scripts/teleop/quest_piper_aero_ik_teleop.py --dry-run
```

关键默认值：

- `--ik-mode osqp_full_pose`
- `--orientation-source wrist_pose`
- `--initial-arm-qpos "0 1.57 -1.3485 0 0 0"`
- `--joint-motion-weights "0.7 1.0 1.0 0.35 0.22 0.08"`
- `--qp-task-weights "1.0 1.0 1.0 1.2 1.2 1.2"`

OSQP IK 会同时处理：

- 末端 task-space 速度跟踪。
- 关节速度限制。
- 关节位置限位。
- 相邻控制周期的 `qdot` 平滑/加速度限制。
- 接近奇异值时自适应增加 damping。

修改 Piper IK 或控制参数后，先运行自动 benchmark：

```bash
python scripts/benchmarks/piper_ik_benchmark.py
```

通过后再录制视频：

```bash
python scripts/benchmarks/piper_ik_benchmark.py \
  --record-video \
  --output-dir outputs/piper_ik_benchmark/final
```

调试时重点看终端里的：

- `ik_status`：OSQP 是否 solved，或是否 fallback 到 DLS。
- `ik_iter`：OSQP 迭代次数。
- `min_sv`：当前任务 Jacobian 最小奇异值。
- `damp`：奇异附近实际使用的 damping。
- `qdot`：当前 IK 输出关节速度。
- `qerr`：MuJoCo position actuator 当前目标误差。

### `quest_so101_aero_ik_teleop.py`

SO101 + Aero Hand 的当前推荐完整遥操作入口。

用途：

- Arm Channel：Quest wrist/root pose 控制 SO101 机械臂。
- Hand Channel：Quest wrist-local landmarks 控制 Aero Hand 手指。
- 默认使用 `models/so101_aero_hand/SO101_aerohand.xml`。
- 默认 IK 模式是 `position_nullspace`。

常用命令：

```bash
python scripts/teleop/quest_so101_aero_ik_teleop.py \
  --host 127.0.0.1 \
  --port 8000 \
  --hand any
```

干跑检查：

```bash
python scripts/teleop/quest_so101_aero_ik_teleop.py --dry-run
```

SO101 只有 5 个 arm DoF，无法稳定满足任意 6D 末端位姿，所以默认策略是：

```text
先保证末端位置，再用 position nullspace 尽量修正姿态
```

不要直接把 Piper 的 6DoF full-pose 参数套到 SO101 上，除非明确要做实验。

## 调试入口

### `quest_arm_channel_so101_ik.py`

仅机械臂 Arm Channel 调试入口，不做 Aero Hand 手指重定向。

用途：

- 验证 Quest wrist/root pose 到 SO101 末端目标的映射。
- 验证 SO101 IK 和坐标轴方向。
- 比完整 SO101 + Aero Hand 遥操作更轻量。

常用命令：

```bash
python scripts/teleop/quest_arm_channel_so101_ik.py \
  --host 127.0.0.1 \
  --port 8000 \
  --hand any
```

干跑检查：

```bash
python scripts/teleop/quest_arm_channel_so101_ik.py --dry-run
```

这个脚本不适合作为 Piper + Aero Hand 的入口。

### `quest_arm_channel_target_ball.py`

坐标轴和 Arm Channel 输入调试入口。它只移动 MuJoCo target ball，不控制真实机器人模型。

用途：

- 检查 Quest 到 MuJoCo/robot base 的平移坐标轴映射 `R_BQ`。
- 验证手向上、向前、向左时目标球是否按预期移动。
- 在机械臂控制异常前，先排查坐标轴方向。

常用命令：

```bash
python scripts/teleop/quest_arm_channel_target_ball.py \
  --host 127.0.0.1 \
  --port 8000 \
  --hand any
```

默认映射：

```text
Quest +Y up      -> MuJoCo +Z up
Quest +Z forward -> MuJoCo +X forward
Quest -X left    -> MuJoCo +Y left
```

如果 target ball 方向都不对，不要先调 IK，先修 `--R_BQ`。

### `quest_tcp_aero_teleop.py`

仅 Aero Hand 手指遥操作入口，不控制任何机械臂。

用途：

- 测试 Quest 21 个 hand landmarks 到 Aero Hand 7D 动作的重定向。
- 验证手指弯曲、拇指外展等手部映射。
- 不使用 Arm Channel，不使用 wrist/root pose 控制机械臂。

常用命令：

```bash
python scripts/teleop/quest_tcp_aero_teleop.py \
  --host 127.0.0.1 \
  --port 8000 \
  --alpha 0.25
```

`--alpha` 是手指动作平滑系数，越大越平滑但越慢。

## 入口选择规则

- 要遥操 Piper + Aero Hand：用 `quest_piper_aero_ik_teleop.py`。
- 要遥操 SO101 + Aero Hand：用 `quest_so101_aero_ik_teleop.py`。
- 只想确认 Quest 到机器人平移方向：先用 `quest_arm_channel_target_ball.py`。
- 只想调 SO101 arm，不想控制手指：用 `quest_arm_channel_so101_ik.py`。
- 只想调 Aero Hand 手指：用 `quest_tcp_aero_teleop.py`。

## 常用按键

完整机械臂遥操作入口里：

- `R`：把当前 Quest wrist/root pose 重新置零到当前机器人末端位姿。
- `P`：暂停/恢复机械臂运动。

## 坐标帧注意事项

- Quest `wrist_pos_world` / `wrist_quat_world` 位于 Quest/Unity 世界追踪坐标帧 `Q`。
- Quest `landmarks_wrist` 位于 wrist-local 手部坐标帧。
- 机械臂控制不要把 Quest 世界原点当作 robot base 原点。
- 手指重定向不要把 `landmarks_wrist` 当作世界坐标。
- 默认 `R_BQ` 映射是：

```text
Quest/Unity Q: +X right, +Y up, +Z forward
Robot base B:  +X forward, +Y left, +Z up

R_BQ =
[[ 0, 0, 1],
 [-1, 0, 0],
 [ 0, 1, 0]]
```

## 修改建议

- 改 Quest 数据解析或坐标帧转换时，优先看 `aero_quest/quest_hand_frame.py`。
- 改完整遥操作循环或通用 IK 接线时，优先看 `quest_aero_arm_ik_teleop.py`。
- 改 DLS 速度控制时看 `aero_quest/arm_teleop.py`；改 OSQP IK 时看 `aero_quest/osqp_ik.py`。
- 改 Aero Hand 手指重定向时，优先看 `aero_quest/retargeting.py`。
- 新增机器人完整遥操作入口时，像现有 SO101/Piper 入口一样复用 `quest_aero_arm_ik_teleop.py`，只覆盖模型、关节、IK 和控制参数默认值。

## 故障排查

- SSH tunnel 出现 `connect failed: Connection refused`：远端 `127.0.0.1:8000` 没有监听程序，或 tunnel 目标端口与 Python `--port` 不一致。先在远端启动调试/遥操作脚本并确认端口。
- `adb reverse` 报 `more than one device/emulator`：用 `adb devices` 找到 Quest serial，并在命令中显式添加 `-s <quest-serial>`。
- Quest HTS 显示 connected，但远端持续输出 `frame_id=None new=False stale=True`：检查 Mac 的 `18000` 监听、`adb reverse --list` 和远端 `8000` 监听。
- Mac 端口冲突：使用 `lsof -nP -iTCP:18000 -sTCP:LISTEN`；不要让 VS Code 或其他程序占用 tunnel 端口。
- MuJoCo 报 `ERROR: could not initialize GLFW`：需要在具有可用图形显示环境的终端运行 viewer。
- 平移方向错误：先运行 `quest_arm_channel_target_ball.py` 验证 `R_BQ`，不要先调整 IK。
- 手掌朝向错误：检查 `aero_quest/quest_hand_frame.py` 的 palm frame 和所选 `--orientation-source`。
