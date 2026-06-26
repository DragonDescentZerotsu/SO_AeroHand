# Aero Quest Sim 智能体说明

本项目当前主线是用 MuJoCo 生成 wet-lab bench 专家轨迹和 LeRobot 训练数据，重点任务是 Piper gripper 从 rack 取 pipette 并交给 Piper + Aero Hand。Meta Quest 遥操作仍保留用于人工验证和历史实验，但不是当前代码结构的中心。

## 文档和记忆规则

- 根目录 `AGENTS.md` 是项目架构、长期约束和跨目录工作流的唯一主记录。
- 子目录有特殊规则时写入就近的 `AGENTS.md`，例如遥操作见 `scripts/teleop/AGENTS.md`，benchmark 见 `scripts/benchmarks/AGENTS.md`。
- `docs/` 用于需要公式、背景或长篇教程的专题文档，不要把同一份操作说明重复维护在多个入口文档中。
- 修改入口、文件名、默认算法或验证流程时，应同步更新相关 `AGENTS.md` 和直接受影响的专题文档。

## 环境初始化

从项目根目录使用 `aero_sim` 环境：

```bash
conda activate aero_sim
python -m pip install -e ".[dev,quest]"
git submodule update --init --recursive
```

主要 Python 依赖在 `pyproject.toml`。`mujoco_menagerie/` 和 `third_party/` 中的模型作为 git submodule 管理。

Quest Hand Tracking Streamer 的 Python SDK 由 `hand-tracking-sdk` 提供。本地如需查看或修改上游 streamer，可克隆到被 git 忽略的 `external/hand-tracking-streamer/`。项目内代码按用途放置：专家轨迹和数据导出放在 `aero_tasks/` 与 `scripts/planning/`，Quest/teleop 逻辑放在 `aero_quest/` 与 `scripts/teleop/`。

## Quest 数据入口

Quest 手部追踪数据通过 Hand Tracking Streamer 应用经 TCP 进入，通常流程如下：

```text
adb reverse tcp:8000 tcp:8000
Quest HTS -> TCP localhost:8000 -> hand-tracking-sdk -> Python scripts
```

主要接收端位于：

- `scripts/04_receive_quest_tcp.py`：最小 TCP 接收器冒烟测试。
- `scripts/teleop/quest_tcp_aero_teleop.py`：仅针对 Aero Hand 的 Quest landmark 重定向。
- `scripts/teleop/quest_aero_arm_ik_teleop.py`：共享的 Quest、机械臂 IK、MuJoCo 和 Aero Hand 遥操作实现；通常不直接运行。
- `scripts/teleop/quest_so101_aero_ik_teleop.py`：当前 SO101 机械臂加 Aero Hand 遥操作。
- `scripts/teleop/quest_piper_aero_ik_teleop.py`：当前 Piper 6DoF 机械臂加 Aero Hand 遥操作，默认使用 `osqp_full_pose` IK。
- `scripts/teleop/quest_arm_channel_so101_ik.py`：当前 Arm Channel 控制的仅机械臂版本。
- `scripts/debug_quest_dual_channel.py`：轻量级解析器/通道调试脚本。

共享的类型化解析和坐标帧转换位于 `aero_quest/quest_hand_frame.py`。

Quest 直接通过 USB 接在运行 MuJoCo 的同一台机器时：

```bash
adb devices
adb reverse tcp:8000 tcp:8000
```

Quest HTS 使用 TCP、`localhost` 或 `127.0.0.1`、端口 `8000`。

### Quest 在本地 Mac、MuJoCo 在远端 workstation

当 Quest 通过 USB 接在本地 Mac，而 MuJoCo 和遥操作 Python 运行在远端 workstation 时，`adb reverse` 必须在连接 Quest 的 Mac 上执行。数据链路为：

```text
Quest HTS localhost:8000
  -> adb reverse
  -> Mac localhost:18000
  -> SSH local port forwarding
  -> remote localhost:8000
  -> Python/MuJoCo
```

Mac 安装 Android platform tools：

```bash
brew install android-platform-tools
```

Mac 建立 SSH 本地端口转发。Mac 使用 `18000`，远端 Python 继续监听 `8000`：

```bash
ssh -N -L 18000:127.0.0.1:8000 tianang@dlf_2
```

确认 Mac 的 SSH tunnel 正在监听：

```bash
lsof -nP -iTCP:18000 -sTCP:LISTEN
```

Mac 查找 Quest serial，并把 Quest 设备内的 `localhost:8000` 转发到 Mac 的 `18000`：

```bash
adb devices
adb -s <quest-serial> reverse tcp:8000 tcp:18000
adb -s <quest-serial> reverse --list
# 当前使用的 Quest serial 是 2G97C5ZHCV042T：
adb -s 2G97C5ZHCV042T reverse tcp:8000 tcp:18000
adb -s 2G97C5ZHCV042T reverse --list
```

成功时应看到类似：

```text
UsbFfs tcp:8000 tcp:18000
```

Quest Hand Tracking Streamer 仍然配置为：

```text
Protocol: TCP
IP/Host: localhost
Port: 8000
Hand: Right
```

远端 workstation 先启动轻量调试接收器：

```bash
conda activate aero_sim
cd /data/tianang/projects/SO_AeroHand
python scripts/debug_quest_dual_channel.py \
  --host 127.0.0.1 \
  --port 8000 \
  --hand any
```

确认持续收到 `frame_id`、wrist pose 和 landmarks 后，再启动完整遥操作。Piper + Aero Hand：

```bash
python scripts/teleop/quest_piper_aero_ik_teleop.py \
  --host 127.0.0.1 \
  --port 8000 \
  --hand any
```

SO101 + Aero Hand：

```bash
python scripts/teleop/quest_so101_aero_ik_teleop.py \
  --host 127.0.0.1 \
  --port 8000 \
  --hand any
```

常见故障：

- SSH 输出 `open failed: connect failed: Connection refused`：远端 `127.0.0.1:8000` 当前没有监听程序，或 SSH tunnel 的目标端口和 Python `--port` 不一致。先在远端启动调试或遥操作脚本。
- `adb reverse` 报 `more than one device/emulator`：使用 `adb devices` 找到 Quest serial，并显式添加 `-s <quest-serial>`。
- Quest HTS 显示 connected，但远端持续显示 `frame_id=None new=False stale=True`：检查 Mac 的 `18000` 监听、`adb reverse --list` 和远端 Python 的 `8000` 监听。
- Mac 端口被占用：使用 `lsof -nP -iTCP:18000 -sTCP:LISTEN` 检查；必要时更换 Mac 端口，并同步修改 SSH `-L` 和 `adb reverse` 的目标端口。

更详细的遥操作入口和参数说明见 `scripts/teleop/AGENTS.md`。

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

- 专家轨迹主线：`aero_tasks/motion_planning.py`、`aero_tasks/task_sampling.py`、`aero_tasks/payload_collision.py`、`aero_tasks/lerobot_export.py`、`scripts/planning/plan_piper_gripper_pipette_handoff.py`、`scripts/planning/generate_piper_pipette_handoff_lerobot.py`、`scripts/planning/preview_lerobot_cameras.py`、`scripts/planning/replay_trajectory.py`。具体命令、数据布局和回放按键见 `scripts/planning/AGENTS.md`。
- 接收：`scripts/04_receive_quest_tcp.py`，完整遥操作在 `scripts/teleop/`。
- 解析和类型化坐标帧模型：`aero_quest/quest_hand_frame.py`。
- 机械臂速度控制和 DLS IK 辅助：`aero_quest/arm_teleop.py`。
- 通用 OSQP IK：`aero_quest/osqp_ik.py`。
- Aero Hand 重定向：`aero_quest/retargeting.py`。
- SO101 + Aero 动作应用：`aero_quest/so101_aero_control.py`。
- MuJoCo landmark 辅助：`aero_quest/mujoco_landmarks.py`。
- Quest 双通道记录、质量分析和回放：`aero_quest/quest_logger.py`、`aero_quest/quest_data_quality.py`、`aero_quest/quest_replay.py`。
- 仿真入口点：`scripts/teleop/quest_so101_aero_ik_teleop.py`、`scripts/teleop/quest_piper_aero_ik_teleop.py`、`scripts/teleop/quest_arm_channel_so101_ik.py`、`scripts/teleop/quest_arm_channel_target_ball.py`。
- Piper IK 自动验证：`scripts/benchmarks/piper_ik_benchmark.py`。默认 full 套件包含固定工作区挑战、100 个确定性随机可达位姿、连续 6D 轨迹，以及固定位置 wrist `+70° -> -70°` 奇异/限位压力测试；修改 Piper IK 或控制参数后应先运行该 benchmark，再进行人工遥操作验证。

更详细的专题资料：

- `docs/quest_dual_channel_pipeline.md`：双通道坐标帧、映射和控制公式。
- `docs/quest_telemetry_layer.md`：Quest 日志、质量检查和回放。
- `docs/formula_retargeting_tutorial.md`：Aero Hand 公式重定向。

## 遥操作入口选择

- `scripts/teleop/quest_so101_aero_ik_teleop.py`：SO101 + Aero Hand。SO101 只有 5 个 arm DoF，所以默认 `--ik-mode position_nullspace`，先保证末端位置，再用剩余自由度尽量跟随姿态。
- `scripts/teleop/quest_piper_aero_ik_teleop.py`：Piper + Aero Hand。Piper 有 6 个 arm DoF，所以默认 `--ik-mode osqp_full_pose`，位置和姿态作为同一个 6D QP 任务求解。
- `scripts/teleop/quest_arm_channel_so101_ik.py`：仅机械臂 Arm Channel 调试，不做 Aero Hand 手指重定向。
- `scripts/teleop/quest_arm_channel_target_ball.py`：只移动 MuJoCo target ball，用来验证 Quest 到机器人坐标轴映射，不适合作为真实机械臂控制器。

## MuJoCo 模型和场景生成

基础机器人模型和任务场景分开管理：

- `models/so101_aero_hand/SO101_aerohand.xml`：基础 SO101 机械臂 + Aero Hand 组合模型。这个文件只描述机器人本体和手，不要把 pipette、rack、桌面任务物体等直接塞进这里。
- `scripts/scenes/build_so101_aero_scene.py`：只负责生成基础机器人模型 `SO101_aerohand.xml`。
- `models/piper_aero_hand/Piper_aerohand.xml`：基础 AgileX Piper 机械臂 + Aero Hand 组合模型。这里保留 Piper 的 `link6/joint6` wrist roll，删除原平行夹爪 `link7/link8`，把 Aero palm 的安装轴对齐到 `link6` 的 `+Z` 末端轴。
- `models/piper_aero_hand/Piper_original_gripper_black.xml`：原始 AgileX Piper 模型的项目内视觉版本，仅把 `link6/link7/link8` gripper 可视 mesh 设为黑色，供左侧原始 gripper 任务实例使用。
- `scripts/scenes/build_piper_aero_scene.py`：生成基础 Piper + Aero Hand 模型，以及左侧原始 Piper 的黑色 gripper 视觉模型。
- `configs/scenes/*.yaml`：任务场景 recipe。这里描述基础模型、机器人实例、要放入的物体、物体初始位姿、是否添加 `freejoint`，以及之后训练用的随机化/任务字段。
- `aero_quest/scene_builder.py`：当前场景组合器。它读取 recipe，把外部 MJCF 机器人或物体通过 MuJoCo `<model>` / `<attach>` 组合到场景中，并按输出目录重写 mesh/model 路径。没有 `base_model` 时会创建空白 scene root，适合多机器人或纯任务场景。它仍在 `aero_quest/` 是历史原因；之后如果继续清理主线，可迁到 `aero_tasks/` 或单独的 scene 包。
- `scripts/scenes/build_scene_from_config.py`：从 `configs/scenes/*.yaml` 生成具体任务场景。
- `models/so101_aero_hand/scenes/*.xml`：生成后的任务场景，例如 `SO101_aerohand_pipette.xml`。
- `models/piper_aero_hand/scenes/*.xml`：Piper 相关任务场景，例如双 Piper 对比和 pipette rack 桌面场景。

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

Piper 双臂桌面 rack/pipette 示例：

```bash
python scripts/scenes/build_scene_from_config.py --config configs/scenes/piper_dual_pipette_rack_table.yaml
python -m mujoco.viewer --mjcf=models/piper_aero_hand/scenes/Piper_dual_pipette_rack_table.xml
```

新增任务场景时，优先新增 `configs/scenes/<task>.yaml`，不要复制粘贴基础机器人 XML。机器人或整机模型实例放在 `model_instances`，桌子、rack 等固定物体放在 `static_models`，pipette 等可抓取物体放在 `objects` 并设置 `freejoint: true`。需要把 rack/pipette 这类已经调好的相对位姿整体挪动或旋转时，使用 `layout_groups` 记录 `source_origin`、`target_origin`、`yaw` 和成员列表；这会在生成 XML 时保持组内相对位姿不变。

需要训练 policy 时，MJCF 负责拓扑和碰撞，episode reset 时再由 Python 环境根据 YAML 中的 `randomize` 字段和任务 sampler 随机化 arm qpos、静态 body pose、object freejoint pose、layout group pose 和 target object。可抓取物体必须放在带 `freejoint` 的 wrapper body 下，否则只是固定场景物体，无法被抓起来。

当前 Piper pipette handoff sampler 位于 `aero_tasks/task_sampling.py`：`--sample-pipette-rack-bar` 以 rack 横梁中心为 offset 参考点，默认沿 rack 局部 `+X` 在整根 `0.255m` 横梁上采样 pipette 初始位置。rack pose 默认以桌面中心 `rack_center_xy=(0,0)` 为参考，在桌面范围内采样 `x=[-0.36,0.36]m`、`y=[-0.12,0.24]m`、`yaw=[-30,30]deg`，并拒绝与两个 Piper 初始状态碰撞的样本；`--fixed-rack-pose` 可关闭 rack pose 随机。每个 episode 的覆盖项写入 `raw/episode_xxxxxx/episode_spec.json`，planner、LeRobot 导出和交互回放都应使用这份 spec 保证场景一致。

当前 `pipette_grasp.yaml` 直接引用本机 `/data/tianang/projects/AutoBio/autobio/model/object/pipette.gen.xml`。如果场景需要脱离这台机器运行，应先把相关 AutoBio MJCF 和 mesh assets 复制或子模块化到本项目，再更新 recipe 的 `source` 路径。

## Quest 遥测和离线回放

需要把 Quest 输入问题与 IK/机器人问题分开时，先记录并分析双通道数据：

```bash
python scripts/record_quest_dual_channel.py \
  --transport tcp \
  --host 0.0.0.0 \
  --port 8000 \
  --out logs/test.jsonl \
  --duration 30
python scripts/analyze_quest_latency.py --log logs/test.jsonl
python scripts/replay_quest_dual_channel.py --log logs/test.jsonl --realtime
```

日志必须继续分开保存 Arm Channel 的 `wrist_pos_world`、`wrist_quat_world` 和 Hand Channel 的 `landmarks_wrist`。实时控制使用 latest-frame buffer，旧帧应被覆盖而不是排队积压。

## 验证流程

修改共享坐标帧、模型或 IK 后，至少运行与改动相关的检查：

```bash
pytest tests/test_quest_hand_frame.py tests/test_so101_aero_model.py
python tests/test_osqp_ik.py
python tests/test_piper_aero_model.py
python scripts/teleop/quest_so101_aero_ik_teleop.py --dry-run
python scripts/teleop/quest_piper_aero_ik_teleop.py --dry-run
python scripts/benchmarks/piper_ik_benchmark.py
```

修改专家轨迹、采样、payload collision、LeRobot 导出或相机配置时，至少运行：

```bash
python -m py_compile aero_tasks/motion_planning.py aero_tasks/task_sampling.py aero_tasks/payload_collision.py aero_tasks/lerobot_export.py scripts/planning/plan_piper_gripper_pipette_handoff.py scripts/planning/generate_piper_pipette_handoff_lerobot.py scripts/planning/preview_lerobot_cameras.py
pytest tests/test_task_sampling.py
pytest tests/test_piper_handoff_success.py
MUJOCO_GL=egl python scripts/planning/preview_lerobot_cameras.py --out-dir outputs/lerobot/camera_preview_smoke --frame 8186
```

窄改动可以只运行直接相关的测试；修改共享遥操作引擎、OSQP IK 或模型拓扑时，应运行完整的对应机器人 dry-run 和 benchmark；修改专家轨迹主线时，应优先跑 planning/export 的编译和预览冒烟测试。

## 仓库布局

- `aero_tasks/`：专家轨迹生成、任务采样、payload 碰撞检查、相机渲染和 LeRobot 导出共享 Python 包。
- `aero_quest/`：Quest 坐标帧、重定向、遥操作、遥测和历史人工控制相关 Python 包。
- `scripts/teleop/`：当前遥操作入口。
- `scripts/scenes/`：基础组合模型和配置场景生成入口。
- `scripts/planning/`：离线 motion planning、合成专家轨迹和任务级规划入口。
- `scripts/benchmarks/`：自动性能与轨迹验证。
- `scripts/legacy/`：只为历史兼容或对照保留，不作为新功能入口。
- `models/`：项目生成的基础模型和任务场景。
- `configs/scenes/`：任务场景 recipe。
- `docs/`：专题设计、公式和教程。
- `tests/`：离线单元测试及 MuJoCo 冒烟测试。
- `mujoco_menagerie/`、`third_party/`：第三方资产和子模块。

`logs/`、`outputs/` 和 `external/` 中的本地运行产物或上游 checkout 不应提交，除非任务明确要求。

面向训练的本地 LeRobot 数据集按 task 分组放在 `outputs/lerobot/<task_name>/<dataset_name>/`，例如 `outputs/lerobot/piper_pipette_handoff/<dataset_name>/`；其中 `raw/` 只用于保存 MuJoCo 原始轨迹和摘要，训练主体读取 `meta/`、`data/`、`videos/`。

## 许可证和第三方资产

项目尚未选择统一许可证。重新分发模型、mesh 或生成的派生资产前，检查 `THIRD_PARTY_NOTICES.md` 以及对应子模块或源资产的许可证，不要默认第三方资产可随项目任意再分发。

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
