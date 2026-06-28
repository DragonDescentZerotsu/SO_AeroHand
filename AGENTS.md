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
- Blender 渲染基础设施：`aero_tasks/blender_render.py` 负责普通 Python 侧 manifest/命令调度，`aero_tasks/blender_scene.py` 负责 Blender 内 MuJoCo geom/camera animation，`aero_tasks/blender_liquid.py` 负责 `wet_state.jsonl` 液面和 tip 液柱 overlay；入口是 `scripts/planning/render_trajectory_blender.py`，Blender worker 是 `scripts/blender/render_trajectory_worker.py`。Blender 默认只渲染 MuJoCo visual groups `0/1/2`，不要把 collision group `3` 当可视模型输出。
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
- `models/piper_aero_hand/scenes/ejectable_pipette_tip_demo.xml`：可弹出 pipette tip 的简化验证模型。它把 tip 作为独立 free body，初始用 `tip_lock` weld 固定在 socket 上；`pipette_ejector` slide joint 行程为 `[-0.0095, 0]m`，当前弹簧设定约为未按下 `2N`、按到底 `5N`。`scripts/debug/demo_ejectable_pipette_tip.py` 会按下 ejector，到阈值后关闭 weld，并沿 `tip_socket_site` 的局部 `-Z` 方向给 tip 初速度。这个文件目前是 demo，不是 `Piper_dual_pipette_rack_table.xml` 的正式任务 pipette。
- `models/piper_aero_hand/scenes/pipette_liquid_transfer_demo.xml`：pipette 吸液/排液可视化验证模型，使用 AutoBio pipette/tip mesh、一个 source tube 和一个 target well。实际 tip 液体渲染由 `scripts/debug/demo_mujoco_pipette_liquid_transfer.py` 在输出目录生成 64 段 frustum liquid proxy 的临时 MJCF，不把这些 proxy 当作正式任务模型拓扑。
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

如果之后把可 eject tip 的 pipette 接入专家轨迹，不要直接用当前 demo MJCF 替换 YAML 的 `source`：当前 scene builder 通过 `<attach model="pipette_model" body="pipette" prefix="pipette_0/">` 只挂指定 body subtree，而 demo 里的 `pipette_tip` 是独立 sibling body。正式模型或 scene builder 扩展必须继续保留 `pipette_0`、`pipette_0_free`、`pipette_0/pipette`、`pipette_0/tip_site`、`pipette_0/pipette_ejector` 和 `pipette_0/pipette_button` 等 planner 依赖名称，并同步处理 tip free body 初始 pose、weld 开关和新增 qpos/ctrl 维度。

## Liquid Plan

BioDexBench 风格的液体转移不做真实 CFD 或微流控仿真。目标是支持生物学正确性评分，而不是精确流体物理：物理仿真继续负责手、pipette、tip、容器和孔板的接触；液体由隐藏 wet-lab 语义状态和可见液体代理共同表示。

当前已完成第一版语义层，核心入口是 `aero_tasks/liquid.py`，测试在 `tests/test_liquid.py` 和 `tests/test_liquid_foundations.py`。液体相关模块按职责拆分：

- `aero_tasks/liquid.py`：语义账本、体积转移、tip 液柱 frustum 体积反推、解析容器几何和通用 `ContainerState.surface()`。
- `aero_tasks/liquid_meshplane.py`：可选 AutoBio meshplane 后端 `MeshPlaneGeometry`。它继承 `ContainerGeometry`，从真实容器 interior mesh 根据 `volume_ul`、重力和加速度求液面 plane；核心语义层不强依赖 AutoBio。
- `aero_tasks/liquid_detection.py`：`tip_site` 与圆形 tube/well/reservoir 接受区域的几何检测，输出 tip 是否在容器内、是否在液面下、signed depth 和 surface metadata。
- `aero_tasks/liquid_eval.py`：第一版隐藏状态 BCS/evaluator，覆盖 `well_ok`、`sample_ok`、`tip_hygiene_ok`、`no_contamination` 和 `volume_ok`。

当前语义层维护这些隐藏状态：

- `tip_attached`、`tip_clean_state`、`tip_sample_id`、`tip_volume_ul`、`tip_capacity_ul`、`air_aspirated_ul`。
- 每个 source tube、reagent reservoir、well 的 `sample_id`、`volume_ul`、`capacity_ul`、`liquid_color`、`contaminated_by`。
- 每次 aspirate/dispense/air_aspirate/spill/touch_forbidden_surface 的事件日志。之后接入专家轨迹时应写入 episode 的 hidden semantic log，供 BCS/evaluator 使用，不作为 policy 输入。

体积转移使用语义账本，不依赖真实流体。`PipetteLiquidController` 根据 plunger/button qpos 增量触发吸液和排液，默认 `PlungerModel(qpos_pressed_m=-0.008, qpos_rest_m=0.0, stroke_volume_ul=200.0)`，即完整吸/排一次为 `200uL`：

```text
aspirate_ul = min(request_ul, source.volume_ul, tip.capacity_ul - tip.volume_ul)
source.volume_ul -= aspirate_ul
tip.volume_ul += aspirate_ul
tip.sample_id = source.sample_id or mixture(sample_id)

dispense_ul = min(request_ul, tip.volume_ul, target.capacity_ul - target.volume_ul)
tip.volume_ul -= dispense_ul
target.volume_ul += dispense_ul
target.sample_id = mixture(target.sample_id, tip.sample_id)
```

当前模块已经处理 tip 容量、空气占位、空吸、未对准 target 时 spill、容器容量上限、sample mixing 和污染标记。正式任务接入时，吸液/排液事件应由 `tip_site` 是否位于有效容器液体区域、plunger/button 行程变化、tip 是否安装且未被禁用共同触发；错误孔位分液、tip 复用和禁忌接触应进入诊断状态。BCS 至少覆盖 `well_ok`、`sample_ok`、`tip_hygiene_ok`、`no_contamination` 和 `volume_ok`。

当前 MuJoCo demo 入口：

```bash
MUJOCO_GL=egl conda run -n aero_sim python scripts/debug/demo_mujoco_pipette_liquid_transfer.py \
  --out-dir outputs/debug_rollouts/mujoco_pipette_liquid_transfer

MUJOCO_GL=egl conda run -n aero_sim python scripts/debug/demo_mujoco_pipette_liquid_transfer.py \
  --liquid-style pale_highlight \
  --out-dir outputs/debug_rollouts/mujoco_pipette_liquid_transfer_pale_highlight

MUJOCO_GL=egl conda run -n aero_sim python scripts/debug/demo_meshplane_pipette_centrifuge_liquid.py \
  --out-dir outputs/debug_rollouts/meshplane_pipette_centrifuge_liquid

MUJOCO_GL=egl conda run -n aero_sim python scripts/debug/demo_liquid_detection_eval.py \
  --out-dir outputs/debug_rollouts/liquid_detection_eval
```

这个脚本输出 `01_mujoco_tip_aspirate_close.mp4`、`02_mujoco_tip_dispense_close.mp4`、`03_mujoco_full_tube_to_well_transfer.mp4` 和 `04_mujoco_plunger_button_full_view.mp4`，并写入 `wet_state.jsonl`、`summary.json` 和临时渲染模型 `generated_pipette_liquid_transfer_frustum.xml`。当前 demo 支持 `--liquid-style blue` 和 `--liquid-style pale_highlight`；`pale_highlight` 使用很淡的透明蓝 `rgba=(0.70, 0.92, 1.0, 0.24)`，并在 MJCF material 上使用较高 `specular/shininess` 近似清液高光。MuJoCo 原生 renderer 只支持透明度和简单高光，不支持真实折射；真实清液折射应留给 Blender/Cycles 路径。

`demo_meshplane_pipette_centrifuge_liquid.py` 是完整 pipette + AutoBio 离心管 meshplane 验证 demo：离心管液面通过 `MeshPlaneGeometry` 调用 AutoBio `ContainerDefinition.from_object_mesh(...)` 和 `MeshPlane.solve_plane_distance(...)`，从真实 interior mesh 及 `volume_ul` 求出，并随 `gravity_world - acceleration_world` 改变 normal；MuJoCo 只画液面 patch、surface normal 和 `tip_site` submerged marker，不再尝试画离心管内完整液体体积。`demo_liquid_detection_eval.py` 是更小的语义回归 demo，用简化圆柱容器展示 `tip_site` 检测、source/tip/target 体积变化和 BCS 从失败到成功。

颜色和样本身份必须保持通用：

- `liquid_color` 是 wet semantic state 的一部分，属于 `ContainerState`、`PipetteTipState` 和每个 sample/reagent，而不是全局常量。新增不同颜色液体时优先在 sample/reagent spec 中给出 `sample_id` 和 `liquid_color`，不要在 renderer 里按容器名硬编码颜色。
- 吸液时 source 的 `liquid_color` 会随 `sample_id` 进入 tip；排液时 tip 的颜色进入 target；混合时当前实现按转移体积线性 blend。之后如需更生物学语义的颜色规则，可把 sample palette / mixture color policy 抽成独立配置，但体积账本和渲染 overlay 仍应读取同一份 wet state。
- MuJoCo 可视化可以有 renderer-only style，例如 `blue` 用于强可见训练检查，`pale_highlight` 用于接近透明清液的人工检查；这些 style 只改变显示用 RGBA/材质，不改变 `volume_ul`、`sample_id` 或 BCS 语义。

可视化实现分两类：

- Pipette tip 内液体：用 capillary-column 近似，液柱固定在 tip local frame 内，液面不随世界重力或加速度找水平。当前 MuJoCo demo 使用 64 段 frustum mesh stack 加一个 clipped boundary cylinder；`FrustumSegment` 按截锥体积反推当前液面高度，避免把体积线性映射成高度。这个 frustum stack 只是 MuJoCo 轻量可视化 proxy，不是隐藏语义状态来源，也不应绑定 Blender 渲染。
- 大容器液体：MuJoCo 中只渲染开放液面，不渲染液面以下 bulk liquid 体积。之前试过用 cylinder/ellipsoid stack 近似 bulk liquid，但会和真实 tube mesh 穿模、和 meshplane 液面相交、透明排序差，容易产生误导；不要把这类 proxy 接入训练主线。当前通用 `aero_tasks/liquid.py` 使用 `ContainerState.surface()` 根据 `volume_ul` 映射液面高度，并根据 `gravity_world - acceleration_world` 估计开放自由液面 normal。第一版已提供 constant-area、cylindrical 和 conical-cylindrical 解析近似，source tube、reservoir、well 可按实际容器选择解析几何、查表或 meshplane 后端。

容器几何后端按分层策略设计：

- 默认使用解析几何：`ConstantAreaGeometry`、`CylindricalGeometry`、`ConicalCylindricalGeometry` 速度快、依赖少、可测试，适合 well plate、reservoir、标准 tube 近似和训练主线。
- 真实形状容器使用可选 meshplane 后端：对 AutoBio 离心管、复杂瓶子或不规则容器，应从 watertight/opening-corrected container mesh 提取 interior mesh，用 `MeshPlaneGeometry.from_trimesh(...)` 构造 geometry，再由 `ContainerState.surface()` 统一调用。MuJoCo 训练渲染只需要 `calculate_plane` 风格的液面 patch；Blender/USD 路径之后可用同一 distance 调 `calculate_mesh` 得到液面以下真实填充 mesh。
- `ContainerState` 仍然只负责语义账本：`volume_ul`、`sample_id`、`liquid_color`、污染和容量限制；具体 `volume -> surface` 由 geometry backend 决定。不要让 renderer 反向决定体积或样本身份。

渲染和数据导出采用双轨：

- MuJoCo 渲染是训练默认路径。后续需要把 demo 中的 wet state provider 和 frustum/container surface overlay 接入 `aero_tasks/lerobot_export.py` 或新增 renderer，使 LeRobot 视频可按 frame 显示 tip 液柱和大容器液面。MuJoCo 不负责大容器完整透明液体体积，也不要为了视觉效果塞入 bulk proxy。
- Blender 渲染用于高质量可视化、论文图和人工检查。当前已有三层基础版：`blender_render.py` 读取任意 planner 输出的 MJCF/qpos 轨迹并生成 Blender manifest，`blender_scene.py` 在 Blender 中重建 MuJoCo geom 和相机动画，`blender_liquid.py` 读取同一份 `wet_state.jsonl` 并叠加大容器液面和 tip 液柱。入口示例：

```bash
python scripts/planning/render_trajectory_blender.py \
  --trajectory outputs/piper_gripper_pipette_handoff/piper_gripper_pipette_handoff_expert.npz \
  --out-dir outputs/debug_rollouts/blender_handoff \
  --camera handoff_mujoco_demo \
  --max-frames 120

conda run -n aero_sim python scripts/debug/demo_blender_real_pipette_centrifuge_liquid.py \
  --out-dir outputs/debug_rollouts/blender_real_pipette_centrifuge_liquid \
  --render --width 960 --height 544 --fps 20 --max-frames 140
```

本地没有 `blender` 可执行文件但 Python 环境可 `import bpy` 时，`blender_render.py` 会自动 fallback 到当前 Python 解释器运行 `scripts/blender/render_trajectory_worker.py`，先渲 PNG 序列再用 `imageio/ffmpeg` 合成 mp4；当前 `aero_sim` 已安装 `bpy==5.0.1` 并使用这条路径。若两者都不可用，入口仍会生成 `blender_render_manifest.json` 和 `render_command.sh` 供其他机器运行。`demo_blender_real_pipette_centrifuge_liquid.py` 使用真实 AutoBio pipette/tip mesh、真实 `centrifuge_1500ul_no_lid_vis` mesh 和 `MeshPlaneGeometry`，输出 `01_blender_real_pipette_centrifuge_full.mp4`、`02_blender_real_tip_close.mp4`、`wet_state.jsonl`、`real_pipette_centrifuge_liquid.npz` 和临时 MJCF；它的 qpos 是检查用可视化轨迹，不是正式专家轨迹。后续高保真液体可继续参考 AutoBio：对 meshplane 容器用 `calculate_mesh(distance)` 生成液面以下真实填充 mesh，并导出或读取 `liquid.usd` animation。不要依赖 MuJoCo 的 `tip_liquid_seg_*` proxy，也不要让训练主线依赖 Blender。

建议 episode 数据布局：

```text
raw/episode_xxxxxx/
  piper_gripper_pipette_handoff_expert.npz
  episode_spec.json
  wet_state.jsonl
  liquid_surfaces.npz
  mujoco_videos/
  blender_videos/
  liquid.usd
```

已完成：语义状态、体积账本、基本解析容器几何、可选 meshplane geometry 后端、tip frustum 体积反推、圆形容器 `tip_site` 检测、第一版 BCS evaluator、MuJoCo tip liquid transfer demo、AutoBio meshplane 离心管液面 demo、检测/BCS demo、通用 Blender 轨迹渲染基础版、Blender wet-state overlay 基础版、真实 pipette/tip + 离心管 Blender 检查 demo 和单元测试。当前专家轨迹 planner 仍主要停留在 pipette 拿取/handoff 阶段，所以还不能把液体逻辑完整接进正式专家轨迹。之后要补：把正式 pipetting planner 的 `tip_site`、容器 registry 和 plunger qpos 接到 `PipetteLiquidController`；把 `wet_state.jsonl`、BCS/evaluator 结果写入 episode；把 tip 液柱和大容器液面 overlay 接入 LeRobot renderer；把 Blender liquid overlay 扩展成 AutoBio 风格 `liquid.usd`/meshplane bulk exporter。AutoBio 可作为算法和渲染管线参考，但引入代码或资产前仍需检查许可证和本项目第三方资产记录。

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
