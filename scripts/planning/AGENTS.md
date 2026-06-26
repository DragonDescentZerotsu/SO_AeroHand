# Planning 脚本说明

本目录放离线动作规划、合成专家轨迹、LeRobot 导出和轨迹检查入口。保持两层边界：

- 通用 IK、碰撞检查、RRT-Connect 等基础能力放在 `aero_tasks/motion_planning.py`。
- 通用 episode spec、静态 body pose/freejoint 初始状态覆盖和 rack 局部横梁采样放在 `aero_tasks/task_sampling.py`。
- 通用 carried-payload pose 测量和 kinematic sweep 碰撞检查放在 `aero_tasks/payload_collision.py`。
- 通用相机渲染、fixed-roll 相机和 LeRobot feature 定义放在 `aero_tasks/lerobot_export.py`。
- 具体任务脚本放在 `scripts/planning/`，只负责读取 scene 中的 body/joint 名称、生成任务目标位姿、调用 planner、导出轨迹/视频。

## 当前入口

### `plan_piper_gripper_pipette_handoff.py`

用于 `models/piper_aero_hand/scenes/Piper_dual_pipette_rack_table.xml`：

1. 左侧原始 Piper gripper 张开。
2. 移动到 pipette 偏上部的侧向 grasp site。默认 `grasp_site_offset_m=0.12`，即从 `pipette_0/pipette` body 原点沿 pipette 长轴上移 12cm。
3. 让 gripper 的 wrist-to-finger-center 轴和 pipette 长轴正交，而不是共线；gripper 的局部 X 轴与 pipette 长轴平行，两指沿局部 Y 轴闭合。
4. 从平行夹爪的对称等效姿态中选择局部 Y 轴平行地面、且与初始姿态旋转差最小的 frame。初始到 pregrasp 的 RRT 路径会逐点用 `joint6` 补偿 wrist roll，并重新检查碰撞。
5. 避免原始 gripper/arm 撞到 pipette rack、桌面和右侧 Piper + Aero Hand。
6. 夹爪闭合。
7. pipette 保持 freejoint，由 MuJoCo 接触和摩擦决定是否被真实夹起。
8. 沿世界 `+Z` 抬高 3cm 离开 rack。
9. 运行 pickup/lift 的 MuJoCo 动力学 rollout，从真实仿真 state 读取 gripper TCP、pipette freejoint wrapper pose 和 hook pose，重新计算 payload/hook 相对 TCP 的 offset。注意 hook reference local 属于 `pipette_0/pipette` body，不属于 wrapper body `pipette_0`；wrapper 只用于 freejoint 和 carried-payload sweep。
10. 从 `piper_aerohand/aero_index_proximal_site` 读取食指 proximal 指节中心，并用 `right_index_proximal_link` 局部 `+Z` 作为指节长轴。
11. Handoff 从真实 `post_pickup` state 重新规划，不再沿用理想几何抓取姿态。目标点来自 proximal 指节中心、8mm 表面 offset 和 15mm 世界 `+Z` offset；接近方向优先水平且垂直于指节轴。
12. Handoff 使用固定 `post_pickup` TCP 作为 transition point：不额外加高、不沿 pickup 轨迹回退，而是从这个点出发逐步转到 handoff 姿态并靠近 `pre_handoff`，最后插入到 `hook_insert`。handoff 不再对已规划路径后处理 wrist roll；roll 候选限制在小角度内，并在 FK 中复算最终 hook 是否能落到目标点，防止 TCP 到 hook 的几何关系被破坏。
13. `hook_settle` 后逐渐张开原始 Piper gripper，沿 `link6` 局部 `-Z` 后退 8cm，再让 Aero Hand 的食指、中指、无名指和小指最大闭合；拇指保持不动。每个闭合关节的目标是硬上限减 `0.01rad`，用 PD 力矩持续驱动，直到达到上限附近或低速稳定。该 hand attachment 没有这些手指的可用 tendon actuator，因此使用关节 PD 力矩。
14. 验证分成两个窗口：后退阶段必须保持 hook 与 proximal 指节接触、脱离原 gripper、且不碰桌面或 rack；最大闭合阶段必须至少有两个指尖碰到 pipette，末帧仍至少两个指尖接触，且不重新碰到原 gripper、桌面或 rack。摘要记录每个关节是达到 limit、停滞时已有指尖/手指链接触，还是未接触的动力学停滞；不要用任意 finger body 接触代替指尖抓握判据。
15. pipette hook reference 使用它初始挂在 rack 上时 `pipette_body_collision_2` 的实际接触点，局部坐标约为 `[-0.02252, -0.00070, 0.17343]m`，不要用正 X 侧的 ejector/pusher mesh 代替。
16. 支持 `--episode-spec <json>` 覆盖 episode 初始静态 body pose 和 freejoint pose。当前 batch sampler 会用它随机 rack pose，并把 pipette 放到 rack 局部横梁上的采样位置。
17. 进入 handoff 动力学 rollout 前会运行 carried-payload sweep：用 gripper TCP 刚性携带当前实际姿态的 pipette，检查 `pre_handoff/hook_insert/hook_settle` 阶段是否撞到 `table_0` 或 `pipette_rack_0`。刚从 rack 抬起时的支撑接触和最终与 Aero Hand 的目标接触不属于这个检查。

运行：

```bash
python scripts/planning/plan_piper_gripper_pipette_handoff.py
```

生成轨迹和摘要：

```text
outputs/piper_gripper_pipette_handoff/piper_gripper_pipette_handoff_expert.npz
outputs/piper_gripper_pipette_handoff/summary.json
```

无显示器远端录制视频时使用 EGL：

```bash
MUJOCO_GL=egl python scripts/planning/plan_piper_gripper_pipette_handoff.py --allow-failed-grasp --record-video
```

输出视频：

```text
outputs/piper_gripper_pipette_handoff/piper_gripper_pipette_handoff_expert.mp4
```

### `replay_trajectory.py`

使用原生 MuJoCo viewer 交互式循环回放导出的完整动力学 `qpos`：

```bash
python scripts/planning/replay_trajectory.py \
  outputs/piper_gripper_pipette_handoff/piper_gripper_pipette_handoff_expert.npz
```

鼠标可以正常旋转、平移和缩放视角。键盘控制为：空格暂停/继续，左右方向键单帧移动，上下方向键调整播放速度，`R` 回到第一帧，`L` 切换循环播放。默认从 MJCF 的 `model.opt.timestep` 推导真实动力学采样率；可通过 `--trajectory-fps` 覆盖。

该脚本恢复轨迹中每一帧的完整 `qpos`，不会重新运行控制器或重新积分动力学，因此会原样显示生成时的机械臂跟踪误差、物体滑动和掉落。默认读取 `.npz` 中的 `model` 字段，也可以用 `--model` 覆盖。

如果轨迹目录存在 `summary.json`，回放器会自动显示 handoff 标记：红球是指节中心，黄球是上方 hook 目标，绿球是 pipette hook reference，蓝线是指节长轴，青线是插入方向，紫线是 hook 到目标的当前误差。使用 `--no-markers` 可关闭；使用 `--summary <path>` 可指定其他摘要。

### `generate_piper_pipette_handoff_lerobot.py`

批量调用单 episode planner，并把成功轨迹导出成 LeRobot v3 风格数据集。默认 task 名称是 `piper_pipette_handoff`，输出根目录按 task 分组：

```bash
MUJOCO_GL=egl python scripts/planning/generate_piper_pipette_handoff_lerobot.py \
  --num-episodes 2 \
  --dataset-name <dataset_name> \
  --sample-pipette-rack-bar \
  --max-attempts-per-episode 4 \
  --width 320 \
  --height 240 \
  --fps 20
```

输出位置：

```text
outputs/lerobot/piper_pipette_handoff/<dataset_name>
```

目录结构：

- `meta/`、`data/`、`videos/` 是 LeRobot 数据集主体。`observation.state` 保存完整 MuJoCo `qpos`，`action` 保存 `ctrl`，`observation.stage_index` 保存规划阶段编号。
- `videos/observation.images.table_overview/`：固定在世界坐标中的主视角，eye 在桌子负 Y 的机械臂安装侧，朝桌面中心看。
- `videos/observation.images.gripper_forward/`：固定在左侧 Piper `piper_original/link6` 局部坐标里，eye 在 gripper 后方略高处，roll 由 link6 局部轴锁定，随 gripper 一起旋转。
- `videos/observation.images.palm_inner/`：固定在 Aero Hand `piper_aerohand/palm` 局部坐标里，从 palm 内侧看 pipette tip 区域；roll 由 palm 局部轴锁定，不再跟随 pipette 或左侧 gripper。当前默认 target 是用样例轨迹末帧的 `pipette_0/tip_site` 换算到 palm local 后填写的，随机化最终姿态后应重新计算或改成动态 target。
- `raw/episode_xxxxxx/` 保存原始 MuJoCo `.npz` 和 `summary.json`，用于回放、排错和复核动力学成功条件。
- `raw/episode_xxxxxx/episode_spec.json` 在启用随机采样时保存该 episode 的初始 rack body pose、pipette freejoint pose 和采样元数据。
- `generation_manifest.json` 保存每个成功 episode 的 seed、attempt index、episode spec、成功指标、payload collision 检查、raw 路径、导出帧数和 camera 名称；失败 attempt 默认删除 raw，只在 manifest 的 `failed_attempts` 中记录摘要，使用 `--keep-failed-raw` 可保留失败目录。

快速验证只写 state/action 时可加 `--skip-render`。统计固定 attempt budget 的随机化成功率时可设置 `--num-episodes <attempts>`、`--max-attempts-per-episode 1`、`--allow-partial`；manifest 会记录 `attempts_completed`、`successful_episodes` 和 `success_rate`。正式训练数据应保留视频。脚本会在写完后重新打开 `LeRobotDataset(repo_id, root=...)`，提前捕获 parquet footer 或视频 metadata 问题。

当前 pipette rack 随机化不重新生成 MJCF：`--sample-pipette-rack-bar` 会从 tuned 初始挂载姿态出发，以 `pipette_rack_0/pipette_rack` 横梁中心为参考，沿 rack 局部 `+X` 在整根横梁长度 `0.255m` 内采样 pipette offset。rack pose 默认以桌面中心 `rack_center_xy=(0,0)` 为参考，在桌面范围内采样 `x=[-0.36,0.36]m`、`y=[-0.12,0.24]m`、`yaw=[-30,30]deg`，并拒绝与两个 Piper 初始状态碰撞的样本；用 `--fixed-rack-pose` 可只采 pipette offset。采样保持 pipette 相对 rack 的局部 Y/Z 和姿态关系。同一个 `episode_spec.json` 会记录 rack pose 和 pipette pose，planner、LeRobot export 和 `replay_trajectory.py` 都会应用这份 spec。

### `sample_piper_handoff_debug_rollouts.py`

随机采样若干条可检查 rollout，成功和动态失败都会保留，不写入 LeRobot 训练集。它会调用单条 planner，并加上 `--allow-failed-grasp --record-video`；如果某次采样在动力学 rollout 前就被 carried-payload sweep 或 IK/碰撞检查拒绝，则只记录到 `debug_manifest.json` 的 `skipped_attempts`，继续采样直到收集到指定数量的 `.npz` trace。

```bash
MUJOCO_GL=egl python scripts/planning/sample_piper_handoff_debug_rollouts.py \
  --run-name debug_random_review_6 \
  --num-rollouts 6 \
  --seed-start 101 \
  --rack-bar-offset-min-m -0.1275 \
  --rack-bar-offset-max-m 0.1275
```

输出位置：

```text
outputs/debug_rollouts/piper_pipette_handoff/<run_name>/
```

每个 `attempt_xxxxxx/` 中保留：

- `episode_spec.json`：该次 rack/pipette 初始随机化。
- `piper_gripper_pipette_handoff_expert.npz`：完整 MuJoCo `qpos/ctrl/labels` trace，可用 `replay_trajectory.py` 回放。
- `summary.json`：成功/失败判据，包括 `dynamic_handoff_success`、`hook_handoff_reached`、`release_survived` 和 `palm_grasp_stable`。`hook_geometrically_reached` 是严格几何判据；`hook_functionally_reached` 允许在目标接触已确认、target/axis 误差严格通过时，使用稍宽的 contact top tolerance，避免稳定挂住的边界样本被误判失败。
- `piper_gripper_pipette_handoff_expert.mp4`：快速检查视频。
- `planner.log`：planner stdout/stderr。

回放某条 trace：

```bash
python scripts/planning/replay_trajectory.py \
  outputs/debug_rollouts/piper_pipette_handoff/<run_name>/attempt_xxxxxx/piper_gripper_pipette_handoff_expert.npz
```

相机参数集中在 `aero_tasks/lerobot_export.py` 的 `DEFAULT_HANDOFF_CAMERAS`：

- `mode="world"`：`eye_offset_world` 是世界坐标 eye，`lookat` 是世界坐标目标点，适合桌面固定相机。
- `mode="body_local"`：`body` 指定相机固定在哪个 body 上，`eye_offset_local` 和 `target_offset_local` 都在该 body 的局部坐标中，但 MuJoCo free camera 会自行确定画面 roll。
- `mode="body_local_fixed_roll"`：同样使用 body 局部 eye/target，但额外用 `up_axis_local` 指定哪个 body 局部方向应投影到画面上方。这个模式适合真实安装在 gripper/palm 上的相机。
- body 局部点到世界点的公式是 `world_point = body_pos_world + body_R @ local_point`。这里 `body_R` 是 3x3 旋转矩阵，列向量分别表示 body 局部 `+X/+Y/+Z` 在 MuJoCo 世界坐标里的方向。

检查相机局部轴和画面时使用预览脚本：

```bash
MUJOCO_GL=egl python scripts/planning/preview_lerobot_cameras.py \
  --out-dir outputs/lerobot/camera_preview_fixed_roll \
  --frame 0 \
  --frame 4000 \
  --frame 8186
```

该脚本会输出每个相机的 PNG 预览和 `camera_debug.json`。诊断 JSON 中的 `body_R_rows` 是旋转矩阵，`body_local_axes_world_columns` 把矩阵列向量拆成局部 `+X/+Y/+Z` 的世界方向，`eye_minus_body_world` 应等于 `body_R @ eye_offset_local`。

## 当前限制

- 当前版本已经移除了 kinematic attachment；`close` 后不会手动绑定 pipette，必须靠 MuJoCo contact/friction 夹起。
- 默认运行同时要求抓取保持、`hook_handoff_reached=true`、`release_survived=true` 和 `palm_grasp_stable=true`。挂接判据保留严格几何诊断 `hook_geometrically_reached`，同时加入功能性判据 `hook_functionally_reached`：确认目标接触后，hook 点到目标点的 3D 误差和指节轴向 offset 仍用严格阈值，顶部 offset 使用稍宽的 contact tolerance，避免稳定挂住的边界样本被误判失败。释放后 pipette 不再接触 gripper、且不落到桌面或 rack。加 `--allow-failed-grasp` 只用于导出失败 rollout 和视频检查。
- 当前 `grasp_site_offset_m=0.12` 的结果可以动态夹起并保持 pipette。成功判据不能只看最终高度，因为 handoff 轨迹可能主动降低末端；应同时检查搬运阶段的抓取点相对误差和双指接触。修改 grasp site、场景或接触参数后必须重新检查 `summary.json` 的 `dynamics` 字段。
- `summary.json` 会记录 `grasp_orientation_delta_deg`、`grasp_local_y_world_z` 和失败候选，便于检查必要姿态变化、水平约束、IK 不可达和 rack 碰撞。
- handoff 阶段会记录 `handoff_transition_mode`、`handoff_transition_point_world`、`handoff_local_y_world_z` 和 `dynamics.max_abs_handoff_local_y_world_z`，用于检查固定 post-pickup transition 和 gripper 局部 Y 轴是否接近平行 global XY 平面。
- Handoff 目标由目标 site/body 的局部几何实时计算，不使用固定世界 X/Y。`summary.json` 记录目标轴、接近轴、roll 搜索、hook 三维误差、目标接触和 rollout 校正历史；场景随机化后应复用同一规则重新求解。
- 平行夹爪夹持近似圆柱 pipette 时需要 `condim=6` 的滚动阻力，否则 handoff 大角度旋转会让 hook 在两指之间滚动，目标位姿不可控。
- 当前宽范围随机检查样本：`outputs/debug_rollouts/piper_pipette_handoff/broad_rack_yaw30_review_20` 使用桌面中心 rack pose 随机化、rack yaw `[-30,30]deg` 和整根横梁 pipette offset 生成 20 条有视频 rollout。按新功能性 hook 判据重算后成功 16/20，失败 4/20；无视频的 skipped attempt 已清理，本地输出按 `success/` 和 `failed/` 分组，便于人工复核。
- OMPL 当前环境未安装；本项目第一版使用 MuJoCo collision checking + SciPy bounded IK + 轻量 RRT-Connect。之后如果安装 OMPL/MoveIt，可替换搜索后端但保留任务脚本接口。
- 如果 `summary.json` 中 `pregrasp_cost/grasp_cost` 偏高，这些值现在表示 TCP 位置误差，说明 gripper center 没有足够接近目标；需要继续改进 grasp frame/site 标定、目标候选采样和末端几何误差检查。
- 当前 LeRobot 批量脚本已支持 rack 横梁上的 pipette 初始位置随机采样、rack 平面位置/yaw 随机化、失败 retry 和 carried-pipette 对 table/rack 的 sweep collision check。颜色/纹理随机化、distractor 采样和多任务 task sampler 仍是下一层。

## 扩展规则

- 新任务优先新增一个 task script，不要把任务条件硬编码进 `aero_tasks/motion_planning.py`。
- 新物体或新机器人需要的 body/joint/site 名称应集中放在 task script 顶部或配置文件里。
- 随机化任务时，先随机 scene/object pose，再复用同一套 IK + collision + RRT pipeline 生成专家轨迹。
- 轨迹导出至少包含 `qpos`、`ctrl`、`labels` 和 `model`，方便回放、训练和人工检查。
- 面向训练的批量数据统一放在 `outputs/lerobot/<task_name>/<dataset_name>/`；每个 task 保持自己的 dataset namespace，避免不同任务 episode 混在同一个目录里。
