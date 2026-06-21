# Planning 脚本说明

本目录放离线动作规划、合成专家轨迹和任务级 motion planning 入口。这里的脚本应尽量把通用规划器和具体任务逻辑分开：

- 通用 IK、碰撞检查、RRT-Connect 等基础能力放在 `aero_quest/motion_planning.py`。
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
9. 从 `piper_aerohand/aero_index_proximal_site` 读取食指 proximal 指节中心，并用 `right_index_proximal_link` 局部 `+Z` 作为指节长轴。
10. 从指节中心沿“世界向上投影到指节法平面”的方向偏置 8mm，再沿世界 `+Z` 上移 15mm，得到上方挂接目标；构造与指节轴垂直的 `link6 +Z` 接近方向并搜索无碰撞 roll。先移动到目标外侧 5cm 的 `pre_handoff`，再固定姿态沿法向执行 `hook_insert`。
11. `hook_settle` 后逐渐张开原始 Piper gripper，沿 `link6` 局部 `-Z` 后退 8cm，再让 Aero Hand 的食指、中指、无名指和小指最大闭合；拇指保持不动。每个闭合关节的目标是硬上限减 `0.01rad`，用 PD 力矩持续驱动，直到达到上限附近或低速稳定。该 hand attachment 没有这些手指的可用 tendon actuator，因此使用关节 PD 力矩。
12. 验证分成两个窗口：后退阶段必须保持 hook 与 proximal 指节接触、脱离原 gripper、且不碰桌面或 rack；最大闭合阶段必须至少有两个指尖碰到 pipette，末帧仍至少两个指尖接触，且不重新碰到原 gripper、桌面或 rack。摘要记录每个关节是达到 limit、停滞时已有指尖/手指链接触，还是未接触的动力学停滞；不要用任意 finger body 接触代替指尖抓握判据。
13. pipette hook reference 使用它初始挂在 rack 上时 `pipette_body_collision_2` 的实际接触点，局部坐标约为 `[-0.02252, -0.00070, 0.17343]m`，不要用正 X 侧的 ejector/pusher mesh 代替。

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

## 当前限制

- 当前版本已经移除了 kinematic attachment；`close` 后不会手动绑定 pipette，必须靠 MuJoCo contact/friction 夹起。
- 默认运行同时要求抓取保持、`hook_handoff_reached=true` 和 `release_survived=true`。挂接判据是：hook 与 proximal 指节真实接触、hook 顶部高度接近上方目标、沿指节长轴基本居中；释放后 pipette 不再接触 gripper、且不落到桌面或 rack。加 `--allow-failed-grasp` 只用于导出失败 rollout 和视频检查。
- 当前 `grasp_site_offset_m=0.12` 的结果可以动态夹起并保持 pipette。成功判据不能只看最终高度，因为 handoff 轨迹可能主动降低末端；应同时检查搬运阶段的抓取点相对误差和双指接触。修改 grasp site、场景或接触参数后必须重新检查 `summary.json` 的 `dynamics` 字段。
- `summary.json` 会记录 `grasp_orientation_delta_deg`、`grasp_local_y_world_z` 和失败候选，便于检查必要姿态变化、水平约束、IK 不可达和 rack 碰撞。
- Handoff 目标由目标 site/body 的局部几何实时计算，不使用固定世界 X/Y。`summary.json` 记录目标轴、接近轴、roll 搜索、hook 三维误差、目标接触和 rollout 校正历史；场景随机化后应复用同一规则重新求解。
- 平行夹爪夹持近似圆柱 pipette 时需要 `condim=6` 的滚动阻力，否则 handoff 大角度旋转会让 hook 在两指之间滚动，目标位姿不可控。
- OMPL 当前环境未安装；本项目第一版使用 MuJoCo collision checking + SciPy bounded IK + 轻量 RRT-Connect。之后如果安装 OMPL/MoveIt，可替换搜索后端但保留任务脚本接口。
- 如果 `summary.json` 中 `pregrasp_cost/grasp_cost` 偏高，这些值现在表示 TCP 位置误差，说明 gripper center 没有足够接近目标；需要继续改进 grasp frame/site 标定、目标候选采样和末端几何误差检查。

## 扩展规则

- 新任务优先新增一个 task script，不要把任务条件硬编码进 `aero_quest/motion_planning.py`。
- 新物体或新机器人需要的 body/joint/site 名称应集中放在 task script 顶部或配置文件里。
- 随机化任务时，先随机 scene/object pose，再复用同一套 IK + collision + RRT pipeline 生成专家轨迹。
- 轨迹导出至少包含 `qpos`、`ctrl`、`labels` 和 `model`，方便回放、训练和人工检查。
