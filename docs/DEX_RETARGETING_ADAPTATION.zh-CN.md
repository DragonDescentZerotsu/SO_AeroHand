# dex-retargeting 对 Aero Quest Sim 的适配整理

本文整理 `dexsuite/dex-retargeting` 中适合本项目的设计，并给出 AeroHand
落地边界。目标不是直接复制该仓库，而是复用其成熟的 retargeting 抽象，同时保持：

- Quest Arm Channel 与 Hand Channel 分离；
- Hand Channel 始终使用 `landmarks_wrist`；
- AeroHand 保持当前语义 7D tendon/actuator 控制接口；
- 现有实时 teleop 的公式 retargeter 不被实验优化器阻塞。

参考版本：`dex-retargeting` 主分支，公开包版本 `0.5.0`。

## 1. 两个项目的关键差异

| 项目 | dex-retargeting | Aero Quest Sim |
|---|---|---|
| 人手输入 | MANO/MediaPipe 等关键点 | Quest 21 点 `landmarks_wrist` |
| 输入坐标 | 由调用方转换为操作手坐标 | 明确为 Quest Wrist 局部帧 |
| 机器人模型 | URDF + Pinocchio | MuJoCo MJCF |
| 优化变量 | URDF 中选定的独立关节 qpos | AeroHand 语义 7D action |
| 传动 | 独立关节、mimic adaptor | tendon/actuator 耦合及被动关节 |
| 几何求值 | Pinocchio FK/Jacobian | MuJoCo step 后读取 landmark sites |
| 实时策略 | sequential warm start + LP filter | 公式映射 + EMA；实验优化器主要离线 |

最重要的差异是 AeroHand 的 7D action 并不等于模型中所有手指关节 qpos。
`aero_quest/mujoco_control.py` 将 7 个语义动作映射到 tendon actuator；PIP/DIP 等关节
随后由 MuJoCo 传动和动力学共同决定。因此直接把 AeroHand URDF 交给
`dex-retargeting` 的 Pinocchio optimizer，可能得到运动学上合理、但无法由当前
actuator 接口真实执行的关节姿态。

## 2. 建议吸收的部分

### 2.1 Vector retargeting 的任务定义

这是最适合 AeroHand 的部分。dex-retargeting 的 VectorOptimizer 不要求人手和机器手
具有相同骨长，而是比较一组“起点 link → 终点 link”的向量。

本项目第一阶段建议使用：

```text
wrist → thumb tip
wrist → index tip
wrist → middle tip
wrist → ring tip
wrist → little tip
thumb tip → index tip
thumb tip → middle tip
thumb tip → ring tip
thumb tip → little tip
```

Quest landmark 对应：

```text
wrist: 0
thumb/index/middle/ring/little tip: 4/8/12/16/20
```

机器人对应点直接使用 `aero_quest/mujoco_landmarks.py` 已有的 21 个 landmark sites。
所有人手和机器人点都先分别经过 `palm_localize()`，不要把 wrist-local Quest 点与
MuJoCo world 点直接相减。

相比当前仅比较五指两两距离的 Baseline 2，显式 3D 向量还能保留指尖方向信息。

### 2.2 Sequential retargeting / warm start

`SeqRetargeting` 的核心思想适合保留：

- 当前帧从上一帧 action 开始优化；
- 优化失败时返回上一帧，而不是输出跳变；
- 统一保存调用次数、优化耗时和最后 loss；
- 提供 `reset()`，在丢帧、重新校准或切换手时清空时序状态。

在本项目中，`last_qpos` 应改名为 `previous_action_7d`。初始值优先使用
`quest_points_to_action_7d()` 的公式结果，而不是关节范围中点。

### 2.3 上一帧正则与鲁棒损失

dex-retargeting 使用 Huber/SmoothL1 减少关键点噪声影响，并加入：

```text
L_smooth = ||action - previous_action||²
```

本项目已有相似的 `w_smooth`，建议保留并统一成：

```text
L_total =
    w_vector * L_vector
  + w_bend   * L_bend
  + w_pinch  * L_pinch
  + w_prior  * ||action - formula_action||²
  + w_smooth * ||action - previous_action||²
```

其中 `formula_action` 是稳定且快速的保底解，`previous_action` 负责时间连续性，两者
不要混为一个 prior。

### 2.4 DexPilot 的 pinch 投影思想

DexPilot optimizer 会在指尖距离低于阈值后进入 projected 状态，并使用更高权重维持
抓取；距离超过另一个阈值才退出。这实际上是带 hysteresis 的 pinch 状态机。

适合移植的不是它完整的 Allegro/URDF 实现，而是：

- `project_dist`：进入 pinch；
- `escape_dist`：退出 pinch，且 `escape_dist > project_dist`；
- pinch 激活后提高 thumb-index 向量权重；
- 必要时把目标 pinch 距离投影到 AeroHand 可达到的最小距离。

这能避免 Quest 距离在阈值附近抖动，也避免强迫 AeroHand 达到其机械结构无法实现的
人手距离。

### 2.5 配置驱动的任务映射

dex-retargeting 把 target joint/link 和 human landmark index 放在 YAML 中，这是值得
采用的。AeroHand 配置不应照搬 URDF 字段，而应描述语义 action 和 MuJoCo sites：

```yaml
retargeting:
  backend: mujoco_action_7d
  source_frame: wrist
  canonicalize: palm_local

  action_names:
    - thumb_abduction
    - thumb_flexion_1
    - thumb_flexion_2
    - index_curl
    - middle_curl
    - ring_curl
    - little_curl

  vector_tasks:
    - {origin_human: 0, target_human: 4,
       origin_robot: aero_wrist_lm, target_robot: aero_thumb_tip_lm, weight: 1.0}
    - {origin_human: 0, target_human: 8,
       origin_robot: aero_wrist_lm, target_robot: aero_index_tip_lm, weight: 1.0}
    - {origin_human: 4, target_human: 8,
       origin_robot: aero_thumb_tip_lm, target_robot: aero_index_tip_lm, weight: 3.0}

  optimization:
    vector_weight: 1.0
    bend_weight: 2.0
    pinch_weight: 8.0
    formula_prior_weight: 0.5
    temporal_weight: 0.1
    max_iterations: 20

  pinch:
    enter_distance_normalized: 0.30
    exit_distance_normalized: 0.40
```

配置中的 robot 名称必须按 MuJoCo site/actuator 名称解析，不依赖数组顺序。

### 2.6 显式名称映射

dex-retargeting 特别提醒不同库的 joint order 可能不同。本项目同样应坚持：

- action 使用 `AERO_ACTION_NAMES`；
- actuator 使用 `AERO_HAND_ACTION_MAP`；
- landmark 使用 `ROBOT_LANDMARK_SITE_NAMES`；
- 不把 MuJoCo actuator、qpos 或 site 数组位置硬编码成跨模型契约。

## 3. 暂时不建议引入的部分

### 3.1 Pinocchio RobotWrapper

原因：

- 项目主模型是 MJCF，不是 URDF；
- Pinocchio FK 不会自然复现 AeroHand tendon 和被动关节响应；
- 会形成 MuJoCo 与 Pinocchio 两套机器人几何真值；
- 当前已有 `get_robot_landmarks_21()`，可直接从执行模型读取真实结果。

### 3.2 NLopt + Torch 的解析 Jacobian 路线

dex-retargeting 通过 Pinocchio Jacobian 为 NLopt 提供梯度。AeroHand 当前优化变量是
actuator action，并且需要 simulation settle；从 action 到 sites 的梯度不是现成的
刚体 Jacobian。

当前不引入新的数值优化路径。若未来重新评估优化器，优先把它作为独立 replay
实验，不连接实时 teleop；在有明确实时收益和周期预算之前，不增加 CEM、代理模型或
训练标签链路。

### 3.3 Free-joint wrist warm start

dex-retargeting 的 `warm_start(wrist_pos, wrist_quat)` 用于带 6D floating root 的整只手。
本项目腕部世界位姿属于 Arm Channel，手指 landmarks 属于 Hand Channel。把 free-joint
wrist 混入手指 optimizer 会破坏双通道边界，因此不采用。

### 3.4 直接添加 `dex_retargeting>=0.5`

当前版本要求 NumPy `>=2.0`，同时引入 Pinocchio、NLopt、Torch 相关优化路径。对本项目
来说依赖明显偏重，且核心 URDF optimizer 并不直接适配 AeroHand tendon action。
建议先实现轻量兼容层；只有未来加入独立关节型机械手时，再作为可选 extra。

## 4. 与现有代码的合并建议

当前有三组相近实现：

- `aero_quest/retargeting.py`：实时公式 baseline；
- `aero_quest/action_optimizer.py`：MuJoCo 7D 离线优化；
- `quest_aerohand_retargeting/src/retargeting/`：实验 Baseline 2/3。

建议最终收敛为：

```text
aero_quest/
  retargeting.py                 # 公式 baseline，实时保底
  retargeting_tasks.py           # 人/机器人向量任务及 YAML 解析
  pinch_state.py                 # enter/exit hysteresis

configs/retargeting/
  aero_hand_vector.yaml

scripts/retargeting/
  evaluate_formula_retargeting.py # replay 上计算 vector/pinch/action 指标
```

`action_optimizer.py` 与 `quest_aerohand_retargeting` 可作为历史实验代码保留，但不应
成为默认运行入口，也不应再维护第二套 Quest frame schema、landmark normalization
或 Aero action mapping。

## 5. 推荐实施顺序

### Phase 1：纯 NumPy 任务层

- 新建 `retargeting_tasks.py`；
- 定义 vector task schema；
- 从 `(21,3)` Quest 和 robot landmarks 提取同形状向量；
- 增加 frame/shape/名称校验；
- 单元测试验证 wrist-local → palm-local 后的平移和旋转不变性。

这一阶段不需要 MuJoCo optimizer，也不会改变 teleop。

### Phase 2：公式映射与调试指标迭代

- 在 replay 中计算 vector error 和 pinch error；
- 使用这些指标调节现有公式映射及动作范围；
- 保持 7D action 的 EMA、超时回退和 latest-frame 架构；
- 不生成 pseudo-label，不引入训练模型依赖。

### Phase 3：pinch hysteresis / projection

- 从 DexPilot 借鉴 enter/exit 双阈值状态；
- 根据 `test_aero_landmark_reachability.py` 测得的机械可达距离设置 projection；
- 对 thumb-index 与 thumb-middle 分别配置，不写死同一个阈值。

### Phase 4：评估后再决定实时化

- 使用相同 Quest NPZ replay 比较公式、vector、pinch；
- 指标至少包括 vector error、pinch error、action jerk、失败率和单帧耗时；
- 只有优化器稳定达到控制周期预算后，才作为可选 teleop backend；
- 实时路径必须保留 latest-frame、超时回退和公式 action fallback。

## 6. 推荐的最小适配接口

```python
class FormulaAeroRetargeter:
    def reset(self) -> None: ...

    def retarget(self, landmarks_wrist: np.ndarray) -> tuple[np.ndarray, dict]:
        """Return semantic action_7d and stable debug features."""
        ...
```

输入名称显式包含 `_wrist`，防止未来误传 world landmarks。输出保持项目现有 7D 语义
action，不向 teleop 暴露模型内部 qpos。

## 7. 一句话结论

对本项目最有价值的是 dex-retargeting 的 **vector task、sequential warm start、
Huber + temporal regularization、DexPilot pinch hysteresis、YAML 名称映射**。不应直接
引入的是它的 **URDF/Pinocchio robot backend、独立 joint qpos 输出和 free-joint wrist
处理**。AeroHand 应继续以 MuJoCo 中真实 tendon actuator 响应作为优化几何真值。

## 8. 当前已落地的适配

项目内现在已有：

- `aero_quest/retargeting_tasks.py`：YAML vector task、名称校验、向量提取和 Huber loss；
- `aero_quest/pinch_state.py`：thumb-index 双阈值 hysteresis；
- `configs/retargeting/aero_hand_vector.yaml`：AeroHand 9 项向量任务；
- `aero_quest/retargeting.py`：公式 7D action 保持主线，并输出稳定 `pinch_active`；
- 实时 skeleton viewer：显示向量和 pinch 指标，但不参与控制。

当前明确舍弃 pseudo-label/训练路线。Vector task 保留为调试和客观误差指标，不作为
MuJoCo 离线优化目标，也不改变 teleop 默认行为。

## 参考

- dex-retargeting repository: <https://github.com/dexsuite/dex-retargeting>
- optimizer implementations:
  <https://github.com/dexsuite/dex-retargeting/blob/main/src/dex_retargeting/optimizer.py>
- sequential wrapper:
  <https://github.com/dexsuite/dex-retargeting/blob/main/src/dex_retargeting/seq_retarget.py>
- Ability Hand vector config:
  <https://github.com/dexsuite/dex-retargeting/blob/main/src/dex_retargeting/configs/teleop/ability_hand_right.yml>
- MIT license:
  <https://github.com/dexsuite/dex-retargeting/blob/main/LICENSE>
