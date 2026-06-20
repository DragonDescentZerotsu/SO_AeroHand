# 真实 AeroHand MuJoCo 仿真接入说明

本文说明 `quest_aerohand_retargeting` 里真实 `AeroHandSimEnv` 如何和 Baseline 1 结合，以及后续 Baseline 2/3 应该怎样复用这条链路。

## 目标

我们要把原来的 placeholder 评估：

```text
7D action -> 手写公式估计 robot_thumb_index_distance
```

替换成真实 MuJoCo 评估：

```text
7D action
  -> MuJoCo ctrl
  -> MuJoCo step
  -> 读取 aero_thumb_tip_site / aero_index_tip_site
  -> robot_thumb_index_distance
```

这样 `robot_thumb_index_distance` 才是 AeroHand 在仿真里的真实指尖距离，而不是一个假公式。

同时现在也读取 AeroHand 的 21 个 `aero_*_lm` landmark site，用来和 Quest 3 的 21 点手骨架做离线形状对比。

## 当前配置

默认配置在：

```text
quest_aerohand_retargeting/configs/default.yaml
```

关键字段：

```yaml
sim:
  model_path: models/piper_aero_hand/Piper_aerohand.xml
  use_placeholder: false
  thumb_tip_site: aero_thumb_tip_site
  index_tip_site: aero_index_tip_site
  middle_tip_site: aero_middle_tip_site
  ring_tip_site: aero_ring_tip_site
  little_tip_site: aero_little_tip_site
  settle_steps: 20
```

含义：

- `model_path`：加载 Piper + AeroHand 组合模型。
- `use_placeholder: false`：使用真实 MuJoCo，不再使用 placeholder 距离公式。
- `thumb_tip_site`：AeroHand 拇指指尖 site。
- `index_tip_site`：AeroHand 食指指尖 site。
- `middle/ring/little_tip_site`：Baseline 2 和可视化需要读取的另外三个指尖 site。
- `settle_steps`：每个 action 写入后 MuJoCo 前进多少步，再读取指尖位置。

## 数据流

真实记录评估脚本：

```text
quest_aerohand_retargeting/scripts/evaluate_recorded_baseline1.py
```

完整链路：

```text
data/teleop_episodes/baseline1_test.jsonl
        ↓
load_quest_dual_channel_jsonl
        ↓
HandFrame.landmarks_wrist
        ↓
DirectPoseMappingRetargeter
        ↓
aero_quest.retargeting.quest_points_to_action_7d
        ↓
7D AeroHand action
        ↓
AeroHandSimEnv.step(action)
        ↓
normalized_aero_hand_to_ctrl
        ↓
mujoco.mj_step
        ↓
data.site_xpos[aero_thumb_tip_site]
data.site_xpos[aero_index_tip_site]
        ↓
robot_thumb_index_distance
```

## 核心代码

### 1. Baseline 1 产生 7D action

文件：

```text
quest_aerohand_retargeting/src/retargeting/baseline1_direct_mapping.py
```

核心逻辑：

```python
action = quest_points_to_action_7d(landmarks_wrist)
```

输出 action 顺序：

```text
[
  thumb_abduction,
  thumb_flexion_1,
  thumb_flexion_2,
  index_curl,
  middle_curl,
  ring_curl,
  little_curl,
]
```

### 2. AeroHandSimEnv 写入 MuJoCo ctrl

文件：

```text
quest_aerohand_retargeting/src/sim/aerohand_env.py
```

核心逻辑：

```python
ctrl = self._base_ctrl.copy()
self.data.ctrl[:] = normalized_aero_hand_to_ctrl(self.model, self.action, ctrl=ctrl)
for _ in range(self.settle_steps):
    mujoco.mj_step(self.model, self.data)
```

这里复用了主项目的映射：

```text
aero_quest.so101_aero_control.normalized_aero_hand_to_ctrl
```

它会把 7D semantic action 映射到 AeroHand tendon / joint actuator：

```text
thumb_abduction -> right_thumb_A_cmc_abd
thumb_flexion_1 -> right_th1_A_tendon
thumb_flexion_2 -> right_th2_A_tendon
index_curl      -> right_index_A_tendon
middle_curl     -> right_middle_A_tendon
ring_curl       -> right_ring_A_tendon
little_curl     -> right_pinky_A_tendon
```

### 3. 读取真实指尖 site

核心逻辑：

```python
thumb_tip = data.site_xpos[thumb_site_id]
index_tip = data.site_xpos[index_site_id]
robot_pinch_distance = np.linalg.norm(thumb_tip - index_tip)
```

当前使用：

```text
aero_thumb_tip_site
aero_index_tip_site
```

这两个 site 已经存在于：

```text
models/piper_aero_hand/Piper_aerohand.xml
models/so101_aero_hand/SO101_aerohand.xml
```

### 4. 读取 AeroHand 21 点骨架 site

为了和 Quest 3 `landmarks_wrist[0:21]` 对齐，`AeroHandSimEnv` 现在按 Quest landmark 顺序读取这些 MuJoCo site：

```text
aero_wrist_lm
aero_thumb_metacarpal_lm
aero_thumb_proximal_lm
aero_thumb_distal_lm
aero_thumb_tip_lm
aero_index_proximal_lm
aero_index_intermediate_lm
aero_index_distal_lm
aero_index_tip_lm
aero_middle_proximal_lm
aero_middle_intermediate_lm
aero_middle_distal_lm
aero_middle_tip_lm
aero_ring_proximal_lm
aero_ring_intermediate_lm
aero_ring_distal_lm
aero_ring_tip_lm
aero_little_proximal_lm
aero_little_intermediate_lm
aero_little_distal_lm
aero_little_tip_lm
```

注意：Quest 的 `landmarks_wrist` 是 Wrist 局部坐标；MuJoCo 的 `hand_landmarks` 是 MuJoCo/world 坐标。可视化脚本会把两边各自平移到 wrist 原点并按自身尺度归一化，所以图里比较的是手型结构，不是同一个空间中的绝对位置。

## 如何运行

对真实 Quest 记录跑 Baseline 1 + MuJoCo 指尖距离评估：

```bash
python quest_aerohand_retargeting/scripts/evaluate_recorded_baseline1.py \
  --input data/teleop_episodes/baseline1_test.jsonl
```

输出目录：

```text
quest_aerohand_retargeting/data/processed/baseline1_recorded/
```

主要输出：

```text
baseline1_metrics.json
baseline1_records.json
baseline1_actions.csv
baseline1_action_curves.png
baseline1_pinch_distances.png
```

其中：

- `human_pinch_distance` 来自 Quest `landmarks_wrist[4]` 和 `landmarks_wrist[8]`。
- `robot_pinch_distance` 来自 MuJoCo `aero_thumb_tip_site` 和 `aero_index_tip_site`。
- `action` 是 Baseline 1 输出的 7D AeroHand action。

画 Quest 3 手骨架和 MuJoCo AeroHand 骨架对比：

```bash
python quest_aerohand_retargeting/scripts/visualize_quest_aero_skeletons.py \
  --input data/teleop_episodes/baseline1_test.jsonl \
  --method baseline1b \
  --num-frames 6
```

输出：

```text
quest_aerohand_retargeting/data/processed/skeleton_compare/baseline1b_skeletons.png
```

也可以指定具体帧：

```bash
python quest_aerohand_retargeting/scripts/visualize_quest_aero_skeletons.py \
  --input data/teleop_episodes/baseline1_test.jsonl \
  --method baseline1b \
  --frame-indices 0,500,1000,1500,2000,2500,3000
```

## 当前结果如何理解

在 `baseline1_test.jsonl` 上，当前结果大致是：

```text
human_pinch_distance mean:        0.0756 m
robot_thumb_index_distance mean:  0.1169 m
robot_thumb_index_distance min:   0.0198 m
robot_thumb_index_distance median:0.1383 m
```

这说明：

- MuJoCo 接入已经生效，因为 robot distance 能随 action 改变。
- Baseline 1 偶尔能把 AeroHand 指尖拉近到约 2cm。
- 但整体中位数仍然很大，说明普通几何 retargeting 并不能稳定完成 pinch。

这个结论正是后续 Baseline 1b / Baseline 2 / Baseline 3 的依据。

## 后续如何复用

### Baseline 1b: Direct Pinch Correction

可以先在 Baseline 1 action 后加一个直接 pinch correction：

```python
pinch_strength = f(human_thumb_index_distance)
action[1] = max(action[1], 0.45 * pinch_strength)
action[2] = max(action[2], 0.55 * pinch_strength)
action[3] = max(action[3], 0.75 * pinch_strength)
```

然后仍然通过：

```python
obs = sim_env.step(action)
robot_dist = obs["robot_pinch_distance"]
```

判断是否真的让 MuJoCo 里的 AeroHand 指尖靠近。

当前代码入口：

```text
quest_aerohand_retargeting/src/retargeting/baseline1_direct_mapping.py
```

类名：

```python
PinchAugmentedDirectRetargeter
```

配置：

```yaml
retargeting:
  baseline1b:
    thumb_abduction_min: 0.20
    thumb_flexion_1_min: 0.45
    thumb_flexion_2_min: 0.55
    index_curl_min: 0.75
    blend: 1.0
```

运行 Baseline 1 和 Baseline 1b 对比：

```bash
python quest_aerohand_retargeting/scripts/evaluate_recorded_baseline1.py \
  --input data/teleop_episodes/baseline1_test.jsonl \
  --method both
```

### Baseline 2: Optimized Vector Retargeting

优化器可以把 `AeroHandSimEnv` 当作 forward model：

```text
给定 action u
  -> sim_env.step(u)
  -> 读 robot fingertip positions
  -> 计算 L_vector / L_smooth / L_limit
```

目标形式：

```text
min_u L_vector + L_smooth + L_limit
```

当前轻量实现：

```text
quest_aerohand_retargeting/src/retargeting/baseline2_optimized_retargeting.py
```

类名：

```python
MuJoCoVectorOptimizedRetargeter
```

它每帧会：

1. 用 Baseline 1 生成一个 seed action。
2. 在 seed action 和上一帧 action 附近采样若干候选 7D action。
3. 对每个候选 action 调用 `AeroHandSimEnv.evaluate_action(action)`。
4. 读取 MuJoCo 里的五个 fingertip site：
   - `aero_thumb_tip_site`
   - `aero_index_tip_site`
   - `aero_middle_tip_site`
   - `aero_ring_tip_site`
   - `aero_little_tip_site`
5. 比较人手和 AeroHand 的五指 fingertip pairwise distance shape。
6. 选择 `L_vector + L_smooth + L_limit` 最小的候选。

运行：

```bash
python quest_aerohand_retargeting/scripts/evaluate_recorded_baseline1.py \
  --input data/teleop_episodes/baseline1_test.jsonl \
  --method baseline2 \
  --max-frames 200
```

完整对比：

```bash
python quest_aerohand_retargeting/scripts/evaluate_recorded_baseline1.py \
  --input data/teleop_episodes/baseline1_test.jsonl \
  --method all
```

注意：`--method all` 会同时跑 Baseline 1、1b、2。Baseline 2 每帧会评估多个 MuJoCo 候选动作，全量记录会明显更慢。调参时建议先加 `--max-frames 200`。

### Baseline 3: Pinch-aware / Contact-aware Retargeting

在 Baseline 2 基础上加：

```text
L_pinch = ||robot_thumb_tip - robot_index_tip|| - human_pinch_distance
```

或更直接：

```text
if human_pinch_event:
    minimize robot_thumb_index_distance
```

目标形式：

```text
min_u L_vector + lambda_pinch * L_pinch + L_smooth + L_limit
```

当前实现：

```text
quest_aerohand_retargeting/src/retargeting/baseline3_contact_aware_retargeting.py
```

类名：

```python
MuJoCoPinchAwareRetargeter
```

它会复用 Baseline 2 的 MuJoCo sampled-action forward model，并额外加入：

```text
L_pinch = (robot_pinch_distance - human_pinch_distance)^2
L_close = robot_pinch_distance^2, only when human pinch_strength is high
```

默认配置：

```yaml
retargeting:
  baseline3:
    num_candidates: 40
    sample_radius: 0.24
    local_radius: 0.10
    seed: 11
    lambda_pinch: 25.0
    lambda_close: 10.0
    pinch_strength_threshold: 0.55
    contact_distance_m: 0.03
```

运行：

```bash
python quest_aerohand_retargeting/scripts/evaluate_recorded_baseline1.py \
  --input data/teleop_episodes/baseline1_test.jsonl \
  --method baseline3 \
  --max-frames 300
```

Baseline 3 的 `records.json` 会额外写入每帧 `objective_terms`，方便看 `L_vector`、`L_pinch`、`L_close` 是否在主导优化。

后续如果 contact geom 稳定，也可以加入：

```python
obs["contact"]
```

作为 contact success 指标或 loss 项。

## 注意事项

1. `landmarks_wrist` 是 Quest wrist-local 坐标，不要当成 MuJoCo world 坐标。
2. `robot_thumb_index_distance` 是 MuJoCo world 中两个 fingertip site 的距离。
3. 当前 `AeroHandSimEnv` 每次 `step` 会连续前进 `settle_steps` 个 MuJoCo step；如果评估太慢，可以降低这个值。
4. 当前评估只控制 AeroHand，arm 保持在模型初始/ctrl midpoint 状态。
5. 如果切换 SO101 模型，只需要改 `model_path`，site 名称仍然可以使用 `aero_thumb_tip_site` / `aero_index_tip_site`。
