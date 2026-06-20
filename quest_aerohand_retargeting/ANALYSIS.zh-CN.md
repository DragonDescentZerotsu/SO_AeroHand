# 当前代码与初版项目大纲的差距分析

## 你已经完成了什么

当前主仓库已经不只是空骨架，已经完成了相当多“真实遥操作主线”的工作：

1. **Quest 数据接收已经有实用实现**
   - `aero_quest/quest_receiver.py` 已经封装了 `hand-tracking-sdk` 的 TCP/UDP 数据接入。
   - `aero_quest/quest_hand_frame.py` 和 `aero_quest/quest_dual_channel.py` 已经明确区分 Arm Channel 和 Hand Channel。
   - `scripts/04_receive_quest_tcp.py`、`scripts/debug_quest_dual_channel.py`、`scripts/record_quest_dual_channel.py`、`scripts/replay_quest_dual_channel.py` 已经覆盖了接收、调试、记录和回放。

2. **关键坐标帧约定已经建立**
   - 当前代码已经明确：`wrist_pos_world` / `wrist_quat_world` 位于 Quest/Unity 世界追踪帧 `Q`。
   - `landmarks_wrist` 位于 Quest 手腕局部帧 `Wrist`。
   - `RelativeWristArmController` 已经支持 `R_BQ`，没有把 Quest 世界原点硬当成机器人基座。

3. **AeroHand 7D 几何 retargeting 已经有可用 baseline**
   - `aero_quest/retargeting.py` 已经把 21 个 Quest landmarks 转成 Aero Hand 语义 7D action：
     `[thumb_abduction, thumb_flexion_1, thumb_flexion_2, index_curl, middle_curl, ring_curl, little_curl]`
   - 这已经比新要求里的 “Baseline 1: Direct Pose Mapping” 更接近真实控制。

4. **MuJoCo 模型和遥操作入口已经存在**
   - SO101 + AeroHand：`scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py`
   - Piper + AeroHand：`scripts/teleop/quest_piper_aero_ik_teleop.py`
   - AeroHand-only：`scripts/teleop/quest_tcp_aero_teleop.py`
   - 还有场景生成器、pipette 任务 YAML、SO101/Piper 组合模型。

5. **记录、质量检查、latency 分析和 replay 已经初步覆盖**
   - `aero_quest/quest_logger.py`
   - `aero_quest/quality_filter.py`
   - `aero_quest/quest_data_quality.py`
   - `scripts/analyze_quest_latency.py`

## 和这次“大纲要求”相比还缺什么

这次要求的重点不是直接提高遥操作控制，而是搭建一个更像研究实验工程的结构，用来比较 retargeting baseline、pinch/contact-aware 方法和 residual correction。差距主要在这里：

1. **缺少独立的 baseline/evaluation 管线**
   - 现有主线偏“实时遥操作能跑起来”。
   - 新要求需要“同一段 Quest/mock 数据 -> 多个 baseline -> 同一组 metrics”的比较框架。
   - 我新增的 `quest_aerohand_retargeting/scripts/evaluate_baselines.py` 正是补这个缺口。

2. **缺少清晰的实验模块边界**
   - 现有功能集中在 `aero_quest/`，偏工程实用。
   - 新要求希望按 `quest_io / preprocessing / visualization / retargeting / sim / residual / evaluation` 分层。
   - 我新增了这套子项目结构，方便后续论文实验或 ablation。

3. **pinch-aware / contact-aware 还没有真正实现**
   - 现有代码有 pinch distance、closure features、Aero 7D action，但还没有显式 `L_pinch` / contact loss 的优化接口。
   - 新骨架里的 `baseline3_contact_aware_retargeting.py` 现在只是接口和轻量 placeholder，后续需要接 MuJoCo fingertip site 和 contact state。

4. **Optimized Vector Retargeting 还没有优化器**
   - 现有 7D retargeting 是公式启发式，不是 `min_u L_vector + L_smooth + L_limit`。
   - 新骨架保留了 objective terms，但还没有 scipy/CEM/finite-difference optimizer。

5. **Simulation Residual Correction 还没有模型**
   - 现有代码没有 residual policy。
   - 新骨架里的 `SimResidualCorrector` 先返回 zero residual，只定义了未来输入输出：
     human pinch features、baseline action、sim thumb/index tip、contact state -> delta action。

6. **visualization overlay 还只是 placeholder**
   - 现有 MuJoCo viewer 能看机器人，但还没有统一显示 human skeleton、retargeting target、actual AeroHand pose 的 overlay。
   - 新骨架里预留了 `visualization/human_skeleton_viewer.py` 和 `visualization/mujoco_overlay.py`。

7. **指标体系还需要接真实任务状态**
   - 当前新增 metrics 已经有 pinch distance、robot distance、fingertip error、success rate、smoothness、latency。
   - 但 object slip rate、contact success、final fingertip error 仍是 placeholder 级别，需要真实 MuJoCo object/contact 数据。

## 这次新增的内容

新增子项目目录：

```text
quest_aerohand_retargeting/
    README.md
    ANALYSIS.zh-CN.md
    MUJOCO_SIM_INTEGRATION.zh-CN.md
    requirements.txt
    configs/default.yaml
    data/raw/.gitkeep
    data/processed/.gitkeep
    scripts/
        record_quest_data.py
        replay_quest_data.py
        run_aerohand_sim.py
        evaluate_baselines.py
    src/
        quest_io/
        preprocessing/
        visualization/
        retargeting/
        sim/
        residual/
        evaluation/
        utils/
```

当前可运行 demo：

```bash
python quest_aerohand_retargeting/scripts/evaluate_baselines.py --config quest_aerohand_retargeting/configs/default.yaml
```

真实 Quest 记录的 Baseline 1 离线评估：

```bash
python quest_aerohand_retargeting/scripts/evaluate_recorded_baseline1.py --input data/teleop_episodes/baseline1_test.jsonl
```

输出：

- `quest_aerohand_retargeting/data/processed/mock_quest_hand.jsonl`
- `quest_aerohand_retargeting/data/processed/mock_baseline_records.json`
- `quest_aerohand_retargeting/data/processed/mock_metrics.json`
- `quest_aerohand_retargeting/data/processed/baseline1_recorded/baseline1_actions.csv`
- `quest_aerohand_retargeting/data/processed/baseline1_recorded/baseline1_metrics.json`
- `quest_aerohand_retargeting/data/processed/baseline1_recorded/baseline1_action_curves.png`
- `quest_aerohand_retargeting/data/processed/baseline1_recorded/baseline1_pinch_distances.png`

## 下一步建议

1. 把 `MockHTSReceiver` 替换/适配到 `aero_quest.quest_receiver.QuestTelemetryReceiver`。
2. 继续扩大完整 21 landmarks 支持，让真实记录和 mock 记录使用同一套 frame schema。
3. Baseline 1 已接到现有 `aero_quest.retargeting.quest_points_to_action_7d`；下一步需要用真实 Quest replay 数据验证动作曲线。
4. MuJoCo 模型里的 `aero_thumb_tip_site` / `aero_index_tip_site` 已接入 `AeroHandSimEnv`；下一步需要用这些真实 site distance 调整 Baseline 1b 或进入 Baseline 2。
5. 用 scipy 或 CEM 实现 Baseline 2 的优化器。
6. 用 MuJoCo contact 和 fingertip distance 实现 Baseline 3 的 pinch/contact loss。
7. 采集更多 replay 数据，跑 baseline 指标对比，再决定 residual correction 的训练数据格式。

更具体的 MuJoCo 接入链路、site 读取方式和后续 Baseline 2/3 复用方式见：

```text
quest_aerohand_retargeting/MUJOCO_SIM_INTEGRATION.zh-CN.md
```
