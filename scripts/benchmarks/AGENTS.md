# Benchmark 脚本说明

## Piper IK 自动验证

入口：

```bash
python scripts/benchmarks/piper_ik_benchmark.py
```

这个脚本不需要 Quest。它会：

- 从有效 Piper 关节配置通过 FK 生成可达的 palm 位姿目标。
- palm 位姿使用 `aero_wrist_site`，即 Aero palm 的安装/腕部基准点，不是指尖或 `grasp_site`。
- 默认运行 `full` 套件：16 个固定挑战、100 个固定随机种子的可达位姿、16 秒连续 6D palm 轨迹，以及固定位置的 wrist pitch `+70° -> -70°` 压力测试。
- 固定挑战覆盖左右/高低工作区、远近伸展、正负 wrist roll/flex、组合姿态和近奇异伸展。
- 固定和随机位姿分别从同一个 home 关节分支开始，避免把任意大步换肘型误判为局部速度 IK 的性能。
- 连续轨迹围绕条件良好的非奇异姿态运行；home 本身接近 6D Jacobian 奇异点，不适合作为位置和姿态同时变化的周期轨迹中心。
- wrist sweep 压力测试故意请求受 joint5 限位约束的不可精确到达姿态。它验证奇异/限位区域是否平滑降低姿态优先级、保护 palm 位置并最终恢复，不把姿态残差误报成控制器失败。
- 使用与 Piper 遥操作相同的 warm-start OSQP IK、关节权重、速度/加速度限制和奇异值 damping。
- 在 MuJoCo 动力学中通过 position actuator 执行，而不是直接写入 qpos。
- 统计到达时间、最终位置误差、最终姿态误差、OSQP 求解时间、迭代次数、最小奇异值和实际 damping。
- 完整结果写入 JSON；不应只根据视频判断性能。

默认通过标准：

```text
固定挑战成功率 = 100%
随机目标成功率 >= 95%
位置误差 <= 10 mm
姿态误差 <= 4 deg
连续保持 >= 0.15 s
每个目标 settle time <= 3 s
连续轨迹 position P95 <= 25 mm
连续轨迹 orientation P95 <= 8 deg
wrist sweep 最大位置误差 <= 40 mm
wrist sweep 最终位置误差 <= 10 mm
OSQP p95 wall time <= 1 ms
OSQP failure = 0
```

输出：

```text
outputs/piper_ik_benchmark/summary.json
```

录制视频：

```bash
python scripts/benchmarks/piper_ik_benchmark.py \
  --record-video \
  --output-dir outputs/piper_ik_benchmark/full
```

视频输出：

```text
outputs/piper_ik_benchmark/full/piper_osqp_ik_benchmark.mp4
```

视频中的黄色球表示目标位置，红绿蓝轴表示目标 palm 朝向。左上角显示目标名称、仿真时间、位置误差、姿态误差和 OSQP 状态。

完整模式的视频不会播放 100 个随机统计目标。它播放原有 5 点连续回归序列、完整 6D 周期轨迹和固定位置 wrist sweep，便于人工检查跟随平滑度与姿态饱和行为；随机覆盖结果保存在 `summary.json`。

常用参数：

- `--suite full|smoke`：完整验证或旧版 5 点快速冒烟测试；默认 `full`。
- `--random-targets` / `--random-seed`：随机覆盖数量和确定性种子；默认 100 和 `20260619`。
- `--timeout`：每个目标的最大仿真时间。
- `--position-tolerance-mm`：位置通过阈值。
- `--orientation-tolerance-deg`：姿态通过阈值。
- `--hold-seconds`：必须连续保持在阈值内的时间。
- `--max-settle-seconds`：允许的最大到达时间。
- `--min-random-success-rate`：随机目标最低成功率。
- `--max-osqp-p95-ms`：OSQP p95 求解时间上限。
- `--trajectory-duration` / `--trajectory-period`：连续轨迹总时长和周期。
- `--trajectory-p95-position-mm` / `--trajectory-p95-orientation-deg`：连续轨迹 P95 误差阈值。
- `--wrist-sweep-max-position-mm` / `--wrist-sweep-final-position-mm`：不可达姿态压力测试的位置保护阈值。
- `--min-command-lead` / `--max-command-lead`：position actuator 命令领先实际关节角的范围。
- `--record-video`：开启离屏视频录制。

目标必须优先由有效关节配置通过 FK 生成。不要随意手写可能不可达的 Cartesian pose，否则 benchmark 会混淆 IK 性能和可达性问题。
