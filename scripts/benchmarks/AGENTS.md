# Benchmark 脚本说明

## Piper IK 自动验证

入口：

```bash
python scripts/benchmarks/piper_ik_benchmark.py
```

这个脚本不需要 Quest。它会：

- 从有效 Piper 关节配置通过 FK 生成可达的 palm 位姿目标。
- palm 位姿使用 `aero_wrist_site`，即 Aero palm 的安装/腕部基准点，不是指尖或 `grasp_site`。
- 连续测试位置、wrist roll、wrist flex 和回到 home 等目标。
- 使用与 Piper 遥操作相同的 warm-start OSQP IK、关节权重、速度/加速度限制和奇异值 damping。
- 在 MuJoCo 动力学中通过 position actuator 执行，而不是直接写入 qpos。
- 统计到达时间、最终位置误差、最终姿态误差、OSQP 求解时间、迭代次数、最小奇异值和实际 damping。
- 任一目标失败或性能不达标时返回非零退出码。

默认通过标准：

```text
位置误差 <= 10 mm
姿态误差 <= 4 deg
连续保持 >= 0.15 s
每个目标 settle time <= 3 s
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
  --output-dir outputs/piper_ik_benchmark/final
```

视频输出：

```text
outputs/piper_ik_benchmark/final/piper_osqp_ik_benchmark.mp4
```

视频中的黄色球表示目标位置，红绿蓝轴表示目标 palm 朝向。左上角显示目标名称、仿真时间、位置误差、姿态误差和 OSQP 状态。

常用参数：

- `--timeout`：每个目标的最大仿真时间。
- `--position-tolerance-mm`：位置通过阈值。
- `--orientation-tolerance-deg`：姿态通过阈值。
- `--hold-seconds`：必须连续保持在阈值内的时间。
- `--max-settle-seconds`：允许的最大到达时间。
- `--max-osqp-p95-ms`：OSQP p95 求解时间上限。
- `--min-command-lead` / `--max-command-lead`：position actuator 命令领先实际关节角的范围。
- `--record-video`：开启离屏视频录制。

目标必须优先由有效关节配置通过 FK 生成。不要随意手写可能不可达的 Cartesian pose，否则 benchmark 会混淆 IK 性能和可达性问题。
