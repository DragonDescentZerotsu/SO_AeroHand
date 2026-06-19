# 待删除：内容已迁移

本文件原本同时承担用户说明和项目记忆，已经与 `AGENTS.md`、代码实现及专门文档产生重复和过时内容。

仍然有效的信息已迁移到：

- `AGENTS.md`：项目架构、环境初始化、代码地图、场景构建、遥测工具、验证流程和项目规则。
- `scripts/teleop/AGENTS.md`：遥操作入口、Mac/SSH/ADB 连接方式、坐标帧、控制参数和故障排查。
- `scripts/benchmarks/AGENTS.md`：Piper IK 自动 benchmark 和视频验证。
- `docs/quest_dual_channel_pipeline.md`：双通道坐标帧和控制公式。
- `docs/quest_telemetry_layer.md`：Quest 数据记录、质量分析和回放。

本次清理发现并丢弃的过时描述包括：

- 项目只支持 SO101。
- SO101 当前仍采用“三关节位置 IK + 两个 wrist 关节单独控制”。
- Piper 默认使用旧的 `full_pose` 模式；当前入口默认是 `osqp_full_pose`。
- Piper 复用 SO101 命名脚本作为通用主逻辑；共享实现现已独立为 `quest_aero_arm_ik_teleop.py`。

确认上述迁移结果后，可以直接删除本文件。
