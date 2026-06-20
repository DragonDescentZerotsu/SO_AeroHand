# 遥操作模块边界

遥操作代码按数据所有权分为三层：

```text
Quest 接收/类型层
├── Arm Channel: wrist_pos_world / wrist_quat_world
└── Hand Channel: landmarks_wrist

独立控制层
├── 机械臂: arm_teleop.py + osqp_ik.py
└── Aero Hand: aero_hand_teleop.py + retargeting.py + mujoco_control.py

组合入口
└── scripts/teleop/quest_so101_aero_nullspace_ik_teleop.py
```

## 机械臂代码

- `aero_quest/arm_teleop.py`：task-space 速度控制、MuJoCo Jacobian 和机械臂接口。
- `aero_quest/osqp_ik.py`：可复用的约束速度 IK。
- `aero_quest/quest_hand_frame.py`：Arm Channel 的相对腕部映射和坐标帧类型。
- `scripts/teleop/quest_arm_channel_so101_ik.py`：SO101 纯机械臂入口。
- `scripts/teleop/quest_arm_channel_piper_ik.py`：Piper 纯机械臂入口。

机械臂层不得导入 `retargeting.py` 或读取 Aero Hand 7D 动作。

## Aero Hand 代码

- `aero_quest/aero_hand_teleop.py`：实时 Hand Channel 状态、平滑、pinch 和任务 profile。
- `aero_quest/retargeting.py`：wrist-local landmarks 到语义 7D 动作。
- `aero_quest/mujoco_control.py`：语义 7D 动作到 Aero Hand actuator。
- `aero_quest/retargeting_tasks.py`：离线 vector task 和误差评估。
- `scripts/teleop/quest_tcp_aero_teleop.py`：Aero Hand 独立入口。

Aero Hand 层只接收 `landmarks_wrist`，不得依赖 Quest 世界位姿、`R_BQ` 或机械臂 IK。

## 组合入口

完整遥操入口只负责：

1. 接收并解析同一 Quest 帧。
2. 把 Arm Channel 送给机械臂控制器。
3. 把 Hand Channel 送给 `AeroHandTeleopChannel`。
4. 在同一个 MuJoCo step 前分别写入 arm 和 hand actuator。

新增机器人时，应新增一个很薄的默认参数包装入口；不要复制 Aero Hand
retargeting。新增手部 profile 时，只修改 Aero Hand 层；不要修改机械臂 IK。
