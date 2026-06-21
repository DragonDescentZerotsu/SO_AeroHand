from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.arm_teleop import DampedLeastSquaresIK, joint_qpos, joint_ranges
from aero_quest.osqp_ik import OSQPIKConfig, OSQPVelocityIK
from scripts.teleop.quest_aero_arm_ik_teleop import set_arm_ctrl_targets, set_arm_qpos


def test_piper_osqp_ik_respects_bounds_and_damps_near_singularity():
    model = mujoco.MjModel.from_xml_path(str(PROJECT_ROOT / "models/piper_aero_hand/Piper_aerohand.xml"))
    data = mujoco.MjData(model)
    data.ctrl[:] = 0.5 * (model.actuator_ctrlrange[:, 0] + model.actuator_ctrlrange[:, 1])
    ik = DampedLeastSquaresIK(
        model,
        ee_site="aero_wrist_site",
        joint_names=["joint1", "joint2", "joint3", "joint4", "joint5", "joint6"],
        damping=0.035,
        max_joint_speed=5.0,
    )
    qhome = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float64)
    set_arm_qpos(model, data, ik.joint_ids, qhome)
    set_arm_ctrl_targets(model, data, ik.joint_ids, qhome)
    mujoco.mj_forward(model, data)

    solver = OSQPVelocityIK(
        joint_count=6,
        task_dimension=6,
        joint_motion_weights=np.array([0.7, 1.0, 1.0, 0.35, 0.22, 0.08], dtype=np.float64),
        task_weights=np.array([1.0, 1.0, 1.0, 1.2, 1.2, 1.2], dtype=np.float64),
        config=OSQPIKConfig(
            base_damping=0.035,
            accel_weight=0.04,
            max_joint_speed=5.0,
            max_joint_accel=120.0,
            singular_damping_threshold=0.10,
            singular_damping_gain=0.10,
        ),
    )
    lower, upper = joint_ranges(model, ik.joint_ids)
    result = solver.solve(
        ik.jacobian(data, control_orientation=True),
        np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float64),
        joint_qpos(model, data, ik.joint_ids),
        lower,
        upper,
        dt=0.005,
    )
    qdot = result.qdot
    qtarget = joint_qpos(model, data, ik.joint_ids) + qdot * 0.005

    assert result.status.lower().startswith("solved")
    assert np.all(np.isfinite(qtarget))
    assert np.all(np.isfinite(qdot))
    assert np.all(np.abs(qdot) <= 5.0 + 1e-6)
    assert np.all(np.abs(qdot) <= 120.0 * 0.005 + 1e-4)
    assert result.min_singular < 0.10
    assert result.effective_damping > ik.damping
    assert 0.0 < result.orientation_scale < 1.0
    assert abs(qdot[5]) > 1e-3

    second = solver.solve(
        ik.jacobian(data, control_orientation=True),
        np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float64),
        joint_qpos(model, data, ik.joint_ids),
        lower,
        upper,
        dt=0.005,
    )
    assert second.status.lower().startswith("solved")
    assert second.wall_time_s < 0.01


def test_osqp_adaptive_orientation_reduces_priority_near_joint_limit():
    solver = OSQPVelocityIK(
        joint_count=6,
        task_dimension=6,
        joint_motion_weights=np.ones(6),
        task_weights=np.ones(6),
        config=OSQPIKConfig(
            adaptive_orientation=True,
            orientation_singularity_threshold=0.05,
            orientation_joint_limit_margin=0.20,
            minimum_orientation_scale=0.08,
        ),
    )
    jacobian = np.eye(6)
    lower = -np.ones(6)
    upper = np.ones(6)
    task_velocity = np.ones(6)

    centered = solver.solve(jacobian, task_velocity, np.zeros(6), lower, upper, dt=0.01)
    solver.reset()
    near_limit = solver.solve(
        jacobian,
        task_velocity,
        np.array([0.95, 0.0, 0.0, 0.0, 0.0, 0.0]),
        lower,
        upper,
        dt=0.01,
    )

    assert centered.orientation_scale == 1.0
    assert np.isclose(near_limit.orientation_scale, 0.25)
    assert np.linalg.norm(near_limit.qdot[:3]) > np.linalg.norm(near_limit.qdot[3:])


if __name__ == "__main__":
    test_piper_osqp_ik_respects_bounds_and_damps_near_singularity()
    test_osqp_adaptive_orientation_reduces_priority_near_joint_limit()
    print("OSQP IK smoke test passed")
