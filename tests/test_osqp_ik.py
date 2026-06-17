from pathlib import Path
import sys

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.arm_teleop import DampedLeastSquaresIK
from scripts.teleop.quest_so101_aero_nullspace_ik_teleop import (
    set_arm_ctrl_targets,
    set_arm_qpos,
    solve_osqp_task_space_ik,
)


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

    qtarget, qdot, _qdot_pos, _qdot_null, diag = solve_osqp_task_space_ik(
        ik,
        data,
        np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0], dtype=np.float64),
        dt=0.005,
        control_orientation=True,
        joint_motion_weights=np.array([0.7, 1.0, 1.0, 0.35, 0.22, 0.08], dtype=np.float64),
        task_weights=np.array([1.0, 1.0, 1.0, 1.2, 1.2, 1.2], dtype=np.float64),
        prev_qdot=np.zeros(6, dtype=np.float64),
        accel_weight=0.04,
        max_joint_accel=120.0,
        singular_damping_threshold=0.10,
        singular_damping_gain=0.10,
    )

    assert diag["status"].lower().startswith("solved")
    assert np.all(np.isfinite(qtarget))
    assert np.all(np.isfinite(qdot))
    assert np.all(np.abs(qdot) <= 5.0 + 1e-6)
    assert np.all(np.abs(qdot) <= 120.0 * 0.005 + 1e-4)
    assert diag["min_singular"] < 0.10
    assert diag["effective_damping"] > ik.damping
    assert abs(qdot[5]) > 1e-3


if __name__ == "__main__":
    test_piper_osqp_ik_respects_bounds_and_damps_near_singularity()
    print("OSQP IK smoke test passed")
