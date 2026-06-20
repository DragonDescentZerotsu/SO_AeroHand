from pathlib import Path

import mujoco
import numpy as np

from aero_quest.arm_teleop import DampedLeastSquaresIK
from aero_quest.osqp_ik import effective_singular_damping
from scripts.teleop.quest_so101_aero_nullspace_ik_teleop import (
    solve_osqp_task_space_ik,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_piper_osqp_ik_respects_acceleration_and_singularity_damping():
    model = mujoco.MjModel.from_xml_path(
        str(PROJECT_ROOT / "models/piper_aero_hand/Piper_aerohand.xml")
    )
    data = mujoco.MjData(model)
    ik = DampedLeastSquaresIK(
        model,
        ee_site="aero_wrist_site",
        joint_names=[f"joint{index}" for index in range(1, 7)],
        damping=0.035,
        max_joint_speed=5.0,
    )
    ik.set_joint_positions(
        data, np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])
    )

    _qtarget, qdot, _qdot_pos, _qdot_null, diagnostics = (
        solve_osqp_task_space_ik(
            ik,
            data,
            np.array([0.0, 0.0, 0.0, 3.0, 0.0, 0.0]),
            dt=0.005,
            control_orientation=True,
            joint_motion_weights=np.array([0.7, 1.0, 1.0, 0.35, 0.22, 0.08]),
            task_weights=np.array([1.0, 1.0, 1.0, 1.2, 1.2, 1.2]),
            prev_qdot=np.zeros(6),
            accel_weight=0.04,
            max_joint_accel=120.0,
            singular_damping_threshold=0.10,
            singular_damping_gain=0.10,
        )
    )

    assert str(diagnostics["status"]).lower().startswith("solved")
    assert np.all(np.isfinite(qdot))
    assert np.all(np.abs(qdot) <= 120.0 * 0.005 + 1e-4)
    assert diagnostics["effective_damping"] >= ik.damping


def test_singular_damping_increases_below_threshold():
    damping, min_singular = effective_singular_damping(
        np.diag([1.0, 0.01]), 0.035, 0.10, 0.10
    )
    assert min_singular == 0.01
    assert damping > 0.035
