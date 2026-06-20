from pathlib import Path
import sys

import mujoco
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from aero_quest.arm_teleop import DampedLeastSquaresIK
from aero_quest.quest_hand_frame import QuestHandFrame
from scripts.teleop.quest_so101_aero_nullspace_ik_teleop import (
    absolute_orientation_target_B,
    calibrate_absolute_orientation_offset,
)


MODEL_PATH = PROJECT_ROOT / "models/piper_aero_hand/Piper_aerohand.xml"
ARM_JOINTS = [f"joint{index}" for index in range(1, 7)]
HOME_QPOS = np.asarray([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0], dtype=np.float64)


def make_piper_ik():
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    ik = DampedLeastSquaresIK(
        model,
        ee_site="aero_wrist_site",
        joint_names=ARM_JOINTS,
        damping=0.05,
        max_joint_speed=1.5,
    )
    ik.set_joint_positions(data, HOME_QPOS)
    return model, data, ik


def test_absolute_orientation_calibration_preserves_aero_mount():
    _model, data, ik = make_piper_ik()
    frame = QuestHandFrame(
        hand_side="Right",
        timestamp_ns=0,
        frame_id=0,
        wrist_pos_world=np.zeros(3),
        wrist_quat_world=np.asarray([0.0, 0.0, 0.0, 1.0]),
        landmarks_wrist=np.asarray(
            [
                [0.0, 0.0, 0.0],
                *([[0.0, 0.0, 0.0]] * 4),
                [1.0, 1.0, 0.0],
                *([[0.0, 0.0, 0.0]] * 3),
                [0.0, 2.0, 0.0],
                *([[0.0, 0.0, 0.0]] * 7),
                [-1.0, 1.0, 0.0],
                *([[0.0, 0.0, 0.0]] * 3),
            ],
            dtype=np.float64,
        ),
    )
    R_BQ = np.asarray([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    _ee_pos_B, ee_R_B = ik.ee_pose(data)
    offset = calibrate_absolute_orientation_offset(frame, R_BQ, "palm_landmarks", ee_R_B)
    target_R_B = absolute_orientation_target_B(frame, R_BQ, "palm_landmarks", offset)
    np.testing.assert_allclose(target_R_B, ee_R_B, atol=1e-9)


def test_osqp_reuses_workspace_and_accumulates_joint_target():
    model, data, ik = make_piper_ik()
    xdot = np.asarray([0.1, -0.05, 0.08, 0.2, -0.1, 0.15], dtype=np.float64)
    q_target_1, _ = ik.solve_osqp(data, xdot, model.opt.timestep, True)
    workspace = ik._osqp_solver._solver
    data.qpos[[model.jnt_qposadr[joint_id] for joint_id in ik.joint_ids]] = q_target_1
    mujoco.mj_forward(model, data)
    q_target_2, _ = ik.solve_osqp(data, xdot, model.opt.timestep, True)

    assert ik._osqp_solver._solver is workspace
    assert np.linalg.norm(q_target_2 - q_target_1) > 0.0
