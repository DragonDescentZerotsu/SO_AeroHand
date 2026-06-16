import argparse
from pathlib import Path

import mujoco
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models/so101_aero_hand/SO101_aerohand.xml"


def parse_args():
    parser = argparse.ArgumentParser(description="Check SO101 + Aero Hand attachment alignment.")
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--tol", type=float, default=1e-6)
    return parser.parse_args()


def site_pose(model, data, name):
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)
    if site_id < 0:
        raise RuntimeError(f"Missing site: {name}")
    return model.site_pos[site_id].copy(), data.site_xpos[site_id].copy()


def body_pose(model, data, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {name}")
    return model.body_pos[body_id].copy(), data.xpos[body_id].copy()


def body_rotation(model, data, name):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        raise RuntimeError(f"Missing body: {name}")
    return data.xmat[body_id].reshape(3, 3).copy()


def main():
    args = parse_args()
    model_path = Path(args.model).expanduser()
    if not model_path.is_absolute():
        model_path = PROJECT_ROOT / model_path

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    gripper_local, gripper_world = site_pose(model, data, "gripperframe")
    attach_local, attach_world = site_pose(model, data, "so101_aero_attach_site")
    wrist_local, wrist_world = site_pose(model, data, "aero_wrist_site")
    wrist_lm_local, wrist_lm_world = site_pose(model, data, "aero_wrist_lm")
    palm_local, palm_world = body_pose(model, data, "palm")

    tetheria_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "tetheria_mount")
    wrist_roll_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "wrist_roll")
    if wrist_roll_joint < 0:
        raise RuntimeError("Missing joint: wrist_roll")
    removed_mount = tetheria_body < 0
    local_ok = np.allclose(wrist_local, wrist_lm_local, atol=args.tol)
    world_ok = np.linalg.norm(wrist_world - wrist_lm_world) <= args.tol
    gripper_R = body_rotation(model, data, "gripper")
    palm_R = body_rotation(model, data, "palm")
    gripper_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper")
    roll_axis_world = gripper_R @ model.jnt_axis[wrist_roll_joint]
    roll_point_world = data.xpos[gripper_body] + gripper_R @ model.jnt_pos[wrist_roll_joint]
    palm_mount_axis_world = -palm_R[:, 2]
    axis_dot = float(np.dot(roll_axis_world, palm_mount_axis_world))
    axis_ok = axis_dot >= 1.0 - args.tol
    attach_delta = attach_world - roll_point_world
    attach_radial = attach_delta - np.dot(attach_delta, roll_axis_world) * roll_axis_world
    attach_axis_dist = float(np.linalg.norm(attach_radial))
    position_ok = attach_axis_dist <= args.tol

    middle_tip = site_pose(model, data, "aero_middle_tip_site")[1]
    thumb_tip = site_pose(model, data, "aero_thumb_tip_site")[1]
    middle_from_wrist = middle_tip - wrist_world
    thumb_from_wrist = thumb_tip - wrist_world

    print(f"model: {model_path}")
    print(f"tetheria_mount_removed: {removed_mount}")
    print(f"gripperframe local: {np.array2string(gripper_local, precision=8)}")
    print(f"aero_attach local:  {np.array2string(attach_local, precision=8)}")
    print(f"palm local:         {np.array2string(palm_local, precision=8)}")
    print(f"aero_wrist local:   {np.array2string(wrist_local, precision=8)}")
    print(f"aero_wrist_lm local:{np.array2string(wrist_lm_local, precision=8)}")
    print(f"gripperframe world: {np.array2string(gripper_world, precision=8)}")
    print(f"aero_attach world:  {np.array2string(attach_world, precision=8)}")
    print(f"palm world:         {np.array2string(palm_world, precision=8)}")
    print(f"aero_wrist world:   {np.array2string(wrist_world, precision=8)}")
    print(f"aero_wrist_lm world:{np.array2string(wrist_lm_world, precision=8)}")
    print(f"dist palm-gripperframe: {np.linalg.norm(palm_world - gripper_world):.12g}")
    print(f"dist wrist-gripperframe:{np.linalg.norm(wrist_world - gripper_world):.12g}")
    print(f"dist wrist-aero_attach: {np.linalg.norm(wrist_world - attach_world):.12g}")
    print(f"roll_axis_world: {np.array2string(roll_axis_world, precision=8)}")
    print(f"roll_point_world: {np.array2string(roll_point_world, precision=8)}")
    print(f"palm_mount_axis_world: {np.array2string(palm_mount_axis_world, precision=8)}")
    print(f"roll_palm_axis_dot: {axis_dot:.12g}")
    print(f"attach_to_roll_axis_dist: {attach_axis_dist:.12g}")
    print(f"middle_tip_from_wrist: {np.array2string(middle_from_wrist, precision=8)}")
    print(f"thumb_tip_from_wrist:  {np.array2string(thumb_from_wrist, precision=8)}")
    print(f"local_alignment_ok: {local_ok}")
    print(f"wrist_landmark_world_ok: {world_ok}")
    print(f"axis_alignment_ok: {axis_ok}")
    print(f"axis_position_ok: {position_ok}")

    if not removed_mount or not local_ok or not world_ok or not axis_ok or not position_ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
