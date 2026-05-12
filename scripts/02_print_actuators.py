from pathlib import Path

import mujoco


MODEL_PATH = Path.home() / "Projects/aero_quest_sim/mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml"


def name_or_none(model, obj_type, obj_id):
    if obj_id < 0:
        return None
    return mujoco.mj_id2name(model, obj_type, obj_id)


def print_actuators(model):
    print("\n=== Actuators ===")
    print(f"model.nu = {model.nu}")
    for i in range(model.nu):
        name = name_or_none(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
        lo, hi = model.actuator_ctrlrange[i]
        trn_type = model.actuator_trntype[i]
        target_id = int(model.actuator_trnid[i][0])

        if trn_type == mujoco.mjtTrn.mjTRN_JOINT:
            target_type = "joint"
            target_name = name_or_none(model, mujoco.mjtObj.mjOBJ_JOINT, target_id)
        elif trn_type == mujoco.mjtTrn.mjTRN_TENDON:
            target_type = "tendon"
            target_name = name_or_none(model, mujoco.mjtObj.mjOBJ_TENDON, target_id)
        else:
            target_type = str(trn_type)
            target_name = str(model.actuator_trnid[i].tolist())

        print(
            f"[{i:02d}] name={name} "
            f"ctrlrange=({lo:.6f}, {hi:.6f}) "
            f"target={target_type}:{target_name}"
        )


def print_joints(model):
    print("\n=== Joints ===")
    print(f"model.njnt = {model.njnt}")
    for i in range(model.njnt):
        name = name_or_none(model, mujoco.mjtObj.mjOBJ_JOINT, i)
        lo, hi = model.jnt_range[i]
        print(f"[{i:02d}] name={name} range=({lo:.6f}, {hi:.6f})")


def print_tendons(model):
    print("\n=== Tendons ===")
    print(f"model.ntendon = {model.ntendon}")
    for i in range(model.ntendon):
        name = name_or_none(model, mujoco.mjtObj.mjOBJ_TENDON, i)
        lo, hi = model.tendon_range[i]
        print(f"[{i:02d}] name={name} range=({lo:.6f}, {hi:.6f})")


def main():
    print(f"Loading model: {MODEL_PATH}")
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    print(f"nq={model.nq} nv={model.nv} nu={model.nu}")
    print_actuators(model)
    print_tendons(model)
    print_joints(model)


if __name__ == "__main__":
    main()
