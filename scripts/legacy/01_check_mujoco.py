import mujoco
import mujoco.viewer
from pathlib import Path

MODEL_PATH = Path.home() / "Projects/aero_quest_sim/mujoco_menagerie/tetheria_aero_hand_open/scene_right.xml"

model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
data = mujoco.MjData(model)

print("Model loaded.")
print("nq =", model.nq)
print("nv =", model.nv)
print("nu =", model.nu)

with mujoco.viewer.launch_passive(model, data) as viewer:
    while viewer.is_running():
        mujoco.mj_step(model, data)
        viewer.sync()
