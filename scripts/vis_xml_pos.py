import mujoco as mj
import mujoco.viewer as mjv
import pathlib

xml_path = pathlib.Path("/Users/huangxiansheng/Project/GMR/assets/unitree_g1/g1_mocap_29dof.xml")
model = mj.MjModel.from_xml_path(str(xml_path))
data  = mj.MjData(model)

mj.mj_forward(model, data)

with mjv.launch_passive(model, data) as viewer:
    while viewer.is_running():
        viewer.sync()
