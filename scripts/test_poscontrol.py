import sys, os
sys.path.insert(0, os.path.expanduser('~/roboai_project'))

import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

N_ARM = 7
SIM_HZ = int(1.0 / model.opt.timestep)
GRIPPER_OPEN = 255.0

q_target = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
print("Actuator limits (deg):")

for i in range(N_ARM):
    lo,hi = model.actuator_ctrlrange[i]
    val = q_target[i]
    status = "OK" if lo <= val <= hi else "OUT OF RANGE"
    print(f"  joint{i+1}: {np.rad2deg(val):8.2f} deg  "
          f"[{np.rad2deg(lo):.1f}, {np.rad2deg(hi):.1f}] {status}")
    
print("launching viewer — close window to exit")
print(f"Target (deg): {np.rad2deg(q_target).round(1)}")

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        data.ctrl[:N_ARM] = q_target
        data.ctrl[7] = GRIPPER_OPEN
        mujoco.mj_step(model, data)
        viewer.sync()
        
        if step % SIM_HZ ==0:
            q = data.qpos[:N_ARM]
            e_norm = np.linalg.norm(q_target - q)
            e_deg = np.rad2deg(q_target - q)
            print(f"t={step*model.opt.timestep:.1f}s | |e|={e_norm:.5f} rad | e(deg)={e_deg.round(3)}")
        step += 1


        