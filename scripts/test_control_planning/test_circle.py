import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import mujoco
import mujoco.viewer
from planning.fk import fk
from planning.ik import dls_ik

MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)

N_ARM   = 7
SIM_HZ  = int(1.0 / model.opt.timestep)
GRIPPER = 255.0

# Circle parameters
center = np.array([0.3069, 0.0, 0.5903])
radius = 0.08
period = 10.0
omega  = 2 * np.pi / period

# Fixed end-effector orientation — same as home pose
q_home = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
T_home = fk(q_home)
R_des  = T_home[:3, :3]

# Initial IK guess — home pose
q_curr = q_home.copy()

print(f"Model loaded | timestep={model.opt.timestep}s | SIM_HZ={SIM_HZ}")
print(f"Circle: center={center} | radius={radius}m | period={period}s")
print("Launching viewer — close window to exit\n")

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        t = step * model.opt.timestep

        # Desired Cartesian position on circle
        p_des = np.array([
            center[0] + radius * np.cos(omega * t),
            center[1] + radius * np.sin(omega * t),
            center[2]
        ])

        # Solve IK online — warm start from previous solution
        result = dls_ik(q_curr, p_des, R_des)
        q_curr = result['q']

        # Send to sim
        data.ctrl[:N_ARM] = q_curr
        data.ctrl[7]      = GRIPPER

        mujoco.mj_step(model, data)
        viewer.sync()

        if step % SIM_HZ == 0:
            T_curr  = fk(q_curr)
            p_curr  = T_curr[:3, 3]
            p_err   = np.linalg.norm(p_des - p_curr) * 1000
            print(f"t={t:.1f}s | p_des={np.round(p_des,4)} | "
                  f"pos_err={p_err:.3f}mm | "
                  f"IK_iter={result['n_iter']} | "
                  f"converged={result['converged']}")

        step += 1