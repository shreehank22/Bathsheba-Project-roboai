import os
import sys
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import mujoco.viewer
from planning.fk import fk
from control.impedance_controller import Impedance_Controller
import time

MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
N_ARM = 7
SIM_HZ = 100


DISTURBANCE_MAG = 30.0

def disturbance(t):
    if 3.0 <= t <= 8.0:
        return np.array([0.0, DISTURBANCE_MAG, 0.0])
    else:
        return np.zeros(3)
T_END = 20.0

# disable actuators
for i in range(N_ARM):
    model.actuator_gainprm[i,0] = 0.0
    model.actuator_biasprm[i,1] = 0.0
    model.actuator_biasprm[i,2] = 0.0


q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
print(f"Start EE: {np.round(fk(q_home)[:3,3],4)}")
data.qpos[:N_ARM] = q_home
data.ctrl[:N_ARM] = q_home
p_des = fk(q_home)[:3,3]
R_des = fk(q_home)[:3,:3]
mujoco.mj_forward(model, data)
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

# gains
Kp = np.diag([100, 100, 100])
Kr = np.diag([100, 100, 100])
Dp = np.diag([20, 20, 20])
Dr = np.diag([20, 20, 20])

Controller = Impedance_Controller(Kp,Kr,Dp,Dr)

log_t  = []
log_err = []
log_p  = []
log_ep = []

print("\nLaunching viewer...")
t_wall_start = time.time()
with mujoco.viewer.launch_passive(model,data) as viewer:
    step = 0
    while viewer.is_running():
        t_global = step * model.opt.timestep
        if t_global >= T_END:
            break

        q  = data.qpos[:N_ARM].copy()
        dq = data.qvel[:N_ARM].copy()

        tau,e_p,e_r = Controller.compute_Torque(q,dq,p_des=p_des,R_des=R_des,v_des=None)
        tau = tau + data.qfrc_bias[:N_ARM]
        data.qfrc_applied[:N_ARM] = tau
        data.ctrl[:N_ARM] = q

        data.xfrc_applied[ee_id, 3:6] = disturbance(t_global)
        mujoco.mj_step(model, data)

        t_wall_elapsed = time.time() - t_wall_start
        if t_global > t_wall_elapsed:
            time.sleep(t_global - t_wall_elapsed)

        viewer.sync()

        if step % SIM_HZ == 0:
            T_curr = fk(data.qpos[:N_ARM])
            p_curr = T_curr[:3,3]
            err = np.linalg.norm(p_curr-p_des)*1000
            if len(log_err) > 0 and log_err[-1] < 1.0 and err >= 1.0:
                print(f">> Deflection START | q={np.round(data.qpos[:N_ARM],4)}")
            if len(log_err) > 0 and log_err[-1] >= 1.0 and err < 1.0:
                print(f">> Deflection END   | q={np.round(data.qpos[:N_ARM],4)}")
            log_t.append(t_global)
            log_err.append(err)
            log_p.append(p_curr.copy())
            log_ep.append(e_p.copy())
            print(f"t={t_global:.1f}s | F={np.linalg.norm(disturbance(t_global)):.1f}N | "f"EE=[{p_curr[0]:.4f}, {p_curr[1]:.4f}, {p_curr[2]:.4f}] m | err={err:.2f}mm")

        step += 1

