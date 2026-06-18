import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import mujoco
import mujoco.viewer
from planning.fk import fk
import matplotlib.pyplot as plt
from control.CS_impedance_controller import Impedance_Controller

MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
N_ARM = 7
SETTLE_TIME = 15.0
THRESH = 1e-3
PLOT_EVERY = 50
T_DIST = 2.0

q_home = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

# disable actuators
for i in range(N_ARM):
    model.actuator_gainprm[i,0] = 0.0
    model.actuator_biasprm[i,1] = 0.0
    model.actuator_biasprm[i,2] = 0.0
mujoco.mj_forward(model,data)

T = fk(q_home)
p_des = T[:3,3]
R_des = T[:3,:3]

# end effector id
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

# controller parameters
K = 200
SIM_DT = model.opt.timestep
D = 2*np.sqrt(K)
Kp = np.diag([K,K,K])
Dp = np.diag([D,D,D])
Kr = np.diag([K,K,K])
Dr = np.diag([D,D,D])

CONTROLLER = Impedance_Controller(Kp,Kr,Dp,Dr)

# reset and initialize

mujoco.mj_resetData(model, data)
data.qpos[:N_ARM] = q_home
mujoco.mj_forward(model, data)

# Logging errors
ep_log = []
er_log = []
time_log = []
settle_time = None
t = 0.0

# Set up real-time plotting
plt.ion()
fig,(ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

ax1.set_title('Position Error (m)')
ax1.set_ylabel('Position Error (mm)')
ax1.set_xlabel('Time (s)')
ax1.axhline(1.0, color='r', linestyle='--', linewidth=1, label='1mm threshold')
ax1.legend()
ax1.grid(True, alpha=0.4)

ax2.set_title('Rotation Error (rad)')
ax2.set_ylabel('Orientation Error (mrad)')
ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.4)

line1, = ax1.plot([], [], 'b-', linewidth=1.5)
line2, = ax2.plot([], [], 'g-', linewidth=1.5)

plt.tight_layout()
plt.show()

# mujoco viewer loop
print("Launching viewer...")
step = 0
with mujoco.viewer.launch_passive(model,data) as viewer:
    viewer.cam.distance = 2.0
    while t < SETTLE_TIME and viewer.is_running():
        q = data.qpos[:N_ARM].copy()
        dq = data.qvel[:N_ARM].copy()
        tau, e_p, e_r = CONTROLLER.compute_Torque(q, dq, p_des, R_des, model, data)
        data.ctrl[:N_ARM] = 0.0
        data.qfrc_applied[:N_ARM] = tau  
        if T_DIST <= t < T_DIST+0.5:
            data.xfrc_applied[ee_id] = [80.0, 0, 0, 0, 0, 0]
        else:
            data.xfrc_applied[ee_id] = np.zeros(6)  
        mujoco.mj_step(model, data)
        viewer.sync()
        
        ep_norm = np.linalg.norm(e_p)
        er_norm = np.linalg.norm(e_r)
        ep_log.append(ep_norm)
        er_log.append(er_norm)
        time_log.append(t)
        
        if settle_time is None and ep_norm < THRESH and t > 0.5:
            settle_time = t
        if step%PLOT_EVERY == 0:
            t_arr = np.array(time_log)
            line1.set_data(t_arr, np.array(ep_log)*1000)  # convert to mm
            line2.set_data(t_arr, np.array(er_log)*1000)  # convert
            ax1.relim()
            ax2.relim()
            ax1.autoscale_view()
            ax2.autoscale_view()
            fig.canvas.flush_events()
        
        step += 1
        t+=SIM_DT
        
        # Final results
ep_log = np.array(ep_log)
er_log = np.array(er_log)

print(f'Steady-state |e_p| : {ep_log[-100:].mean()*1000:.3f} mm')
print(f'Steady-state |e_r| : {er_log[-100:].mean()*1000:.3f} mrad')
print(f'Peak        |e_p|  : {ep_log.max()*1000:.3f} mm')
print(f'Peak        |e_r|  : {er_log.max()*1000:.3f} mrad')
print(f'Settling time      : {settle_time:.3f} s' if settle_time else 'Did not settle')

plt.ioff()
plt.savefig(os.path.expanduser('~/Bathsheba-Project-roboai/assets/test1_setpoint.png'), dpi=150)
plt.show()

