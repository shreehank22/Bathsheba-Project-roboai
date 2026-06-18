import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import mujoco
import mujoco.viewer
from planning.fk import fk
import matplotlib.pyplot as plt
from control.CS_impedance_controller import CS_Impedance_Controller
from scipy.spatial.transform import Rotation
# Load model and data
MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
N_ARM = 7
SETTLE_TIME = 7.0
K=200
THRESH = 1e-3
PLOT_EVERY = 50
SIM_DT = model.opt.timestep

# disabble position actuators
for i in range(N_ARM):
    model.actuator_gainprm[i,0] = 0.0
    model.actuator_biasprm[i,1] = 0.0
    model.actuator_biasprm[i,2] = 0.0
mujoco.mj_forward(model,data)

# Desired end-effector pose
q_home = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
T = fk(q_home)
p_des = T[:3,3]
R_des = T[:3,:3]

ROTATION_CASES = [('Rz +30deg', Rotation.from_euler('z',  30, degrees=True).as_matrix() @ R_des),
                  ('Rz -30deg', Rotation.from_euler('z', -30, degrees=True).as_matrix() @ R_des),
                  ('Ry +30deg', Rotation.from_euler('y',  30, degrees=True).as_matrix() @ R_des),
                  ('Ry -30deg', Rotation.from_euler('y', -30, degrees=True).as_matrix() @ R_des),
                  ('Rx +30deg', Rotation.from_euler('x',  30, degrees=True).as_matrix() @ R_des),
                  ('Rx -30deg', Rotation.from_euler('x', -30, degrees=True).as_matrix() @ R_des)]

# Controller parameters
D = 2*np.sqrt(K)
Kp = np.diag([K]*3)
Kr = np.diag([K]*3)
Dp = np.diag([D]*3)
Dr = np.diag([D]*3)

CONTROLLER = CS_Impedance_Controller(Kp,Kr,Dp,Dr)

# storing results
results = {}

for label,R_des_new in ROTATION_CASES:
    print("\n Running the visualizer")
    mujoco.mj_resetData(model,data)
    data.qpos[:N_ARM] = q_home
    data.qvel[:N_ARM] = np.ones(N_ARM)*1.0
    mujoco.mj_forward(model,data)
    ep_log = []
    er_log = []
    t_log = []
    settle_r = None
    t,step=0.0,0
    with mujoco.viewer.launch_passive(model,data) as viewer:
        viewer.cam.distance = 2.0
        while t < SETTLE_TIME and viewer.is_running():
            q=data.qpos[:N_ARM].copy()
            dq = data.qvel[:N_ARM].copy()
            tau,e_p,e_r = CONTROLLER.compute_Torque(q,dq,p_des,R_des_new,model,data)
            data.ctrl[:N_ARM] = 0.0
            data.qfrc_applied[:N_ARM] = tau
            mujoco.mj_step(model,data)
            viewer.sync()
            ep_norm = np.linalg.norm(e_p)
            er_norm = np.linalg.norm(e_r)
            ep_log.append(ep_norm)
            er_log.append(er_norm)
            t_log.append(t)
            if settle_r is None and t>0.5 and er_norm<THRESH:
                settle_r = t
            step+=1
            t = step*SIM_DT
    results[label] = {
        'ep': np.array(ep_log),
        'er': np.array(er_log),
        't':  np.array(t_log),
        'settle_r': settle_r,
    }
    print(f'  Peak |e_p| : {np.array(ep_log).max()*1000:.2f} mm')
    print(f'  Peak |e_r| : {np.array(er_log).max()*1000:.2f} mrad')
    print(f'  Settling   : {settle_r:.3f} s' if settle_r else '  Did not settle')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7))

for i, (label, r) in enumerate(results.items()):
    color = plt.cm.tab10(i / 6)
    ax1.plot(r['t'], r['ep']*1000, color=color, linewidth=1.8, label=label)
    ax2.plot(r['t'], r['er']*1000, color=color, linewidth=1.8, label=label)
    if r['settle_r']:
        ax2.axvline(r['settle_r'], color=color, linestyle=':', linewidth=1)

ax1.axhline(THRESH*1000, color='k', linestyle='--', linewidth=1, label='1mm threshold')
ax1.set_title('Position Error |e_p| — should stay near zero')
ax1.set_ylabel('mm')
ax1.set_xlabel('Time (s)')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.4)

ax2.axhline(THRESH*1000, color='k', linestyle='--', linewidth=1, label='1mrad threshold')
ax2.set_title('Orientation Error |e_r| — should decay to zero')
ax2.set_ylabel('mrad')
ax2.set_xlabel('Time (s)')
ax2.legend(fontsize=8)
ax2.grid(True, alpha=0.4)

plt.tight_layout()
plt.savefig(os.path.expanduser('~/Bathsheba-Project-roboai/assets/test5_orientation_tracking.png'), dpi=150)
plt.show()
print('Plot saved.')



            
        