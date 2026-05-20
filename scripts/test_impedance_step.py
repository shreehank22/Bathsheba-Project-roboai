import sys
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
import matplotlib.pyplot as plt
import numpy as np
import mujoco
import mujoco.viewer
from planning.fk import fk
from control.impedance_controller import Impedance_Controller

MODEL_PATH = '/home/shreehan/mujoco_menagerie/franka_emika_panda/scene_pick.xml'
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
N_ARM = 7
SIM_HZ = int(1.0 / model.opt.timestep)

for i in range(N_ARM):
    model.actuator_gainprm[i,0] = 0.0
    model.actuator_biasprm[i,1] = 0.0
    model.actuator_biasprm[i,2] = 0.0

q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
T = fk(q_home)
p_home = T[:3,3]
p_des = p_home + np.array([0.1,0,0])
R_des = np.array([[1,0,0],[0,-1,0],[0,0,-1]],dtype=np.float64)
print(f"Start EE: {np.round(p_home, 4)}")
print(f"Target  : {np.round(p_des, 4)}")

# gains
Kp = np.diag([2000,2000,2000])
Kr = np.diag([200,200,200])
Dp = np.diag([89,89,89])
Dr = np.diag([20,20,20])

controller = Impedance_Controller(Kp,Kr,Dp,Dr)

# Initial configs
data.qpos[:N_ARM] = q_home
data.qvel[:N_ARM] = 0
data.ctrl[:N_ARM] = q_home
mujoco.mj_forward(model,data)

MAX_TIME = 10.0
log_t   = []
log_err = []
log_p   = []

print("\nLaunching viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        t_global = step * model.opt.timestep
    
        q = data.qpos[:N_ARM].copy()
        dq = data.qvel[:N_ARM].copy()
    
        tau,e_p,e_r = controller.compute_Torque(q,dq,p_des=p_des,R_des=R_des,v_des=None)
        tau = tau + data.qfrc_bias[:N_ARM]
        data.qfrc_applied[:N_ARM] = tau
        data.ctrl[:N_ARM] = q
        
        mujoco.mj_step(model, data)
        viewer.sync()
    
        if step % SIM_HZ == 0:
            T_curr = fk(data.qpos[:N_ARM])
            p_curr = T_curr[:3, 3]
            err = np.linalg.norm(p_curr-p_des)*1000            
            log_t.append(t_global)
            log_err.append(err)
            log_p.append(p_curr.copy())
            print(f"t={t_global:.1f}s | EE={np.round(p_curr,3)} | err={err:.2f}mm")
        if t_global >= MAX_TIME:
            print("Done.")
            break

        step += 1
    

# plot
log_t   = np.array(log_t)
log_err = np.array(log_err)
log_p   = np.array(log_p)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(log_t, log_err, 'k-', linewidth=1.5)
axes[0].axhline(y=0, color='g', linestyle='--', linewidth=1)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Position error (mm)')
axes[0].set_title('Step Response — Position Error')
axes[0].grid(True)

labels = ['X', 'Y', 'Z']
for i, label in enumerate(labels):
    axes[1].plot(log_t, log_p[:, i], label=f'{label} actual')
    axes[1].axhline(y=p_des[i], linestyle='--', linewidth=1, label=f'{label} desired')
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Position (m)')
axes[1].set_title('EE Position vs Desired')
axes[1].legend()
axes[1].grid(True)

plt.suptitle('Impedance Control — Step Response')
plt.tight_layout()
plt.savefig('/tmp/step_response.png', dpi=150)
plt.show()
print("Plot saved to /tmp/step_response.png")
    