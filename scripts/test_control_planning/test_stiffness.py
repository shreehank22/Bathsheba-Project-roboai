import sys
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
import numpy as np
import mujoco
import matplotlib.pyplot as plt
from planning.fk import fk
from control.impedance_controller import Impedance_Controller
import mujoco.viewer as mjviewer
import os

STIFFNESS = [50, 100, 200, 400, 800] 
SETTLE_TIME = 3.0
FORCE_TIME = 5.0
TOTAL_TIME = SETTLE_TIME + FORCE_TIME + 1.0
FORCE_MAGNITUDE = 5.0
MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
N_ARM = 7
SIM_DT = model.opt.timestep

q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
T = fk(q_home)
p_des,R_des = T[:3,3],T[:3,:3]
print("Desired Position:", np.round(p_des,4))
print("Desired Orientation:", np.round(R_des,4))


# get ee body id
ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

# Compute Lambda at home config
from planning.ik import geometric_jacobian
mujoco.mj_resetData(model, data)
data.qpos[:N_ARM] = q_home
mujoco.mj_forward(model, data)
J_home = geometric_jacobian(q_home)
M_full = np.zeros((model.nv, model.nv))
mujoco.mj_fullM(model, M_full, data.qM)
M_home = M_full[:N_ARM, :N_ARM]
Lambda = np.linalg.inv(J_home @ np.linalg.inv(M_home) @ J_home.T)
lambda_yy = Lambda[1, 1]
print(f'Lambda_yy = {lambda_yy:.4f}')

# main control loop
results = []
for k in STIFFNESS:
    mujoco.mj_resetData(model, data)
    data.qpos[:N_ARM] = q_home
    mujoco.mj_forward(model, data)
    deflections = []
    print(f"\nTesting stiffness: {k}")
    Kp = np.diag([k,k,k])
    Kr = np.diag([k,k,k])
    Dp = np.diag([2*np.sqrt(k), 2*np.sqrt(k), 2*np.sqrt(k)])
    Dr = np.diag([2*np.sqrt(k), 2*np.sqrt(k), 2*np.sqrt(k)])
    CONTROLLER = Impedance_Controller(Kp,Kr,Dp,Dr)
    t=0.0
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        while t < TOTAL_TIME and viewer.is_running():
            q,dq = data.qpos[:N_ARM],data.qvel[:N_ARM]
            if t < SETTLE_TIME:
                f_ext = np.zeros(3)    
            elif t < SETTLE_TIME + FORCE_TIME:
                f_ext = np.array([0.0,FORCE_MAGNITUDE,0.0])
            else:
                f_ext = np.zeros(3)
            tau, e_p, e_r = CONTROLLER.compute_Torque(q, dq, p_des, R_des, model, data)
            if t > SETTLE_TIME + FORCE_TIME - 0.5 and t < SETTLE_TIME + FORCE_TIME:
                deflections.append(np.linalg.norm(e_p))
            data.qfrc_applied[:N_ARM] = tau  # added gravity compensation
            data.ctrl[:N_ARM] = q
            data.xfrc_applied[ee_id] = [f_ext[0], f_ext[1], f_ext[2], 0, 0, 0]
            mujoco.mj_step(model, data)
            viewer.sync()
            t += SIM_DT
    avg_deflections = np.mean(deflections)
    expected = FORCE_MAGNITUDE / (lambda_yy * k)
    error_pct = abs(avg_deflections - expected) / expected * 100
    results.append({'k': k, 'measured': avg_deflections, 'expected': expected})
    print(f"k={k:4d} | measured={avg_deflections*1000:6.2f}mm | "f"expected={expected*1000:6.2f}mm | error={error_pct:.1f}%")

# ── Plot ──────────────────────────────────────────────────────────────────────
k_vals   = [r['k']             for r in results]
measured = [r['measured']*1000 for r in results]
expected = [r['expected']*1000 for r in results]

plt.figure(figsize=(8, 5))
plt.plot(k_vals, measured, 'bo-', linewidth=2, markersize=8, label='Measured')
plt.plot(k_vals, expected, 'r--', linewidth=2, label='Expected F/(λ·k)')
plt.xlabel('Stiffness Kp (N/m)', fontsize=12)
plt.ylabel('Steady-state deflection (mm)', fontsize=12)
plt.title('Impedance Stiffness Sweep', fontsize=13)
plt.legend(fontsize=11)
plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('/home/shreehan/Bathsheba-Project-roboai/assets/stiffness_sweep.png', dpi=150)
plt.show()
print("Plot saved.")