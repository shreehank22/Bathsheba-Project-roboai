import os
import sys

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from control.CS_impedance_controller import CS_Impedance_Controller
from planning.fk import fk

MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

N_ARM  = 7
SIM_DT = model.opt.timestep
ee_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

# zero the actuator gains and biases for torque control
for i in range(N_ARM):
    model.actuator_gainprm[i, 0] = 0.0
    model.actuator_biasprm[i, 1] = 0.0
    model.actuator_biasprm[i, 2] = 0.0


# set home position
q_home = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
T_home = fk(q_home)

# retrieving position and orientation of the end-effector
p_des = T_home[:3, 3]
R_des = T_home[:3, :3]

Kp = np.diag([1000.0]*3)
Kr = np.diag([100.0]*3)
Dp = np.diag([2*np.sqrt(1000.0)]*3)
Dr = np.diag([2*np.sqrt(100.0)]*3)


# gains for nullspace control
Kn = np.diag([20.0]*N_ARM)
Dn = np.diag([2*np.sqrt(20.0)]*N_ARM)

q_null = q_home

CS_CONTROLLER = CS_Impedance_Controller(Kp,Kr,Dp,Dr,Kn,Dn,q_null=q_null)

mujoco.mj_resetData(model, data)
data.qpos[:N_ARM] = q_home
mujoco.mj_forward(model, data)

link4_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'link4')
force = np.array([0, 20.0, 0, 0, 0, 0])
t_start = 1.0
t_end = 1.5
TOTAL_TIME = 4.0
q_log,t_log,e_p_log,e_r_log = [], [], [], []


# run the loop
with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    t=0.0
    while viewer.is_running() and t < TOTAL_TIME:
        q= data.qpos[:N_ARM].copy()
        dq = data.qvel[:N_ARM].copy()
        mujoco.mj_step1(model, data)

        tau,e_p,e_r = CS_CONTROLLER.compute_Torque(q, dq, p_des, R_des, model, data)
        data.qfrc_applied[:N_ARM] = tau

        # inject an external force on link4 during the specified time interval
        if t_start <= t < t_end:
            data.xfrc_applied[link4_id] = force
        else:
            data.xfrc_applied[link4_id] = np.zeros(6)

        # call mj_step2 to advance the simulation
        mujoco.mj_step2(model, data)

        # update the viewer
        viewer.sync()
        q_log.append(q.copy())
        t_log.append(t)
        e_p_log.append(e_p.copy())
        e_r_log.append(e_r.copy())
        t += SIM_DT

# post-processing and plotting
q_log = np.array(q_log)
t_log = np.array(t_log)
e_p_log = np.array(e_p_log)
e_r_log = np.array(e_r_log)

ref_idx = np.where(t_log < t_start)[0][-1]
q_ref = q_log[ref_idx]
q_drift = q_log - q_ref

print(f"final ||e_p|| [mm]: {np.linalg.norm(e_p_log[-1])*1000:.4f}")
print(f"final ||e_r|| [rad]: {np.linalg.norm(e_r_log[-1]):.6f}")
print("final joint drift [rad]:", np.round(q_drift[-1], 5))

# plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 10), sharex=True)

# EE position error
axes[0].plot(t_log, np.linalg.norm(e_p_log, axis=1)*1000, label="||e_p||")
for k, lbl in enumerate(["x", "y", "z"]):
    axes[0].plot(t_log, e_p_log[:, k]*1000, "--", lw=0.8, label=f"e_p {lbl}")
axes[0].set_ylabel("position error [mm]")
axes[0].set_title("End-effector error under elbow push")

# EE orientation error
axes[1].plot(t_log, np.linalg.norm(e_r_log, axis=1), label="||e_r||")
for k, lbl in enumerate(["x", "y", "z"]):
    axes[1].plot(t_log, e_r_log[:, k], "--", lw=0.8, label=f"e_r {lbl}")
axes[1].set_ylabel("orientation error [rad]")

# joint drift
for j in range(N_ARM):
    axes[2].plot(t_log, q_drift[:, j], label=f"q{j+1}")
axes[2].set_ylabel("q - q_ref [rad]")
axes[2].set_xlabel("time [s]")
axes[2].set_title("Joint drift (null-space motion)")

for ax in axes:
    ax.axvspan(t_start, t_end, color="red", alpha=0.15)
    ax.axhline(0, color="k", lw=0.5)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=2)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "assets", "test7_nullspace_drift_after.png"), dpi=150)
plt.show()