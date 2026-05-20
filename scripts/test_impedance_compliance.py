import sys
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
import matplotlib.pyplot as plt
import numpy as np
import mujoco
from planning.cartesian_trajectory import CartesianTrajectory
import mujoco.viewer
from planning.ik import dls_ik
from planning.fk import fk
from control.impedance_controller import Impedance_Controller
import time

MODEL_PATH = '/home/shreehan/mujoco_menagerie/franka_emika_panda/scene_pick.xml'
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
N_ARM = 7
SIM_HZ = 100


DISTURBANCE_MAG = 150.0
T1, T2, T3, T4, T_END = 3.0, 7.0, 10.0, 14.0, 17.0

def disturbance(t):
    if T1<=t<=T2:
        return np.array([DISTURBANCE_MAG, 0, 0])
    elif T3<=t<=T4:
        return np.array([0,-DISTURBANCE_MAG, 0])
    else:
        return np.zeros(3)

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
Kp = np.diag([200,200,200])
Kr = np.diag([200,200,200])
Dp = np.diag([80, 80, 80]) 
Dr = np.diag([20,20,20])

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
            log_t.append(t_global)
            log_err.append(err)
            log_p.append(p_curr.copy())
            log_ep.append(e_p.copy())
            print(f"t={t_global:.1f}s | F={np.linalg.norm(disturbance(t_global)):.1f}N | "f"EE=[{p_curr[0]:.4f}, {p_curr[1]:.4f}, {p_curr[2]:.4f}] m | err={err:.2f}mm")

        step += 1

# ── Convert logs ───────────────────────────────────────────────────────────
log_t  = np.array(log_t)
log_err = np.array(log_err)
log_p  = np.array(log_p)
log_ep = np.array(log_ep)

e_ss = (DISTURBANCE_MAG / 200.0) * 1000

# ── Plots ──────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

ax1  = axes[0]
ax1b = ax1.twinx()
log_fmag = [np.linalg.norm(disturbance(t)) for t in log_t]
ax1.plot(log_t, log_err, 'k-', linewidth=1.5, label='|e_p| (mm)')
ax1b.plot(log_t, log_fmag, 'r--', linewidth=1.2, label='|F_ext| (N)')
ax1.set_ylabel('Position error (mm)')
ax1b.set_ylabel('Applied force (N)', color='r')
ax1.set_title('Compliance Test — EE Error under External Disturbance')
ax1.legend(loc='upper left')
ax1b.legend(loc='upper right')
ax1.grid(True, alpha=0.4)

axes[1].plot(log_t, log_ep[:,0]*1000, label='e_x (mm)')
axes[1].plot(log_t, log_ep[:,1]*1000, label='e_y (mm)')
axes[1].plot(log_t, log_ep[:,2]*1000, label='e_z (mm)')
axes[1].axhline( e_ss, color='r', linestyle=':', alpha=0.6, label=f'Predicted +e_ss: {e_ss:.1f}mm')
axes[1].axhline(-e_ss, color='b', linestyle=':', alpha=0.6, label=f'Predicted -e_ss: -{e_ss:.1f}mm')
axes[1].set_ylabel('Per-axis error (mm)')
axes[1].set_title('Per-Axis Position Deviation from Setpoint')
axes[1].legend(fontsize=8)
axes[1].grid(True, alpha=0.4)

axes[2].plot(log_p[:,0]*1000, log_p[:,1]*1000, 'k-', linewidth=1.2, label='EE path')
axes[2].scatter([p_des[0]*1000], [p_des[1]*1000], c='green', s=100, zorder=5, label='Setpoint')
axes[2].set_xlabel('X (mm)')
axes[2].set_ylabel('Y (mm)')
axes[2].set_title('EE Trajectory in XY Plane')
axes[2].set_aspect('equal')
axes[2].legend()
axes[2].grid(True, alpha=0.4)

for ax in axes:
    ax.axvspan(T1, T2, alpha=0.08, color='red')
    ax.axvspan(T3, T4, alpha=0.08, color='blue')

plt.tight_layout()
plt.savefig('/tmp/impedance_compliance.png', dpi=150)
plt.show()
print("Plot saved to /tmp/impedance_compliance.png")

# ── Metrics ────────────────────────────────────────────────────────────────
def mean_err_window(t_start, t_end):
    mask = (log_t >= t_start) & (log_t < t_end)
    return log_err[mask].mean() if mask.any() else float('nan')

print("\n── Compliance Metrics ──────────────────────────────────────")
print(f"  Baseline error            : {mean_err_window(0, T1):.2f} mm")
print(f"  SS deflection +X          : {mean_err_window(T2-2, T2):.2f} mm  (predicted {e_ss:.1f} mm)")
print(f"  Return error after +X     : {mean_err_window(T2+1, T3):.2f} mm")
print(f"  SS deflection -Y          : {mean_err_window(T4-2, T4):.2f} mm  (predicted {e_ss:.1f} mm)")
print(f"  Return error after -Y     : {mean_err_window(T4+1, T_END):.2f} mm")
print("─────────────────────────────────────────────────────────────")