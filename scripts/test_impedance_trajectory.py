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

MODEL_PATH = '/home/shreehan/mujoco_menagerie/franka_emika_panda/scene_pick.xml'
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
N_ARM = 7
SIM_HZ = int(1.0 / model.opt.timestep)

R_des = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

waypoints = np.array([
    [0.4, -0.2, 0.6],
    [0.5,  0.0, 0.6],
    [0.4,  0.2, 0.6],
    [0.3,  0.0, 0.6],
    [0.4, -0.2, 0.6],
])
timestamps = np.array([0.0, 4.0, 8.0, 12.0, 16.0])

for i in range(N_ARM):
    model.actuator_gainprm[i,0] = 0.0
    model.actuator_biasprm[i,1] = 0.0
    model.actuator_biasprm[i,2] = 0.0

trajectory = CartesianTrajectory(waypoints,timestamps,R_des)

q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
result = dls_ik(q_home, waypoints[0], R_des, lambda_max=0.01, epsilon=0.05)
q_current = result['q']
data.qpos[:N_ARM] = q_current
data.ctrl[:N_ARM] = q_current
mujoco.mj_forward(model, data)
print(f"Start EE: {waypoints[0]}")
print(f"Target  : {waypoints[4]}")

# gains
Kp = np.diag([2000,2000,2000])
Kr = np.diag([200,200,200])
Dp = np.diag([89,89,89])
Dr = np.diag([20,20,20])

Controller = Impedance_Controller(Kp,Kr,Dp,Dr)


log_t = []
log_err = []
log_p = []
log_p_des = []

print("\nLaunching viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        t_global = step * model.opt.timestep

        q = data.qpos[:N_ARM].copy()
        dq = data.qvel[:N_ARM].copy()

        p_des, pd_des, R_d = trajectory.evaluate(t_global)

        tau, e_p, e_r = Controller.compute_Torque(q, dq, p_des, R_d)
        tau = tau + data.qfrc_bias[:N_ARM]
        data.qfrc_applied[:N_ARM] = tau
        data.ctrl[:N_ARM] = q

        mujoco.mj_step(model, data)
        viewer.sync()

        if step % SIM_HZ == 0:
            T_curr = fk(q)
            p_curr = T_curr[:3, 3]
            err = np.linalg.norm(p_curr - p_des) * 1000
            log_t.append(t_global)
            log_err.append(err)
            log_p.append(p_curr.copy())
            log_p_des.append(p_des.copy())
            print(f"t={t_global:.1f}s | EE={np.round(p_curr,3)} | err={err:.2f}mm")

        if trajectory.is_done(t_global):
            print("Trajectory complete.")
            break

        step += 1

log_t = np.array(log_t)
log_err = np.array(log_err)
log_p = np.array(log_p)
log_p_des = np.array(log_p_des)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(log_t, log_err, 'k-', linewidth=1.5)
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Tracking error (mm)')
axes[0].set_title('Impedance — Trajectory Tracking Error')
axes[0].grid(True)

axes[1].plot(log_p_des[:, 0], log_p_des[:, 1], 'b--', linewidth=2, label='Desired')
axes[1].plot(log_p[:, 0], log_p[:, 1], 'r-', linewidth=1.5, label='Actual')
axes[1].scatter(waypoints[:, 0], waypoints[:, 1], c='green', s=80, zorder=5, label='Waypoints')
axes[1].set_xlabel('X (m)')
axes[1].set_ylabel('Y (m)')
axes[1].set_title('XY Plane Trajectory')
axes[1].legend()
axes[1].grid(True)
axes[1].set_aspect('equal')

plt.suptitle('Impedance Control — Trajectory Tracking')
plt.tight_layout()
plt.savefig('/tmp/impedance_trajectory.png', dpi=150)
plt.show()
print("Plot saved to /tmp/impedance_trajectory.png")
