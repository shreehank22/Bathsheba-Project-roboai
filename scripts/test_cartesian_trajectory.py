import os
import sys
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
import numpy as np
import mujoco
import mujoco.viewer
from planning.cartesian_trajectory import CartesianTrajectory
from planning.ik import dls_ik
from planning.fk import fk
import matplotlib.pyplot as plt

MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
N_ARM = 7
SIM_HZ = int(1.0 / model.opt.timestep)

R_des = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)

waypoints = np.array([
    [0.4, -0.2, 0.6],
    [0.5,  0.0, 0.6],
    [0.4,  0.5, 0.6],
    [0.3,  0.0, 0.6],
    [0.4, -0.2, 0.6],
])
timestamps = np.array([0.0, 4.0, 8.0, 12.0, 16.0])

trajectory = CartesianTrajectory(waypoints, timestamps, R_des)

q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
result = dls_ik(q_home, waypoints[0], R_des, lambda_max=0.01, epsilon=0.05)
q_current = result['q']
data.qpos[:N_ARM] = q_current
data.ctrl[:N_ARM] = q_current
mujoco.mj_forward(model, data)

t_start = None
phase = 'track'
log_t = []
log_p_actual = []
log_p_des = []
q_des = q_current.copy()

print("Launching viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        t_global = step * model.opt.timestep
        if t_start is None:
            t_start = t_global
        t_local = t_global - t_start

        if phase == 'track':
            p_des, pd_des, R_d = trajectory.evaluate(t_local)
            result = dls_ik(q_current, p_des, R_d, lambda_max=0.01, epsilon=0.05)
            q_des = result['q']
            q_current = data.qpos[:N_ARM].copy()
            if trajectory.is_done(t_local):
                phase = 'done'
                print("Trajectory complete!")
                break

        data.qfrc_applied[:N_ARM] = data.qfrc_bias[:N_ARM]
        data.ctrl[:N_ARM] = q_des
        mujoco.mj_step(model, data)
        viewer.sync()

        if step % SIM_HZ == 0:
            T_curr = fk(data.qpos[:N_ARM])
            p_curr = T_curr[:3, 3]
            p_des_now, _, _ = trajectory.evaluate(t_local)
            log_t.append(t_local)
            log_p_actual.append(p_curr.copy())
            log_p_des.append(p_des_now.copy())
        step += 1



log_t        = np.array(log_t)
log_p_actual = np.array(log_p_actual)
log_p_des    = np.array(log_p_des)
errors       = np.linalg.norm(log_p_actual - log_p_des, axis=1) * 1000

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# XY plane tracking
axes[0].plot(log_p_des[:, 0],    log_p_des[:, 1],    'b--', label='Desired', linewidth=2)
axes[0].plot(log_p_actual[:, 0], log_p_actual[:, 1], 'r-',  label='Actual',  linewidth=1.5)
axes[0].scatter(waypoints[:, 0], waypoints[:, 1], c='green', zorder=5, s=80, label='Waypoints')
axes[0].set_xlabel('X (m)')
axes[0].set_ylabel('Y (m)')
axes[0].set_title('XY Plane Trajectory')
axes[0].legend()
axes[0].grid(True)
axes[0].set_aspect('equal')

# tracking error over time
axes[1].plot(log_t, errors, 'k-', linewidth=1.5)
axes[1].set_xlabel('Time (s)')
axes[1].set_ylabel('Tracking error (mm)')
axes[1].set_title('Tracking Error over Time')
axes[1].grid(True)

plt.suptitle('Cartesian Trajectory Tracking with Gravity Compensation')
plt.tight_layout()
plt.savefig('/tmp/tracking_plot.png', dpi=150)
plt.show()
print("Plot saved to /tmp/tracking_plot.png")