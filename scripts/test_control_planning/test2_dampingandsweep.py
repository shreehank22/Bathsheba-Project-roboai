import os
import sys
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
import numpy as np
import mujoco
import mujoco.viewer
from planning.fk import fk
import matplotlib.pyplot as plt
from control.impedance_controller import Impedance_Controller

MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
N_ARM = 7
SETTLE_TIME = 8.0
THRESH = 1e-3
K = 200
SIM_DT  = model.opt.timestep

q_home = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

for i in range(N_ARM):
    model.actuator_gainprm[i,0] = 0.0
    model.actuator_biasprm[i,1] = 0.0
    model.actuator_biasprm[i,2] = 0.0

T = fk(q_home)
p_des = T[:3,3]
R_des = T[:3,:3]

DAMPING_CASES = [
    ('Underdamped (0.5x)', 0.5  * 2 * np.sqrt(K)),
    ('Critical(1.0x)', 1.0  * 2 * np.sqrt(K)),
    ('Overdamped(2.0x)', 2.0  * 2 * np.sqrt(K)),
]

results = {}

for label, d_val in DAMPING_CASES:
    print(f'\nRunning: {label}')
    mujoco.mj_resetData(model, data)
    data.qpos[:N_ARM] = q_home
    data.qvel[:N_ARM] = np.ones(N_ARM) * 1.0  # add this line
    mujoco.mj_forward(model, data)

    Kp = np.diag([K]*3)
    Kr = np.diag([K]*3)
    Dp = np.diag([d_val]*3)
    Dr = np.diag([d_val]*3)
    CONTROLLER = Impedance_Controller(Kp, Kr, Dp, Dr)

    ep_log, t_log = [], []
    settle_time = None
    t = 0.0
    step = 0

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 2.0
        while t < SETTLE_TIME and viewer.is_running():
            q = data.qpos[:N_ARM].copy()
            dq = data.qvel[:N_ARM].copy()
            tau,e_p,e_r = CONTROLLER.compute_Torque(q, dq, p_des, R_des, model, data)
            data.ctrl[:N_ARM] = 0.0
            data.qfrc_applied[:N_ARM] = tau
            mujoco.mj_step(model, data)
            viewer.sync()

            ep_norm = np.linalg.norm(e_p)
            ep_log.append(ep_norm)
            t_log.append(t)

            if settle_time is None and t > 0.2 and ep_norm < THRESH:
                settle_time = t

            step += 1
            t += SIM_DT

    results[label] = {
        'ep':np.array(ep_log),
        't':np.array(t_log),
        'settle_time': settle_time,
        'peak':np.array(ep_log).max(),
    }
    st = f'{settle_time:.3f} s' if settle_time else 'Did not settle'
    print(f'  Peak |e_p| : {results[label]["peak"]*1000:.2f} mm')
    print(f'  Settling time: {st}')

# plot all three overlaid
fig, ax = plt.subplots(figsize=(10, 5))
colors = ['r', 'b', 'g']
for (label, _), color in zip(DAMPING_CASES, colors):
    r = results[label]
    ax.plot(r['t'], r['ep']*1000, color=color, linewidth=1.8, label=label)
    if r['settle_time']:
        ax.axvline(r['settle_time'], color=color, linestyle=':', linewidth=1)

ax.axhline(THRESH*1000, color='k', linestyle='--', linewidth=1, label='1mm threshold')
ax.set_title(f'Step Response — Damping Sweep (K={K} N/m)')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position Error |e_p| (mm)')
ax.legend()
ax.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.expanduser('~/Bathsheba-Project-roboai/assets/test3_damping_sweep.png'), dpi=150)
plt.show()
print('\nPlot saved.')