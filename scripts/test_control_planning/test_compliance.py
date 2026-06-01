import sys
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
import numpy as np
import mujoco
import mujoco.viewer
import os
import matplotlib.pyplot as plt
from planning.fk import fk
from control.impedance_controller import Impedance_Controller

MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
N_ARM = 7
SIM_DT = model.opt.timestep

q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
mujoco.mj_resetData(model, data)
data.qpos[:N_ARM] = q_home
mujoco.mj_forward(model, data)
T     = fk(q_home)
p_des = T[:3,3].copy()
R_des = T[:3,:3].copy()
print(f'Desired EE pos: {np.round(p_des,4)}')

ee_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

GAIN_SETS = [
    {'label': 'A: k=200  z=1.0', 'k': 200,  'd': 2*np.sqrt(200) },
    {'label': 'B: k=800  z=1.0', 'k': 800,  'd': 2*np.sqrt(800) },
    {'label': 'C: k=800  z=2.0', 'k': 800,  'd': 4*np.sqrt(800) },
    {'label': 'D: k=800  z=0.5', 'k': 800,  'd': np.sqrt(800)   },
]

SETTLE_TIME     = 1.0
FORCE_TIME      = 2.0
RETURN_TIME     = 20.0
TOTAL_TIME      = SETTLE_TIME + FORCE_TIME + RETURN_TIME
FORCE_MAGNITUDE = 120.0
f_dir           = np.array([1.0, 0.0, 0.0])
THRESHOLD       = 5.0

all_results = []

for gs in GAIN_SETS:
    k = gs['k']
    d = gs['d']
    label = gs['label']
    Kp = np.diag([k, k, k]); Kr = np.diag([k, k, k])
    Dp = np.diag([d, d, d]); Dr = np.diag([d, d, d])
    ctrl = Impedance_Controller(Kp, Kr, Dp, Dr)

    mujoco.mj_resetData(model, data)
    data.qpos[:N_ARM] = q_home
    mujoco.mj_forward(model, data)

    t_log=[]; e_p_log=[]; settling_time=None
    force_off_time = SETTLE_TIME + FORCE_TIME

    print(f'\nRunning {label}')
    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.distance = 1.5
        t = 0.0
        while t < TOTAL_TIME and viewer.is_running():
            q  = data.qpos[:N_ARM].copy()
            dq = data.qvel[:N_ARM].copy()

            f_ext = f_dir*FORCE_MAGNITUDE if SETTLE_TIME<=t<force_off_time else np.zeros(3)

            tau, e_p, e_r = ctrl.compute_Torque(q, dq, p_des, R_des, model, data)
            data.qfrc_applied[:N_ARM] = tau 
            data.ctrl[:N_ARM]         = q
            data.xfrc_applied[ee_id]  = [f_ext[0], f_ext[1], f_ext[2], 0, 0, 0]

            mujoco.mj_step(model, data)
            viewer.sync()

            e_norm = np.linalg.norm(e_p)*1000
            t_log.append(t); e_p_log.append(e_norm)

            if t > force_off_time and e_norm < THRESHOLD and settling_time is None:
                settling_time = t - force_off_time
                print(f'  Settled to <{THRESHOLD}mm at t={t:.2f}s (settling={settling_time:.2f}s)')

            t += SIM_DT

    if settling_time is None:
        print(f'  Did not settle within {RETURN_TIME}s')

    all_results.append({'label': label, 't': t_log, 'e': e_p_log,
                        'settling_time': settling_time, 'k': k, 'd': d})

# ── Plot ──────────────────────────────────────────────────────────────────────
plt.figure(figsize=(10, 5))
colors = ['b', 'r', 'g', 'm']
for res, col in zip(all_results, colors):
    plt.plot(res['t'], res['e'], color=col, linewidth=2, label=res['label'])
plt.axvspan(SETTLE_TIME, SETTLE_TIME+FORCE_TIME, alpha=0.1, color='red', label='Force applied')
plt.axhline(THRESHOLD, color='k', linestyle='--', linewidth=1, label=f'Threshold {THRESHOLD}mm')
plt.axhline(0, color='k', linestyle='--', linewidth=0.5)
plt.xlabel('Time (s)'); plt.ylabel('Deflection magnitude (mm)')
plt.title(f'Controller Tuning — Gain Comparison  F={FORCE_MAGNITUDE}N')
plt.legend(); plt.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig('/home/shreehan/Bathsheba-Project-roboai/assets/controller_tuning.png', dpi=150)
plt.show()

# ── Summary table ─────────────────────────────────────────────────────────────
print('\n' + '='*55)
print(f'{"Gain Set":<20} {"k":>6} {"d":>6} {"Settling (s)":>14}')
print('='*55)
for res in all_results:
    label = res['label']
    st = f'{res["settling_time"]:.2f}s' if res['settling_time'] else 'DNF'
    print(f'{label:<20} {res["k"]:>6} {res["d"]:>6.1f} {st:>14}')
print('='*55)
print('Plot saved.')