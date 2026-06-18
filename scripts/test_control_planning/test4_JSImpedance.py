import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import numpy as np
import mujoco
import mujoco.viewer
from planning.fk import fk
import matplotlib.pyplot as plt
from control.JS_impedance_controller import JS_Impedance_Controller

MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
SETTLE_TIME = 3.0
FORCE_TIME  = 0.2
RETURN_TIME = 5.0
TOTAL_TIME  = SETTLE_TIME + FORCE_TIME + RETURN_TIME
FORCE_MAG = 60.0
N_ARM  = 7
SIM_DT = model.opt.timestep
ee_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

for i in range(N_ARM):
    model.actuator_gainprm[i, 0] = 0.0
    model.actuator_biasprm[i, 1] = 0.0
    model.actuator_biasprm[i, 2] = 0.0
mujoco.mj_forward(model, data)

q_home = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

K_JS = 500
D_JS = 2*np.sqrt(K_JS)
k_js = np.diag([K_JS]*N_ARM)
d_js = np.diag([D_JS]*N_ARM)
JS_CONTROLLER = JS_Impedance_Controller(k_js, d_js, q_home)

mujoco.mj_resetData(model, data)
data.qpos[:N_ARM] = q_home
mujoco.mj_forward(model, data)

js_eq_log, js_ee_log, js_t_log, js_q_log = [], [], [], []
t, step = 0.0, 0

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    while t < TOTAL_TIME and viewer.is_running():
        q  = data.qpos[:N_ARM].copy()
        dq = data.qvel[:N_ARM].copy()
        tau,e_q = JS_CONTROLLER.compute_Torque(q, dq, model, data)
        data.ctrl[:N_ARM] = 0.0
        data.qfrc_applied[:N_ARM] = tau
        if SETTLE_TIME <= t < SETTLE_TIME + FORCE_TIME:
            data.xfrc_applied[ee_id] = [0, FORCE_MAG, 0, 0, 0, 0]
        else:
            data.xfrc_applied[ee_id] = [0, 0, 0, 0, 0, 0]
        mujoco.mj_step(model, data)
        viewer.sync()

        js_eq_log.append(np.linalg.norm(e_q))
        js_ee_log.append(fk(q)[:3, 3].copy())
        js_q_log.append(q.copy())
        js_t_log.append(t)

        step += 1
        t = step * SIM_DT

js_eq_log = np.array(js_eq_log)
js_ee_log = np.array(js_ee_log)
js_q_log = np.array(js_q_log)
js_t_log = np.array(js_t_log)

# per-joint deflection analysis
force_mask = (js_t_log >= SETTLE_TIME) & (js_t_log < SETTLE_TIME + FORCE_TIME)
q_during = js_q_log[force_mask]
q_baseline = js_q_log[0]

print(f'\n=== JS Impedance Controller ===')
print(f'Peak |e_q|: {js_eq_log.max():.6f} rad')
print(f'Steady-state |e_q|: {js_eq_log[-100:].mean():.6f} rad')
print(f'Peak EE Y-disp: {(js_ee_log[:,1]-js_ee_log[0,1]).max()*1000:.2f} mm')
print(f'\n=== Per-Joint Deflection under {FORCE_MAG}N Y-force ===')
for i in range(N_ARM):
    peak = np.abs(q_during[:, i]-q_baseline[i]).max()
    print(f'Joint {i+1}:{np.degrees(peak):8.4f} deg  ({peak:.6f} rad)')

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6))

ax1.plot(js_t_log, js_eq_log, 'r-', linewidth=1.8)
ax1.axvline(SETTLE_TIME,color='gray',linestyle=':',linewidth=1,label='Force ON')
ax1.axvline(SETTLE_TIME+FORCE_TIME, color='orange', linestyle=':', linewidth=1, label='Force OFF')
ax1.set_title('JS Impedance — Joint Error |e_q|')
ax1.set_ylabel('rad'); ax1.set_xlabel('Time (s)')
ax1.legend(); ax1.grid(True, alpha=0.4)

ax2.plot(js_t_log, (js_ee_log[:,1]-js_ee_log[0,1])*1000, 'r-', linewidth=1.8)
ax2.axvline(SETTLE_TIME,color='gray',linestyle=':',linewidth=1)
ax2.axvline(SETTLE_TIME+FORCE_TIME, color='orange', linestyle=':', linewidth=1)
ax2.set_title(f'JS Impedance — EE Y-displacement under {FORCE_MAG}N disturbance')
ax2.set_ylabel('mm'); ax2.set_xlabel('Time (s)')
ax2.grid(True, alpha=0.4)
plt.tight_layout()
plt.savefig(os.path.expanduser('~/Bathsheba-Project-roboai/assets/test6_js_only.png'), dpi=150)
plt.show()

fig2, axes = plt.subplots(7, 1, figsize=(10, 14), sharex=True)
for i in range(N_ARM):
    axes[i].plot(js_t_log,np.degrees(js_q_log[:, i]), 'r-', linewidth=1.2)
    axes[i].axvline(SETTLE_TIME,color='gray',linestyle=':', linewidth=1)
    axes[i].axvline(SETTLE_TIME+FORCE_TIME, color='orange', linestyle=':', linewidth=1)
    axes[i].set_ylabel(f'J{i+1} (deg)', fontsize=8)
    axes[i].grid(True, alpha=0.4)
    axes[i].axhline(np.degrees(q_home[i]), color='k', linestyle='--', linewidth=0.8)

axes[0].set_title(f'Per-Joint Angles under {FORCE_MAG}N Y-force — JS Impedance')
axes[-1].set_xlabel('Time (s)')
plt.tight_layout()
plt.savefig(os.path.expanduser('~/Bathsheba-Project-roboai/assets/test6_js_joint_angles.png'), dpi=150)
plt.show()