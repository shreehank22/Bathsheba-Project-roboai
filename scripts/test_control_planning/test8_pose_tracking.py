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
from planning.ik import geometric_jacobian
from planning.PoseTrajectory import PoseTrajectory

MODEL_PATH = os.path.join(os.environ.get('MUJOCO_MENAGERIE', os.path.expanduser('~/mujoco_menagerie')), 'franka_emika_panda', 'scene.xml')
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)

N_ARM  = 7
SIM_DT = model.opt.timestep
ee_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'hand')

# set HEADLESS=1 to run without the viewer
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

# zero the actuator gains and biases for torque control
for i in range(N_ARM):
    model.actuator_gainprm[i, 0] = 0.0
    model.actuator_biasprm[i, 1] = 0.0
    model.actuator_biasprm[i, 2] = 0.0

print("dof_damping[:7]      =", model.dof_damping[:N_ARM])
print("dof_frictionloss[:7] =", model.dof_frictionloss[:N_ARM])

# ablation D: set ZERO_DAMPING=1 to remove joint damping from the model
ZERO_DAMPING = os.environ.get("ZERO_DAMPING", "0") == "1"
if ZERO_DAMPING:
    model.dof_damping[:N_ARM] = 0.0


# set home position
q_home = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])
T_home = fk(q_home)

# retrieving position and orientation of the end-effector
p_des = T_home[:3, 3]
R_des = T_home[:3, :3]

# rotational stiffness; override with KR=200 for the doubled-Kr ablation
KR = float(os.environ.get("KR", "100"))

# controller options; override with USE_NULLSPACE=0 / COMPENSATE=0
USE_NULLSPACE = os.environ.get("USE_NULLSPACE", "1") == "1"
COMPENSATE = os.environ.get("COMPENSATE", "1") == "1"

PLOT_SUFFIX = (("_ns" if USE_NULLSPACE else "") + ("_comp" if COMPENSATE else "")
               + ("" if KR == 100.0 else f"_Kr{KR:g}") + ("_nodamping" if ZERO_DAMPING else ""))
CONFIG_LABEL = ", ".join([("nullspace" if USE_NULLSPACE else "no nullspace"),
                          ("passive comp" if COMPENSATE else "no passive comp")]
                         + ([f"Kr={KR:g}"] if KR != 100.0 else [])
                         + (["zero joint damping"] if ZERO_DAMPING else []))

Kp = np.diag([1000.0]*3)
Kr = np.diag([KR]*3)
Dp = np.diag([2*np.sqrt(1000.0)]*3)
Dr = np.diag([2*np.sqrt(KR)]*3)


# gains for nullspace control
Kn = np.diag([20.0]*N_ARM)
Dn = np.diag([2*np.sqrt(20.0)]*N_ARM)

q_null = q_home

# defining the controller; USE_NULLSPACE and COMPENSATE select null-space control and passive torque compensation
if USE_NULLSPACE:
    CS_CONTROLLER = CS_Impedance_Controller(Kp, Kr, Dp, Dr, Kn, Dn, q_null=q_null, compensation_passive=COMPENSATE)
else:
    CS_CONTROLLER = CS_Impedance_Controller(Kp, Kr, Dp, Dr, compensation_passive=COMPENSATE)


# goal pose: 0.15 m away from home, rotated 30 deg about world z
p_goal = p_des + np.array([0.10, 0.10, -0.05])
yaw = np.deg2rad(30.0)
Rz = np.array([[np.cos(yaw), -np.sin(yaw), 0.0],[np.sin(yaw),  np.cos(yaw), 0.0],[0.0,0.0,1.0]])

R_goal = Rz @ R_des

T_MOVE = 4.0
T_HOLD = 1.5
TOTAL_TIME = T_MOVE + T_HOLD

traj = PoseTrajectory(p_des, R_des, p_goal, R_goal, T_MOVE)

def run_trial(use_feedforward):
    mujoco.mj_resetData(model, data)
    data.qpos[:N_ARM] = q_home
    mujoco.mj_forward(model, data)

    t_log, e_p_log, e_r_log, p_cmd_log, p_act_log = [], [], [], [], []
    w_des_log, w_act_log = [], []

    def step(t):
        q = data.qpos[:N_ARM].copy()
        dq = data.qvel[:N_ARM].copy()
        mujoco.mj_step1(model, data)

        p_cmd, R_cmd, v_cmd = traj.evaluate(t)
        v_des = v_cmd if use_feedforward else None
        tau, e_p, e_r = CS_CONTROLLER.compute_Torque(q, dq, p_cmd, R_cmd, model, data, v_des=v_des)
        data.qfrc_applied[:N_ARM] = tau

        mujoco.mj_step2(model, data)

        t_log.append(t)
        e_p_log.append(e_p.copy())
        e_r_log.append(e_r.copy())
        p_cmd_log.append(p_cmd.copy())
        p_act_log.append(p_cmd - e_p)   # actual EE position from the controller's fk(q)
        w_des_log.append(v_cmd[3:].copy())
        w_act_log.append(geometric_jacobian(q)[3:] @ dq)

    t = 0.0
    if HEADLESS:
        while t < TOTAL_TIME:
            step(t)
            t += SIM_DT
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 2.0
            while viewer.is_running() and t < TOTAL_TIME:
                step(t)
                viewer.sync()
                t += SIM_DT

    return dict(t=np.array(t_log), e_p=np.array(e_p_log), e_r=np.array(e_r_log),
                p_cmd=np.array(p_cmd_log), p_act=np.array(p_act_log),
                w_des=np.array(w_des_log), w_act=np.array(w_act_log))


results = {"with v_des": run_trial(use_feedforward=True),"v_des = None": run_trial(use_feedforward=False),}

# summary of results
for name, r in results.items():
    ep = np.linalg.norm(r["e_p"], axis=1)
    er = np.linalg.norm(r["e_r"], axis=1)
    moving = r["t"] < T_MOVE
    print(f"[{name}]")
    print(f"  moving: peak ||e_p|| = {ep[moving].max()*1000:.3f} mm, RMS = {np.sqrt(np.mean(ep[moving]**2))*1000:.3f} mm")
    print(f"  moving: peak ||e_r|| = {er[moving].max():.5f} rad, RMS = {np.sqrt(np.mean(er[moving]**2)):.5f} rad")
    print(f"  final:  ||e_p|| = {ep[-1]*1000:.4f} mm, ||e_r|| = {er[-1]:.6f} rad")

# plotting
fig, axes = plt.subplots(3, 1, figsize=(10, 11), sharex=True)
styles = {"with v_des": "-", "v_des = None": "--"}

for name, r in results.items():
    axes[0].plot(r["t"], np.linalg.norm(r["e_p"], axis=1)*1000, styles[name], label=name)
    axes[1].plot(r["t"], np.linalg.norm(r["e_r"], axis=1), styles[name], label=name)
axes[0].set_ylabel("||e_p|| [mm]")
axes[0].set_title(f"Pose tracking ({CONFIG_LABEL}): 0.15 m + 30 deg yaw over 4 s, then 1.5 s hold")
axes[1].set_ylabel("||e_r|| [rad]")

# commanded vs actual position
r_ff = results["with v_des"]
r_no = results["v_des = None"]
for k, (lbl, c) in enumerate(zip(["x", "y", "z"], ["C0", "C1", "C2"])):
    axes[2].plot(r_ff["t"], r_ff["p_cmd"][:, k], color=c, lw=2.5, alpha=0.35, label=f"{lbl} cmd")
    axes[2].plot(r_ff["t"], r_ff["p_act"][:, k], color=c, lw=1.0, label=f"{lbl} actual (v_des)")
    axes[2].plot(r_no["t"], r_no["p_act"][:, k], color=c, lw=1.0, ls="--", label=f"{lbl} actual (None)")
axes[2].set_ylabel("EE position [m]")
axes[2].set_xlabel("time [s]")
axes[2].legend(loc="upper right", fontsize=8, ncol=3)
for ax in axes:
    ax.axvline(T_MOVE, color="k", lw=0.8, ls=":")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8, ncol=3)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "assets", f"test8_pose_tracking{PLOT_SUFFIX}.png"), dpi=150)

# angular velocity diagnostic: omega_des (trajectory) vs omega_actual (J[3:] @ dq)
fig2, axes2 = plt.subplots(2, 1, figsize=(10, 7), sharex=True)
for name, r in results.items():
    wd_norm = np.linalg.norm(r["w_des"], axis=1)
    wa_norm = np.linalg.norm(r["w_act"], axis=1)
    axes2[0].plot(r["t"], wa_norm, styles[name], label=f"||w_actual|| ({name})")

    # the angle is undefined when either vector is ~0, so mask those samples
    valid = (wd_norm > 0.02*wd_norm.max()) & (wa_norm > 1e-6)
    cos_a = np.sum(r["w_des"]*r["w_act"], axis=1) / np.maximum(wd_norm*wa_norm, 1e-12)
    angle = np.where(valid, np.degrees(np.arccos(np.clip(cos_a, -1, 1))), np.nan)
    axes2[1].plot(r["t"], angle, styles[name], label=name)
    print(f"[{name}] angle(w_des, w_actual) while moving: max = {np.nanmax(angle):.2f} deg, "
          f"mean = {np.nanmean(angle):.2f} deg; peak ||w_des|| = {wd_norm.max():.4f}, peak ||w_actual|| = {wa_norm.max():.4f} rad/s")
axes2[0].plot(r_ff["t"], np.linalg.norm(r_ff["w_des"], axis=1), color="k", lw=2.5, alpha=0.3, label="||w_des||")
axes2[0].set_ylabel("||w|| [rad/s]")
axes2[0].set_title(f"Angular velocity ({CONFIG_LABEL}): trajectory vs actual")
axes2[1].set_ylabel("angle(w_des, w_actual) [deg]")
axes2[1].set_xlabel("time [s]")
for ax in axes2:
    ax.axvline(T_MOVE, color="k", lw=0.8, ls=":")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)

plt.tight_layout()
plt.savefig(os.path.join(PROJECT_ROOT, "assets", f"test8_omega_diagnostic{PLOT_SUFFIX}.png"), dpi=150)
if not HEADLESS:
    plt.show()
