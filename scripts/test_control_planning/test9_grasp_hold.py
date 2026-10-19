"""
T9: grasp-hold stability under different contact settings.

The arm picks the cube with the CS impedance controller (pre-grasp -> descend -> close -> lift 0.20 m),
then holds it for HOLD_DURATION. During the hold it logs, every step:
  - the cube position minus the EE position (rel_z shows contact creep),
  - the cube tilt: the angle between the cube's z-axis and world z (shows rotation about the grasp axis).
Each contact variant patches the compiled pick scene, so they can be compared on identical motion.
The cube target comes from ground truth (freejoint qpos), so perception does not affect the comparison.

Result (30 s hold, Kp=1000, Kr=100, passive compensation on; plot: assets/test9_grasp_hold.png):

| variant                          | noslip | cone      | cube condim / friction | creep, first 10 s | tilt: start -> 10 s (-> 30 s) | outcome           |
|----------------------------------|--------|-----------|------------------------|-------------------|-------------------------------|-------------------|
| baseline                         | 0      | pyramidal | 3 / 1 0.005 0.0001     | 8.91 mm           | 0.75 -> 2.67 deg              | fell after 22.4 s |
| noslip4                          | 4      | pyramidal | 3 / 1 0.005 0.0001     | 0.17 mm           | 0.56 -> 1.40 (-> 2.84) deg    | held, tilting     |
| torsion                          | 0      | pyramidal | 4 / 1 0.02 0.0001      | 13.20 mm          | 0.52 -> 1.03 deg              | fell after 14.6 s |
| noslip4_torsion                  | 4      | pyramidal | 4 / 1 0.02 0.0001      | 13.20 mm          | 0.47 -> 0.47 deg              | fell after 14.0 s |
| noslip4_torsion0.005             | 4      | pyramidal | 4 / 1 0.005 0.0001     | 13.22 mm          | 0.46 -> 0.47 deg              | fell after 14.0 s |
| noslip10 (overlaps noslip4)      | 10     | pyramidal | 3 / 1 0.005 0.0001     | 0.17 mm           | 0.56 -> 1.41 (-> 2.84) deg    | held, tilting     |
| noslip4_torsion_elliptic (final) | 4      | elliptic  | 4 / 1 0.02 0.0001      | 0.02 mm           | 0.47 -> 0.49 (-> 0.54) deg    | held              |

Takeaway: the slip is both creep and rotation. noslip removes most of the creep but not the slow
rotation; noslip10 tracks noslip4 to within 0.001 mm and 0.006 deg, so more iterations do not help. Torsional friction (condim 4) stops the rotation but, with
MuJoCo's default pyramidal friction cone, makes the sliding worse, even with noslip. With the elliptic
cone, noslip and torsional friction together hold the cube. The final row is what
envs/pick_scene.py and assets/scene_pick.xml now use.

Usage:
  HEADLESS=1 python scripts/test_control_planning/test9_grasp_hold.py
  VARIANTS=baseline,noslip4_torsion_elliptic HEADLESS=1 python ...      (subset; names are the keys of VARIANTS below)
"""
import os
import sys

PROJECT_ROOT = os.environ.get("PROJECT_ROOT", os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
sys.path.insert(0, PROJECT_ROOT)

import numpy as np
import mujoco
import mujoco.viewer
import matplotlib.pyplot as plt

from control.CS_impedance_controller import CS_Impedance_Controller
from envs.pick_scene import load_pick_scene
from planning.fk import fk
from planning.ik import dls_ik
from planning.PoseTrajectory import PoseTrajectory

# set HEADLESS=1 to run without the viewer
HEADLESS = os.environ.get("HEADLESS", "0") == "1"

N_ARM = 7
FINGER_LENGTH = 0.11
PREGRASP_HEIGHT = 0.25
LIFT_HEIGHT = 0.20
HOLD_TIME = 0.5
HOLD_DURATION = 30.0
CUBE_REST_Z = 0.425

q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4], dtype=np.float64)
R_down = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)

# contact variants: (noslip_iterations, friction cone, cube condim, cube friction [sliding, torsional, rolling])
PYR, ELL = mujoco.mjtCone.mjCONE_PYRAMIDAL, mujoco.mjtCone.mjCONE_ELLIPTIC
VARIANTS = {
    "baseline":             (0,  PYR, 3, [1, 0.005, 0.0001]),
    "noslip4":              (4,  PYR, 3, [1, 0.005, 0.0001]),
    "torsion":              (0,  PYR, 4, [1, 0.02,  0.0001]),
    "noslip4_torsion":      (4,  PYR, 4, [1, 0.02,  0.0001]),
    "noslip4_torsion0.005": (4,  PYR, 4, [1, 0.005, 0.0001]),
    "noslip10":             (10, PYR, 3, [1, 0.005, 0.0001]),
    "noslip4_torsion_elliptic": (4, ELL, 4, [1, 0.02, 0.0001]),   # final: what the scene now uses
}
selected = os.environ.get("VARIANTS", ",".join(VARIANTS)).split(",")

Kp = np.diag([1000.0]*3)
Kr = np.diag([100.0]*3)
Dp = np.diag([2*np.sqrt(1000.0)]*3)
Dr = np.diag([2*np.sqrt(100.0)]*3)
Kn = np.diag([20.0]*N_ARM)
Dn = np.diag([2*np.sqrt(20.0)]*N_ARM)


def build_model(variant):
    noslip, cone, condim, friction = VARIANTS[variant]
    model = load_pick_scene()
    model.opt.noslip_iterations = noslip
    model.opt.cone = cone
    g = model.body_geomadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")]
    model.geom_condim[g] = condim
    model.geom_friction[g] = friction

    # zero the actuator gains and biases for torque control
    for i in range(N_ARM):
        model.actuator_gainprm[i, 0] = 0.0
        model.actuator_biasprm[i, 1] = 0.0
        model.actuator_biasprm[i, 2] = 0.0
    return model


def run_variant(variant):
    model = build_model(variant)
    data = mujoco.MjData(model)
    data.qpos[:N_ARM] = q_home
    mujoco.mj_forward(model, data)
    dt = model.opt.timestep

    controller = CS_Impedance_Controller(Kp, Kr, Dp, Dr, Kn, Dn, q_null=q_home, compensation_passive=True)
    cube_adr = model.jnt_qposadr[model.body_jntadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cube")]]
    p_cube = data.qpos[cube_adr:cube_adr+3].copy()

    # waypoints: (target, duration, gripper during the move, settle time after)
    p_pre   = p_cube + [0, 0, PREGRASP_HEIGHT]
    p_grasp = p_cube + [0, 0, FINGER_LENGTH]
    p_lift  = p_grasp + [0, 0, LIFT_HEIGHT]
    plan = [(p_pre, 3.5, 255.0, HOLD_TIME), (p_grasp, 2.0, 255.0, HOLD_TIME),
            (p_grasp, None, 0.0, 1.0),                      # close the gripper in place
            (p_lift, 2.0, 0.0, 0.0)]

    T0 = fk(q_home)
    p_end, R_end = T0[:3, 3].copy(), T0[:3, :3].copy()
    log = dict(t=[], rel=[], tilt=[])
    state = dict(stage=0, traj=None, t0=0.0, gripper=255.0, t_hold=None, fell=False)

    def start_stage(t):
        nonlocal p_end, R_end
        target, T, gripper, _ = plan[state["stage"]]
        state["gripper"], state["t0"] = gripper, t
        if T is not None:
            state["traj"] = PoseTrajectory(p_end, R_end, target, R_down, T)
            result = dls_ik(controller.q_null, target, R_down)
            if result["converged"]:
                controller.q_null = result["q"]
            p_end, R_end = target.copy(), R_down.copy()

    def step(t):
        q = data.qpos[:N_ARM].copy()
        dq = data.qvel[:N_ARM].copy()
        mujoco.mj_step1(model, data)

        # sequencing
        if state["t_hold"] is None:
            if state["stage"] == 0 and state["traj"] is None and t == 0.0:
                start_stage(t)
            _, T, _, settle = plan[state["stage"]]
            if t - state["t0"] >= (T or 0.0) + settle:
                state["traj"] = None
                state["stage"] += 1
                if state["stage"] < len(plan):
                    start_stage(t)
                else:
                    state["t_hold"] = t

        if state["traj"] is not None:
            p_ref, R_ref, v_ref = state["traj"].evaluate(t - state["t0"])
        else:
            p_ref, R_ref, v_ref = p_end, R_end, np.zeros(6)

        tau, _, _ = controller.compute_Torque(q, dq, p_ref, R_ref, model, data, v_des=v_ref)
        data.qfrc_applied[:N_ARM] = tau
        data.ctrl[7] = state["gripper"]
        mujoco.mj_step2(model, data)

        if state["t_hold"] is not None:
            rel = data.qpos[cube_adr:cube_adr+3] - fk(data.qpos[:N_ARM])[:3, 3]
            R = np.zeros(9)
            mujoco.mju_quat2Mat(R, data.qpos[cube_adr+3:cube_adr+7])
            log["t"].append(t - state["t_hold"])
            log["rel"].append(rel)
            log["tilt"].append(np.degrees(np.arccos(np.clip(R[8], -1, 1))))
            if data.qpos[cube_adr+2] < CUBE_REST_Z + 0.02:
                state["fell"] = True

    def done(t):
        return state["fell"] or (state["t_hold"] is not None and t - state["t_hold"] >= HOLD_DURATION)

    t = 0.0
    if HEADLESS:
        while not done(t):
            step(t)
            t += dt
    else:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.distance = 2.0
            while viewer.is_running() and not done(t):
                step(t)
                viewer.sync()
                t += dt

    out = {k: np.array(v) for k, v in log.items()}
    out["fell"] = state["fell"]
    return out


results = {}
for name in selected:
    print(f"===== {name}  (noslip, cone, condim, friction) = {VARIANTS[name]}")
    r = run_variant(name)
    results[name] = r
    if len(r["t"]) == 0:
        print("  never reached the hold")
        continue
    rel_z = r["rel"][:, 2] * 1000
    for k in np.searchsorted(r["t"], np.arange(0.0, r["t"][-1], 5.0)):
        print(f"  t_hold {r['t'][k]:5.1f} s | rel_z {rel_z[k]:8.2f} mm | tilt {r['tilt'][k]:6.3f} deg")
    outcome = f"FELL after {r['t'][-1]:.1f} s" if r["fell"] else f"held {r['t'][-1]:.0f} s"
    k10 = min(np.searchsorted(r["t"], 10.0), len(r["t"]) - 1)
    print(f"  creep over first 10 s {rel_z[0]-rel_z[k10]:.2f} mm | tilt {r['tilt'][0]:.2f} deg -> "
          f"{r['tilt'][k10]:.2f} deg at 10 s | {outcome}")

# plotting
LABELS = {"noslip4_torsion": "noslip4 + torsion",
          "noslip4_torsion0.005": "noslip4 + torsion 0.005",
          "noslip4_torsion_elliptic": "noslip4 + torsion + elliptic (final)"}

fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
plotted = {}
for name, r in results.items():
    if len(r["t"]) == 0:
        continue
    label = LABELS.get(name, name) + (" (fell)" if r["fell"] else "")
    creep = (r["rel"][:, 2] - r["rel"][0, 2]) * 1000

    # a run within 0.01 mm / 0.01 deg of an earlier one would hide under it: dash it and say so in the legend
    style = "-"
    for other, (o_t, o_creep, o_tilt) in plotted.items():
        if len(o_t) == len(r["t"]) and np.allclose(o_creep, creep, atol=0.01) and np.allclose(o_tilt, r["tilt"], atol=0.01):
            label += f" (overlaps {LABELS.get(other, other)})"
            style = "--"
            break
    plotted[name] = (r["t"], creep, r["tilt"])

    line, = axes[0].plot(r["t"], creep, style, label=label, lw=2.0 if style == "--" else 1.5)
    axes[1].plot(r["t"], r["tilt"], style, color=line.get_color(), lw=2.0 if style == "--" else 1.5)
    if r["fell"]:
        for ax in axes:
            ax.axvline(r["t"][-1], color=line.get_color(), ls=":", lw=1)
# fixed ranges: a fall shoots off the axes and is marked by the dotted line at its time
axes[0].set_ylim(-30, 2)
axes[1].set_ylim(0, 8)
axes[0].set_ylabel("rel_z creep from hold start [mm]")
axes[0].set_title("T9: cube vs end effector during a 30 s hold, by contact setting")
axes[1].set_ylabel("cube tilt [deg]")
axes[1].set_xlabel("time since hold start [s]")
axes[0].legend(loc="center left", bbox_to_anchor=(1.01, 0.5), fontsize=8)
for ax in axes:
    ax.grid(True, alpha=0.3)

plt.tight_layout()
suffix = "" if len(selected) == len(VARIANTS) else "_" + "_".join(selected)
plt.savefig(os.path.join(PROJECT_ROOT, "assets", f"test9_grasp_hold{suffix}.png"), dpi=150)
if not HEADLESS:
    plt.show()
