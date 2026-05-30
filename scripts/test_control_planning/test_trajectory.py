import sys, os
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
from planning.trajectory import JointTrajectory

import numpy as np
import mujoco
import mujoco.viewer

MODEL_PATH = os.path.expanduser('~/mujoco_menagerie/franka_emika_panda/scene.xml')

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
print(f"Model loaded | timestep={model.opt.timestep}s | nu={model.nu}")

N_ARM = 7
SIM_HZ = int(1.0/model.opt.timestep)
GRIPPER = 255.0
DURATION = 5.0

home_pose = np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4])

# waypoints for trajectory test

WAYPOINTS = [
    np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]),
    np.array([0.0, -np.pi/6, 0.0, -2*np.pi/3, 0.0, np.pi/3, np.pi/6]),
    np.array([0.0, -np.pi/3, 0.0, -5*np.pi/6, 0.0, np.pi/6, np.pi/3]),
    np.array([0.0, -np.pi/4, 0.0, -3*np.pi/4, 0.0, np.pi/2, np.pi/4]),
]

# waypoints limits check

print("=== Waypoint Limit Check ===")
for w_idx, w in enumerate(WAYPOINTS):
    for i in range(N_ARM):
        lo, hi = model.actuator_ctrlrange[i]
        if not (lo <= w[i] <= hi):
            print(f"  W{w_idx} joint{i+1}: {np.rad2deg(w[i]):.1f} deg OUT OF RANGE [{np.rad2deg(lo):.1f}, {np.rad2deg(hi):.1f}]")
print("  All OK" )

seg_idx = 0
trajectory = JointTrajectory(WAYPOINTS[seg_idx], WAYPOINTS[seg_idx+1], DURATION)
print("launching viewer — close window to exit")
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        t_global = step*model.opt.timestep
        t_local = t_global - seg_idx*DURATION
        if trajectory.is_done(t_local):
            seg_idx += 1
            if seg_idx >= len(WAYPOINTS)-1:
                print("Trajectory complete!")
                break
            trajectory = JointTrajectory(WAYPOINTS[seg_idx], WAYPOINTS[seg_idx+1], DURATION)
            print(f"Starting segment {seg_idx} -> {seg_idx+1}")
        q_des, qd_des, _ = trajectory.evaluate(t_local)
        data.ctrl[:N_ARM] = q_des
        data.ctrl[7] = GRIPPER
        mujoco.mj_step(model, data)
        viewer.sync()
        if step % SIM_HZ ==0:
            q = data.qpos[:N_ARM]
            e_norm = np.linalg.norm(q_des - q)
            e_deg = np.rad2deg(q_des - q)
            print(f"t={t_global:.1f}s | seg={seg_idx} | |e|={e_norm:.5f} rad | e(deg)={e_deg.round(3)}")
        step += 1