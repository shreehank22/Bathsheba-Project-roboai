import sys
sys.path.insert(0, '/home/shreehan/roboai_project')

import cv2
import numpy as np
import mujoco
from perception.detector import segment_red, camera_intrinsics, pixel_to_world

from planning.trajectory import JointTrajectory
from planning.ik import dls_ik
from planning.fk import fk

MODEL_PATH = '/home/shreehan/mujoco_menagerie/franka_emika_panda/scene_pick.xml'
CAM_NAME = 'fixed_cam'
H,W=480,640
CUBE_HALF_SIZE = 0.05/2


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)
N_ARM = 7
SIM_HZ = int(1.0 / model.opt.timestep)
GRIPPER = 255.0   # open

# Render RGB
renderer = mujoco.Renderer(model,height=H,width=W)
renderer.update_scene(data,camera = CAM_NAME)
rgb_image = renderer.render()

# Render depth
renderer.enable_depth_rendering()
renderer.update_scene(data,camera = CAM_NAME)
depth_image = renderer.render()
renderer.disable_depth_rendering()

# Execute the perception module
mask,centroid,detected = segment_red(rgb_image)
print("Mask and Centroid:", centroid, detected)

if not detected:
    print("Cube not detected — exiting")
    exit()

u,v = centroid
depth = depth_image[v,u]
print("Depth at centroid:", depth)

K = camera_intrinsics(model,CAM_NAME,H,W)
p_cube = pixel_to_world(u,v,depth,K,model,data,CAM_NAME)
p_cube[2] -= CUBE_HALF_SIZE
print("3D World Coordinates:", p_cube)

print(f"Cube detected at pixel : {centroid}")
print(f"Cube 3D position       : {np.round(p_cube, 4)}")
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
p_cube  = data.xpos[cube_id].copy()
print(f"Ground truth cube position: {np.round(p_cube, 4)}")

# Grasping pose development 

PREGRASP_OFFSET = 0.20
pregrasp_pos = p_cube.copy()
pregrasp_pos[2] += PREGRASP_OFFSET
print(f"Pre-grasp position    : {np.round(pregrasp_pos, 4)}")

R_des = np.array([[ 1,  0,  0],
                  [ 0, -1,  0],
                  [ 0,  0, -1]], dtype=np.float64)

q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
result = dls_ik(q_home, pregrasp_pos, R_des)


if not result['converged']:
    print("IK did not converge — exiting")
    exit()
q_pregrasp = result['q']
print(f"Pre-grasp joint angles: {np.round(q_pregrasp, 4)}")
print(f"Pre-grasp position error: {result['pos_err']:.4f} m"
      f", orientation error: {result['rot_err']:.4f} rad"
      f", iterations: {result['n_iter']}")

# Trajectory and execution
q_start = data.qpos[:N_ARM].copy()
traj = JointTrajectory(q_start, q_pregrasp, T=3.0)


FINGER_LENGTH = 0.11
p_grasp = p_cube.copy()
p_grasp[2] += FINGER_LENGTH

result_grasp = dls_ik(q_pregrasp, p_grasp, R_des)
if not result_grasp['converged']:
    print("Grasp IK did not converge — exiting")
    exit()
q_grasp = result_grasp['q']
print(f"Grasp position     : {np.round(p_grasp, 4)}")
print(f"Grasp IK error     : {result_grasp['pos_err']*1000:.2f} mm")

print("\nLaunching viewer...")
import mujoco.viewer

HOLD_TIME  = 0.5
traj_descend = None
t_hold_start = None
traj_lift = None
q_lift = None
traj_place = None
q_place = None
t_start = 0.0
phase = 'pregrasp'

with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        t_global = step*model.opt.timestep
        t_local = t_global-t_start

        if phase == 'pregrasp':
            q_des, _, _ = traj.evaluate(t_local)
            if traj.is_done(t_local):
                phase = 'hold'
                t_hold_start = t_global
                print("Pre-grasp reached — holding...")

        elif phase == 'hold':
            q_des = q_pregrasp
            if t_global - t_hold_start >= HOLD_TIME:
                phase = 'descend'
                t_start = t_global
                traj_descend = JointTrajectory(q_pregrasp, q_grasp, T=2.0)
                print("Starting descend...")

        elif phase == 'descend':
            t_local = t_global - t_start
            q_des, _, _ = traj_descend.evaluate(t_local)
            if traj_descend.is_done(t_local):
                phase = 'grasp'
                t_hold_start = t_global 
                print("Grasp position reached!")

        elif phase == 'grasp':
            q_des = q_grasp
            if t_global - t_hold_start >= HOLD_TIME:  
                phase = 'close_gripper'
                t_start = t_global
                print("Closing Gripper")
        elif phase == 'close_gripper':
            q_des = q_grasp
            GRIPPER = 0.0   # close
            if t_global - t_start >= 1.0:
                p_lift = p_grasp.copy()
                p_lift[2] += 0.20
                result_lift = dls_ik(q_grasp, p_lift, R_des)
                q_lift = result_lift['q']
                traj_lift = JointTrajectory(q_grasp, q_lift, T=2.0)
                phase = 'lift'
                t_start = t_global
                print("Lifting...")
        elif phase == 'lift':
            if traj_lift is None:
                pass
            else:
                t_local = t_global-t_start
                q_des, _, _ = traj_lift.evaluate(t_local)
                GRIPPER = 0.0
                if traj_lift.is_done(t_local):
                    phase = 'lifted'
                    print("Lift complete!")
        elif phase == 'lifted':
            q_des = q_lift
            GRIPPER = 0.0
            if traj_place is None:
                p_place_above = np.array([0.3,0.2,0.625])
                result_place = dls_ik(q_lift, p_place_above, R_des)
                q_place = result_place['q']
                traj_place = JointTrajectory(q_lift, q_place, T=2.0)
                phase = 'move_to_place'
                t_start = t_global
                print("Moving to place position...")
        elif phase == 'move_to_place':
            t_local = t_global-t_start
            q_des, _, _ = traj_place.evaluate(t_local)
            GRIPPER = 0.0
            if traj_place.is_done(t_local):
                phase = 'place'
                t_hold_start = t_global
                print("Place position reached!")
        elif phase == 'place':
            q_des   = q_place
            GRIPPER = 255.0   # open immediately
            if t_global - t_hold_start >= 1.0:
                print("Cube released!")
                phase = 'done'


        data.ctrl[:N_ARM] = q_des
        data.ctrl[7] = GRIPPER

        mujoco.mj_step(model, data)
        viewer.sync()

        if step % SIM_HZ == 0:
            q_curr = data.qpos[:N_ARM]
            T_curr = fk(q_curr)
            p_curr = T_curr[:3, 3]
            print(f"t={t_global:.1f}s | phase={phase} | EE={np.round(p_curr,3)}")

        step += 1