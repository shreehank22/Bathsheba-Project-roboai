import os
import cv2
import threading

import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import mujoco
import mujoco.viewer
from perception.detector import segment_red, segment_green, camera_intrinsics, pixel_to_world, pixel_to_plane
from perception.camera_stream import push_frames, start_stream, stop_stream
from planning.fk import fk
from envs.pick_scene import load_pick_scene
from control.CS_impedance_controller import CS_Impedance_Controller
from planning.PoseTrajectory import PoseTrajectory
from planning.ik import dls_ik

CAM_NAME = 'fixed_cam'
CAM_NAME2 = 'fixed_cam_opposite'
H,W = 480, 640
CUBE_HALF_SIZE = 0.05/2
PATCH_Z = 0.4015
PREGRASP_HEIGHT = 0.25   # height above the cube; 0.25 is where the wrist camera was validated
HOLD_TIME = 0.5
FINGER_LENGTH = 0.11
CUBE_REST_Z = 0.425      # cube centre height resting on the table
ENABLE_FIXED_CAM = True  # live fixed-camera window
ENABLE_WRIST_CAM = True  # live wrist-camera window
ENABLE_OPPOSITE_CAM = True  # live opposite-camera window
STREAM_EVERY = 10        # push camera frames every N sim steps


# default configs
q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4], dtype=np.float64)
R_des_default = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)

model = load_pick_scene()
N_ARM = 7
data = mujoco.MjData(model)
data.qpos[:N_ARM] = q_home
mujoco.mj_forward(model, data)
SIM_HZ = int(1/model.opt.timestep)
SIM_DT = model.opt.timestep
GRIPPER = 255.0
PLACE_ABOVE_Z = 0.63
PLACE_CLEARANCE = 0.003

# Fixed camera perception setup
renderer_rgb = mujoco.Renderer(model, height=H, width=W)
renderer_rgb.update_scene(data, camera=CAM_NAME)
rgb_image = renderer_rgb.render()

renderer_depth = mujoco.Renderer(model, height=H, width=W)
renderer_depth.enable_depth_rendering()
renderer_depth.update_scene(data, camera=CAM_NAME)
depth_image = renderer_depth.render()

K = camera_intrinsics(model,CAM_NAME,height=H,width=W)

# detection of the cube in the initial frame
mask,centroid,detected = segment_red(rgb_image)
print(f"Initial detection: Detected={detected}, Centroid={centroid}")

if not detected:
    print("No object detected in the initial frame. Exiting.");exit(1)
    
u,v = centroid
depth = depth_image[v,u]

p_cube = pixel_to_world(u,v,depth,K,model,data,CAM_NAME)
p_cube[2] -= CUBE_HALF_SIZE
print(f"Initial cube position: {p_cube}")

# detection of the green patch in the initial frame
mask_green,centroid_green,detected_green = segment_green(rgb_image)

if not detected_green:
    print("No green patch detected in the initial frame. Exiting.");exit(1)

print(f"Green patch detection: Detected={detected_green}, Centroid={centroid_green}")

p_patch = pixel_to_plane(centroid_green[0], centroid_green[1], K, model, data, CAM_NAME, PATCH_Z)
if p_patch is None:
    print("Failed to project green patch to the plane. Exiting.");exit(1)

print(f"Green patch position: {p_patch}")

# ground-truth check against the place_patch geom (valid since mj_forward has run)
patch_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'place_patch')
p_patch_true = data.geom_xpos[patch_id]

print(f"Green patch true position: {p_patch_true}")
print(f"Green patch x-y error: {np.linalg.norm(p_patch[:2] - p_patch_true[:2])*1000:.2f} mm")

# setting up the impedance controller with nullspace control and passive joint torque compensation
Kp = np.diag([1000.0]*3)
Kr = np.diag([100.0]*3)
Dp = np.diag([2*np.sqrt(1000.0)]*3)
Dr = np.diag([2*np.sqrt(100.0)]*3)
Kn = np.diag([20.0]*N_ARM)
Dn = np.diag([2*np.sqrt(20.0)]*N_ARM)
q_null = q_home # setting the nullspace target configuration to the home position


CS_CONTROLLER = CS_Impedance_Controller(Kp, Kr, Dp, Dr, Kn, Dn, q_null=q_null, compensation_passive=True)

# zero the actuator gains and biases for torque control
for i in range(N_ARM):
    model.actuator_gainprm[i, 0] = 0.0
    model.actuator_biasprm[i, 1] = 0.0
    model.actuator_biasprm[i, 2] = 0.0

print("dof_damping[:7] =", model.dof_damping[:N_ARM])
print("dof_frictionloss[:7] =", model.dof_frictionloss[:N_ARM])

# wrist camera renderers and intrinsics
renderer_wrist = mujoco.Renderer(model, height=H, width=W)
renderer_depth_wrist = mujoco.Renderer(model, height=H, width=W)
renderer_depth_wrist.enable_depth_rendering()
K_wrist = camera_intrinsics(model, 'wrist_cam', H, W)

# opposite camera renderers and intrinsics
if ENABLE_OPPOSITE_CAM:
    renderer_opp = mujoco.Renderer(model, height=H, width=W)
    renderer_depth_opp = mujoco.Renderer(model, height=H, width=W)
    renderer_depth_opp.enable_depth_rendering()
    K_opp = camera_intrinsics(model, CAM_NAME2, H, W)
else:
    renderer_opp = None
    renderer_depth_opp = None
    K_opp = None


# cube freejoint qpos address, for the ground-truth error
cube_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
cube_qpos_adr = model.jnt_qposadr[model.body_jntadr[cube_body_id]]

# defining p_hold and R_hold
T_home = fk(q_home)
p_hold = T_home[:3,3]
R_hold = T_home[:3,:3]


# Initializing the loop for the impedance control with nullspace control and passive joint torque compensation
phase = 'pregrasp'
t_phase = 0.0
traj = None
p_end = p_hold.copy()
R_end = R_hold.copy()

def start_move(p_from, R_from, p_target, R_target, T, t_now):
    traj = PoseTrajectory(p_from, R_from, p_target, R_target, T)

    # posture target for the null space: IK solution at the goal pose
    result = dls_ik(CS_CONTROLLER.q_null, p_target, R_target)
    if result['converged']:
        CS_CONTROLLER.q_null = result['q']
    else:
        print(f"Warning: posture IK did not converge (pos err {result['pos_err']*1000:.2f} mm), keeping previous q_null")

    return traj, p_target.copy(), R_target.copy(), t_now

def estimate_yaw(mask):
    contours,_ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    rect = cv2.minAreaRect(max(contours, key=cv2.contourArea))
    angle = rect[2]
    w, h = rect[1]
    if w < h:
        angle += 90
    angle = ((angle+45)%90)-45
    return np.deg2rad(-angle)
 
def make_R_des(yaw):
    c,s= np.cos(yaw), np.sin(yaw)
    R_des = np.array([[c, -s, 0],[s, c, 0],[0, 0, 1]], dtype=np.float64) @ R_des_default
    return R_des


# live camera windows; renderer_rgb is free after the initial detection, so it renders the fixed view
stream_on = ENABLE_FIXED_CAM or ENABLE_WRIST_CAM
stop_event = threading.Event()
if stream_on:
    start_stream(stop_event)

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    t=0.0
    step = 0
    while viewer.is_running():
        q = data.qpos[:N_ARM].copy()
        dq = data.qvel[:N_ARM].copy()
        mujoco.mj_step1(model, data)

        if traj is not None:
            p_ref,R_ref,v_ref = traj.evaluate(t-t_phase)
        else:
            p_ref,R_ref,v_ref = p_end,R_end,np.zeros(6)

        if phase == 'pregrasp' and traj is None:
            p_target = p_cube + np.array([0, 0, PREGRASP_HEIGHT])
            traj,p_end,R_end,t_phase = start_move(p_end,R_end,p_target,R_des_default,3.5,t)
            print(f"[{t:5.2f}s] moving to pre-grasp {np.round(p_target, 3)}")

        elif phase == 'pregrasp' and traj.is_done(t - t_phase):
            traj = None
            phase = 'hold'
            t_phase = t
            print(f"[{t:5.2f}s] pre-grasp reached, holding")

        elif phase == 'hold' and t-t_phase >= HOLD_TIME:
            phase = 'wrist_detect'
            print(f"[{t:5.2f}s] hold done, running wrist detection")

        elif phase == 'wrist_detect':
            renderer_wrist.update_scene(data, camera='wrist_cam')
            wrist_img = renderer_wrist.render()
            renderer_depth_wrist.update_scene(data, camera='wrist_cam')
            depth_wrist = renderer_depth_wrist.render()
            mask_w, centroid_w, detected_w = segment_red(wrist_img)

            if detected_w:
                uw, vw = centroid_w
                p_cube_refined = pixel_to_world(uw, vw, depth_wrist[vw, uw], K_wrist, model, data, 'wrist_cam')
                p_cube_refined[2] -= CUBE_HALF_SIZE
                yaw_est = estimate_yaw(mask_w)
                R_grasp = make_R_des(yaw_est)
                p_grasp = p_cube_refined + np.array([0, 0, FINGER_LENGTH])

                p_cube_true = data.qpos[cube_qpos_adr:cube_qpos_adr+3]
                err_xy = np.linalg.norm(p_cube_refined[:2] - p_cube_true[:2])*1000
                print(f"[{t:5.2f}s] wrist refined cube: {np.round(p_cube_refined, 4)} | true: {np.round(p_cube_true, 4)} | x-y err {err_xy:.2f} mm")
                print(f"[{t:5.2f}s] estimated yaw: {np.degrees(yaw_est):.1f} deg")
            else:
                print(f"[{t:5.2f}s] wrist detection failed, keeping fixed-cam estimate")
                R_grasp = R_des_default.copy()
                p_grasp = p_cube + np.array([0, 0, FINGER_LENGTH])

            # descend -> grasp -> close -> lift
            phase = 'descend'

        elif phase == 'descend' and traj is None:
            # any yaw correction happens during this descent
            traj,p_end,R_end,t_phase = start_move(p_end,R_end,p_grasp,R_grasp,2.0,t)
            print(f"[{t:5.2f}s] descending to grasp {np.round(p_grasp, 3)}")

        elif phase == 'descend' and traj.is_done(t - t_phase):
            traj = None
            phase = 'grasp_settle'
            t_phase = t
            z_actual = fk(q)[2,3]
            print(f"[{t:5.2f}s] grasp pose reached | commanded z {p_grasp[2]:.4f} | actual z {z_actual:.4f} | sag {(p_grasp[2]-z_actual)*1000:.2f} mm")

        elif phase == 'grasp_settle' and t - t_phase >= HOLD_TIME:
            GRIPPER = 0.0
            phase = 'closing'
            t_phase = t
            print(f"[{t:5.2f}s] closing gripper")

        elif phase == 'closing' and t - t_phase >= 1.0:

            p_lift = p_grasp + np.array([0, 0, 0.20])
            traj,p_end,R_end,t_phase = start_move(p_end,R_end,p_lift,R_grasp,2.0,t)
            phase = 'lift'
            print(f"[{t:5.2f}s] lifting to {np.round(p_lift, 3)}")

        elif phase == 'lift' and traj.is_done(t - t_phase):

            traj = None
            cube_z = data.qpos[cube_qpos_adr+2]
            lifted = cube_z - CUBE_REST_Z
            if lifted > 0.10:
                print(f"[{t:5.2f}s] grasp SUCCEEDED: cube z {cube_z:.4f} ({lifted*1000:.1f} mm above rest)")
                phase = 'move_to_place'
            else:
                print(f"[{t:5.2f}s] grasp FAILED: cube z {cube_z:.4f} ({lifted*1000:.1f} mm above rest), aborting place")
                phase = 'failed'   # terminal: holds p_end

        elif phase == 'move_to_place' and traj is None:


            # place-above target: detected patch x-y at PLACE_ABOVE_Z (the retreat reuses it)
            p_place_above = np.array([p_patch[0], p_patch[1], PLACE_ABOVE_Z])


            # long sideways move with the cube, so a bit more time
            traj,p_end,R_end,t_phase = start_move(p_end,R_end,p_place_above,R_grasp,2.5,t)
            print(f"[{t:5.2f}s] moving to place-above {np.round(p_place_above, 3)}")

        elif phase == 'move_to_place' and traj.is_done(t - t_phase):
            traj = None
            phase = 'descend_place'
            print(f"[{t:5.2f}s] above the patch, descending to place")

        elif phase == 'descend_place' and traj is None:

            # place-down target: cube bottom just clears the patch
            p_place_down = np.array([p_patch[0], p_patch[1], PATCH_Z + CUBE_HALF_SIZE + FINGER_LENGTH + PLACE_CLEARANCE])
            traj,p_end,R_end,t_phase = start_move(p_end,R_end,p_place_down,R_grasp,2.0,t)
            print(f"[{t:5.2f}s] descending to place-down {np.round(p_place_down, 4)}")

        elif phase == 'descend_place' and traj.is_done(t - t_phase):
            traj = None
            z_actual = fk(q)[2,3]
            print(f"[{t:5.2f}s] place pose reached | commanded z {p_place_down[2]:.4f} | actual z {z_actual:.4f} | sag {(p_place_down[2]-z_actual)*1000:.2f} mm")
            GRIPPER = 255.0
            phase = 'release'
            t_phase = t
            print(f"[{t:5.2f}s] opening gripper")

        elif phase == 'release' and t - t_phase >= 1.0:
            # fingers open and cube settled: back up to the place-above pose

            traj,p_end,R_end,t_phase = start_move(p_end,R_end,p_place_above,R_grasp,2.0,t)
            phase = 'retreat'
            print(f"[{t:5.2f}s] retreating to {np.round(p_place_above, 3)}")

        elif phase == 'retreat' and traj.is_done(t - t_phase):
            traj = None
            p_cube_final = data.qpos[cube_qpos_adr:cube_qpos_adr+3].copy()
            err_place = np.linalg.norm(p_cube_final[:2] - p_patch_true[:2])*1000
            R_cube = np.zeros(9)
            mujoco.mju_quat2Mat(R_cube, data.qpos[cube_qpos_adr+3:cube_qpos_adr+7])
            tilt = np.degrees(np.arccos(np.clip(R_cube.reshape(3,3)[2,2], -1, 1)))

            print(f"[{t:5.2f}s] cube final position: {np.round(p_cube_final, 4)} | patch true centre: {np.round(p_patch_true, 4)}")
            print(f"[{t:5.2f}s] place x-y error: {err_place:.2f} mm | cube final tilt: {tilt:.3f} deg")
            
            phase = 'done'   # terminal: holds p_end
            t_phase = t
            print(f"[{t:5.2f}s] task complete, holding")

        tau,e_p,e_r = CS_CONTROLLER.compute_Torque(q,dq,p_ref,R_ref,model,data,v_des=v_ref)
        data.qfrc_applied[:N_ARM] = tau
        data.ctrl[7] = GRIPPER
        mujoco.mj_step2(model, data)
        if step % SIM_HZ == 0:
            print(f"t = {t:5.2f} s | EE pos = {np.round(p_ref-e_p, 4)} | ||e_p|| = {np.linalg.norm(e_p)*1000:.3f} mm")
        viewer.sync()

        if stream_on and step % STREAM_EVERY == 0:
            fixed_img, wrist_img, opp_img = None, None, None
            if ENABLE_FIXED_CAM:
                renderer_rgb.update_scene(data, camera=CAM_NAME)
                fixed_img = renderer_rgb.render()
            if ENABLE_WRIST_CAM:
                renderer_wrist.update_scene(data, camera='wrist_cam')
                wrist_img = renderer_wrist.render()
            if ENABLE_OPPOSITE_CAM:
                renderer_opp.update_scene(data, camera=CAM_NAME2)
                opp_img = renderer_opp.render()
            push_frames(fixed_img, wrist_img, opp_img)

        step += 1
        t += SIM_DT

if stream_on:
    stop_stream(stop_event)
