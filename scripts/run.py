import os 
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import threading
import cv2
import numpy as np
import mujoco
import mujoco.viewer
from perception.detector import segment_red, camera_intrinsics, pixel_to_world
from perception.camera_stream import push_frames, start_stream, stop_stream
from planning.trajectory import JointTrajectory
from planning.ik import dls_ik
from planning.fk import fk

MENAGERIE_PATH = os.environ.get('MUJOCO_MENAGERIE', os.path.expanduser('~/mujoco_menagerie'))
MODEL_PATH = os.path.join(MENAGERIE_PATH, 'franka_emika_panda', 'scene_pick.xml')
CAM_NAME = 'fixed_cam'
H,W = 480, 640
CUBE_HALF_SIZE = 0.05/2
PREGRASP_OFFSET = 0.2
FINGER_LENGTH = 0.11
OBSERVE_HEIGHT = 0.25
HOLD_TIME = 0.5

R_des_default = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4], dtype=np.float64)

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

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)  
mujoco.mj_forward(model, data)
N_ARM = 7
SIM_HZ = int(1/model.opt.timestep)
GRIPPER = 255.0

# Fixed camera perception setup
renderer_rgb = mujoco.Renderer(model, height=H, width=W)
renderer_rgb.update_scene(data, camera=CAM_NAME)
rgb_image = renderer_rgb.render()
 
renderer_depth = mujoco.Renderer(model, height=H, width=W)
renderer_depth.enable_depth_rendering()
renderer_depth.update_scene(data, camera=CAM_NAME)
depth_image = renderer_depth.render()

mask,centroid,detected = segment_red(rgb_image)
print(f"Initial detection: Detected={detected}, Centroid={centroid}")
if not detected:
    print("No object detected in the initial frame. Exiting.");exit(1)
u,v = centroid
depth = depth_image[v,u]

K = camera_intrinsics(model,CAM_NAME,height=H,width=W)
p_cube = pixel_to_world(u,v,depth,K,model,data,CAM_NAME)
p_cube[2] -= CUBE_HALF_SIZE
print(f"Initial cube position: {p_cube}")


# Inverse Kinematics planning 
pregrasp_pos = p_cube.copy()
pregrasp_pos[2] += PREGRASP_OFFSET
result = dls_ik(q_home, pregrasp_pos, R_des_default)
if not result['converged']:
    print("IK failed to find a solution. Exiting.");exit()
q_pregrasp = result['q']
print(f"Pre-grasp position: {np.round(pregrasp_pos,3)}, Joint angles: {np.round(q_pregrasp,3)}")

p_grasp = p_cube.copy()
p_grasp[2] += FINGER_LENGTH
result_grasp = dls_ik(q_pregrasp, p_grasp, R_des_default)

if not result_grasp['converged']:
    print("IK failed to find a grasp solution. Exiting.");exit()
q_grasp = result_grasp['q']
R_des = R_des_default.copy()
print(f"Grasp IK error: {result_grasp['pos_err']*1000:.2f} mm")

p_observe = p_cube.copy()
p_observe[2] += OBSERVE_HEIGHT
result_observe = dls_ik(q_pregrasp, p_observe, R_des_default)
if not result_observe['converged']:
    print("IK failed to find an observe solution. Exiting.");exit()
q_observe = result_observe['q']

q_start = data.qpos[:N_ARM].copy()
traj = JointTrajectory(q_start, q_pregrasp, 3.0)

renderer_fixed = mujoco.Renderer(model, height=H, width=W)
renderer_wrist = mujoco.Renderer(model, height=H, width=W)
renderer_depth_wrist = mujoco.Renderer(model, height=H, width=W)
renderer_depth_wrist.enable_depth_rendering()
 
stop_event = threading.Event()
start_stream(stop_event)

traj_descend = None
traj_observe = None
traj_lift = None
traj_place = None
q_lift = None
q_place = None
t_hold_start = None
t_start = 0.0
phase = 'pregrasp'
wrist_detected = False

print("\nLaunching viewer...")
with mujoco.viewer.launch_passive(model, data) as viewer:
    step = 0
    while viewer.is_running():
        t_global = step * model.opt.timestep
        t_local  = t_global - t_start
 
        if phase == 'pregrasp':
            q_des, _, _ = traj.evaluate(t_local)
            if traj.is_done(t_local):
                phase = 'hold'
                t_hold_start = t_global
                print("Pre-grasp reached — holding...")
 
        elif phase == 'hold':
            q_des = q_pregrasp
            if t_global - t_hold_start >= HOLD_TIME:
                phase = 'observe'
                t_start = t_global
                traj_observe = JointTrajectory(q_pregrasp, q_observe, T=2.0)
                print("Moving to observation pose...")
 
        elif phase == 'observe':
            q_des, _, _ = traj_observe.evaluate(t_global - t_start)
            if traj_observe.is_done(t_global - t_start):
                phase = 'wrist_detect'
                t_hold_start = t_global
                print("Observation pose reached — running wrist detection...")
 
        elif phase == 'wrist_detect':
            q_des = q_observe
            if not wrist_detected:
                renderer_wrist.update_scene(data, camera='wrist_cam')
                wrist_img = renderer_wrist.render()
                cv2.imwrite('/tmp/wrist_detect_live.png',
                            cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
                renderer_depth_wrist.update_scene(data, camera='wrist_cam')
                depth_wrist = renderer_depth_wrist.render()
                mask_w, centroid_w, detected_w = segment_red(wrist_img)
 
                if detected_w:
                    wrist_detected = True
                    K_wrist = camera_intrinsics(model, 'wrist_cam', H, W)
                    uw, vw = centroid_w
                    p_cube_refined = pixel_to_world(uw, vw, depth_wrist[vw, uw],
                                                       K_wrist, model, data, 'wrist_cam')
                    p_cube_refined[2] -= CUBE_HALF_SIZE
                    yaw_est = estimate_yaw(mask_w)
                    R_des  = make_R_des(yaw_est)
 
                    print(f"Wrist cam refined pos : {np.round(p_cube_refined, 4)}")
                    print(f"Estimated yaw: {np.degrees(yaw_est):.1f} deg")
                    print(f"R_des_grasp:\n{np.round(R_des, 3)}")
 
                    p_grasp = p_cube_refined.copy()
                    p_grasp[2] += FINGER_LENGTH
                    result_grasp = dls_ik(q_observe, p_grasp, R_des)
                    q_grasp = result_grasp['q']
                    traj_descend = JointTrajectory(q_observe, q_grasp, T=2.0)
                    phase = 'descend'
                    t_start = t_global
                    print("Wrist detection done — descending to refined grasp...")
 
            if not wrist_detected:
                if t_global - t_hold_start >= 2.0:
                    print("Wrist detection failed — using fixed cam estimate")
                    traj_descend = JointTrajectory(q_observe, q_grasp, T=2.0)
                    phase = 'descend'
                    t_start = t_global
 
        elif phase == 'descend':
            q_des,_,_ = traj_descend.evaluate(t_global - t_start)
            if traj_descend.is_done(t_global - t_start):
                phase = 'grasp'
                t_hold_start = t_global
                print("Grasp position reached!")
 
        elif phase == 'grasp':
            q_des = q_grasp
            if t_global - t_hold_start >= HOLD_TIME:
                phase = 'close_gripper'
                t_start = t_global
                print("Closing gripper...")
 
        elif phase == 'close_gripper':
            q_des = q_grasp
            GRIPPER = 0.0
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
            q_des, _, _ = traj_lift.evaluate(t_global - t_start)
            GRIPPER = 0.0
            if traj_lift.is_done(t_global - t_start):
                phase = 'lifted'
                print("Lift complete!")
 
        elif phase == 'lifted':
            q_des = q_lift
            GRIPPER = 0.0
            if traj_place is None:
                p_place_above = np.array([0.3, 0.2, 0.625])
                result_place = dls_ik(q_lift, p_place_above, R_des_default)
                q_place = result_place['q']
                traj_place = JointTrajectory(q_lift, q_place, T=2.0)
                phase = 'move_to_place'
                t_start = t_global
                print("Moving to place position...")
 
        elif phase == 'move_to_place':
            q_des, _, _ = traj_place.evaluate(t_global - t_start)
            GRIPPER = 0.0
            if traj_place.is_done(t_global - t_start):
                phase  = 'place'
                t_hold_start = t_global
                print("Place position reached!")
 
        elif phase == 'place':
            q_des = q_place
            GRIPPER = 255.0
            if t_global - t_hold_start >= 1.0:
                print("Cube released!")
                phase = 'done'
 
        elif phase == 'done':
            q_des = q_place
            print("Task completed — exiting")
            break
 
        data.ctrl[:N_ARM] = q_des
        data.ctrl[7] = GRIPPER
        mujoco.mj_step(model, data)
        viewer.sync()
 
        if step % 10 == 0:
            renderer_fixed.update_scene(data, camera=CAM_NAME)
            fixed_img = renderer_fixed.render()
            renderer_wrist.update_scene(data, camera='wrist_cam')
            wrist_img = renderer_wrist.render()
            push_frames(fixed_img, wrist_img)
 
        if step % SIM_HZ == 0:
            q_curr = data.qpos[:N_ARM]
            T_curr = fk(q_curr)
            print(f"t={t_global:.1f}s | phase={phase} | EE={np.round(T_curr[:3,3], 3)}")
 
        step += 1
 
stop_stream(stop_event)