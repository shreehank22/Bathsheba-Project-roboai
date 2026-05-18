import sys
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
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

MODEL_PATH = '/home/shreehan/mujoco_menagerie/franka_emika_panda/scene_pick.xml'
CAM_NAME = 'fixed_cam'
H, W = 480, 640
CUBE_HALF_SIZE = 0.05 / 2
PREGRASP_OFFSET = 0.20
FINGER_LENGTH = 0.11
OBSERVE_HEIGHT = 0.25
HOLD_TIME = 0.5

model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)
N_ARM = 7
SIM_HZ = int(1.0 / model.opt.timestep)
GRIPPER = 255.0

# --- Fixed cam perception ---
renderer_rgb = mujoco.Renderer(model, height=H, width=W)
renderer_rgb.update_scene(data, camera=CAM_NAME)
rgb_image = renderer_rgb.render()

renderer_depth = mujoco.Renderer(model, height=H, width=W)
renderer_depth.enable_depth_rendering()
renderer_depth.update_scene(data, camera=CAM_NAME)
depth_image = renderer_depth.render()

mask, centroid, detected = segment_red(rgb_image)
print("Mask and Centroid:", centroid, detected)
if not detected:
    print("Cube not detected — exiting"); exit()

u, v = centroid
depth = depth_image[v, u]
K = camera_intrinsics(model, CAM_NAME, H, W)
p_cube = pixel_to_world(u, v, depth, K, model, data, CAM_NAME)
p_cube[2] -= CUBE_HALF_SIZE

cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
p_cube = data.xpos[cube_id].copy()
print(f"Cube 3D position: {np.round(p_cube, 4)}")

# --- IK ---
R_des = np.array([[1, 0, 0], [0, -1, 0], [0, 0, -1]], dtype=np.float64)
q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])

pregrasp_pos = p_cube.copy()
pregrasp_pos[2] += PREGRASP_OFFSET
result = dls_ik(q_home, pregrasp_pos, R_des)
if not result['converged']:
    print("Pregrasp IK did not converge — exiting"); exit()
q_pregrasp = result['q']
print(f"Pre-grasp position: {np.round(pregrasp_pos, 4)}")

p_grasp = p_cube.copy()
p_grasp[2] += FINGER_LENGTH
result_grasp = dls_ik(q_pregrasp, p_grasp, R_des)
if not result_grasp['converged']:
    print("Grasp IK did not converge — exiting"); exit()
q_grasp = result_grasp['q']
print(f"Grasp IK error: {result_grasp['pos_err']*1000:.2f} mm")

p_observe = p_cube.copy()
p_observe[2] += OBSERVE_HEIGHT
result_obs = dls_ik(q_pregrasp, p_observe, R_des)
if not result_obs['converged']:
    print("Observe IK did not converge — exiting"); exit()
q_observe = result_obs['q']

q_start = data.qpos[:N_ARM].copy()
traj = JointTrajectory(q_start, q_pregrasp, T=3.0)

# --- Renderers ---
renderer_fixed = mujoco.Renderer(model, height=H, width=W)
renderer_wrist = mujoco.Renderer(model, height=H, width=W)
renderer_depth_wrist = mujoco.Renderer(model, height=H, width=W)
renderer_depth_wrist.enable_depth_rendering()
stop_event = threading.Event()
start_stream(stop_event)

# --- Phase state ---
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
        t_local = t_global - t_start

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
                cv2.imwrite('/tmp/wrist_detect_live.png', cv2.cvtColor(wrist_img, cv2.COLOR_RGB2BGR))
                mask, centroid, detected = segment_red(wrist_img)
            if detected:
                wrist_detected = True
                u_w, v_w = centroid
                renderer_depth_wrist.update_scene(data, camera='wrist_cam')
                depth_wrist = renderer_depth_wrist.render()
                d = depth_wrist[v_w, u_w]
                K_wrist = camera_intrinsics(model, 'wrist_cam', H, W)
                p_cube_refined = pixel_to_world(u_w, v_w, d, K_wrist, model, data, 'wrist_cam')
                p_cube_refined[2] -= CUBE_HALF_SIZE
                print(f"Wrist cam refined cube: {np.round(p_cube_refined, 4)}")
                p_grasp = p_cube_refined.copy()
                p_grasp[2] += FINGER_LENGTH
                result_grasp = dls_ik(q_observe, p_grasp, R_des)
                q_grasp = result_grasp['q']
                traj_descend = JointTrajectory(q_observe, q_grasp, T=2.0)
                phase = 'descend'
                t_start = t_global
                print("Wrist detection done — descending to refined grasp...")
            else:
                if t_global - t_hold_start >= 2.0:
                    print("Wrist detection failed — using fixed cam estimate")
                    traj_descend = JointTrajectory(q_observe, q_grasp, T=2.0)
                    phase = 'descend'
                    t_start = t_global

        elif phase == 'descend':
            q_des, _, _ = traj_descend.evaluate(t_global - t_start)
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
                result_place = dls_ik(q_lift, p_place_above, R_des)
                q_place = result_place['q']
                traj_place = JointTrajectory(q_lift, q_place, T=2.0)
                phase = 'move_to_place'
                t_start = t_global
                print("Moving to place position...")

        elif phase == 'move_to_place':
            q_des, _, _ = traj_place.evaluate(t_global - t_start)
            GRIPPER = 0.0
            if traj_place.is_done(t_global - t_start):
                phase = 'place'
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
            print(f"t={t_global:.1f}s | phase={phase} | EE={np.round(T_curr[:3,3],3)}")

        step += 1

stop_stream(stop_event)