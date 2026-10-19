import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import mujoco
import mujoco.viewer
from perception.detector import segment_red, segment_green, camera_intrinsics, pixel_to_world, pixel_to_plane
from planning.fk import fk
from envs.pick_scene import load_pick_scene
from control.CS_impedance_controller import CS_Impedance_Controller

CAM_NAME = 'fixed_cam'
H,W = 480, 640
CUBE_HALF_SIZE = 0.05/2
PATCH_Z = 0.4015

q_home = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4], dtype=np.float64)

model = load_pick_scene()
N_ARM = 7
data = mujoco.MjData(model)
data.qpos[:N_ARM] = q_home
mujoco.mj_forward(model, data)
SIM_HZ = int(1/model.opt.timestep)
SIM_DT = model.opt.timestep
GRIPPER = 255.0

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

# defining p_hold and R_hold
T_home = fk(q_home)
p_hold = T_home[:3,3]
R_hold = T_home[:3,:3]


# Initializing the loop for the impedance control with nullspace control and passive joint torque compensation

with mujoco.viewer.launch_passive(model, data) as viewer:
    viewer.cam.distance = 2.0
    t=0.0
    step = 0
    while viewer.is_running():
        q = data.qpos[:N_ARM].copy()
        dq = data.qvel[:N_ARM].copy()
        mujoco.mj_step1(model, data)
        tau,e_p,e_r = CS_CONTROLLER.compute_Torque(q, dq, p_hold, R_hold, model, data)
        data.qfrc_applied[:N_ARM] = tau
        data.ctrl[7] = GRIPPER
        mujoco.mj_step2(model, data)
        if step % SIM_HZ == 0:
            print(f"t = {t:5.2f} s | EE pos = {np.round(p_hold - e_p, 4)} | ||e_p|| = {np.linalg.norm(e_p)*1000:.3f} mm")
        viewer.sync()
        step += 1
        t += SIM_DT
