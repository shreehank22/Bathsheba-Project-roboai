import sys
sys.path.insert(0, '/home/shreehan/roboai_project')

import cv2
import numpy as np
import mujoco
from perception.detector import segment_red, camera_intrinsics, pixel_to_world

MODEL_PATH = '/home/shreehan/mujoco_menagerie/franka_emika_panda/scene_pick.xml'
CAM_NAME = 'fixed_cam'
H,W=480,640
CUBE_HALF_SIZE = 0.05/2


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

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

# If detected
if detected:
    u,v=centroid
    depth = depth_image[v,u]
    print("Depth at centroid:", depth)
    K = camera_intrinsics(model, CAM_NAME, H, W)
    p_world = pixel_to_world(u,v,depth,K,model,data,CAM_NAME)
    p_world[2] -= CUBE_HALF_SIZE
    print("3D World Coordinates:", p_world)

    # Ground Truth
    cube_id = mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'cube')
    p_true = data.xpos[cube_id]
    print(f"Ground Truth World Coordinates: {np.round(p_true,4)}")
    print(f'Error (mm): {np.linalg.norm(p_world - p_true)*1000:.2f}')

    # Save results
    debug = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
    cv2.circle(debug, centroid, 10, (0,255,0), -1)
    cv2.putText(debug, f'cube {centroid}', (u+10, v),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    cv2.imwrite('/home/shreehan/roboai_project/assets/detection_debug.png', debug)
    print('Debug image saved')
