import sys

from sympy import centroid
sys.path.insert(0, '/home/shreehan/roboai_project')

import cv2
import numpy as np
import mujoco
from perception.detector import segment_red

MODEL_PATH = '/home/shreehan/mujoco_menagerie/franka_emika_panda/scene_pick.xml'


model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

renderer = mujoco.Renderer(model)
renderer.update_scene(data,camera = "fixed_cam")
rgb_image = renderer.render()

# Execute the perception module
mask,centroid,detected = segment_red(rgb_image)
print("Mask and Centroid:", centroid, detected)

# Save debug images
debug = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)
if detected:
    cv2.circle(debug, centroid, 10, (0,255,0), -1)
    cv2.putText(debug, f'cube {centroid}', (centroid[0]+10, centroid[1]),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

cv2.imwrite('/home/shreehan/roboai_project/assets/detection_debug.png', debug)
cv2.imwrite('/home/shreehan/roboai_project/assets/mask_debug.png', mask)
print('Debug images saved to assets/')