import mujoco
import numpy as np

def check_grasp(model, data, gripper_site_name, object_geom_name, threshold=0.01):
    gripper_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, gripper_site_name)
    object_geom_id = mujoco.m