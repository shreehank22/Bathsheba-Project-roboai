import cv2
import mujoco
import numpy as np


def segment_red(rgb_image):
    rgb_image = np.ascontiguousarray(rgb_image, dtype=np.uint8)
    hsv_image = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2HSV)
    mask1= cv2.inRange(hsv_image, (0, 100, 100), (10, 255, 255))
    mask2 = cv2.inRange(hsv_image, (170, 100, 100), (180, 255, 255))
    mask = cv2.bitwise_or(mask1, mask2)


    # binary mask of the red color
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # find contours 
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask,None,False
    largest = max(contours,key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:  
        return mask,None,False

    # centroid
    M = cv2.moments(largest)
    if M['m00'] == 0:
        return mask,None,False
    u = int(M['m10'] / M['m00'])
    v = int(M['m01'] / M['m00'])

    return mask,(u,v),True

def camera_intrinsics(model, cam_name, height, width):
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    fovy=np.deg2rad(model.cam_fovy[cam_id])
    fy = (height/2)/np.tan(fovy/2)
    fx = fy
    cx = width/2
    cy = height/2
    K = np.array([[fx, 0, cx],
              [ 0,fy, cy],
              [ 0, 0,  1]])
    return K

def pixel_to_world(u,v,depth,K,model,data,cam_name):

    # back-project to camera coordinates
    X_c= (u-K[0,2])*depth/K[0,0]
    Y_c= -(v-K[1,2])*depth/K[1,1]
    Z_c= -depth
    p_cam = np.array([X_c,Y_c,Z_c])

    # get camera pose in world frame
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    R_cam = data.cam_xmat[cam_id].reshape(3,3)
    t_cam = data.cam_xpos[cam_id]
    
    # Transform from camera to world coordinates
    p_world = R_cam @ p_cam + t_cam
    return p_world

def segment_blue(rgb_image):
    rgb_image = np.ascontiguousarray(rgb_image,dtype=np.uint8)
    hsv_image = cv2.cvtColor(rgb_image,cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv_image, (110, 150, 50), (130, 255, 255))
    
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    # find contours 
    contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) == 0:
        return mask,None,False
    largest = max(contours,key=cv2.contourArea)
    if cv2.contourArea(largest) < 50:  
        return mask,None,False

    M = cv2.moments(largest)
    if M['m00'] == 0:
        return mask,None,False
    u = int(M['m10'] / M['m00'])
    v = int(M['m01'] / M['m00'])

    return mask,(u,v),True
    

