import numpy as np
import cv2 
import mujoco


def cube_keypoints(half_size):
    return np.array([
        [-half_size, -half_size, -half_size],
        [half_size, -half_size, -half_size],
        [half_size, half_size, -half_size],
        [-half_size, half_size, -half_size],
        [-half_size, -half_size, half_size],
        [half_size, -half_size, half_size],
        [half_size, half_size, half_size],
        [-half_size, half_size, half_size],
    ], dtype=np.float32)

def detect_corners(mask,rgb):
    mask_u8 = mask.astype(np.uint8)
    corners = cv2.goodFeaturesToTrack(mask_u8,maxCorners=8,qualityLevel=0.01,minDistance=10)
    if corners is None or len(corners) < 4:
        return None,False
    corners = corners.reshape(-1, 2).astype(np.float32)
    return corners,True

def project_keypoints(corners_3d,K,R_init,T_init):
    rvec,_=cv2.Rodrigues(R_init)
    projected,_=cv2.projectPoints(corners_3d,rvec,T_init,K,None)
    return projected.reshape(-1,2).astype(np.float32)

def match_corners(detected_2d,projected_2d,corners_3d,threshold=50.0):
    image_points = []
    object_points = []
    for i,proj in enumerate(projected_2d):
        dists = np.linalg.norm(detected_2d - proj, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < threshold:
            image_points.append(detected_2d[min_idx])
            object_points.append(corners_3d[i])
    if len(image_points) < 4:
        return None,None,False
    return np.array(image_points, dtype=np.float32), np.array(object_points, dtype=np.float32), True

def estimate_pose(image_points,object_points,K):
    success,rvec,tvec,inliers = cv2.solvePnPRansac(object_points,image_points,K,None,reprojectionError=8.0,confidence=0.99,flags=cv2.SOLVEPNP_ITERATIVE)
    if not success or inliers is None or len(inliers) < 4:
        return None,None,None,False
    _,rvec,tvec = cv2.solvePnP(object_points,image_points,K,None,rvec=rvec,tvec=tvec,useExtrinsicGuess=True,flags=cv2.SOLVEPNP_ITERATIVE)
    R_est,_=cv2.Rodrigues(rvec)
    return rvec,tvec,R_est.astype(np.float64),True

def pose_estimation(R_cam_obj,t_cam_obj,model,data,cam_name):
    R_conv = np.array([
    [1,  0,  0],
    [0, -1,  0],
    [0,  0, -1]], dtype=np.float64)
    
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    R_wc = data.cam_xmat[cam_id].reshape(3,3)
    t_wc = data.cam_xpos[cam_id]
    R_muj = R_conv @ R_cam_obj
    t_muj = R_conv @ t_cam_obj.flatten()
    
    R_wo = R_wc @ R_muj
    t_wo = R_wc @ t_muj + t_wc
    return R_wo, t_wo

def estimate_rotation_pca(mask, depth, K, model, data, cam_name):
    from perception.detector import pixel_to_world
    ys, xs = np.where(mask > 0)
    if len(xs) < 10:
        return np.eye(3)

    pts = []
    for u, v in zip(xs[::3], ys[::3]):  # subsample every 3rd pixel
        d = depth[v, u]
        if d <= 0:
            continue
        p = pixel_to_world(u, v, d, K, model, data, cam_name)
        pts.append(p)
    if len(pts) < 6:
        return np.eye(3)
    pts = np.array(pts)
    pts -= pts.mean(axis=0)
    _, _, Vt = np.linalg.svd(pts, full_matrices=False)
    R_pca = Vt.T  
    if np.linalg.det(R_pca) < 0:
        R_pca[:, 2] *= -1
    return R_pca.astype(np.float64)

def estimate_cube_pose(rgb,depth,mask,model,data,cam_name,half_size,p_cube_init,K):
    
    corners_2d,found = detect_corners(mask,rgb)
    if not found:
        return {"success": False,'reason': 'No corners detected'}
    
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    R_wc = data.cam_xmat[cam_id].reshape(3,3)
    t_wc = data.cam_xpos[cam_id]
    R_conv = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
    
    
    # Obtain 3d points
    corners_3d = cube_keypoints(half_size)
    t_init = R_conv@(R_wc.T @ (p_cube_init - t_wc))
    R_init = R_conv @ R_wc.T
    projected = project_keypoints(corners_3d, K, R_init, t_init)
    
    h,w = rgb.shape[:2]
    visible_mask=((projected[:, 0] >= 0) & (projected[:, 0] < w) & (projected[:, 1] >= 0) & (projected[:, 1] < h))
    proj_visible = projected[visible_mask]
    obj_visible = corners_3d[visible_mask]
  
    if len(proj_visible) < 4:
        return {'success': False, 'reason': 'insufficient projected corners in view'}
    img_pts = []
    obj_pts = []
    used=set()
    
    for i,proj in enumerate(proj_visible):
        dists = np.linalg.norm(corners_2d - proj, axis=1)
        min_idx = np.argmin(dists)
        if dists[min_idx] < 25.0 and min_idx not in used:
            img_pts.append(corners_2d[min_idx])
            obj_pts.append(obj_visible[i])
            used.add(min_idx)
    
    if len(img_pts) < 4:
        return {'success': False, 'reason': 'insufficient corner matches'}
    
    img_pts = np.array(img_pts, dtype=np.float32)
    obj_pts = np.array(obj_pts, dtype=np.float32)
    
    # Solving PnP for pose estimation
    rvec,tvec,R_est,pose_found = estimate_pose(img_pts,obj_pts,K)
    if not pose_found:
        return {'success': False, 'reason': 'pose estimation failed'}
    
    R_wo,t_wo = pose_estimation(R_est,tvec,model,data,cam_name)
    
    proj_check,_=cv2.projectPoints(obj_pts,rvec,tvec,K,None)
    err = np.mean(np.linalg.norm(proj_check.reshape(-1,2) - img_pts, axis=1))
    T=np.eye(4)
    T[:3,:3]=R_wo
    T[:3,3]=t_wo

    return{'success': True, 'R': R_wo, 't': t_wo, 'T': T, 'Reprojection_Error': err,'n_correspondences':len(img_pts),}

