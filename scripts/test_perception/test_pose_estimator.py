import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

import numpy as np
import mujoco
import cv2
from scipy.linalg import logm

from perception.detector import segment_red, camera_intrinsics, pixel_to_world
from perception.pose_estimator import estimate_cube_pose, cube_keypoints, detect_corners, project_keypoints

# ── Config ───────────────────────────────────────────────────────────────────
MENAGERIE_PATH = os.environ.get('MUJOCO_MENAGERIE', os.path.expanduser('~/mujoco_menagerie'))
MODEL_PATH = os.path.join(MENAGERIE_PATH, 'franka_emika_panda', 'scene_pick.xml')
CAM_NAME   = 'fixed_cam'
H, W       = 480, 640
HALF_SIZE  = 0.025

# ── Setup ────────────────────────────────────────────────────────────────────
model = mujoco.MjModel.from_xml_path(MODEL_PATH)
data  = mujoco.MjData(model)
mujoco.mj_forward(model, data)

# ── Render ───────────────────────────────────────────────────────────────────
renderer = mujoco.Renderer(model, height=H, width=W)
renderer.update_scene(data, camera=CAM_NAME)
rgb = renderer.render()

renderer.enable_depth_rendering()
renderer.update_scene(data, camera=CAM_NAME)
depth = renderer.render()
renderer.disable_depth_rendering()

# ── Coarse position ───────────────────────────────────────────────────────────
mask, centroid, detected = segment_red(rgb)
if not detected:
    print("Cube not detected — exiting"); exit()

u, v     = centroid
d        = depth[v, u]
K        = camera_intrinsics(model, CAM_NAME, H, W)
p_coarse = pixel_to_world(u, v, d, K, model, data, CAM_NAME)
p_coarse[2] -= HALF_SIZE

# ── Camera info ───────────────────────────────────────────────────────────────
cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
R_wc   = data.cam_xmat[cam_id].reshape(3, 3)
t_wc   = data.cam_xpos[cam_id]
print(f"R_wc     :\n{np.round(R_wc, 4)}")
print(f"t_wc     : {np.round(t_wc, 4)}")
print(f"p_coarse : {np.round(p_coarse, 4)}")

# ── Convention debug ──────────────────────────────────────────────────────────
R_conv = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)
t_raw  = R_wc.T @ (p_coarse - t_wc)
t_conv = R_conv @ t_raw
print(f"t_init_cam (raw, MuJoCo) : {np.round(t_raw, 4)}")
print(f"t_init_cam (after R_conv): {np.round(t_conv, 4)}")

# ── Corner debug ──────────────────────────────────────────────────────────────
corners_3d        = cube_keypoints(HALF_SIZE)
corners_2d, found = detect_corners(mask, rgb)
print(f"Corners detected  : {found}")
if found:
    print(f"Detected corners  :\n{corners_2d}")

# Project with R_conv applied (what estimate_cube_pose uses)
R_init    = R_conv @ R_wc.T
t_init    = t_conv
projected = project_keypoints(corners_3d, K, R_init, t_init)
print(f"Projected corners :\n{projected}")

debug = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
if found:
    for pt in corners_2d:
        cv2.circle(debug, tuple(pt.astype(int)), 5, (0, 255, 0), -1)  # green = detected
for pt in projected:
    cv2.circle(debug, tuple(pt.astype(int)), 5, (0, 0, 255), -1)      # red = projected
cv2.imwrite(os.path.join(PROJECT_ROOT, 'assets', 'corner_debug.png'), debug)
print("Corner debug image saved.")

# ── 6-DoF pose estimation ────────────────────────────────────────────────────
result = estimate_cube_pose(rgb, depth, mask, model, data, CAM_NAME, HALF_SIZE, p_coarse, K)
if not result['success']:
    print(f"Pose estimation failed: {result['reason']}"); exit()

# ── Ground truth ──────────────────────────────────────────────────────────────
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
t_gt    = data.xpos[cube_id]
R_gt    = data.xmat[cube_id].reshape(3, 3)

# ── Errors ───────────────────────────────────────────────────────────────────
t_err     = np.linalg.norm(result['t'] - t_gt) * 1000
R_rel     = result['R'].T @ R_gt
R_err_deg = np.degrees(np.linalg.norm(logm(R_rel), 'fro') / np.sqrt(2))

# ── Report ───────────────────────────────────────────────────────────────────
print("=" * 45)
print(f"  Correspondences    : {result['n_correspondences']}")
print(f"  Reprojection error : {result['Reprojection_Error']:.2f} px")
print(f"  Translation error  : {t_err:.2f} mm")
print(f"  t estimated        : {np.round(result['t'], 4)}")
print(f"  t ground truth     : {np.round(t_gt, 4)}")
print(f"  R estimated        :\n{np.round(result['R'], 3)}")
print(f"  R ground truth     :\n{np.round(R_gt, 3)}")
print(f"  Rotation error     : {R_err_deg:.2f} deg")
print("=" * 45)

# ── Verify camera-frame pose directly ────────────────────────────────────────
from perception.pose_estimator import estimate_pose, pose_estimation
import cv2

# Recompute manually to inspect intermediate values
cam_id  = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, CAM_NAME)
R_wc    = data.cam_xmat[cam_id].reshape(3, 3)
t_wc    = data.cam_xpos[cam_id]
R_conv  = np.array([[1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float64)

# What solvePnP returns in camera frame
rvec_raw = cv2.Rodrigues(result['R'])[0]   # just for display
print(f"\nR_wo from result :\n{np.round(result['R'], 4)}")
print(f"t_wo from result : {np.round(result['t'], 4)}")

# Manually invert: t_in_cam = R_wc.T @ (t_wo - t_wc)
t_back_cam = R_wc.T @ (result['t'] - t_wc)
print(f"t back-projected to MuJoCo cam frame : {np.round(t_back_cam, 4)}")
t_back_ocv = R_conv @ t_back_cam
print(f"t back-projected to OpenCV cam frame : {np.round(t_back_ocv, 4)}")

# Ground truth in camera frame
cube_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'cube')
t_gt_cam = R_wc.T @ (data.xpos[cube_id] - t_wc)
t_gt_ocv = R_conv @ t_gt_cam
print(f"t_gt in MuJoCo cam frame : {np.round(t_gt_cam, 4)}")
print(f"t_gt in OpenCV cam frame : {np.round(t_gt_ocv, 4)}")
debug = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
cv2.imwrite(os.path.join(PROJECT_ROOT, 'assets', 'pose_debug.png'), debug)
print("Pose debug image saved.")
# ── Display images ────────────────────────────────────────────────────────────
cv2.imshow('RGB', cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
cv2.imshow('Corner Debug', cv2.imread(os.path.join(PROJECT_ROOT, 'assets', 'corner_debug.png')))
cv2.imshow('Pose Debug', cv2.imread(os.path.join(PROJECT_ROOT, 'assets', 'pose_debug.png')))
cv2.waitKey(0)
cv2.destroyAllWindows()