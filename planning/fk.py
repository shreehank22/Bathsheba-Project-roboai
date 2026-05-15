import numpy as np
import sys, os
sys.path.insert(0, os.path.expanduser('~/roboai_project'))
from utils.transform import dh_transform

DH_PARAMS = [
    [0,0.333,-np.pi/2],
    [0,0,np.pi/2],
    [0.0825,0.316,np.pi/2],
    [-0.0825,0,-np.pi/2],
    [0,0.384,np.pi/2],
    [0.088,0,np.pi/2],
    [0,0.107,0 ],
]

R_EE_OFFSET = np.array([[np.cos(-np.pi/4),-np.sin(-np.pi/4),0,0],[np.sin(-np.pi/4),np.cos(-np.pi/4),0,0],[0,0,1,0],[0,0,0,1]])

def fk(q:np.ndarray):
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = DH_PARAMS[i]
        theta = q[i]
        T_i = dh_transform(a, d, alpha, theta)
        T = T @ T_i
    T = T @ R_EE_OFFSET
    return T
def fk_all_frames(q):
    frames = []
    T = np.eye(4)
    for i in range(7):
        a, d, alpha = DH_PARAMS[i]
        T = T @ dh_transform(a, d, alpha, q[i])
        frames.append(T.copy())
    return frames


