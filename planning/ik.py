import numpy as np
from planning.fk import fk, fk_all_frames

def log_SO3(R):
    trace_val = (np.trace(R) - 1) / 2
    theta = np.arccos(np.clip(trace_val, -1, 1))
    if abs(theta) < 1e-7:
        skew = 0.5 * (R - R.T)
    elif abs(theta - np.pi) < 1e-4:
        A = (R + np.eye(3)) / 2
        n = np.sqrt(np.maximum(0, np.diag(A)))
        if A[0,1] < 0: n[1] = -n[1]
        if A[0,2] < 0: n[2] = -n[2]
        return np.pi * n
    else:
        skew = (theta / (2*np.sin(theta))) * (R - R.T)
    return np.array([skew[2,1], skew[0,2], skew[1,0]])

def geometric_jacobian(q):
    frames = fk_all_frames(q)
    p_ee = frames[-1][:3,3]
    J = np.zeros((6,7))
    z0 = np.array([0,0,1], dtype=np.float64)
    p0 = np.zeros(3)
    for i in range(7):
        z_prev = z0 if i == 0 else frames[i-1][:3,2]
        p_prev = p0 if i == 0 else frames[i-1][:3,3]
        r = p_ee - p_prev
        J[:3,i] = np.cross(z_prev, r)
        J[3:6,i] = z_prev
    return J

def dls_ik(q0, p_des, R_des, k=1.0, lambda_max=0.01,
           epsilon=0.05, max_iter=400, tol=1e-6):
    q = q0.copy()
    for i in range(max_iter):
        T_curr = fk(q)
        p_curr = T_curr[:3,3]
        R_curr = T_curr[:3,:3]
        e_p = p_des - p_curr
        e_r = log_SO3(R_des @ R_curr.T)
        e = np.concatenate([e_p, e_r])
        pos_err = np.linalg.norm(e_p)
        rot_err = np.linalg.norm(e_r)
        if np.linalg.norm(e) < tol:
            return dict(q=q, converged=True, n_iter=i+1,
                        pos_err=pos_err, rot_err=rot_err)
        J = geometric_jacobian(q)
        U, S, Vt = np.linalg.svd(J, full_matrices=False)
        V = Vt.T
        lam = lambda_max * np.exp(-(S/epsilon)**2)
        S_inv = np.zeros((6,6))
        for j in range(6):
            S_inv[j,j] = S[j] / (S[j]**2 + lam[j]**2)
        dq = V @ S_inv @ U.T @ e
        scaled = k * dq
        step_limit = 0.5
        if np.linalg.norm(scaled) > step_limit:
            scaled = scaled * step_limit / np.linalg.norm(scaled)
        q = q + scaled
    T_curr = fk(q)
    pos_err = np.linalg.norm(p_des - T_curr[:3,3])
    rot_err = np.linalg.norm(log_SO3(R_des @ T_curr[:3,:3].T))
    return dict(q=q, converged=False, n_iter=max_iter,
                pos_err=pos_err, rot_err=rot_err)