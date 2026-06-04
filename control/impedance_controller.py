import mujoco
import numpy as np
from planning.ik import geometric_jacobian, log_SO3
from planning.fk import fk


class Impedance_Controller:
    def __init__(self,Kp,Kr,Dp,Dr):
        self.Kp = Kp
        self.Kr = Kr
        self.Dp = Dp
        self.Dr = Dr
    def compute_Torque(self,q,dq,p_des,R_des,model,data,v_des=None):
        if v_des is None:
            v_des = np.zeros(6)
        nv = model.nv
        M_full = np.zeros((nv, nv))
        mujoco.mj_fullM(model, M_full, data.qM)
        n=len(q)
        M = M_full[:n, :n]
        T = fk(q)
        p = T[:3,3]
        R = T[:3,:3]
        J = geometric_jacobian(q)
        M_inv = np.linalg.inv(M)
        alpha = 1e-4
        lambda_m = np.linalg.inv(J @ M_inv @ J.T + alpha * np.eye(6))
        v = J@dq
        e_p = p_des - p
        e_r = log_SO3(R_des@R.T)
        e_v = v_des - v
        F = lambda_m @ np.concatenate([self.Kp@e_p + self.Dp@e_v[:3],self.Kr@e_r + self.Dr@e_v[3:]])
        tau = J.T@F + data.qfrc_bias[:n]
        return tau,e_p,e_r