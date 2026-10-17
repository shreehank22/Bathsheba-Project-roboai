import mujoco
import numpy as np
from planning.ik import geometric_jacobian, log_SO3
from planning.fk import fk


class CS_Impedance_Controller:
    def __init__(self,Kp,Kr,Dp,Dr,Kn=None,Dn=None,q_null=None):
        self.Kp = Kp
        self.Kr = Kr
        self.Dp = Dp
        self.Dr = Dr
        self.Kn = Kn
        self.Dn = Dn

        if q_null is not None:
            if Kn is None or Dn is None:
                raise ValueError("If q_null is provided, Kn and Dn must also be provided.")
            q_null = np.asarray(q_null,dtype=float)
            if q_null.shape != (7,):
                raise ValueError("q_null must be a 7-dimensional vector.")
        self.q_null = q_null

    def compute_Torque(self,q,dq,p_des,R_des,model,data,v_des=None):
        if v_des is None:
            v_des = np.zeros(6)
        nv = model.nv
        M_full = np.zeros((nv, nv))
        mujoco.mj_fullM(model, data, M_full)
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

    
        # if a nullspace target configuration is provided, compute the nullspace torque
        if self.q_null is not None:

            # compute the nullspace projection matrix
            J_bar = M_inv @ J.T @ lambda_m
            N = np.eye(n) - J_bar @ J
            e_null = self.q_null - q

            # PD control in the nullspace
            tau_null = self.Kn @ e_null - self.Dn @ dq  

            # project the nullspace torque into the nullspace of the Jacobian
            tau += N.T @ tau_null  


        return tau,e_p,e_r