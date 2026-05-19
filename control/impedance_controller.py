import numpy as np
from planning.ik import geometric_jacobian, log_SO3
from planning.fk import fk

class Impedance_Controller:
    def __init__(self,Kp,Kr,Dp,Dr):
        self.Kp = Kp
        self.Kr = Kr
        self.Dp = Dp
        self.Dr = Dr
    def compute_Torque(self,q,dq,p_des,R_des,v_des=None):
        if v_des is None:
            v_des = np.zeros(6)
        T = fk(q)
        p = T[:3,3]
        R = T[:3,:3]
        J = geometric_jacobian(q)
        v = J@dq
        e_p = p_des - p
        e_r = log_SO3(R_des@R.T)
        e_v = v_des - v
        F = np.concatenate([self.Kp@e_p+self.Dp@e_v[:3],self.Kr@e_r+self.Dr@e_v[3:]])
        tau = J.T@F
        return tau,e_p,e_r