import mujoco
import numpy as np

class JS_Impedance_Controller:
    def __init__(self,Kp,Dp,q_rest):
        self.Kp = Kp
        self.Dp = Dp
        self.q_rest = q_rest
    def compute_Torque(self,q,dq,model,data):
        n = len(q)
        g = data.qfrc_bias[:n]
        e_q = self.q_rest - q
        tau = self.Kp@e_q-self.Dp@dq + g
        return tau,e_q