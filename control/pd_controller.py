import numpy as np
class PDController:
    def __init__(self,kp : np.ndarray, kd : np.ndarray):
        self.kp = np.asarray(kp)
        self.kd = np.asarray(kd)
    def compute_torque(self,q,qd,q_des,qd_des=None):
        if qd_des is None:
            qd_des = np.zeros_like(qd)
        error = q_des - q
        error_dot = qd_des - qd
        torque = self.kp * error + self.kd * error_dot
        return torque