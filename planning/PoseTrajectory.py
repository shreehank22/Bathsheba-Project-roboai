import numpy as np

from planning.ik import log_SO3, exponential_map_SO3

class PoseTrajectory:
    
    def __init__(self,p0,R0,p1,R1,T):

        self.p0 = np.array(p0, dtype=float)
        self.R0 = np.array(R0, dtype=float)
        self.p1 = np.array(p1, dtype=float)
        self.R1 = np.array(R1, dtype=float)
        if T <= 0:
            raise ValueError(f"T must be positive, got {T}")
        
        self.T = T
        self.del_pos = self.p1 - self.p0
        self.del_orient = log_SO3(self.R1 @ self.R0.T)
        self.rot_angle = np.linalg.norm(self.del_orient)


    def evaluate(self, t):
        tau = np.clip(t/self.T, 0, 1)

        # quintic scaling for smooth start and stop
        s = 10 * tau**3 - 15 * tau**4 + 6 * tau**5
        s_dot = (30 * tau**2 - 60 * tau**3 + 30 * tau**4) / self.T
        position = self.p0 + s * self.del_pos
        orientation = exponential_map_SO3(s * self.del_orient)@self.R0
        velocity = np.concatenate([s_dot * self.del_pos, s_dot * self.del_orient])
        return position, orientation, velocity

    
    def is_done(self, t):
        return t >= self.T