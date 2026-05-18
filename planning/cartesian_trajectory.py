from scipy.interpolate import CubicSpline
import numpy as np


class CartesianTrajectory:
    def __init__(self,waypoints,timestamps,R_des):
        self.waypoints = waypoints
        self.timestamps = timestamps
        self.R_des = R_des
        self.T = timestamps[-1] 
        self.spline = CubicSpline(self.timestamps, self.waypoints, bc_type='clamped')
        
    def evaluate(self, t):
        t = np.clip(t,0,self.T)
        p_des = self.spline(t)
        pd_des = self.spline(t, 1)
        return p_des, pd_des, self.R_des
    def is_done(self, t):
        return t >= self.T