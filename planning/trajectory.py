import numpy as np

class JointTrajectory:
    def __init__(self,q0,qf,T):
        self.q0 = q0
        self.qf = qf
        self.T = T
        self.n = len(q0)
        self.coeffs = self._solve_coefficients()
    def _solve_coefficients(self):
        coeffs = np.zeros((self.n,6))
        for i in range(self.n):
            a0 = self.q0[i]
            a1 = 0.0
            a2 = 0.0
            a3 = (10*(self.qf[i]-self.q0[i]))/(self.T**3)
            a4 = (-15*(self.qf[i]-self.q0[i]))/(self.T**4)
            a5 = (6*(self.qf[i]-self.q0[i]))/(self.T**5)
            coeffs[i] = [a0,a1,a2,a3,a4,a5]
        return coeffs
    def evaluate(self,t):
        q = np.zeros(self.n)
        qd = np.zeros(self.n)
        qdd = np.zeros(self.n)
        t = np.clip(t,0,self.T)
        for i in range(self.n):
            a0,a1,a2,a3,a4,a5 = self.coeffs[i]
            q[i] = a0 + a1*t + a2*t**2 + a3*t**3 + a4*t**4 + a5*t**5
            qd[i] = a1 + 2*a2*t + 3*a3*t**2 + 4*a4*t**3 + 5*a5*t**4
            qdd[i] = 2*a2 + 6*a3*t + 12*a4*t**2 + 20*a5*t**3
        return q,qd,qdd
    def is_done(self,t):
        return t >= self.T
