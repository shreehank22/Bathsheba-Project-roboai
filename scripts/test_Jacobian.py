import sys
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
import numpy as np
from planning.fk import fk
from planning.ik import geometric_jacobian, log_SO3

# Franka joint limits (rad)
Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

N   = 50
EPS = 1e-5

errors_frob = []
errors_lin  = []
errors_ang  = []

for trial in range(N):
    q = np.random.uniform(Q_MIN, Q_MAX)

    # analytical jacobian
    J_analytical = geometric_jacobian(q)

    # numerical jacobian
    J_numerical = np.zeros((6, 7))
    for i in range(7):
        e_i     = np.zeros(7)
        e_i[i]  = 1.0

        q_plus  = q + EPS * e_i
        q_minus = q - EPS * e_i

        T_plus  = fk(q_plus)
        T_minus = fk(q_minus)

        p_plus  = T_plus[:3, 3]
        p_minus = T_minus[:3, 3]
        R_plus  = T_plus[:3, :3]
        R_minus = T_minus[:3, :3]

        J_numerical[:3, i] = (p_plus - p_minus) / (2 * EPS)
        J_numerical[3:, i] = log_SO3(R_plus @ R_minus.T) / (2 * EPS)

    err_frob = np.linalg.norm(J_analytical - J_numerical, 'fro')
    err_lin  = np.linalg.norm(J_analytical[:3] - J_numerical[:3], 'fro')
    err_ang  = np.linalg.norm(J_analytical[3:] - J_numerical[3:], 'fro')

    errors_frob.append(err_frob)
    errors_lin.append(err_lin)
    errors_ang.append(err_ang)

errors_frob = np.array(errors_frob)
errors_lin  = np.array(errors_lin)
errors_ang  = np.array(errors_ang)

print(f"{'Metric':<12} {'Mean':>12} {'Max':>12} {'Std':>12}")
print("-" * 50)
print(f"{'Frobenius':<12} {errors_frob.mean():>12.6f} {errors_frob.max():>12.6f} {errors_frob.std():>12.6f}")
print(f"{'Linear':<12} {errors_lin.mean():>12.6f} {errors_lin.max():>12.6f} {errors_lin.std():>12.6f}")
print(f"{'Angular':<12} {errors_ang.mean():>12.6f} {errors_ang.max():>12.6f} {errors_ang.std():>12.6f}")

# worst case config
worst_idx = np.argmax(errors_frob)
print(f"\nWorst config index: {worst_idx} | Frobenius error: {errors_frob[worst_idx]:.6f}")