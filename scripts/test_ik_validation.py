import sys
sys.path.insert(0, '/home/shreehan/Bathsheba-Project-roboai')
import numpy as np
from planning.fk import fk
from planning.ik import dls_ik, log_SO3

Q_MIN = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
Q_MAX = np.array([ 2.8973,  1.7628,  2.8973, -0.0698,  2.8973,  3.7525,  2.8973])

N = 1000
results = []
def ik_with_restart(p_des, R_des, q_target, n_restarts=5):
    for _ in range(n_restarts):
        q0 = np.clip(q_target + np.random.uniform(-0.3, 0.3, 7), Q_MIN, Q_MAX)
        result = dls_ik(q0, p_des, R_des, lambda_max=0.01, epsilon=0.05, max_iter=300)
        if result['converged']:
            return result
    return result
for trial in range(N):
    q_target = np.random.uniform(Q_MIN, Q_MAX)
    T_target = fk(q_target)
    p_des = T_target[:3, 3]
    R_des = T_target[:3, :3]

    q0 = q_target + np.random.uniform(-0.3, 0.3, 7)
    q0 = np.clip(q0, Q_MIN, Q_MAX)
    result = ik_with_restart(p_des, R_des, q_target)

    T_result = fk(result['q'])
    p_result = T_result[:3, 3]
    R_result = T_result[:3, :3]

    pos_err = np.linalg.norm(p_des - p_result) * 1000
    rot_err = np.degrees(np.linalg.norm(log_SO3(R_des @ R_result.T)))

    results.append({
        'converged': result['converged'],
        'n_iter':    result['n_iter'],
        'pos_err':   pos_err,
        'rot_err':   rot_err,
    })

converged     = [r for r in results if r['converged']]
not_converged = [r for r in results if not r['converged']]

pos_errs  = np.array([r['pos_err'] for r in results])
rot_errs  = np.array([r['rot_err'] for r in results])
iters     = np.array([r['n_iter']  for r in results])

conv_pos  = np.array([r['pos_err'] for r in converged]) if converged else np.array([])
conv_rot  = np.array([r['rot_err'] for r in converged]) if converged else np.array([])
conv_iter = np.array([r['n_iter']  for r in converged]) if converged else np.array([])

print(f"Trials         : {N}")
print(f"Converged      : {len(converged)} / {N} ({100*len(converged)/N:.1f}%)")
print(f"Not converged  : {len(not_converged)}")
print()
print(f"{'Metric':<20} {'Mean':>10} {'Max':>10} {'Std':>10}")
print("-" * 55)
print(f"{'Pos error (mm)':<20} {pos_errs.mean():>10.4f} {pos_errs.max():>10.4f} {pos_errs.std():>10.4f}")
print(f"{'Rot error (deg)':<20} {rot_errs.mean():>10.4f} {rot_errs.max():>10.4f} {rot_errs.std():>10.4f}")
print(f"{'Iterations':<20} {iters.mean():>10.1f} {iters.max():>10.0f} {iters.std():>10.1f}")
print()
if len(converged) > 0:
    print("Converged subset:")
    print(f"{'Pos error (mm)':<20} {conv_pos.mean():>10.4f} {conv_pos.max():>10.4f} {conv_pos.std():>10.4f}")
    print(f"{'Rot error (deg)':<20} {conv_rot.mean():>10.4f} {conv_rot.max():>10.4f} {conv_rot.std():>10.4f}")
    print(f"{'Iterations':<20} {conv_iter.mean():>10.1f} {conv_iter.max():>10.0f} {conv_iter.std():>10.1f}")

for i, r in enumerate(results):
    if not r['converged']:
        print(f"Failed trial {i}: pos_err={r['pos_err']:.4f}mm  n_iter={r['n_iter']}")