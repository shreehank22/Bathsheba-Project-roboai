import numpy as np

def dh_transform(a, d, alpha, theta):
    ct, st = np.cos(theta), np.sin(theta)
    ca, sa = np.cos(alpha), np.sin(alpha)

    return np.array([
        [ct,  -st*ca,  st*sa,  a*ct],
        [st,   ct*ca, -ct*sa,  a*st],
        [ 0,      sa,     ca,     d],
        [ 0,       0,      0,     1]
    ])