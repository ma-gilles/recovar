"""Independent NumPy RELION conventions for diagnostic probes.

Keep these references separate from production geometry implementations.
"""

import numpy as np


def euler_matrix(rot_d, tilt_d, psi_d):
    rot = np.deg2rad(rot_d)
    tilt = np.deg2rad(tilt_d)
    psi = np.deg2rad(psi_d)
    ca, sa = np.cos(rot), np.sin(rot)
    cb, sb = np.cos(tilt), np.sin(tilt)
    cg, sg = np.cos(psi), np.sin(psi)
    cc = cb * ca
    cs = cb * sa
    sc = sb * ca
    ss = sb * sa
    return np.array(
        [
            [cg * cc - sg * sa, cg * cs + sg * ca, -cg * sb],
            [-sg * cc - cg * sa, -sg * cs + cg * ca, sg * sb],
            [sc, ss, cb],
        ]
    )

