"""Independent NumPy RELION conventions for diagnostic probes.

Keep these references separate from production geometry implementations.
"""

import re

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



def read_initial_noise_variance(fixture_dir, n: int) -> np.ndarray:
    txt = (fixture_dir / "run_it000_model.star").read_text()
    m = re.search(r"data_model_optics_group_1\n(.*?)(?:\ndata_)", txt, re.DOTALL)
    v = np.zeros(n, dtype=np.float64)
    for line in m.group(1).strip().split("\n"):
        toks = line.split()
        if len(toks) == 3:
            try:
                v[int(toks[0])] = float(toks[2])
            except ValueError:
                pass
    return v


def centered_correlation(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    return float(np.real(np.vdot(af, bf)) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))
