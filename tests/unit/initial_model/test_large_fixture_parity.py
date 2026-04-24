"""Large-fixture parity tests — 5000 particles × 256² box.

Tests the same parity gates as test_bootstrap_iref_fixture.py and
test_reconstruct_grad_fixture.py but at a scale ~125× larger per iter
than the 500-particle 64²-box fixture. Validates that the machine-
precision parity of F8b scales to real-sized VDAM jobs.

Fixture: /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/slurm_7288808/.../particles.star
RELION dumps: /scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_big_dump_j1/
"""

from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest

BIG_STAR = Path(
    "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/"
    "slurm_7288808/pytest-of-mg6942/pytest-0/"
    "test_pipeline_gpu_memory_stres0/gpu_stress_256_5k/dataset/test_dataset/particles.star"
)
BIG_MRCS = BIG_STAR.with_name("particles.256.mrcs")
BIG_DUMP_DIR = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_big_dump_j1")


pytestmark = pytest.mark.unit

requires_big_fixture = pytest.mark.skipif(
    not (BIG_STAR.exists() and BIG_MRCS.exists() and BIG_DUMP_DIR.exists()),
    reason="5k-particle 256-pixel fixture or RELION dumps not available",
)


def _read_bin(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        nz, ny, nx = struct.unpack("qqq", f.read(24))
        pos = f.tell()
        f.seek(0, 2)
        rem = f.tell() - pos
        f.seek(pos)
        if nz * ny * nx == 0:
            return np.zeros(0)
        bp = rem // (nz * ny * nx)
        dt = np.complex128 if bp == 16 else np.float64
        return np.fromfile(f, dtype=dt, count=nz * ny * nx).reshape(nz, ny, nx)


def _cc(a, b):
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    return float(np.dot(af, bf) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


@requires_big_fixture
def test_bootstrap_iref_big_fixture():
    """F6 parity on 5k particles at box 256.

    Current result: CC=0.936 vs RELION iter-0 class reference. Amplitude
    matches within 3% (ours std=2.22e-3 vs target 2.28e-3). Imaginary-
    part correlation is near zero (-0.06) while real-part CC=0.97 —
    suggests a scale-dependent phase convention still to be traced. The
    small fixture at box 64 reaches CC=0.9993 so the binding is correct
    for that scale.

    This test's gate is CC > 0.85 to catch regressions while the phase
    mystery is diagnosed.
    """
    import mrcfile

    from recovar.data_io.starfile import read_star
    from recovar.em.initial_model.bootstrap_iref import reorder_particles_relion_style
    from recovar.relion_bind import _relion_bind_core as bind

    with mrcfile.open(BIG_MRCS, permissive=True) as m:
        stack = np.ascontiguousarray(np.asarray(m.data, dtype=np.float64))
    main, optics = read_star(str(BIG_STAR))
    voltage = float(optics["_rlnVoltage"].iloc[0])
    Cs = float(optics["_rlnSphericalAberration"].iloc[0])
    Q0 = float(optics["_rlnAmplitudeContrast"].iloc[0])
    angpix = float(optics["_rlnImagePixelSize"].iloc[0])
    ori = int(optics["_rlnImageSize"].iloc[0])
    defU = np.array([float(r["_rlnDefocusU"]) for _, r in main.iterrows()])
    defV = np.array([float(r["_rlnDefocusV"]) for _, r in main.iterrows()])
    defA = np.array([float(r["_rlnDefocusAngle"]) for _, r in main.iterrows()])
    phase = np.array([float(r.get("_rlnPhaseShift", 0.0)) for _, r in main.iterrows()])

    # RELION iterates particles in _rlnMicrographName lex order with a cap
    # of minimum_nr_particles_sigma2_noise=1000 per optics group. At N=5000
    # this selects a DIFFERENT first-1000 than plain row order.
    stack, defU, defV, defA, phase = reorder_particles_relion_style(main, stack, defU, defV, defA, phase)

    Iref = np.asarray(
        bind.vdam_bootstrap_iref(
            stack,
            defU,
            defV,
            defA,
            phase,
            voltage,
            Cs,
            Q0,
            angpix,
            ori,
            1,
            400.0,
            5.0,
            True,
            True,
            1776701668,
            1,
            1,
            18,
            1000,  # cs=18, cap=1000
        )
    )

    target = _read_bin(BIG_DUMP_DIR / "iref_c0_after_reconstruct.bin")
    cc = _cc(Iref[0], target)
    print(
        f"\nF6 LARGE FIXTURE (N=5000, box=256, cs=18, pad=1, cap=1000):\n"
        f"  CC         = {cc:+.6f}\n"
        f"  max|diff|  = {np.abs(Iref[0] - target).max():.3e}\n"
        f"  our std    = {Iref[0].std():.3e}\n"
        f"  target std = {target.std():.3e}\n"
        f"  amplitude ratio = {Iref[0].std() / target.std():.4f}"
    )
    # Machine-precision gate with lex-sorted particle order.
    assert cc > 0.9999, f"F6 big fixture CC below machine precision: {cc:.4f}"


@requires_big_fixture
def test_reconstruct_grad_big_fixture():
    """F8b machine-precision parity on 5k particles at box 256.

    This is the real gate: the reconstructGrad binding called with
    RELION's exact iter-1 M-step inputs (post-applyMomenta BP data,
    mom1_noise_power, etc.) produces the same output Iref[iclass] to
    machine precision.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    iref_before = _read_bin(BIG_DUMP_DIR / "mstep_it1_c0_iref_before.bin")
    bp_data_post = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_data_post_applymomenta.bin")
    bp_weight = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_weight.bin")
    mom1np = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_mom1_noise_power.bin").ravel()
    target = _read_bin(BIG_DUMP_DIR / "mstep_it1_c0_iref_after.bin")

    meta = {}
    for line in (BIG_DUMP_DIR / "mstep_it1_c0_meta.txt").read_text().strip().split("\n"):
        k, v = line.split("=")
        try:
            meta[k] = float(v)
        except ValueError:
            meta[k] = v

    ori = iref_before.shape[0]
    fsc = np.zeros(ori // 2 + 1, dtype=np.float64)
    out = np.asarray(
        bind.vdam_reconstruct_grad(
            iref_before.astype(np.float64),
            bp_data_post.astype(np.complex128),
            bp_weight.astype(np.float64),
            fsc,
            meta["effective_stepsize"],
            meta["tau2_fudge_factor"],
            ori,
            1,
            1,
            int(meta["bpref_r_max"]),
            meta["min_resol_shell"],
            False,
            bool(int(meta["bpref_skip_gridding"])),
            mom1np,
        )
    )

    cc = _cc(out, target)
    max_abs = float(np.abs(out - target).max())
    rel_err = max_abs / max(np.abs(target).max(), 1e-30)

    print(
        f"\nF8b LARGE FIXTURE (N=5000, box=256, RELION iter-1 M-step):\n"
        f"  CC        = {cc:+.6f}\n"
        f"  max|diff| = {max_abs:.3e}\n"
        f"  rel err   = {rel_err:.3e}\n"
        f"  ours std  = {out.std():.3e}\n"
        f"  target std= {target.std():.3e}"
    )

    # Machine-precision gate — should match at box 256 same as at box 64.
    assert cc > 0.9999, f"F8b big fixture CC below machine precision: {cc:.4f}"
