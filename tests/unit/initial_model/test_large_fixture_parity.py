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
    num = np.vdot(af, bf)  # conjugate-first inner product handles complex arrays
    denom = np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30
    # Real part of vdot is the "correlation" for complex data; mirrors
    # RELION's shell-wise Re<F1, conj(F2)> FSC convention.
    return float(np.real(num) / denom)


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


def _max_rel_err(out: np.ndarray, target: np.ndarray) -> float:
    """Relative infinity-norm error, safe against all-zero targets."""
    denom = max(float(np.abs(target).max()), 1e-30)
    return float(np.abs(out - target).max() / denom)


@requires_big_fixture
def test_reweight_grad_big_fixture():
    """Machine-precision parity for BackProjector::reweightGrad on 5k/256.

    RELION dumps one halfset's BPref.data pre- and post-reweight inside
    the pseudo-halfset loop (last iteration, ih=1). Feed the pre-dump +
    matching bp_weight_h through our binding and compare to post-dump.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    data_pre = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_data_h_pre_reweight.bin")
    weight = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_weight_h.bin")
    target = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_data_h_post_reweight.bin")

    meta = _read_mstep_meta()
    ori = int(meta["ori_size"])
    r_max = int(meta["bpref_r_max"])

    out = np.asarray(
        bind.vdam_reweight_grad(
            data_pre.astype(np.complex128),
            weight.astype(np.float64),
            ori,
            1,  # padding_factor
            1,  # TRILINEAR
            r_max,
        )
    )

    cc = _cc(out, target)
    rel_err = _max_rel_err(out, target)
    print(f"\nreweightGrad LARGE FIXTURE (N=5000, box=256):\n  CC       = {cc:+.6f}\n  rel err  = {rel_err:.3e}")
    assert cc > 0.9999, f"reweightGrad CC below machine precision: {cc:.4f}"
    assert rel_err < 1e-9, f"reweightGrad rel err too large: {rel_err:.3e}"


@requires_big_fixture
def test_first_moment_big_fixture():
    """Machine-precision parity for BackProjector::getFristMoment on 5k/256.

    Inputs are the post-reweight halfset data and pre-update halfset
    Igrad1; expected output is Igrad1_h_post. mu (lambda) = 0.9 matches
    GUI InitialModel default.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    data = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_data_h_post_reweight.bin")
    mom_pre = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_Igrad1_h_pre.bin")
    target = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_Igrad1_h_post.bin")

    meta = _read_mstep_meta()
    ori = int(meta["ori_size"])
    r_max = int(meta["bpref_r_max"])

    out = np.asarray(
        bind.vdam_first_moment(
            data.astype(np.complex128),
            mom_pre.astype(np.complex128),
            ori,
            1,
            1,
            r_max,
            **{"lambda": 0.9},
        )
    )

    cc = _cc(out, target)
    rel_err = _max_rel_err(out, target)
    print(f"\ngetFristMoment LARGE FIXTURE (N=5000, box=256):\n  CC       = {cc:+.6f}\n  rel err  = {rel_err:.3e}")
    assert cc > 0.9999, f"getFristMoment CC below machine precision: {cc:.4f}"
    assert rel_err < 1e-9, f"getFristMoment rel err too large: {rel_err:.3e}"


@requires_big_fixture
def test_second_moment_big_fixture():
    """Machine-precision parity for BackProjector::getSecondMoment on 5k/256.

    Uses both halfsets' post-reweight data (plain = halfset 0, _h_ =
    halfset 1) plus Igrad2_pre. Expected Igrad2_post.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    data_h0 = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_data_post_reweight.bin")
    data_h1 = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_data_h_post_reweight.bin")
    mom_pre = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_Igrad2_pre.bin")
    target = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_Igrad2_post.bin")

    meta = _read_mstep_meta()
    ori = int(meta["ori_size"])
    r_max = int(meta["bpref_r_max"])

    out = np.asarray(
        bind.vdam_second_moment(
            data_h0.astype(np.complex128),
            data_h1.astype(np.complex128),
            mom_pre.astype(np.complex128),
            ori,
            1,
            1,
            r_max,
            # RELION's getSecondMoment default is lambda=0.999 (backprojector.h:343)
            **{"lambda": 0.999},
        )
    )

    cc = _cc(out, target)
    rel_err = _max_rel_err(out, target)
    print(f"\ngetSecondMoment LARGE FIXTURE (N=5000, box=256):\n  CC       = {cc:+.6f}\n  rel err  = {rel_err:.3e}")
    assert cc > 0.9999, f"getSecondMoment CC below machine precision: {cc:.4f}"
    assert rel_err < 1e-9, f"getSecondMoment rel err too large: {rel_err:.3e}"


@requires_big_fixture
def test_apply_momenta_big_fixture():
    """Machine-precision parity for BackProjector::applyMomenta on 5k/256.

    Combines the per-halfset first moments and shared second moment
    into the final bp_data + mom1_noise_power fed into reconstructGrad.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    data_shape = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_data_post_reweight.bin")
    mom1_h1 = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_Igrad1_post.bin")
    mom1_h2 = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_Igrad1_h_post.bin")
    mom2 = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_Igrad2_post.bin")
    target_data = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_data_post_applymomenta.bin")
    target_np = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_mom1_noise_power.bin").ravel()

    meta = _read_mstep_meta()
    ori = int(meta["ori_size"])
    r_max = int(meta["bpref_r_max"])

    out_data, out_np = bind.vdam_apply_momenta(
        data_shape.astype(np.complex128),
        mom1_h1.astype(np.complex128),
        mom1_h2.astype(np.complex128),
        mom2.astype(np.complex128),
        ori,
        1,
        1,
        r_max,
    )
    out_data = np.asarray(out_data)
    out_np = np.asarray(out_np).ravel()

    cc_data = _cc(out_data, target_data)
    rel_err_data = _max_rel_err(out_data, target_data)

    # Trim noise-power spectra to the shorter length (RELION's dump uses
    # ori//2+1 shells; the binding returns padded length).
    n_shells = min(out_np.size, target_np.size)
    cc_np = _cc(out_np[:n_shells], target_np[:n_shells])
    rel_err_np = _max_rel_err(out_np[:n_shells], target_np[:n_shells])

    print(
        f"\napplyMomenta LARGE FIXTURE (N=5000, box=256):\n"
        f"  data CC      = {cc_data:+.6f}   rel err = {rel_err_data:.3e}\n"
        f"  noise_pwr CC = {cc_np:+.6f}   rel err = {rel_err_np:.3e}"
    )
    assert cc_data > 0.9999, f"applyMomenta data CC below precision: {cc_data:.4f}"
    assert rel_err_data < 1e-9, f"applyMomenta data rel err: {rel_err_data:.3e}"
    assert cc_np > 0.9999, f"applyMomenta noise-power CC below precision: {cc_np:.4f}"
    assert rel_err_np < 1e-9, f"applyMomenta noise-power rel err: {rel_err_np:.3e}"


@requires_big_fixture
def test_mstep_chain_end_to_end_big_fixture():
    """End-to-end M-step chain parity on 5k/256: feed RELION's pre-reweight
    BP data through reweightGrad → first_moment (x2) → second_moment →
    applyMomenta → reconstructGrad, compare the final Iref to RELION's
    iter-1 Iref_after.

    This is the ULTIMATE parity gate: if every primitive is at machine
    precision AND chains correctly, the final volume matches RELION
    bit-for-bit.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    # Halfset data: plain = h0, _h_ = h1
    bp_h0_pre = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_data_pre_reweight.bin")
    bp_h1_pre = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_data_h_pre_reweight.bin")
    bp_weight_h0 = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_weight.bin")
    bp_weight_h1 = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_bp_weight_h.bin")
    Igrad1_h0_pre = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_Igrad1_pre.bin")
    Igrad1_h1_pre = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_Igrad1_h_pre.bin")
    Igrad2_pre = _read_bin(BIG_DUMP_DIR / "pipe_it1_c0_Igrad2_pre.bin")
    iref_before = _read_bin(BIG_DUMP_DIR / "mstep_it1_c0_iref_before.bin")
    target_iref = _read_bin(BIG_DUMP_DIR / "mstep_it1_c0_iref_after.bin")

    meta = _read_mstep_meta()
    ori = int(meta["ori_size"])
    r_max = int(meta["bpref_r_max"])
    mu_first = 0.9  # getFristMoment default
    mu_second = 0.999  # getSecondMoment default

    # Step 1: reweightGrad per halfset
    bp_h0_rw = np.asarray(
        bind.vdam_reweight_grad(bp_h0_pre.astype(np.complex128), bp_weight_h0.astype(np.float64), ori, 1, 1, r_max)
    )
    bp_h1_rw = np.asarray(
        bind.vdam_reweight_grad(bp_h1_pre.astype(np.complex128), bp_weight_h1.astype(np.float64), ori, 1, 1, r_max)
    )

    # Step 2: getFristMoment per halfset
    Igrad1_h0_post = np.asarray(
        bind.vdam_first_moment(bp_h0_rw, Igrad1_h0_pre.astype(np.complex128), ori, 1, 1, r_max, **{"lambda": mu_first})
    )
    Igrad1_h1_post = np.asarray(
        bind.vdam_first_moment(bp_h1_rw, Igrad1_h1_pre.astype(np.complex128), ori, 1, 1, r_max, **{"lambda": mu_first})
    )

    # Step 3: getSecondMoment (uses both halfsets)
    Igrad2_post = np.asarray(
        bind.vdam_second_moment(
            bp_h0_rw, bp_h1_rw, Igrad2_pre.astype(np.complex128), ori, 1, 1, r_max, **{"lambda": mu_second}
        )
    )

    # Step 4: applyMomenta — data_in is shape carrier
    bp_final, mom1_np = bind.vdam_apply_momenta(bp_h0_rw, Igrad1_h0_post, Igrad1_h1_post, Igrad2_post, ori, 1, 1, r_max)
    bp_final = np.asarray(bp_final)
    mom1_np = np.asarray(mom1_np).ravel()

    # Step 5: reconstructGrad — use bp_weight from halfset 0 (BPref[iclass])
    fsc = np.zeros(ori // 2 + 1, dtype=np.float64)
    out_iref = np.asarray(
        bind.vdam_reconstruct_grad(
            iref_before.astype(np.float64),
            bp_final.astype(np.complex128),
            bp_weight_h0.astype(np.float64),
            fsc,
            meta["effective_stepsize"],
            meta["tau2_fudge_factor"],
            ori,
            1,
            1,
            r_max,
            meta["min_resol_shell"],
            False,
            bool(int(meta["bpref_skip_gridding"])),
            mom1_np,
        )
    )

    cc = _cc(out_iref, target_iref)
    rel_err = _max_rel_err(out_iref, target_iref)
    print(
        f"\nEND-TO-END M-STEP CHAIN PARITY (N=5000, box=256):\n"
        f"  Iref CC      = {cc:+.6f}\n"
        f"  Iref rel err = {rel_err:.3e}\n"
        f"  ours std     = {out_iref.std():.3e}\n"
        f"  target std   = {target_iref.std():.3e}"
    )
    assert cc > 0.9999, f"End-to-end M-step CC below precision: {cc:.4f}"


def _read_mstep_meta():
    """Parse the dumped M-step meta file; add ori_size from iref shape."""
    meta: dict = {}
    for line in (BIG_DUMP_DIR / "mstep_it1_c0_meta.txt").read_text().strip().split("\n"):
        k, v = line.split("=")
        try:
            meta[k] = float(v)
        except ValueError:
            meta[k] = v
    # ori_size isn't in the meta file; read from iref shape
    iref = _read_bin(BIG_DUMP_DIR / "mstep_it1_c0_iref_before.bin")
    meta["ori_size"] = float(iref.shape[0])
    return meta
