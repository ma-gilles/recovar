"""Phase B probe: feed RELION-exact post-perturbation post-oversampling
rotations + translations through our GPU run_em, compare BPref output
to RELION's iter-1 dump.

This isolates the SCORING + BACKPROJECTION components from grid-generation
gaps (perturbation, oversampling, ordering). Whatever CC we measure is the
maximum achievable without porting Phase C1/C2.

Usage:
  pixi run python scripts/probe_estep_exact_rotations.py

Requires:
  - RELION instrumented dumps at /scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small/
    (produced by docs/patches/relion_estep_dump.patch + relion_refine run)
  - The InitialModel fixture at FIXTURE_DIR / PARTICLES_STAR
"""

from __future__ import annotations

import re
import struct
from pathlib import Path

import numpy as np

FIXTURE_DIR = Path("/scratch/gpfs/GILLES/mg6942/tmp/relion_initialmodel_64_20260420_121428_8956_run")
PARTICLES_STAR = Path(
    "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/"
    "slurm_7178672/pytest-of-mg6942/pytest-0/test_pipeline_spa_gpu0/"
    "gpu_spa/test_dataset/particles.star"
)
DUMP_DIR = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small")
RELION_BPREF = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_debug_dump")


def _read_bin(p: Path) -> np.ndarray:
    with open(p, "rb") as f:
        nz, ny, nx = struct.unpack("qqq", f.read(24))
        pos = f.tell()
        f.seek(0, 2)
        rem = f.tell() - pos
        f.seek(pos)
        bp = rem // (nz * ny * nx)
        dt = np.complex128 if bp == 16 else np.float64
        return np.fromfile(f, dtype=dt, count=nz * ny * nx).reshape(nz, ny, nx)


def _cc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    return float(np.real(np.vdot(af, bf)) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def _read_iter0_sigma2(n: int) -> np.ndarray:
    txt = (FIXTURE_DIR / "run_it000_model.star").read_text()
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


def _euler_to_matrix(rot_deg: float, tilt_deg: float, psi_deg: float) -> np.ndarray:
    """RELION Euler convention (rot, tilt, psi) → 3x3 matrix."""
    rot = np.deg2rad(rot_deg)
    tilt = np.deg2rad(tilt_deg)
    psi = np.deg2rad(psi_deg)
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


def main() -> None:
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.em.dense_single_volume.em_engine import run_em
    from recovar.em.initial_model.gpu_pipeline import _split_halfset_particle_ids
    from recovar.reconstruction.noise import make_radial_noise
    from recovar.utils.helpers import load_relion_volume

    # Load RELION-exact post-perturb post-oversample rotations + translations
    with open(DUMP_DIR / "p0_oversampled_eulers.bin", "rb") as f:
        h = struct.unpack("qqq", f.read(24))
        eulers = np.fromfile(f, dtype=np.float64, count=h[0] * h[1] * h[2]).reshape(-1, 3)
    with open(DUMP_DIR / "p0_oversampled_translations.bin", "rb") as f:
        h = struct.unpack("qqq", f.read(24))
        trans = np.fromfile(f, dtype=np.float64, count=h[0] * h[1] * h[2]).reshape(-1, 3)
    rotations = np.array([_euler_to_matrix(*e) for e in eulers]).astype(np.float32)
    translations = trans[:, :2].astype(np.float32)

    print(f"RELION-exact grid: {rotations.shape[0]} rotations × {translations.shape[0]} translations")

    ds = load_dataset(str(PARTICLES_STAR), lazy=False)
    ori = int(ds.grid_size)
    iref_real = np.asarray(load_relion_volume(str(FIXTURE_DIR / "run_it000_class001.mrc")), dtype=np.float64)
    iref_ft = np.asarray(ftu.get_dft3(jnp.asarray(iref_real))).reshape(-1)
    sigma2 = _read_iter0_sigma2(ori // 2 + 1)
    n4 = ori**4
    nv = np.asarray(make_radial_noise(sigma2 * n4, (ori, ori))).astype(np.float32).reshape(-1)
    r_max = 14
    current_size = 28

    mean_ft_j = jnp.asarray(iref_ft, dtype=jnp.complex64)
    mean_var_j = jnp.asarray((np.abs(iref_ft) ** 2).astype(np.float32))

    h0_ids, _ = _split_halfset_particle_ids(ds.n_images)
    ds_h0 = ds.subset(h0_ids)

    print(f"Running E-step on {ds_h0.n_images} h0 particles...")
    result = run_em(
        ds_h0,
        mean=mean_ft_j,
        mean_variance=mean_var_j,
        noise_variance=jnp.asarray(nv),
        rotations=jnp.asarray(rotations),
        translations=jnp.asarray(translations),
        disc_type="linear_interp",
        image_batch_size=10,
        rotation_block_size=200,
        current_size=current_size,
        projection_padding_factor=1,
        reconstruction_padding_factor=1,
        half_spectrum_scoring=True,
        return_stats=True,
        relion_firstiter_score_mode="gaussian",  # matches RELION dump (do_firstiter_cc=0)
    )
    Ft_y = np.asarray(result[2])
    Ft_ctf = np.asarray(result[3])

    # BPref layout (RELION frame): -transpose((2, 0, 1)) of centered slab
    N = ori
    hp = r_max + 1
    c = N // 2
    Fy = Ft_y.reshape(N, N, N)
    Fc = Ft_ctf.reshape(N, N, N)
    sl = (slice(c - hp, c + hp + 1), slice(c, c + hp + 1), slice(c - hp, c + hp + 1))
    bp_data = -np.transpose(Fy[sl], (2, 0, 1)).astype(np.complex128)
    bp_weight = np.transpose(Fc[sl], (2, 0, 1)).real.astype(np.float64)

    target_data = _read_bin(RELION_BPREF / "pipe_it1_c0_bp_data_pre_reweight.bin")
    target_weight = _read_bin(RELION_BPREF / "pipe_it1_c0_bp_weight.bin")

    print("\nB-PROBE RESULT (RELION-exact rotations + translations, h0):")
    print(f"  bp_data    CC = {_cc(bp_data, target_data):+.6f}")
    print(f"  bp_weight  CC = {_cc(bp_weight, target_weight):+.6f}")
    print(f"  ‖ours‖/‖target‖ data   = {np.linalg.norm(bp_data) / np.linalg.norm(target_data):.4e}")
    print(f"  ‖ours‖/‖target‖ weight = {np.linalg.norm(bp_weight) / np.linalg.norm(target_weight):.4e}")


if __name__ == "__main__":
    main()
