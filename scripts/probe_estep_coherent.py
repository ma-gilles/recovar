"""Phase B coherent end-to-end probe.

Drives our GPU run_em on the small RELION InitialModel fixture using
RELION's exact post-perturbation oversampled rotations + translations
(read from the dump produced by docs/patches/relion_estep_dump.patch),
converts to BPref layout via run_em_output_to_bpref, and compares to
RELION's iter-1 BPref dump from the SAME RELION run (perturbation
matches).

Apply ALL corrections discovered during Phase B:
  - RELION-sorted halfset assignment (`_split_halfset_particle_ids`
    with micrograph_names supplied)
  - R_from_relion conversion of dumped Eulers to recovar frame
  - relion_firstiter_score_mode='gaussian' (matches RELION's
    do_firstiter_cc=0 default)
  - half_spectrum_scoring=True

How to regenerate dumps:
  RECOVAR_DEBUG_ESTEP_DIR=$DIR RECOVAR_DEBUG_DUMP_DIR=$DIR \\
    relion_refine ... --random_seed 1234567 \\
    (see scripts/run_relion_dump_small.sh once added)

Result baseline (2026-04-25): bp_data CC = +0.59. Closing the residual
~0.41 gap requires image preprocessing parity (mask/normcorr/scale/
sigma² scaling) — Phase C work.
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
DUMP = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small")


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


def main() -> None:
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.data_io.cryoem_dataset import load_dataset
    from recovar.data_io.starfile import read_star
    from recovar.em.dense_single_volume.em_engine import run_em
    from recovar.em.initial_model.dense_adapter import split_pseudo_halfset_particle_ids as _split_halfset_particle_ids
    from recovar.reconstruction.noise import make_radial_noise
    from recovar.utils.helpers import R_from_relion, load_relion_volume

    with open(DUMP / "p0_oversampled_eulers.bin", "rb") as f:
        h = struct.unpack("qqq", f.read(24))
        eulers = np.fromfile(f, dtype=np.float64, count=h[0] * h[1] * h[2]).reshape(-1, 3)
    with open(DUMP / "p0_oversampled_translations.bin", "rb") as f:
        h = struct.unpack("qqq", f.read(24))
        trans = np.fromfile(f, dtype=np.float64, count=h[0] * h[1] * h[2]).reshape(-1, 3)
    rotations_recovar = R_from_relion(eulers).astype(np.float32)
    translations = trans[:, :2].astype(np.float32)

    ds = load_dataset(str(PARTICLES_STAR), lazy=False)
    ori = int(ds.grid_size)
    # NEW: enable RELION-exact preprocessing (set_relion_image_mask) — closes
    # the per-pixel Fimg gap from CC=+0.949 to +0.997 vs RELION's exp_Fimg.
    backend = ds.image_source.backend if hasattr(ds.image_source, "backend") else None
    if backend is not None and hasattr(backend, "set_relion_image_mask"):
        pixel_size = float(ds.voxel_size)
        backend.set_relion_image_mask(
            pixel_size=pixel_size,
            particle_diameter_ang=544.0,
            width_mask_edge_px=5.0,
        )
        print(f"Enabled RELION-exact mask (pixel_size={pixel_size}, particle_diameter=544, edge=5)")
    iref_real = np.asarray(load_relion_volume(str(FIXTURE_DIR / "run_it000_class001.mrc")), dtype=np.float64)
    # ROUND-10 FIX: apply RELION's gridding correction (sinc² pre-divide) to
    # the real-space volume before FFT. This compensates for trilinear interp
    # smoothing in Fourier space, lifting projection CC from +0.997 to +1.0
    # bit-exact vs RELION's PPref.data.
    from recovar.core.relion_project import gridding_correct_volume_real

    iref_real_corrected = np.asarray(
        gridding_correct_volume_real(jnp.asarray(iref_real), ori_size=ori, padding_factor=1)
    )
    # /N² for FFT normalisation; -1 to flip CTF-sign-induced cross-term.
    iref_ft = -np.asarray(ftu.get_dft3(jnp.asarray(iref_real_corrected))).reshape(-1) / (ori**2)
    sigma2 = _read_iter0_sigma2(ori // 2 + 1)
    n4 = ori**4
    nv = np.asarray(make_radial_noise(sigma2 * n4, (ori, ori))).astype(np.float32).reshape(-1)
    r_max = 14
    current_size = 28

    main_in, _ = read_star(str(PARTICLES_STAR))
    mic_names = np.asarray(main_in["_rlnMicrographName"].tolist())
    h0_ids, _ = _split_halfset_particle_ids(ds.n_images, micrograph_names=mic_names)
    ds_h0 = ds.subset(h0_ids)

    print(
        f"Coherent E-step probe: {ds_h0.n_images} h0 particles × "
        f"{rotations_recovar.shape[0]} rotations × {translations.shape[0]} translations"
    )
    result = run_em(
        ds_h0,
        mean=jnp.asarray(iref_ft, dtype=jnp.complex64),
        mean_variance=jnp.asarray((np.abs(iref_ft) ** 2).astype(np.float32)),
        noise_variance=jnp.asarray(nv),
        rotations=jnp.asarray(rotations_recovar),
        translations=jnp.asarray(translations),
        disc_type="linear_interp",
        image_batch_size=10,
        rotation_block_size=200,
        current_size=current_size,
        projection_padding_factor=1,
        reconstruction_padding_factor=1,
        half_spectrum_scoring=True,
        return_stats=True,
        # InitialModel coherent fixture has do_firstiter_cc=0 → gaussian
        relion_firstiter_score_mode="gaussian",
        score_with_masked_images=True,
    )
    Ft_y = np.asarray(result[2])
    Ft_ctf = np.asarray(result[3])

    N = ori
    hp = r_max + 1
    c = N // 2
    Fy = Ft_y.reshape(N, N, N)
    Fc = Ft_ctf.reshape(N, N, N)
    sl = (slice(c - hp, c + hp + 1), slice(c, c + hp + 1), slice(c - hp, c + hp + 1))
    bp_data = -np.transpose(Fy[sl], (2, 0, 1)).astype(np.complex128)
    bp_weight = np.transpose(Fc[sl], (2, 0, 1)).real.astype(np.float64)

    target_data = _read_bin(DUMP / "pipe_it1_c0_bp_data_pre_reweight.bin")
    target_weight = _read_bin(DUMP / "pipe_it1_c0_bp_weight.bin")

    print("COHERENT B-PROBE (single RELION run, all corrections):")
    print(f"  bp_data CC    = {_cc(bp_data, target_data):+.6f}")
    print(f"  bp_weight CC  = {_cc(bp_weight, target_weight):+.6f}")
    print(f"  ‖ours‖/‖target‖ data   = {np.linalg.norm(bp_data) / np.linalg.norm(target_data):.4e}")
    print(f"  ‖ours‖/‖target‖ weight = {np.linalg.norm(bp_weight) / np.linalg.norm(target_weight):.4e}")


if __name__ == "__main__":
    main()
