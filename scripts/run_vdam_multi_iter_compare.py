"""Multi-iter VDAM convergence comparison: recovar vs RELION trajectory.

Drives `run_iter_gpu_vdam` for N iterations starting from RELION's iter-0 Iref
(shared, deterministic), then compares each iter's Iref to RELION's iter-K
output via FSC + per-shell CC. The bootstrap noise that dominates at iter 1
washes out by iter ~5-10; convergence parity is the meaningful metric.

Usage (fixture set inside; CLI optional):
    pixi run python scripts/run_vdam_multi_iter_compare.py [--iters 30]
"""

from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.data_io.cryoem_dataset import load_dataset
from recovar.data_io.starfile import read_star
from recovar.em.initial_model.gpu_pipeline import run_iter_gpu_vdam
from recovar.em.sampling import (
    apply_relion_translation_perturbation,
    get_oversampled_rotation_grid_from_samples,
    get_oversampled_translation_grid,
    get_translation_grid,
    read_relion_perturbation_from_sampling_star,
)
from recovar.utils.helpers import load_relion_volume

RELION_RUN = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_multi_iter_run")
RELION_DUMP = Path("/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_multi_iter_dump")
PARTICLES_STAR = Path(
    "/scratch/gpfs/GILLES/mg6942/recovar_dev/recovar/.tmp/"
    "slurm_7178672/pytest-of-mg6942/pytest-0/test_pipeline_spa_gpu0/"
    "gpu_spa/test_dataset/particles.star"
)


def read_sigma2(model_star: Path, n_shells: int) -> np.ndarray:
    txt = model_star.read_text()
    m = re.search(r"data_model_optics_group_1\n(.*?)(?:\ndata_)", txt, re.DOTALL)
    v = np.zeros(n_shells, dtype=np.float64)
    for line in m.group(1).strip().split("\n"):
        toks = line.split()
        if len(toks) == 3:
            try:
                v[int(toks[0])] = float(toks[2])
            except ValueError:
                pass
    return v


def read_grad_stepsize(optimiser_star: Path) -> float:
    txt = optimiser_star.read_text()
    m = re.search(r"_rlnGradCurrentStepsize\s+([-0-9.eE+]+)", txt)
    return float(m.group(1)) if m else 0.5


def read_current_size(model_star: Path) -> int:
    txt = model_star.read_text()
    m = re.search(r"_rlnCurrentImageSize\s+(\d+)", txt)
    return int(m.group(1)) if m else 28


def read_sigma_offset_Ang(model_star: Path) -> float:
    txt = model_star.read_text()
    m = re.search(r"_rlnSigmaOffsetsAngst\s+([-0-9.eE+]+)", txt)
    return float(m.group(1)) if m else 6.4


def read_sgd_subset_size(optimiser_star: Path) -> int:
    """RELION's per-iter mini-batch size. -1 means use full dataset."""
    txt = optimiser_star.read_text()
    m = re.search(r"_rlnSgdSubsetSize\s+([-0-9]+)", txt)
    return int(m.group(1)) if m else -1


def read_sorted_idx_dump(p: Path) -> tuple[np.ndarray, int]:
    """Load RELION's per-iter `mydata.sorted_idx` (particle order) + subset_size."""
    import struct as _s

    with open(p, "rb") as f:
        n, sz = _s.unpack("qq", f.read(16))
        order = np.fromfile(f, dtype=np.int64, count=n)
    return order, int(sz)


def read_random_perturbation(sampling_star: Path) -> float:
    pert, _ = read_relion_perturbation_from_sampling_star(str(sampling_star))
    return pert


def cc(a: np.ndarray, b: np.ndarray) -> float:
    af = a.ravel() - a.mean()
    bf = b.ravel() - b.mean()
    return float(np.real(np.vdot(af, bf)) / (np.linalg.norm(af) * np.linalg.norm(bf) + 1e-30))


def fsc_shells(v1: np.ndarray, v2: np.ndarray) -> np.ndarray:
    N = v1.shape[0]
    F1 = np.asarray(ftu.get_dft3(jnp.asarray(v1)))
    F2 = np.asarray(ftu.get_dft3(jnp.asarray(v2)))
    kz, ky, kx = np.meshgrid(
        np.arange(N) - N // 2,
        np.arange(N) - N // 2,
        np.arange(N) - N // 2,
        indexing="ij",
    )
    r = np.round(np.sqrt(kz**2 + ky**2 + kx**2)).astype(int)
    fsc = np.zeros(N // 2 + 1)
    for sh in range(N // 2 + 1):
        mask = r == sh
        if mask.sum() > 0:
            n = np.sum(F1[mask] * np.conj(F2[mask]))
            d = np.sqrt(np.sum(np.abs(F1[mask]) ** 2)) * np.sqrt(np.sum(np.abs(F2[mask]) ** 2))
            fsc[sh] = float(np.real(n) / (d + 1e-30))
    return fsc


def fsc_threshold_shell(fsc: np.ndarray, thresh: float = 0.143) -> int:
    """Smallest shell where FSC drops below threshold (= resolution shell)."""
    for sh in range(1, len(fsc)):
        if fsc[sh] < thresh:
            return sh
    return len(fsc) - 1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument(
        "--out",
        default="/scratch/gpfs/GILLES/mg6942/_agent_scratch/recovar_multi_iter_compare",
    )
    args = parser.parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True, parents=True)
    (out_dir / "SAFE_TO_DELETE").touch()

    # Setup: dataset + RELION-sorted halfsets + RELION-exact eulers/translations.
    ds = load_dataset(str(PARTICLES_STAR), lazy=False)
    ori = int(ds.grid_size)
    pixel_A = float(ds.voxel_size)
    backend = ds.image_source.backend if hasattr(ds.image_source, "backend") else None
    if backend is not None and hasattr(backend, "set_relion_image_mask"):
        backend.set_relion_image_mask(pixel_size=pixel_A, particle_diameter_ang=544.0, width_mask_edge_px=5.0)

    main_in, _ = read_star(str(PARTICLES_STAR))
    mic_names_full = np.asarray(main_in["_rlnMicrographName"].tolist())

    # Random seed used by RELION (must match `--random_seed` of the reference run)
    RELION_SEED = 1234567

    # Read RELION's particle order at the START of expectation (after the
    # natural micrograph-sort that RELION applies before iter 1). RELION calls
    # `mydata.randomiseParticlesOrder(random_seed+iter, do_split, subset_size)`
    # which permutes a NATURAL-SORT (= alphabetical micrograph order) baseline.
    # We approximate this by:
    #   1. Sort particles by micrograph name (lex stable) → "natural order"
    #   2. Use vdam_randomise_particles_order(N, seed=random_seed+iter) for shuffle

    nat_order = np.argsort(mic_names_full, kind="stable")  # RELION's "natural" particle order

    # Use RELION's iter-0 as initialization (deterministic --j 1 trajectory)
    iref_real = np.asarray(load_relion_volume(str(RELION_RUN / "run_it000_class001.mrc")), dtype=np.float64)

    # Initial state (no momenta yet)
    Igrad1 = None
    Igrad2 = None
    iref_curr = iref_real.copy()

    # Load reference trajectory
    relion_irefs = {
        k: np.asarray(
            load_relion_volume(str(RELION_RUN / f"run_it{k:03d}_class001.mrc")),
            dtype=np.float64,
        )
        for k in range(1, args.iters + 1)
        if (RELION_RUN / f"run_it{k:03d}_class001.mrc").exists()
    }
    print(f"RELION trajectory: {len(relion_irefs)} iterations available")

    # Track convergence
    rows = []
    log_path = out_dir / "convergence_log.txt"
    log_f = log_path.open("w")
    log_f.write("iter cc_full norm_ratio fsc05_shell fsc143_shell relion_fsc05_shell relion_fsc143_shell time_s\n")

    for it in range(1, args.iters + 1):
        # Per-iter sampling.star info (RELION wrote one per iter; iter k uses run_it{k:03d}_sampling.star)
        sampling_star = RELION_RUN / f"run_it{it:03d}_sampling.star"
        if not sampling_star.exists():
            sampling_star = RELION_RUN / f"run_it{it - 1:03d}_sampling.star"
        if sampling_star.exists():
            random_perturbation = read_random_perturbation(sampling_star)
        else:
            random_perturbation = 0.0

        # Build oversampled rotation+translation grids (HEALPix order 1 + oversample 1)
        coarse_indices = np.arange(48 * 12, dtype=np.int64)
        rotations, _ = get_oversampled_rotation_grid_from_samples(
            coarse_indices,
            parent_nside_level=1,
            oversampling_order=1,
            random_perturbation=random_perturbation,
        )
        rotations = rotations.astype(np.float32)
        coarse_translations = get_translation_grid(max_pixel=6, pixel_offset=2).astype(np.float32)
        translations, _ = get_oversampled_translation_grid(coarse_translations, pixel_offset=2, oversampling_order=1)
        translations = apply_relion_translation_perturbation(
            translations.astype(np.float32),
            random_perturbation,
            offset_step_pixels=2.0,
        )

        # Per-iter sigma2/current_size/stepsize/sigma_offset from RELION's k-1 model.star
        prev_model = RELION_RUN / f"run_it{it - 1:03d}_model.star"
        prev_optim = RELION_RUN / f"run_it{it - 1:03d}_optimiser.star"
        if not prev_model.exists() or not prev_optim.exists():
            print(f"  iter {it}: missing RELION dumps, stopping")
            break
        sigma2 = read_sigma2(prev_model, ori // 2 + 1)
        current_size = read_current_size(prev_model) if it > 1 else 28
        # _rlnGradCurrentStepsize is at the END of iter K (so iter K's stepsize = run_itK_optimiser.star)
        ss_optim = RELION_RUN / f"run_it{it:03d}_optimiser.star"
        grad_stepsize = read_grad_stepsize(ss_optim) if ss_optim.exists() else 0.5
        sigma_offset_Ang = read_sigma_offset_Ang(prev_model) if prev_model.exists() else 6.4

        # RELION's per-iter mini-batch subset_size and sorted_idx (= particle
        # order). We dump these from a patched RELION so recovar uses the
        # SAME particles per iter that RELION used.
        sorted_idx_path = RELION_DUMP / f"sorted_idx_iter{it:03d}.bin"
        if not sorted_idx_path.exists():
            print(f"  iter {it}: missing sorted_idx dump {sorted_idx_path}, stopping")
            break
        order, sgd_subset_size = read_sorted_idx_dump(sorted_idx_path)
        n_full = ds.n_images
        if sgd_subset_size > 0 and sgd_subset_size < n_full:
            subset_ids = order[:sgd_subset_size]
        else:
            subset_ids = order  # full dataset in RELION's shuffled order
        ds_subset = ds.subset(np.asarray(subset_ids, dtype=np.int64))
        # RELION's pseudo_halfsets assigns halfset = (sorted_index % 2) — i.e.,
        # alternates particles in the SHUFFLED order. We pass `micrograph_names=None`
        # to `_split_halfset_particle_ids` to use natural-order [0::2]/[1::2] on
        # the subset (which IS the shuffled order at this point), matching RELION.
        mic_names_subset = None
        print(
            f"  iter {it}: subset={len(subset_ids)}/{n_full}  stepsize={grad_stepsize:.4f}  current_size={current_size}",
            flush=True,
        )

        t0 = time.time()
        try:
            iref_next, Igrad1, Igrad2, stats = run_iter_gpu_vdam(
                ds_subset,
                iref_curr,
                sigma2,
                rotations,
                translations,
                current_size=current_size,
                iter=it,
                Igrad1=Igrad1,
                Igrad2=Igrad2,
                grad_stepsize=grad_stepsize,
                image_batch_size=50,
                rotation_block_size=100,
                half_spectrum_scoring=True,
                padding_factor=1,
                pseudo_halfsets=True,
                apply_gridding_correction=True,
                iref_ft_scale=1.0 / (ori**2),
                iref_ft_sign=-1.0,
                score_with_masked_images=True,
                relion_firstiter_score_mode="gaussian",
                sigma_offset_Ang=sigma_offset_Ang,
                accumulate_noise=False,
                sparse_pass2=True,
                micrograph_names=mic_names_subset,
            )
        except Exception as e:
            print(f"  iter {it}: ERROR {e}")
            import traceback

            traceback.print_exc()
            break
        t = time.time() - t0
        iref_curr = np.asarray(iref_next)

        if it in relion_irefs:
            relion_iref = relion_irefs[it]
            cc_full = cc(iref_curr, relion_iref)
            norm_ratio = float(np.linalg.norm(iref_curr) / np.linalg.norm(relion_iref))
            f = fsc_shells(iref_curr, relion_iref)
            sh05 = fsc_threshold_shell(f, 0.5)
            sh143 = fsc_threshold_shell(f, 0.143)
            # RELION's self-FSC for reference shape: iter k vs iter k-1 in RELION's trajectory
            relion_prev = relion_irefs.get(it - 1, iref_real if it == 1 else None)
            rfsc = fsc_shells(relion_iref, relion_prev) if relion_prev is not None else f
            rsh05 = fsc_threshold_shell(rfsc, 0.5)
            rsh143 = fsc_threshold_shell(rfsc, 0.143)
            line = f"{it:3d}  CC={cc_full:+.4f}  norm_ratio={norm_ratio:.3f}  recovar_FSC@0.5={sh05}  @0.143={sh143}  RELION_inc_FSC@0.5={rsh05}  @0.143={rsh143}  t={t:.1f}s"
            print(line)
            log_f.write(f"{it} {cc_full:.6f} {norm_ratio:.6f} {sh05} {sh143} {rsh05} {rsh143} {t:.3f}\n")
            log_f.flush()
            rows.append((it, cc_full, norm_ratio, sh05, sh143))
            # Save iter-N iref for inspection
            np.save(out_dir / f"recovar_iref_it{it:03d}.npy", iref_curr)
        else:
            print(f"  iter {it}: no RELION reference, t={t:.1f}s")
            log_f.write(f"{it} - - - - - - {t:.3f}\n")
            log_f.flush()

    log_f.close()
    print(f"\nLog saved to {log_path}")


if __name__ == "__main__":
    main()
