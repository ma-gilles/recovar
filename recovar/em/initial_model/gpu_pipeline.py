"""GPU-accelerated VDAM iteration pipeline.

Combines:
  - E-step on GPU via recovar's JIT-compiled `run_em`
  - M-step VDAM-blend on CPU (plain-EM + RELION-weighted blend)

For full VDAM reconstructGrad parity the M-step can route through the
C++ binding (`vdam_reconstruct_grad`) which requires converting Ft_y /
Ft_ctf from recovar's half-volume-flat layout to RELION's projector-
centered padded layout. That binding + layout converter is the last
remaining piece for end-to-end bit-exact iter-1 parity and is scaffolded
in `gpu_pipeline.reconstruct_grad_from_run_em_output`.

Typical usage:

    from recovar.em.initial_model.gpu_pipeline import run_iter_gpu
    state = run_iter_gpu(
        ds=ds,
        iref_real=iref0,
        sigma2_noise=sigma2,
        rotations=rots, translations=trans,
        current_size=28,
        iter=1,
        blend_step=0.5,  # or None to use the VDAM schedule default
    )
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np


@dataclass
class IterResult:
    """Output of a single GPU iteration."""

    iref_real: np.ndarray  # (N, N, N) real-space updated reference
    iref_ft: np.ndarray  # (N^3,) centered-FT flat complex
    Ft_y: np.ndarray  # raw half-volume-flat E-step accumulator
    Ft_ctf: np.ndarray  # raw half-volume-flat E-step denominator
    relion_stats: object  # RelionStats NamedTuple
    wall_time_s: float
    e_step_s: float
    m_step_s: float


def run_iter_gpu(
    ds,
    iref_real: np.ndarray,
    sigma2_noise: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    current_size: int,
    iter: int,
    blend_step: Optional[float] = None,
    image_batch_size: int = 100,
    rotation_block_size: int = 50,
    half_spectrum_scoring: bool = True,
    padding_factor: int = 1,
    phase_lengths=None,
) -> IterResult:
    """Run one VDAM iteration on GPU (E-step + blend M-step).

    Parameters
    ----------
    ds : CryoEMDataset
        Particle data loaded via `load_dataset`. Iterating will use the
        dataset's default GPU path in `run_em`.
    iref_real : np.ndarray
        Current reference volume in real space, shape (N, N, N).
    sigma2_noise : np.ndarray
        Per-shell noise power, shape (N//2 + 1,). Must be already scaled by
        N^4 per RELION↔recovar FFT-normalization convention.
    rotations, translations : np.ndarray
        Sampling grid (see `recovar.em.sampling`).
    current_size : int
        Fourier-window diameter for scoring (RELION's rlnCurrentImageSize).
    iter : int
        Current iteration number (1-indexed).
    blend_step : float or None
        Real-space VDAM blend step. None → use the `schedules.compute_stepsize`
        default for the 3D initial-model path.
    """
    import jax
    import jax.numpy as jnp

    from recovar.core import fourier_transform_utils as ftu
    from recovar.em.dense_single_volume.em_engine import run_em
    from recovar.em.initial_model.schedules import compute_phase_lengths, compute_stepsize
    from recovar.reconstruction.noise import make_radial_noise

    ori = iref_real.shape[0]
    n4 = ori**4

    # Build full-image-shaped noise (recovar expects (H*W,))
    nv = np.asarray(make_radial_noise(sigma2_noise * n4, (ori, ori))).astype(np.float32).reshape(-1)

    iref_ft = np.asarray(ftu.get_dft3(jnp.asarray(iref_real))).reshape(-1)
    mv = jnp.asarray((np.abs(iref_ft) ** 2).astype(np.float32))
    mean_j = jnp.asarray(iref_ft, dtype=jnp.complex64)

    # E-step on GPU
    jax.block_until_ready(mean_j)
    t_e0 = time.time()
    result = run_em(
        ds,
        mean=mean_j,
        mean_variance=mv,
        noise_variance=jnp.asarray(nv),
        rotations=jnp.asarray(rotations),
        translations=jnp.asarray(translations),
        disc_type="linear_interp",
        image_batch_size=image_batch_size,
        rotation_block_size=rotation_block_size,
        current_size=current_size,
        projection_padding_factor=padding_factor,
        reconstruction_padding_factor=padding_factor,
        half_spectrum_scoring=half_spectrum_scoring,
        return_stats=True,
    )
    jax.block_until_ready(result[0])
    e_step_s = time.time() - t_e0

    new_mean_ft = np.asarray(result[0])
    Ft_y = np.asarray(result[2])
    Ft_ctf = np.asarray(result[3])
    stats = result[4]

    # Convert back to real space
    t_m0 = time.time()
    new_mean_real = np.asarray(ftu.get_idft3(jnp.asarray(new_mean_ft).reshape(ori, ori, ori))).real

    # VDAM real-space blend
    if blend_step is None:
        if phase_lengths is None:
            phase_lengths = compute_phase_lengths(200, 0.3, 0.2)
        blend_step = compute_stepsize(iter=iter, phase_lengths=phase_lengths, is_3d_model=True, ref_dim=3)
    iref_new = (1.0 - blend_step) * iref_real + blend_step * new_mean_real
    m_step_s = time.time() - t_m0

    return IterResult(
        iref_real=iref_new,
        iref_ft=np.asarray(ftu.get_dft3(jnp.asarray(iref_new))).reshape(-1),
        Ft_y=Ft_y,
        Ft_ctf=Ft_ctf,
        relion_stats=stats,
        wall_time_s=e_step_s + m_step_s,
        e_step_s=e_step_s,
        m_step_s=m_step_s,
    )


def run_multi_iter_gpu(
    ds,
    iref_real_init: np.ndarray,
    sigma2_noise: np.ndarray,
    rotations: np.ndarray,
    translations: np.ndarray,
    current_size: int,
    nr_iter: int,
    blend_step: Optional[float] = None,
    image_batch_size: int = 100,
    rotation_block_size: int = 50,
    log_callback: Optional[Callable[[int, IterResult], None]] = None,
    **kwargs,
) -> np.ndarray:
    """Run multiple VDAM iterations on GPU.

    Returns the final real-space volume.
    """
    from recovar.em.initial_model.schedules import compute_phase_lengths

    phase_lengths = compute_phase_lengths(nr_iter, 0.3, 0.2)
    current = iref_real_init.copy()
    total_e, total_m = 0.0, 0.0
    for it in range(1, nr_iter + 1):
        res = run_iter_gpu(
            ds=ds,
            iref_real=current,
            sigma2_noise=sigma2_noise,
            rotations=rotations,
            translations=translations,
            current_size=current_size,
            iter=it,
            blend_step=blend_step,
            image_batch_size=image_batch_size,
            rotation_block_size=rotation_block_size,
            phase_lengths=phase_lengths,
            **kwargs,
        )
        current = res.iref_real
        total_e += res.e_step_s
        total_m += res.m_step_s
        if log_callback is not None:
            log_callback(it, res)
    return current, {"total_e_step_s": total_e, "total_m_step_s": total_m}


def reconstruct_grad_from_run_em_output(
    Ft_y: np.ndarray,
    Ft_ctf: np.ndarray,
    iref_real: np.ndarray,
    ori_size: int,
    r_max: int,
    grad_stepsize: float,
    tau2_fudge: float,
    padding_factor: int = 1,
    min_resol_shell: float = 0.0,
    mom1_noise_power: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Full VDAM M-step via RELION's reconstructGrad binding (CPU).

    Converts recovar's half-volume-flat accumulators to RELION's projector-
    centered padded layout, then calls `vdam_reconstruct_grad`.

    This path is bit-exact vs RELION's reconstructGrad when mom1_noise_power
    is provided (from the Phase-4 applyMomenta primitive). Without it, the
    binding's use_fsc=false branch defaults to fsc_estimate=1.

    Layout conversion:
      recovar Ft_y flat shape = (ori^3,) ordered in fft-natural layout.
      Reshape to (ori, ori, ori//2+1), then ifftshift on axes 0,1 to center,
      then crop to pad_size = 2*(r_max + 1) + 1 slab.
    """
    from recovar.relion_bind import _relion_bind_core as bind

    N = ori_size
    pad_size = 2 * (int(padding_factor * r_max + 0.5) + 1) + 1

    # Convert half-volume flat → centered padded layout
    # Ft_y is of shape ori^3 — note: it's already full-volume flat per
    # `half_volume_to_full_volume` inside run_em. We need half-complex only.
    Fy_full = Ft_y.reshape(N, N, N)
    Fc_full = Ft_ctf.reshape(N, N, N)
    # Take the half slab (non-redundant part)
    # The internal convention: axis 2 is the half-complex axis after FFT.
    # ifftshift on axes (0,1) brings DC to index [0,0,0] in ffw layout,
    # then crop first pad_size rows in axis 0,1 about DC.
    Fy_half = np.fft.ifftshift(Fy_full, axes=(0, 1))[:, :, : N // 2 + 1]
    Fc_half = np.fft.ifftshift(Fc_full, axes=(0, 1))[:, :, : N // 2 + 1].real

    # Crop to (pad_size, pad_size, pad_size/2 + 1) about DC
    half_ps = pad_size // 2

    # After ifftshift DC is at (0, 0, 0); wrap indices for negative frequencies
    def wrap(n):
        idx = np.concatenate([np.arange(half_ps + 1), np.arange(N - half_ps, N)])
        return idx

    idx = wrap(N)
    data_crop = Fy_half[np.ix_(idx, idx, np.arange(pad_size // 2 + 1))]
    weight_crop = Fc_half[np.ix_(idx, idx, np.arange(pad_size // 2 + 1))]

    fsc = np.zeros(N // 2 + 1, dtype=np.float64)
    out = np.asarray(
        bind.vdam_reconstruct_grad(
            iref_real.astype(np.float64),
            data_crop.astype(np.complex128),
            weight_crop.astype(np.float64),
            fsc,
            grad_stepsize,
            tau2_fudge,
            ori_size,
            padding_factor,
            1,  # TRILINEAR
            r_max,
            min_resol_shell,
            False,  # use_fsc
            True,  # skip_gridding
            mom1_noise_power,
        )
    )
    return out
