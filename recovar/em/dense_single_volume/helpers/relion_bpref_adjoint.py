"""Bit-exact RELION BPref accumulation via the relion_bind C++ binding.

The K-class M-step's adjoint (recovar.cuda_backproject.backproject called
through adjoint_slice_volume_half) produces a +0.74 BPref CC ceiling vs
RELION's BackProjector::set2DFourierTransform. Per the prior diagnostic
in `project_initial_model_estep_diag_2026_04_28`, the per-pose stencil
itself is bit-exact (single-pose comparison shows |data|.max identical),
but multi-particle accumulation diverges due to subtle weight-scaling /
Hermitian / interpolation differences.

This helper bypasses recovar's adjoint and uses
`bind.get_backprojector_data` which calls RELION's exact
`bp.set2DFourierTransform()` per image.

Usage: env-gated through `RECOVAR_BPREF_VIA_BINDING=1` in
`sparse_pass2_bucketed.py`.
"""

from __future__ import annotations

import numpy as np


def accumulate_bpref_via_binding(
    summed_flat: np.ndarray,  # (B*R, n_recon) complex
    ctf_probs_flat: np.ndarray,  # (B*R, n_recon) real
    rotations: np.ndarray,  # (B*R, 3, 3) real
    recon_window_indices: np.ndarray,  # (n_recon,) int — flat indices into (N, N//2+1)
    image_shape: tuple,  # (N, N)
) -> tuple[np.ndarray, np.ndarray]:
    """Reproduce one bucket's adjoint contribution via RELION's binding.

    Returns BPref-layout slabs (data, weight) that can be EMBEDDED into
    recovar's centered (N, N, N//2+1) accumulator via `bpref_to_run_em_output`.

    Parameters
    ----------
    summed_flat : (B*R, n_recon) complex
        Per-pose posterior-weighted Fimg×CTF/σ² values at recon_window_indices.
        Used as the IMAGE input to the binding (becomes the bp.data accumulator).
    ctf_probs_flat : (B*R, n_recon) real
        Per-pose posterior-weighted CTF²/σ² values. Used as a SECOND binding
        call's image (so its output bp.data == sum(ctf_probs at voxel) = Ft_ctf).
    rotations : (B*R, 3, 3) real
        Per-pose rotation matrices (in projector frame).
    recon_window_indices : (n_recon,) int
        Flat indices into (N, N//2+1) selecting the windowed pixels.
    image_shape : (N, N)
        Full half-image dimensions.

    Returns
    -------
    bp_data_slab : (pad, pad, pad//2+1) complex
        RELION BPref data slab from accumulating summed.
    bp_weight_slab : (pad, pad, pad//2+1) real
        RELION BPref data slab from accumulating ctf_probs (= Ft_ctf).
    """
    from recovar.relion_bind._relion_bind_core import get_backprojector_data

    n_pose = int(summed_flat.shape[0])
    if rotations.shape[0] != n_pose:
        raise ValueError(f"rotations[0]={rotations.shape[0]} != n_pose={n_pose}")
    N = int(image_shape[0])
    if image_shape[0] != image_shape[1]:
        raise ValueError(f"square image required, got {image_shape}")
    nx_half = N // 2 + 1

    # Scatter windowed slices into full (n_pose, N, nx_half) 2D rfft layout.
    images = np.zeros((n_pose, N, nx_half), dtype=np.complex128)
    images.reshape(n_pose, -1)[:, recon_window_indices] = np.asarray(summed_flat, dtype=np.complex128)

    weights_for_ft_y = np.ones((n_pose, N, nx_half), dtype=np.float64)

    rotations_d = np.ascontiguousarray(np.asarray(rotations, dtype=np.float64))

    # Call 1: compute Ft_y = sum over poses of (image_at_pose) projected to 3D
    # bp.data += img × Mweight (Mweight=1 here) → bp.data = sum(images at voxel) = Ft_y
    data_y, _ = get_backprojector_data(images, rotations_d, weights_for_ft_y, N, 1, 1)

    # Call 2: compute Ft_ctf = sum over poses of (ctf_probs_at_pose)
    images_ctf = np.zeros((n_pose, N, nx_half), dtype=np.complex128)
    images_ctf.reshape(n_pose, -1)[:, recon_window_indices] = np.asarray(ctf_probs_flat, dtype=np.float64).astype(
        np.complex128
    )
    data_ctf, _ = get_backprojector_data(images_ctf, rotations_d, weights_for_ft_y, N, 1, 1)

    return data_y.copy(), data_ctf.real.copy()


def embed_bpref_slab_into_centered(
    bp_data: np.ndarray,
    bp_weight: np.ndarray,
    target_n: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Center-embed a (pad, pad, pad//2+1) RELION BPref slab into recovar's
    (N, N, N//2+1) centered Fourier layout. RELION's pad_size is typically
    N+3 (one boundary pixel each side); the boundary contribution is dropped.
    """
    pad = int(bp_data.shape[0])
    if bp_data.shape != bp_weight.shape:
        raise ValueError(f"data/weight shape mismatch: {bp_data.shape} vs {bp_weight.shape}")
    if bp_data.shape[1] != pad or bp_data.shape[2] != pad // 2 + 1:
        raise ValueError(f"unexpected BPref slab shape {bp_data.shape}")
    if pad < target_n:
        raise ValueError(f"pad_size={pad} < target_n={target_n}; cannot embed")
    out_data = np.zeros((target_n, target_n, target_n // 2 + 1), dtype=bp_data.dtype)
    out_weight = np.zeros((target_n, target_n, target_n // 2 + 1), dtype=bp_weight.dtype)
    # RELION's BPref slab is centered: index pad_size//2 corresponds to k=0.
    # Recovar's centered (N, N, N//2+1) has index N//2 = k=0 on the y/z axes.
    # The x axis is half-rfft: index 0 = k=0.
    pad_center = pad // 2
    target_center = target_n // 2
    half_target = target_n // 2  # number of voxels each side of center on y/z
    # On x: take indices [0 : target_n//2 + 1]
    nx_target = target_n // 2 + 1
    # Crop bp slab to match. Take central (target_n) on y, z and (nx_target) on x.
    pad_y_start = pad_center - target_center
    pad_y_end = pad_y_start + target_n
    pad_z_start = pad_center - target_center
    pad_z_end = pad_z_start + target_n
    out_data[:, :, :nx_target] = bp_data[pad_z_start:pad_z_end, pad_y_start:pad_y_end, :nx_target]
    out_weight[:, :, :nx_target] = bp_weight[pad_z_start:pad_z_end, pad_y_start:pad_y_end, :nx_target]
    return out_data, out_weight
