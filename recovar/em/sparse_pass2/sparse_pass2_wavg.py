"""RELION wavg rectangle and low-shell power terms of the sparse bucketed pass 2.

The wavg rectangle that reproduces RELION's per-image weighted-average
normalization terms (atomic, sequential and direct modes), the weighted
image power shells and the low-shell noise/norm replacements derived from
them. ``sparse_pass2_bucketed`` builds the rectangle per bucket.
"""

from __future__ import annotations

from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.env_flags import parse_env_flag
from recovar.em.helpers.fourier_window import make_fourier_window_indices_np, relion_fftw_order_for_square_score_window
from recovar.em.helpers.half_spectrum import bin_shell_values_jax, make_relion_noise_shell_indices_half
from recovar.em.sparse_pass2.sparse_pass2_policy import _RELION_POWERCLASS_SPECTRUM_NORM_ENV

_RELION_WAVG_SEQUENTIAL_CUDA_ENV = "RECOVAR_K1_RELION_WAVG_SEQUENTIAL_CUDA"


class RelionWavgRectangle(NamedTuple):
    """Static mapping for RELION's full cropped Wavg CUDA pixel stream."""

    centered_indices: np.ndarray
    exact_positions: np.ndarray
    shell_indices: np.ndarray


def _weighted_image_power_shells_and_per_image(
    processed_half,
    shell_indices_half,
    support_mass,
    *,
    shell_count: int,
    norm_unweighted_shell_cutoff: int | None = None,
    norm_unweighted_high_shell=None,
    include_unweighted_high_shell: bool = True,
    valid_image_mask=None,
    source_faithful_spectrum_norm: bool | None = None,
):
    """Accumulate image power for noise shells and per-image norm correction.

    Resolves the environment-controlled reduction policy on the host and runs
    the arithmetic in :func:`_weighted_image_power_shells_and_per_image_core`.
    """

    if source_faithful_spectrum_norm is None:
        source_faithful_spectrum_norm = parse_env_flag(
            _RELION_POWERCLASS_SPECTRUM_NORM_ENV,
            default=False,
        )
    source_faithful_spectrum_norm = bool(source_faithful_spectrum_norm)
    deterministic_norm_reduction = source_faithful_spectrum_norm or parse_env_flag(
        "RECOVAR_K1_RELION_DETERMINISTIC_NORM_REDUCTION",
        default=False,
    )
    return _weighted_image_power_shells_and_per_image_core(
        processed_half,
        shell_indices_half,
        support_mass,
        norm_unweighted_high_shell,
        valid_image_mask,
        shell_count=int(shell_count),
        norm_unweighted_shell_cutoff=(
            None if norm_unweighted_shell_cutoff is None else int(norm_unweighted_shell_cutoff)
        ),
        include_unweighted_high_shell=bool(include_unweighted_high_shell),
        disable_cuda_binning=parse_env_flag("RECOVAR_DISABLE_CUDA", default=False),
        deterministic_norm_reduction=bool(deterministic_norm_reduction),
    )


@partial(
    jax.jit,
    static_argnames=(
        "shell_count",
        "norm_unweighted_shell_cutoff",
        "include_unweighted_high_shell",
        "disable_cuda_binning",
        "deterministic_norm_reduction",
    ),
)
def _weighted_image_power_shells_and_per_image_core(
    processed_half,
    shell_indices_half,
    support_mass,
    norm_unweighted_high_shell,
    valid_image_mask,
    *,
    shell_count: int,
    norm_unweighted_shell_cutoff: int | None,
    include_unweighted_high_shell: bool,
    disable_cuda_binning: bool,
    deterministic_norm_reduction: bool,
):
    """Accumulate image power for noise shells and per-image norm correction.

    Inside the current model size, shell sums use the same significant-support
    mass as the A2/XA residual terms.  Above it, RELION adds ``power_img`` once
    per particle outside the class/posterior loop, so both the noise spectrum
    and norm-correction tail are unweighted.  Fourier pixels assigned to the
    shell-binning sentinel are excluded.
    """

    pixel_power = jnp.abs(processed_half) ** 2
    mass = jnp.asarray(support_mass, dtype=pixel_power.dtype)
    shell_indices_half = jnp.asarray(shell_indices_half)
    valid_norm_shell = (shell_indices_half >= 0) & (shell_indices_half < int(shell_count))
    shell_mass = jnp.where(valid_norm_shell[None, :], mass[:, None], 0.0)
    norm_mass = jnp.where(valid_norm_shell[None, :], mass[:, None], 0.0)
    if norm_unweighted_shell_cutoff is not None:
        full_mass = (
            jnp.ones_like(mass)
            if valid_image_mask is None
            else jnp.asarray(valid_image_mask, dtype=pixel_power.dtype)
        )
        unweighted_shell = valid_norm_shell & (shell_indices_half > int(norm_unweighted_shell_cutoff))
        high_shell_mass = full_mass if include_unweighted_high_shell else jnp.zeros_like(full_mass)
        shell_mass = jnp.where(unweighted_shell[None, :], high_shell_mass[:, None], shell_mass)
        norm_mass = jnp.where(unweighted_shell[None, :], high_shell_mass[:, None], norm_mass)
    weighted_pixel_power = pixel_power * shell_mass
    # Keep the image-power reduction in the producer precision.  RELION's
    # RFLOAT path is binary64 for the double-precision oracle, and narrowing
    # here perturbs both the shell noise statistics and the per-image norm
    # correction before either is accumulated into the host float64 totals.
    weighted_half = jnp.sum(weighted_pixel_power, axis=0)
    if disable_cuda_binning:
        # ``bins.at[indices].add`` lowers to unordered GPU atomics even when
        # RECOVAR's custom CUDA path is explicitly disabled.  Keep this
        # fallback repeatable by reducing each shell independently instead.
        shell_ids = jnp.arange(int(shell_count), dtype=shell_indices_half.dtype)
        weighted_shells = jnp.sum(
            jnp.where(
                shell_indices_half[None, :] == shell_ids[:, None],
                weighted_half[None, :],
                0.0,
            ),
            axis=1,
        )
    else:
        weighted_shells = bin_shell_values_jax(weighted_half, shell_indices_half, shell_count)
    norm_reduction_dtype = jnp.float64 if deterministic_norm_reduction else pixel_power.dtype
    weighted_per_image = jnp.sum(
        (pixel_power * norm_mass).astype(norm_reduction_dtype),
        axis=-1,
    )
    if norm_unweighted_high_shell is not None and include_unweighted_high_shell:
        if norm_unweighted_shell_cutoff is None:
            raise ValueError("a replacement high-shell norm term requires a shell cutoff")
        replacement_high = jnp.asarray(norm_unweighted_high_shell, dtype=norm_reduction_dtype)
        if replacement_high.shape != mass.shape:
            raise ValueError(
                "replacement high-shell norm term must match the particle axis, got "
                f"{replacement_high.shape} for {mass.shape}"
            )
        generic_high = jnp.sum(
            jnp.where(unweighted_shell[None, :], pixel_power, 0.0).astype(
                norm_reduction_dtype
            ),
            axis=-1,
        )
        # Preserve the current-size norm path and the separate shell/noise
        # reduction. Replace only the unweighted high-shell norm term
        # with RELION powerClass's divide-before-square float32 arithmetic.
        weighted_per_image = jax.lax.optimization_barrier(weighted_per_image)
        weighted_per_image = weighted_per_image + full_mass * (replacement_high - generic_high)
    return weighted_shells, weighted_per_image.astype(norm_reduction_dtype)


def _make_relion_wavg_rectangle(
    image_shape,
    current_size,
    recon_window_indices,
    *,
    reconstruction_current_size=None,
):
    """Map active reconstruction pixels into RELION's complete Wavg crop.

    ``exact_positions`` retains its historical field name, but may describe
    either the exact BackProjector disk or RELION InitialModel's rounded-shell
    Wavg support. The supplied reconstruction indices select that contract.

    RELION may remap the particle-image ``current_size`` for an optics group
    while retaining the model-coordinate radius for Projector/BackProjector.
    The rectangle and its rounded noise-shell mask therefore use the particle
    size, while the exact projected terms use ``reconstruction_current_size``.
    """

    image_shape = tuple(int(value) for value in image_shape)
    current_size = int(current_size)
    model_current_size = (
        current_size
        if reconstruction_current_size is None
        else int(reconstruction_current_size)
    )
    if model_current_size > current_size:
        raise ValueError(
            "RELION Wavg model support cannot exceed the particle-image crop: "
            f"model={model_current_size}, image={current_size}"
        )
    rectangle_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        square=True,
        include_dc=True,
    )
    rectangle_order = relion_fftw_order_for_square_score_window(
        image_shape,
        current_size,
        rectangle_indices,
    )
    rectangle_indices = rectangle_indices[rectangle_order]
    rounded_indices, _ = make_fourier_window_indices_np(
        image_shape,
        current_size,
        include_dc=True,
        exact_radius=False,
    )
    exact_indices, _ = make_fourier_window_indices_np(
        image_shape,
        model_current_size,
        include_dc=True,
        exact_radius=True,
    )
    recon_indices = np.asarray(recon_window_indices, dtype=np.int32).reshape(-1)
    exact_support = np.array_equal(np.sort(recon_indices), exact_indices)
    rounded_support = np.array_equal(np.sort(recon_indices), rounded_indices)
    if not (exact_support or rounded_support):
        raise ValueError(
            "RELION Wavg rectangle requires a complete exact-radius or rounded-shell "
            "reconstruction window: "
            f"got {recon_indices.size} pixels, expected {exact_indices.size} or "
            f"{rounded_indices.size}"
        )

    rectangle_position = {
        int(centered_index): position
        for position, centered_index in enumerate(rectangle_indices.tolist())
    }
    try:
        exact_positions = np.asarray(
            [rectangle_position[int(index)] for index in recon_indices],
            dtype=np.int32,
        )
        rounded_positions = np.asarray(
            [rectangle_position[int(index)] for index in rounded_indices],
            dtype=np.int32,
        )
    except KeyError as error:
        raise ValueError("RELION Wavg support is not contained in its square crop") from error

    shell_indices_half = np.asarray(
        make_relion_noise_shell_indices_half(image_shape),
        dtype=np.int32,
    )
    rectangle_shells = shell_indices_half[rectangle_indices]
    valid = np.zeros(rectangle_indices.size, dtype=bool)
    valid[rounded_positions] = True
    rectangle_shells = np.where(valid, rectangle_shells, -1).astype(np.int32)
    expected_rectangle_size = current_size * (current_size // 2 + 1)
    if rectangle_indices.size != expected_rectangle_size:
        raise ValueError(
            "RELION Wavg square crop topology changed: "
            f"got {rectangle_indices.size}, expected {expected_rectangle_size}"
        )
    if np.unique(exact_positions).size != exact_positions.size:
        raise ValueError("RELION Wavg reconstruction position mapping is not bijective")
    return RelionWavgRectangle(
        centered_indices=rectangle_indices.astype(np.int32, copy=False),
        exact_positions=exact_positions,
        shell_indices=rectangle_shells,
    )


def _make_stable_relion_wavg_rectangle(image_shape, shape_plan):
    """Pack a logical Wavg rectangle before its physical-capacity tail.

    RELION's Wavg kernel walks the dense FFTW rectangle, whereas RECOVAR's
    shared projection path stores a compact disk.  Stable shapes are exact
    only when both streams retain their original logical order.  The first
    ``logical_rectangle_pixels`` entries below are therefore byte-for-byte the
    ordinary logical rectangle.  Physical-only pixels follow it and receive a
    sentinel shell; runtime CUDA bounds must never issue that tail.
    """

    logical_recon = shape_plan.packed_indices_np("recon")[
        : shape_plan.logical_reconstruction_pixels
    ]
    physical_recon = shape_plan.packed_indices_np("recon")
    logical = _make_relion_wavg_rectangle(
        image_shape,
        shape_plan.logical_current_size,
        logical_recon,
    )
    physical = _make_relion_wavg_rectangle(
        image_shape,
        shape_plan.physical_current_size,
        physical_recon,
    )

    logical_set = set(map(int, logical.centered_indices.tolist()))
    physical_tail = np.asarray(
        [
            int(index)
            for index in physical.centered_indices.tolist()
            if int(index) not in logical_set
        ],
        dtype=np.int32,
    )
    packed_rectangle = np.concatenate(
        (logical.centered_indices, physical_tail),
    ).astype(np.int32, copy=False)
    if packed_rectangle.size != shape_plan.physical_rectangle_pixels:
        raise ValueError(
            "stable RELION Wavg rectangle does not fill its physical capacity: "
            f"got {packed_rectangle.size}, expected {shape_plan.physical_rectangle_pixels}"
        )
    logical_count = shape_plan.logical_rectangle_pixels
    recon_tail_count = (
        shape_plan.physical_reconstruction_pixels
        - shape_plan.logical_reconstruction_pixels
    )
    rectangle_tail_count = packed_rectangle.size - logical_count
    if recon_tail_count > rectangle_tail_count:
        raise ValueError(
            "stable RELION Wavg rectangle tail cannot hold its reconstruction "
            f"tail: recon={recon_tail_count}, rectangle={rectangle_tail_count}"
        )
    # Some pixels newly admitted by the larger physical radius still lie in
    # the *logical* square rectangle (for example, immediately outside its
    # exact-radius disk).  Mapping those pixels by coordinate would make the
    # logical native Wavg/BPref loops consume physical-only values.  Logical
    # reconstruction rows retain their exact FFTW positions; all capacity-only
    # rows instead receive arbitrary unique storage in the inert rectangle
    # tail, whose coordinates are intentionally never issued.
    recon_positions = np.concatenate(
        (
            logical.exact_positions,
            np.arange(
                logical_count,
                logical_count + recon_tail_count,
                dtype=np.int32,
            ),
        )
    ).astype(np.int32, copy=False)
    rectangle_shells = np.full(packed_rectangle.size, -1, dtype=np.int32)
    rectangle_shells[:logical_count] = logical.shell_indices
    return RelionWavgRectangle(
        centered_indices=packed_rectangle,
        exact_positions=recon_positions,
        shell_indices=rectangle_shells,
    )


def _select_optional_wavg_exact_pixels(values, rectangle):
    """Select exact-radius Wavg pixels when the atomic diagnostic is active."""

    if values is None or rectangle is None:
        return None
    return values[:, rectangle.exact_positions]


def _relion_wavg_rectangle_image_power(raw_shifted, posterior):
    """Shared rectangle power contraction; keep the original F32/barrier order."""
    raw_shifted = jnp.asarray(raw_shifted, dtype=jnp.complex64)
    posterior = jnp.asarray(posterior, dtype=jnp.float32)
    shifted_power = (raw_shifted.real * raw_shifted.real).astype(jnp.float32)
    shifted_power = jax.lax.optimization_barrier(shifted_power)
    shifted_power = (shifted_power + raw_shifted.imag * raw_shifted.imag).astype(jnp.float32)
    image_power = jnp.einsum(
        "brt,btp->brp",
        posterior,
        shifted_power,
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    return image_power


@jax.jit
def _relion_wavg_rectangle_triplet_terms(
    exact_triplet_terms,
    raw_shifted_rectangle,
    posterior,
    exact_positions,
):
    """Embed exact projection terms in the full native Wavg issue stream."""

    exact_terms = jnp.asarray(exact_triplet_terms, dtype=jnp.float32)
    raw_shifted = jnp.asarray(raw_shifted_rectangle, dtype=jnp.complex64)
    posterior = jnp.asarray(posterior, dtype=jnp.float32)
    exact_positions = jnp.asarray(exact_positions, dtype=jnp.int32)
    if exact_terms.ndim != 4 or exact_terms.shape[-1] != 3:
        raise ValueError(f"exact Wavg terms must have shape (B,R,P,3), got {exact_terms.shape}")
    if raw_shifted.ndim != 3 or posterior.shape != exact_terms.shape[:2] + (raw_shifted.shape[1],):
        raise ValueError(
            "Wavg rectangle translations/posteriors do not match exact terms: "
            f"terms={exact_terms.shape}, shifted={raw_shifted.shape}, posterior={posterior.shape}"
        )
    if exact_positions.shape != (exact_terms.shape[2],):
        raise ValueError(
            "Wavg exact-position mapping does not match the projected pixel axis: "
            f"positions={exact_positions.shape}, terms={exact_terms.shape}"
        )

    image_power = _relion_wavg_rectangle_image_power(raw_shifted, posterior)
    rectangle_terms = jnp.zeros(
        exact_terms.shape[:2] + (raw_shifted.shape[-1], 3),
        dtype=jnp.float32,
    )
    rectangle_terms = rectangle_terms.at[..., 2].set(image_power)
    return rectangle_terms.at[:, :, exact_positions, :].set(exact_terms)


def _relion_wavg_direct_norm_per_image(
    atomic_diff2_per_pixel,
    shell_indices,
    high_shell_power,
):
    """Reproduce RELION's sequential host norm sum from the direct Wavg buffer."""

    atomic_diff2 = np.asarray(atomic_diff2_per_pixel, dtype=np.float32)
    shells = np.asarray(shell_indices, dtype=np.int32).reshape(-1)
    high_shell = np.asarray(high_shell_power, dtype=np.float64).reshape(-1)
    if atomic_diff2.ndim != 2 or atomic_diff2.shape[1] != shells.size:
        raise ValueError(
            "atomic Wavg diff2 must have shape (images, rectangle pixels), got "
            f"{atomic_diff2.shape} for {shells.shape}"
        )
    if high_shell.shape != (atomic_diff2.shape[0],):
        raise ValueError(
            "RELION high-shell norm term must match the image axis, got "
            f"{high_shell.shape} for {atomic_diff2.shape}"
        )
    valid = shells >= 0
    output = np.zeros(atomic_diff2.shape[0], dtype=np.float64)
    for image_row in range(atomic_diff2.shape[0]):
        current_size_sum = np.float64(0.0)
        for value in atomic_diff2[image_row, valid]:
            current_size_sum += np.float64(value)
        output[image_row] = current_size_sum + high_shell[image_row]
    return output


def _relion_wavg_atomic_triplet_terms(
    proj,
    proj_abs2,
    summed_shifted,
    ctf_posterior,
    noise_variance,
    scale,
    raw_shifted_images,
    posterior,
):
    """Form per-rotation Wavg ``[XA, AA, diff2]`` float32 atomic operands.

    RELION accumulates all three quantities in one CUDA thread after its
    translation loop. XA and AA are returned in scale-correction units;
    diff2 stays in the raw residual units used by ``wsum_sigma2_noise``.
    """

    proj = jnp.asarray(proj, dtype=jnp.complex64)
    proj_abs2 = jnp.asarray(proj_abs2, dtype=jnp.float32)
    summed_shifted = jnp.asarray(summed_shifted, dtype=jnp.complex64)
    ctf_posterior = jnp.asarray(ctf_posterior, dtype=jnp.float32)
    noise_variance = jnp.asarray(noise_variance, dtype=jnp.float32).reshape(-1)
    scale = jnp.asarray(scale, dtype=jnp.float32).reshape(-1)
    raw_shifted_images = jnp.asarray(raw_shifted_images, dtype=jnp.complex64)
    posterior = jnp.asarray(posterior, dtype=jnp.float32)

    ctf_has_mass = ctf_posterior != 0.0
    ctf_posterior_raw = jnp.where(
        ctf_has_mass,
        ctf_posterior * noise_variance[None, None, :],
        0.0,
    )
    aa_raw = jnp.where(ctf_has_mass, proj_abs2 * ctf_posterior_raw, 0.0).astype(
        jnp.float32
    )
    cross_has_mass = summed_shifted != 0.0
    cross = jnp.where(cross_has_mass, proj * jnp.conj(summed_shifted), 0.0)
    xa_raw = (noise_variance[None, None, :] * cross.real).astype(jnp.float32)
    safe_scale = jnp.maximum(scale, jnp.asarray(1e-30, dtype=jnp.float32))
    # Wavg emits all three atomics at every pixel inside current_size. RELION
    # applies its lower-resolution scale-correction cutoff only when the host
    # later consumes XA/AA; masking here changes the CUDA issue stream.
    xa = (xa_raw / safe_scale[:, None, None]).astype(jnp.float32)
    aa = (aa_raw / (safe_scale[:, None, None] ** 2)).astype(jnp.float32)

    # RELION's g_img input is the raw translated preprocessed image, not the
    # CTF/noise-weighted BPref numerator used by RECOVAR's adjoint path.
    shifted_power = (raw_shifted_images.real * raw_shifted_images.real).astype(jnp.float32)
    shifted_power = jax.lax.optimization_barrier(shifted_power)
    shifted_power = (
        shifted_power + raw_shifted_images.imag * raw_shifted_images.imag
    ).astype(jnp.float32)
    image_power = jnp.einsum(
        "brt,btp->brp",
        posterior,
        shifted_power,
        preferred_element_type=jnp.float32,
    ).astype(jnp.float32)
    diff2 = (
        (image_power + aa_raw)
        - jnp.asarray(2.0, dtype=jnp.float32) * xa_raw
    ).astype(jnp.float32)
    return jnp.stack((xa, aa, diff2), axis=-1)


@jax.jit
def _relion_wavg_sequential_triplet_terms_jax(
    proj,
    raw_ctf,
    scale,
    raw_shifted_images,
    posterior,
):
    """Reproduce RELION Wavg's translation-loop float32 accumulators.

    RELION forms the CTF-and-scale-corrected reference once per rotation and
    pixel, then visits translations in storage order.  It accumulates squared
    residual, XA, and AA separately in float32 before issuing the three
    rotation-level atomics.  Forming ``image_power + AA - 2 * XA`` after three
    independent reductions is algebraically equivalent but not numerically
    equivalent at this boundary.

    XA and AA are returned in RELION's host scale-correction units.  The
    residual remains in the raw units consumed by ``wsum_sigma2_noise``.
    """

    proj = jnp.asarray(proj, dtype=jnp.complex64)
    raw_ctf = jnp.asarray(raw_ctf, dtype=jnp.float32)
    scale = jnp.asarray(scale, dtype=jnp.float32).reshape(-1)
    raw_shifted_images = jnp.asarray(raw_shifted_images, dtype=jnp.complex64)
    posterior = jnp.asarray(posterior, dtype=jnp.float32)
    if proj.ndim != 3:
        raise ValueError(f"Wavg projections must have shape (B,R,P), got {proj.shape}")
    if raw_ctf.shape != (proj.shape[0], proj.shape[2]):
        raise ValueError(
            "Wavg raw CTF must match projection batch/pixel axes, got "
            f"{raw_ctf.shape} for {proj.shape}"
        )
    if raw_shifted_images.ndim != 3:
        raise ValueError(
            "Wavg shifted images must have shape (B,T,P), got "
            f"{raw_shifted_images.shape}"
        )
    if raw_shifted_images.shape[0] != proj.shape[0] or raw_shifted_images.shape[2] != proj.shape[2]:
        raise ValueError(
            "Wavg shifted-image batch/pixel axes must match projections, got "
            f"{raw_shifted_images.shape} for {proj.shape}"
        )
    if posterior.shape != proj.shape[:2] + (raw_shifted_images.shape[1],):
        raise ValueError(
            "Wavg posterior must match projection rotations and image translations, got "
            f"{posterior.shape} for proj={proj.shape}, shifted={raw_shifted_images.shape}"
        )

    ctf_with_scale = (raw_ctf * scale[:, None]).astype(jnp.float32)
    ref_real = (proj.real * ctf_with_scale[:, None, :]).astype(jnp.float32)
    ref_imag = (proj.imag * ctf_with_scale[:, None, :]).astype(jnp.float32)
    zeros = jnp.zeros_like(ref_real, dtype=jnp.float32)

    def add_translation(translation_index, accumulators):
        xa_acc, aa_acc, diff2_acc = accumulators
        weight = posterior[:, :, translation_index]
        trans_real = raw_shifted_images[:, translation_index, :].real
        trans_imag = raw_shifted_images[:, translation_index, :].imag
        diff_real = (ref_real - trans_real[:, None, :]).astype(jnp.float32)
        diff_imag = (ref_imag - trans_imag[:, None, :]).astype(jnp.float32)
        diff_abs2 = (diff_real * diff_real).astype(jnp.float32)
        diff_abs2 = jax.lax.optimization_barrier(diff_abs2)
        diff_abs2 = (diff_abs2 + diff_imag * diff_imag).astype(jnp.float32)
        cross = (ref_real * trans_real[:, None, :]).astype(jnp.float32)
        cross = jax.lax.optimization_barrier(cross)
        cross = (cross + ref_imag * trans_imag[:, None, :]).astype(jnp.float32)
        ref_abs2 = (ref_real * ref_real).astype(jnp.float32)
        ref_abs2 = jax.lax.optimization_barrier(ref_abs2)
        ref_abs2 = (ref_abs2 + ref_imag * ref_imag).astype(jnp.float32)
        weighted = weight[:, :, None]
        return (
            (xa_acc + weighted * cross).astype(jnp.float32),
            (aa_acc + weighted * ref_abs2).astype(jnp.float32),
            (diff2_acc + weighted * diff_abs2).astype(jnp.float32),
        )

    xa_raw, aa_raw, diff2 = jax.lax.fori_loop(
        0,
        raw_shifted_images.shape[1],
        add_translation,
        (zeros, zeros, zeros),
    )
    safe_scale = jnp.maximum(scale, jnp.asarray(1e-30, dtype=jnp.float32))
    xa = (xa_raw / safe_scale[:, None, None]).astype(jnp.float32)
    aa = (aa_raw / (safe_scale[:, None, None] ** 2)).astype(jnp.float32)
    return jnp.stack((xa, aa, diff2), axis=-1)


def _relion_wavg_sequential_triplet_terms(
    proj,
    raw_ctf,
    scale,
    raw_shifted_images,
    posterior,
    *,
    relion_wavg_sequential_cuda: bool | None = None,
    logical_pixel_count=None,
):
    """Dispatch the shared Wavg translation-order reduction.

    The CUDA path is an explicit performance discriminator.  It keeps the
    same image/rotation/pixel ownership and sequential translation arithmetic
    as the JAX reference while avoiding one XLA loop-body launch per
    translation.  Both local EM and VDAM reach this helper through the shared
    exact-local pass-2 implementation.  A typed policy overrides the legacy
    environment gate; ``None`` preserves its existing behavior.
    """

    use_cuda = (
        parse_env_flag(_RELION_WAVG_SEQUENTIAL_CUDA_ENV, default=False)
        if relion_wavg_sequential_cuda is None
        else bool(relion_wavg_sequential_cuda)
    )
    if use_cuda:
        from recovar import cuda_backproject

        if logical_pixel_count is not None:
            return cuda_backproject.relion_wavg_sequential_runtime_triplet_f32(
                jnp.asarray(proj, dtype=jnp.complex64),
                jnp.asarray(raw_ctf, dtype=jnp.float32),
                jnp.asarray(scale, dtype=jnp.float32).reshape(-1),
                jnp.asarray(raw_shifted_images, dtype=jnp.complex64),
                jnp.asarray(posterior, dtype=jnp.float32),
                jnp.asarray(logical_pixel_count, dtype=jnp.int32),
            )
        return cuda_backproject.relion_wavg_sequential_triplet_f32(
            jnp.asarray(proj, dtype=jnp.complex64),
            jnp.asarray(raw_ctf, dtype=jnp.float32),
            jnp.asarray(scale, dtype=jnp.float32).reshape(-1),
            jnp.asarray(raw_shifted_images, dtype=jnp.complex64),
            jnp.asarray(posterior, dtype=jnp.float32),
        )
    if logical_pixel_count is not None:
        raise ValueError(
            "stable Fourier-window Wavg requires the runtime-bound CUDA reducer"
        )
    return _relion_wavg_sequential_triplet_terms_jax(
        proj,
        raw_ctf,
        scale,
        raw_shifted_images,
        posterior,
    )


def _replace_low_shell_noise_with_relion_wavg_direct_residual(
    residual_shells,
    image_power_shells,
    atomic_diff2_per_pixel,
    shell_indices,
    *,
    exclusive_shell_stop: int,
):
    """Replace complete low shells with fused Wavg ``diff2`` atomics.

    The fused value already contains image power, A2, and -2*XA.  Therefore
    its covered shells replace both RECOVAR noise-stat components.
    ``exclusive_shell_stop`` is expressed in shell-number coordinates. The
    caller supplies RELION's complete rectangular Wavg buffer, so the rounded
    cutoff shell is replaced in full even though the reconstruction window
    contains only its exact-radius subset.
    """

    residual = np.asarray(residual_shells, dtype=np.float64).copy()
    image_power = np.asarray(image_power_shells, dtype=np.float64).copy()
    atomic_diff2 = np.asarray(atomic_diff2_per_pixel, dtype=np.float32)
    shells = np.asarray(shell_indices, dtype=np.int32).reshape(-1)
    if residual.ndim != 1 or image_power.shape != residual.shape:
        raise ValueError(
            "noise residual and image-power shells must be matching vectors, got "
            f"{residual.shape} and {image_power.shape}"
        )
    if atomic_diff2.ndim != 2 or atomic_diff2.shape[1] != shells.size:
        raise ValueError(
            "atomic Wavg diff2 must have shape (images, pixels) matching shell indices, got "
            f"{atomic_diff2.shape} and {shells.shape}"
        )
    shell_stop = min(max(0, int(exclusive_shell_stop)), residual.size)
    valid = (shells >= 0) & (shells < shell_stop)
    direct_shells = np.zeros_like(residual)
    if np.any(valid):
        # Preserve physical particle order, then reconstruction-window pixel
        # order.  The per-pixel values have already undergone RELION-style
        # float32 rotation atomics on device.
        for image_row in range(atomic_diff2.shape[0]):
            np.add.at(
                direct_shells,
                shells[valid],
                atomic_diff2[image_row, valid].astype(np.float64),
            )
    residual[:shell_stop] = direct_shells[:shell_stop]
    image_power[:shell_stop] = 0.0
    return residual, image_power


@jax.jit
def _translated_wavg_low_shell_power_pixels(
    shifted_score,
    translation_posterior,
    shell_indices,
    shell_cutoff,
):
    """Return RELION-Wavg-style low-shell image-power pixels per image.

    RELION forms ``wdiff2`` after translating each image and preserves one
    float32 accumulator per Fourier pixel until the host-side normalization
    sum.  Computing image power from the untranslated image is algebraically
    equivalent only in exact arithmetic; the CUDA translation phase makes the
    distinction observable in float32.  Keep the per-pixel boundary here so
    callers can reproduce RELION's host float64/RFLOAT summation order.
    """

    shifted_score = jnp.asarray(shifted_score, dtype=jnp.complex64)
    translation_posterior = jnp.asarray(translation_posterior, dtype=jnp.float32)
    shell_indices = jnp.asarray(shell_indices, dtype=jnp.int32)
    if shifted_score.ndim != 3:
        raise ValueError(f"translated Wavg images must have shape (B,T,P), got {shifted_score.shape}")
    if translation_posterior.shape != shifted_score.shape[:2]:
        raise ValueError(
            "translation posterior must match translated Wavg batch/translation axes, got "
            f"{translation_posterior.shape} for {shifted_score.shape}"
        )
    if shell_indices.shape != (shifted_score.shape[-1],):
        raise ValueError(
            "translated Wavg shell indices must match the pixel axis, got "
            f"{shell_indices.shape} for {shifted_score.shape}"
        )

    pixel_power = shifted_score.real * shifted_score.real
    pixel_power = jax.lax.optimization_barrier(pixel_power)
    pixel_power = pixel_power + shifted_score.imag * shifted_score.imag
    weighted_pixels = jnp.sum(
        translation_posterior[:, :, None] * pixel_power,
        axis=1,
        dtype=jnp.float32,
    )
    valid_low_shell = (shell_indices >= 0) & (shell_indices <= jnp.asarray(shell_cutoff))
    return jnp.where(valid_low_shell[None, :], weighted_pixels, 0.0).astype(jnp.float32)


def _relion_cuda_translate_wavg_norm_images(
    processed_score_half,
    translation_angles,
    score_window_indices,
    image_shape,
):
    """Translate the raw masked image at RELION's Wavg input boundary."""

    from recovar import cuda_backproject

    processed_score_half = jnp.asarray(processed_score_half, dtype=jnp.complex64)
    score_window_indices = jnp.asarray(score_window_indices, dtype=jnp.int32)
    translation_angles = jnp.asarray(translation_angles, dtype=jnp.float32)
    translated = cuda_backproject.relion_translate_score_f32(
        processed_score_half[:, score_window_indices],
        translation_angles,
        score_window_indices,
        image_shape,
    )
    return translated.reshape(
        processed_score_half.shape[0],
        translation_angles.shape[0],
        score_window_indices.shape[0],
    )


def _replace_untranslated_low_shell_norm_power(
    weighted_img_per_image,
    processed_score_half,
    shifted_score,
    translation_posterior,
    shell_indices_half,
    score_window_indices,
    *,
    shell_cutoff: int,
):
    """Replace RECOVAR's untranslated low-shell norm power with Wavg power."""

    score_window_indices = jnp.asarray(score_window_indices, dtype=jnp.int32)
    window_shell_indices = jnp.asarray(shell_indices_half, dtype=jnp.int32)[score_window_indices]
    shifted_score = jnp.asarray(shifted_score, dtype=jnp.complex64)
    translated_pixels = _translated_wavg_low_shell_power_pixels(
        shifted_score,
        translation_posterior,
        window_shell_indices,
        jnp.asarray(shell_cutoff, dtype=jnp.int32),
    )

    processed_window = jnp.asarray(processed_score_half, dtype=jnp.complex64)[:, score_window_indices]
    untranslated_power = jnp.abs(processed_window) ** 2
    support_mass = jnp.sum(
        jnp.asarray(translation_posterior, dtype=jnp.float32),
        axis=1,
        dtype=jnp.float32,
    )
    valid_low_shell = (window_shell_indices >= 0) & (window_shell_indices <= int(shell_cutoff))
    untranslated_pixels = jnp.where(
        valid_low_shell[None, :],
        untranslated_power * support_mass[:, None],
        0.0,
    ).astype(jnp.float32)

    # RELION copies its per-pixel float32 Wavg accumulators to the host and
    # adds them into an RFLOAT normalization scalar in pixel order.
    translated_host = np.asarray(jax.block_until_ready(translated_pixels), dtype=np.float32)
    untranslated_host = np.asarray(jax.block_until_ready(untranslated_pixels), dtype=np.float32)
    adjustment = np.sum(translated_host, axis=-1, dtype=np.float64) - np.sum(
        untranslated_host,
        axis=-1,
        dtype=np.float64,
    )
    return jnp.asarray(weighted_img_per_image, dtype=jnp.float64) + jnp.asarray(
        adjustment,
        dtype=jnp.float64,
    )
