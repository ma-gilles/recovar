"""Per-image pass-2 operands prepared once per half and kept on the device (T16).

Why this module exists
----------------------
The device-resident K=1 driver used to call
:func:`~recovar.em.sparse_pass2.sparse_pass2_bucket_io._prepare_bucket_io` once
per capacity chunk. That call is per-image work -- a RELION FFT, the soft mask,
the CTF and noise algebra, the corrections and the pre-centering phase -- and it
was repeated for every chunk an image appears in, at an image occupancy of
0.38-0.66 at the early state, where a chunk is padded to 87 image slots to hold
about 33 valid images. It was 71.5% of the 135769 CUDA launches of one hp3 half
and a 43-44 ms host gap in front of every chunk, while contributing 0.32 s of
the half's 4.97 s of kernel time: the chunk loop was launch-bound, not
kernel-bound.

Every operand it produced is per-image pure except the translation tiling, so
this module runs the per-image half once for a whole set of images
(:func:`prepare_resident_half_operands`) and keeps the result on the device.
The chunk loop then only gathers rows by image id
(:func:`gather_resident_chunk_operands`). The translation tiling does not become
resident -- a ``[images, translations, pixels]`` tile for a whole half is tens
of gigabytes -- it disappears instead: T15's
``relion_translate_sum_flat_rows_f32`` applies the translation inside the M-step
reduction, so the M-step reads the *unshifted* per-image operands this module
stores.

What is stored, and in which convention
---------------------------------------
``_prepare_bucket_io`` builds two translated reconstruction operands with two
different primitives, and the kernel has to be told which:

* the reconstruction operand uses ``relion_translate_bpref_f32`` when
  ``relion_exact_bpref_operands`` is on -- RELION's BPref rotation, with the
  weighted CTF multiplied *after* the rotation. This module therefore stores the
  raw BPref image and the weighted CTF separately (``recon_image`` and
  ``recon_weight``), which is exactly the pair the kernel's BPref mode takes;
* the noise operand always uses ``relion_translate_score_f32`` (``recon_image``
  with no weight), so it is stored as one complex array.

Scope. The preparation covers the production K=1 configuration only and fails
closed elsewhere (:class:`ResidentOperandsUnsupported`), because a silently
different translate convention would change every reconstruction pixel. The
driver keeps the per-chunk path as the oracle for the configurations this module
refuses and for the ``...RESIDENT_OPERANDS=0`` comparison arm.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from recovar.em.helpers.batch_fetch import fetch_indexed_batch
from recovar.em.helpers.dtype_policy import DensePrecisionPolicy
from recovar.em.helpers.half_spectrum import make_shell_indices_half
from recovar.em.sparse_pass2.sparse_pass2_bucket_io import prepare_unshifted_bucket_operands
from recovar.em.sparse_pass2.sparse_pass2_scoring import _relion_powerclass_noise_terms
from recovar.em.sparse_pass2.sparse_pass2_wavg import _relion_cuda_translate_wavg_norm_images

logger = logging.getLogger(__name__)

RESIDENT_OPERANDS_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_OPERANDS"
# Host cap on the resident per-image operands of one half, as a fraction of the
# device's memory. The operands scale with the half's image count, so a large
# particle count at a large box can outgrow the budget; the driver then keeps
# the per-chunk preparation and says so, instead of failing part way through.
_RESIDENT_OPERAND_BYTES_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_OPERAND_MAX_BYTES"
_RESIDENT_OPERAND_DEVICE_FRACTION = 0.10
_DEFAULT_RESIDENT_OPERAND_MAX_BYTES = 6 * 1024**3
_PREPARE_IMAGE_BATCH_ENV = "RECOVAR_SPARSE_PASS2_RESIDENT_OPERAND_IMAGE_BATCH"
_DEFAULT_PREPARE_IMAGE_BATCH = 256

__all__ = [
    "RESIDENT_OPERANDS_ENV",
    "ResidentHalfOperands",
    "ResidentOperandsUnsupported",
    "gather_resident_chunk_operands",
    "prepare_resident_half_operands",
    "resident_half_operand_bytes",
    "resident_operands_max_bytes",
]


class ResidentOperandsUnsupported(NotImplementedError):
    """The once-per-half preparation does not cover this pass's configuration."""


@dataclass(frozen=True)
class ResidentHalfOperands:
    """Every per-image operand of one half, on the device, in image order.

    Row ``i`` belongs to the half's image ``i``: the driver's candidate tables
    are built in that same order, so a chunk's image slots are a contiguous
    range of these rows and the per-chunk gather is a take by image id.

    Fields
    ------
    score_input, corr_img_score, highres_xi2_half, translation_prior
        The scoring operands, already gathered to the score window. These are
        what ``_prepare_bucket_io`` returned as ``direct_score_input``,
        ``ctf2_over_nv_half`` and the ``powerClass`` tail.
    recon_image, recon_weight
        The unshifted reconstruction operand of the M-step. ``recon_weight`` is
        the BPref weighted CTF and is ``None`` outside the exact-BPref
        configuration, which is also how the T15 kernel selects its translate
        convention: with a weight it reproduces ``relion_translate_bpref_f32``,
        without one ``relion_translate_score_f32``.
    noise_image
        The unshifted noise operand (``score_weighted_half`` on the
        reconstruction window), always in the score convention.
    ctf2_over_nv_recon, direct_ctf_rfloat_recon
        The M-step CTF operands on the reconstruction window.
    processed_image_half
        The raw processed half image RELION's Wavg and power-spectrum terms
        read; the full packed half, because the image-power shells are binned
        over it.
    relion_norm_high_shell, scale, group_ids
        Per-image scalars of the statistics tail. ``relion_norm_high_shell`` is
        the one operand here whose value is not reproducible bit for bit: its
        shell binning is a scatter-add over duplicate indices, which races. Two
        calls on the same array in one process differ by about 6e-8 relative,
        and a different batch size differs by the same amount, so preparing it
        once per half is inside its own run-to-run spread. Under
        ``RECOVAR_EM_DETERMINISTIC_REDUCTIONS=1`` the binning becomes a
        fixed-order masked reduction and every one of those comparisons is
        exact, which is how the driver's verification arm checks it.
    """

    n_images: int
    n_score_pixels: int
    n_recon_pixels: int
    n_half_pixels: int
    n_fine_trans: int
    score_input: jax.Array
    corr_img_score: jax.Array
    highres_xi2_half: jax.Array | None
    translation_prior: jax.Array
    recon_image: jax.Array
    recon_weight: jax.Array | None
    noise_image: jax.Array
    ctf2_over_nv_recon: jax.Array
    direct_ctf_rfloat_recon: jax.Array | None
    processed_image_half: jax.Array
    relion_norm_high_shell: jax.Array | None
    scale: jax.Array
    group_ids: jax.Array

    def __post_init__(self):
        score_shape = (self.n_images, self.n_score_pixels)
        recon_shape = (self.n_images, self.n_recon_pixels)
        for name, expected in (
            ("score_input", score_shape),
            ("corr_img_score", score_shape),
            ("recon_image", recon_shape),
            ("noise_image", recon_shape),
            ("ctf2_over_nv_recon", recon_shape),
            ("processed_image_half", (self.n_images, self.n_half_pixels)),
            ("translation_prior", (self.n_images, self.n_fine_trans)),
        ):
            value = getattr(self, name)
            if tuple(value.shape) != expected:
                raise ValueError(f"{name} must have shape {expected}, got {tuple(value.shape)}")
        for name in ("recon_weight", "direct_ctf_rfloat_recon"):
            value = getattr(self, name)
            if value is not None and tuple(value.shape) != recon_shape:
                raise ValueError(f"{name} must have shape {recon_shape}, got {tuple(value.shape)}")

    def nbytes(self) -> dict:
        """Device bytes of each resident array, plus their total."""

        parts = {}
        for name in (
            "score_input",
            "corr_img_score",
            "highres_xi2_half",
            "translation_prior",
            "recon_image",
            "recon_weight",
            "noise_image",
            "ctf2_over_nv_recon",
            "direct_ctf_rfloat_recon",
            "processed_image_half",
            "relion_norm_high_shell",
            "scale",
            "group_ids",
        ):
            value = getattr(self, name)
            parts[name] = 0 if value is None else int(value.size) * int(value.dtype.itemsize)
        parts["total"] = sum(parts.values())
        return parts


def resident_half_operand_bytes(
    *,
    n_images: int,
    n_score_pixels: int,
    n_recon_pixels: int,
    n_half_pixels: int,
    n_fine_trans: int,
    score_complex_bytes: int = 8,
    real_bytes: int = 4,
    rfloat_ctf_bytes: int = 8,
) -> int:
    """Host estimate of the resident operand bytes, before any device work.

    Used by the driver's admission check, so an over-budget half keeps the
    per-chunk preparation rather than allocating and failing.
    """

    n_images = int(n_images)
    return int(
        n_images
        * (
            int(n_score_pixels) * (int(score_complex_bytes) + int(real_bytes))
            + int(n_recon_pixels)
            * (2 * int(score_complex_bytes) + 2 * int(real_bytes) + int(rfloat_ctf_bytes))
            + int(n_half_pixels) * int(score_complex_bytes)
            + int(n_fine_trans) * int(real_bytes)
            + 4 * int(real_bytes)
        )
    )


def resident_half_operand_avals(
    *,
    n_images: int,
    n_score_pixels: int,
    n_recon_pixels: int,
    n_half_pixels: int,
    n_fine_trans: int,
    score_complex_dtype,
    score_real_dtype,
    acc_real_dtype,
    rfloat_ctf_dtype=None,
    has_recon_weight: bool,
    has_direct_ctf_rfloat: bool,
    has_highres_xi2: bool = True,
    has_relion_norm_high_shell: bool = True,
) -> "ResidentHalfOperands":
    """The half's operands as avals, without preparing them (P4-J).

    Same inputs as :func:`resident_half_operand_bytes`, which the driver's
    admission check already computes before any device work: the shapes of this
    half's operands are decided by the current size, the window and the image
    count, all of them known before pass 1 runs. This returns them as
    ``jax.ShapeDtypeStruct`` so a program that consumes the operands can be
    lowered and compiled ahead of the preparation that fills them.

    Only the SHAPES are encoded here. The dtypes are the caller's, because the
    precision policy owns them and a second copy of that decision would be a
    second place to get it wrong; ``has_*`` say which optional operands the
    configuration produces.

    ``rfloat_ctf_dtype`` is separate and defaults to float64 because that is
    what :func:`resident_half_operand_bytes` assumes for the exact RELION CTF
    (its ``rfloat_ctf_bytes`` default is 8, and the driver leaves it at the
    default while passing the policy's dtypes for everything else). Whether the
    stored array really is float64 is not settled here: the CPU test only holds
    this function and the byte estimate to the same story, and the GPU test
    against :func:`prepare_resident_half_operands` is what decides it. If they
    disagree, the admission check is over-estimating by four bytes per
    reconstruction pixel per image, which is conservative and therefore safe,
    but this function would be wrong and the GPU test is how that surfaces.

    Nothing here allocates or touches a device buffer.
    """

    n_images = int(n_images)
    score_shape = (n_images, int(n_score_pixels))
    recon_shape = (n_images, int(n_recon_pixels))
    half_shape = (n_images, int(n_half_pixels))
    per_image = (n_images,)

    def aval(shape, dtype):
        return jax.ShapeDtypeStruct(tuple(shape), jnp.dtype(dtype))

    return ResidentHalfOperands(
        n_images=n_images,
        n_score_pixels=int(n_score_pixels),
        n_recon_pixels=int(n_recon_pixels),
        n_half_pixels=int(n_half_pixels),
        n_fine_trans=int(n_fine_trans),
        score_input=aval(score_shape, score_complex_dtype),
        corr_img_score=aval(score_shape, score_real_dtype),
        highres_xi2_half=aval(per_image, score_real_dtype) if has_highres_xi2 else None,
        translation_prior=aval((n_images, int(n_fine_trans)), score_real_dtype),
        recon_image=aval(recon_shape, score_complex_dtype),
        recon_weight=aval(recon_shape, acc_real_dtype) if has_recon_weight else None,
        noise_image=aval(recon_shape, score_complex_dtype),
        ctf2_over_nv_recon=aval(recon_shape, acc_real_dtype),
        direct_ctf_rfloat_recon=(
            aval(recon_shape, rfloat_ctf_dtype if rfloat_ctf_dtype is not None else jnp.float64)
            if has_direct_ctf_rfloat
            else None
        ),
        processed_image_half=aval(half_shape, score_complex_dtype),
        relion_norm_high_shell=(
            aval(per_image, score_real_dtype) if has_relion_norm_high_shell else None
        ),
        scale=aval(per_image, jnp.float32),
        group_ids=aval(per_image, jnp.int32),
    )


def resident_operands_max_bytes(device_memory_bytes: int | None = None) -> int:
    """Budget for one half's resident per-image operands."""

    override = os.environ.get(_RESIDENT_OPERAND_BYTES_ENV, "").strip()
    if override:
        value = int(override)
        if value <= 0:
            raise ValueError(f"{_RESIDENT_OPERAND_BYTES_ENV} must be positive, got {value}")
        return value
    if device_memory_bytes is None:
        return _DEFAULT_RESIDENT_OPERAND_MAX_BYTES
    return max(1, int(float(device_memory_bytes) * _RESIDENT_OPERAND_DEVICE_FRACTION))


def _prepare_image_batch_size() -> int:
    raw = os.environ.get(_PREPARE_IMAGE_BATCH_ENV, "").strip()
    if not raw:
        return _DEFAULT_PREPARE_IMAGE_BATCH
    value = int(raw)
    if value <= 0:
        raise ValueError(f"{_PREPARE_IMAGE_BATCH_ENV} must be positive, got {value}")
    return value


def _require_supported(condition: bool, message: str) -> None:
    if not condition:
        raise ResidentOperandsUnsupported(
            "the once-per-half resident operand preparation does not cover " + message
        )


@jax.jit
def _concatenate_and_reorder(parts: tuple, reorder: jax.Array) -> jax.Array:
    """Concatenate one named per-batch operand and put it in dataset order.

    Pure data movement: a concatenation and a gather, no arithmetic, so the
    bytes are the bytes the two eager dispatches produced. As two eager
    programs this was two traces, two lowerings and two XLA compilations per
    named operand per half -- 129 of them on the early state, because each name
    has its own pixel extent and dtype -- where the fused pair is one.
    """

    return jnp.concatenate(parts, axis=0)[reorder]


class _BatchWindowInputs(NamedTuple):
    """The per-image-batch arrays the window gather and cast consume.

    ``weighted_ctf_half`` and ``ctf_half_rfloat`` are ``None`` on the paths
    that do not produce them; ``None`` is a pytree structure, so those paths
    key their own program instead of carrying a dead operand.
    """

    ctf2_over_nv_half: jax.Array
    sparse_score_input_half: jax.Array
    processed_score_half_for_noise: jax.Array
    recon_input_half: jax.Array
    weighted_ctf_half: jax.Array | None
    score_weighted_half: jax.Array
    ctf2_over_nv_recon_half: jax.Array
    ctf_half_rfloat: jax.Array | None
    dc_mask: jax.Array | None
    score_indices: jax.Array
    recon_indices: jax.Array


@partial(
    jax.jit,
    static_argnames=("mask_dc", "score_real_dtype", "score_complex_dtype", "acc_real_dtype"),
)
def _batch_window_operands(
    arrays: _BatchWindowInputs,
    *,
    mask_dc: bool,
    score_real_dtype,
    score_complex_dtype,
    acc_real_dtype,
) -> dict:
    """The window gather and cast of one image batch, as one program.

    Same statements, same order, same dtypes as the loose dispatch: a DC mask
    on the score correction, seven ``values[:, indices]`` gathers and their
    casts. Nothing here is arithmetic, so the values cannot move; what leaves
    the host is the dispatch count. Each ``values[:, indices]`` was five eager
    primitives, not one -- JAX normalizes a fancy index eagerly (``add``,
    ``broadcast_in_dim``, ``select_n``) before the gather -- and the driver
    runs one of these per image batch per half, which on the early and hp3
    states was 1400 eager dispatches per iteration for a constant index array.

    ``weighted_ctf_half is None`` selects the non-BPref reconstruction operand
    and ``ctf_half_rfloat is None`` the path with no direct RFLOAT CTF, exactly
    as the Python branches did.
    """

    ctf2_score = arrays.ctf2_over_nv_half
    if mask_dc:
        ctf2_score = jnp.where(arrays.dc_mask[None, :], 0.0, ctf2_score)
    score_indices = arrays.score_indices
    recon_indices = arrays.recon_indices
    batch_arrays = {
        "score_input": arrays.sparse_score_input_half[:, score_indices],
        "corr_img_score": ctf2_score[:, score_indices].astype(score_real_dtype),
        "processed_image_half": arrays.processed_score_half_for_noise,
        "recon_image": jnp.asarray(
            arrays.recon_input_half[:, recon_indices], dtype=score_complex_dtype
        ),
        "noise_image": jnp.asarray(
            arrays.score_weighted_half[:, recon_indices], dtype=score_complex_dtype
        ),
        "ctf2_over_nv_recon": arrays.ctf2_over_nv_recon_half[:, recon_indices],
    }
    if arrays.weighted_ctf_half is not None:
        batch_arrays["recon_weight"] = jnp.asarray(
            arrays.weighted_ctf_half[:, recon_indices], dtype=acc_real_dtype
        )
    if arrays.ctf_half_rfloat is not None:
        batch_arrays["direct_ctf_rfloat_recon"] = arrays.ctf_half_rfloat[:, recon_indices]
    return batch_arrays


def prepare_resident_half_operands(
    experiment_dataset,
    image_indices,
    *,
    bucket_io_kwargs: dict,
    window_indices,
    recon_window_indices,
    image_shape,
    current_size,
    n_fine_trans: int,
    use_exact_relion_gaussian: bool,
    accumulate_noise: bool,
    source_faithful_spectrum_norm: bool,
    fine_translation_prior_2d=None,
    scale_corrections_np=None,
    group_ids_np=None,
    precision_policy: DensePrecisionPolicy | None = None,
    image_batch_size: int | None = None,
) -> ResidentHalfOperands:
    """Run the per-image preparation once for ``image_indices`` and keep it resident.

    ``bucket_io_kwargs`` is the driver's own ``_prepare_bucket_io`` keyword set,
    so this call reads exactly the configuration the per-chunk path would have
    read; the translation-only keywords are ignored here because nothing in the
    per-image half depends on them.

    ``image_batch_size`` only decides how many images one preparation call
    covers. The preparation is per image, so it does not change any value; the
    last batch is short rather than padded, unlike the per-chunk path, which had
    to pad to a capacity class to keep one program per class.
    """

    image_indices = np.asarray(image_indices)
    if image_indices.ndim != 1 or image_indices.size == 0:
        raise ValueError(f"image_indices must be a non-empty 1-D array, got {image_indices.shape}")
    n_images = int(image_indices.shape[0])
    position_of = {int(index): position for position, index in enumerate(image_indices.tolist())}
    if len(position_of) != n_images:
        raise ValueError("image_indices must not repeat an image")

    kwargs = dict(bucket_io_kwargs)
    relion_exact_bpref_operands = bool(kwargs.get("relion_exact_bpref_operands", False))
    score_with_masked_images = bool(kwargs.get("score_with_masked_images", False))
    half_spectrum_scoring = bool(kwargs.get("half_spectrum_scoring", False))
    use_float64_scoring = bool(kwargs.get("use_float64_scoring", False))
    score_mode = kwargs.get("score_mode", "gaussian")

    # The M-step reads the unshifted operands through T15's float32 kernel, and
    # that kernel knows two translate conventions: BPref for the reconstruction
    # operand and score for the noise operand. Without masked scoring the two
    # operands are the same array in the per-chunk path, which means the noise
    # sum is taken over the *BPref*-translated tile; the kernel cannot express
    # that pairing, so refuse it rather than change the arithmetic.
    _require_supported(score_with_masked_images, "unmasked scoring (score_with_masked_images=0)")
    _require_supported(not use_float64_scoring, "float64 scoring")
    _require_supported(score_mode == "gaussian", f"score_mode={score_mode!r}")
    _require_supported(
        kwargs.get("relion_score_translation_angles", None) is not None,
        "a pass without RELION translation angles",
    )
    _require_supported(not bool(kwargs.get("score_only", False)), "score-only preparation")
    _require_supported(window_indices is not None, "an unwindowed score spectrum")
    _require_supported(recon_window_indices is not None, "an unwindowed reconstruction spectrum")

    precision_policy = precision_policy or DensePrecisionPolicy(
        use_float64_scoring=use_float64_scoring
    )
    score_real_dtype = precision_policy.score_real_dtype
    score_complex_dtype = precision_policy.score_complex_dtype

    unshifted_kwargs = dict(
        noise_variance_half=kwargs["noise_variance_half"],
        config=kwargs["config"],
        score_with_masked_images=score_with_masked_images,
        image_corrections=kwargs["image_corrections"],
        scale_corrections=kwargs["scale_corrections"],
        image_pre_shifts=kwargs["image_pre_shifts"],
        use_float64_scoring=use_float64_scoring,
        return_direct_scoring_io=True,
        score_only=False,
        score_mode=score_mode,
        window_indices=window_indices,
        relion_exact_normalized_cc_operands=bool(
            kwargs.get("relion_exact_normalized_cc_operands", False)
        ),
        relion_exact_bpref_operands=relion_exact_bpref_operands,
    )

    score_indices = jnp.asarray(window_indices, dtype=jnp.int32)
    recon_indices = jnp.asarray(recon_window_indices, dtype=jnp.int32)
    dc_mask = None
    if half_spectrum_scoring:
        dc_mask = jnp.asarray(make_shell_indices_half(image_shape)) == 0

    # Batch outputs stay on the device and are concatenated once; only the
    # dataset's returned order comes back to the host, as a permutation.
    batches: list[dict] = []
    fetched_order: list[np.ndarray] = []
    optional_available: dict[str, bool | None] = {
        "recon_weight": None,
        "direct_ctf_rfloat_recon": None,
        "highres_xi2_half": None,
        "relion_norm_high_shell": None,
    }
    batch_size = int(image_batch_size or _prepare_image_batch_size())

    for start in range(0, n_images, batch_size):
        batch_image_indices = image_indices[start : start + batch_size]
        batch_data, ctf_params, fetched_indices = fetch_indexed_batch(
            experiment_dataset, batch_image_indices
        )
        fetched_indices = np.asarray(fetched_indices)
        unshifted = prepare_unshifted_bucket_operands(
            experiment_dataset,
            jnp.asarray(batch_data),
            ctf_params,
            fetched_indices,
            **unshifted_kwargs,
        )

        batch_arrays = _batch_window_operands(
            _BatchWindowInputs(
                ctf2_over_nv_half=unshifted.ctf2_over_nv_half,
                sparse_score_input_half=unshifted.sparse_score_input_half,
                processed_score_half_for_noise=unshifted.processed_score_half_for_noise,
                recon_input_half=(
                    unshifted.recon_bpref_input_half
                    if relion_exact_bpref_operands
                    else unshifted.recon_weighted_half
                ),
                weighted_ctf_half=(
                    unshifted.weighted_ctf_half if relion_exact_bpref_operands else None
                ),
                score_weighted_half=unshifted.score_weighted_half,
                ctf2_over_nv_recon_half=unshifted.ctf2_over_nv_recon_half,
                ctf_half_rfloat=unshifted.ctf_half_rfloat,
                dc_mask=dc_mask,
                score_indices=score_indices,
                recon_indices=recon_indices,
            ),
            mask_dc=bool(half_spectrum_scoring and not unshifted.use_normalized_cc),
            score_real_dtype=jnp.dtype(score_real_dtype),
            score_complex_dtype=jnp.dtype(score_complex_dtype),
            acc_real_dtype=jnp.dtype(unshifted.acc_real_dtype),
        )

        highres_xi2_half, relion_norm_high_shell = _relion_powerclass_noise_terms(
            unshifted.processed_score_half_for_noise,
            image_shape=image_shape,
            current_size=current_size,
            use_exact_relion_gaussian=use_exact_relion_gaussian,
            accumulate_noise=accumulate_noise,
            source_faithful_spectrum_norm=source_faithful_spectrum_norm,
        )
        if highres_xi2_half is not None:
            batch_arrays["highres_xi2_half"] = highres_xi2_half
        if relion_norm_high_shell is not None:
            batch_arrays["relion_norm_high_shell"] = relion_norm_high_shell

        for name, flag in optional_available.items():
            present = name in batch_arrays
            if flag is None:
                optional_available[name] = present
            elif flag != present:
                raise ValueError(f"{name} availability changed between image batches")

        batches.append(batch_arrays)
        fetched_order.append(fetched_indices)

    fetched_all = np.concatenate(fetched_order)
    if fetched_all.shape[0] != n_images:
        raise ValueError(
            f"the dataset returned {fetched_all.shape[0]} images for {n_images} requested"
        )
    destination = np.empty(n_images, dtype=np.int64)
    for order, dataset_index in enumerate(fetched_all.tolist()):
        position = position_of.get(int(dataset_index))
        if position is None:
            raise ValueError(
                f"the dataset returned image {dataset_index}, which is not in image_indices"
            )
        destination[order] = position
    inverse = np.empty(n_images, dtype=np.int64)
    inverse[destination] = np.arange(n_images, dtype=np.int64)
    if np.unique(destination).size != n_images:
        raise ValueError("the dataset did not return every requested image exactly once")
    reorder = jnp.asarray(inverse)

    def stack(name, required=False):
        present = name in batches[0]
        if required and not present:
            raise ValueError(f"the per-image preparation did not produce {name}")
        if not present or not optional_available.get(name, True):
            return None
        return _concatenate_and_reorder(tuple(batch[name] for batch in batches), reorder)

    if fine_translation_prior_2d is None:
        translation_prior = jnp.zeros((n_images, int(n_fine_trans)), dtype=score_real_dtype)
    else:
        translation_prior = jnp.asarray(
            np.asarray(fine_translation_prior_2d)[image_indices], dtype=score_real_dtype
        )

    scale = np.ones(n_images, dtype=np.float32)
    if scale_corrections_np is not None:
        scale[:] = np.asarray(scale_corrections_np, dtype=np.float32)[image_indices]
    group_ids = np.full(n_images, -1, dtype=np.int32)
    if group_ids_np is not None:
        group_ids[:] = np.asarray(group_ids_np, dtype=np.int32)[image_indices]

    score_input = stack("score_input", required=True)
    recon_image = stack("recon_image", required=True)
    processed_image_half = stack("processed_image_half", required=True)
    operands = ResidentHalfOperands(
        n_images=n_images,
        n_score_pixels=int(score_input.shape[1]),
        n_recon_pixels=int(recon_image.shape[1]),
        n_half_pixels=int(processed_image_half.shape[1]),
        n_fine_trans=int(n_fine_trans),
        score_input=score_input,
        corr_img_score=stack("corr_img_score", required=True),
        highres_xi2_half=stack("highres_xi2_half"),
        translation_prior=translation_prior,
        recon_image=recon_image,
        recon_weight=stack("recon_weight"),
        noise_image=stack("noise_image", required=True),
        ctf2_over_nv_recon=stack("ctf2_over_nv_recon", required=True),
        direct_ctf_rfloat_recon=stack("direct_ctf_rfloat_recon"),
        processed_image_half=processed_image_half,
        relion_norm_high_shell=stack("relion_norm_high_shell"),
        scale=jnp.asarray(scale),
        group_ids=jnp.asarray(group_ids),
    )
    logger.info(
        "Resident pass-2 per-half operands: %d images, %d score / %d recon / %d half pixels, "
        "%.2f GiB resident (%d preparation calls)",
        operands.n_images,
        operands.n_score_pixels,
        operands.n_recon_pixels,
        operands.n_half_pixels,
        operands.nbytes()["total"] / float(1024**3),
        (n_images + batch_size - 1) // batch_size,
    )
    # A separate line, not an addition to the one above: the census parsers key
    # on that line's text. The batch size and the last batch's remainder are
    # what set the leading extent of every program inside the preparation, so
    # P4-A's shape census could not classify those extents without them.
    logger.info(
        "Resident pass-2 per-half operand batching: image_batch_size=%d, "
        "last_batch_images=%d",
        batch_size,
        n_images - batch_size * ((n_images - 1) // batch_size) if n_images else 0,
    )
    return operands


def _gather_rows(values, safe_slots, valid, fill=0):
    """Take one row per chunk slot, with the padded slots set to ``fill``."""

    if values is None:
        return None
    values = jnp.asarray(values)
    gathered = values[safe_slots]
    mask = valid.reshape((-1,) + (1,) * (gathered.ndim - 1))
    return jnp.where(mask, gathered, jnp.asarray(fill, dtype=gathered.dtype))


@partial(jax.jit, static_argnames=("image_shape",))
def _gather_chunk_arrays(
    image_slots,
    score_input,
    corr_img_score,
    highres_xi2_half,
    translation_prior,
    recon_image,
    recon_weight,
    noise_image,
    ctf2_over_nv_recon,
    direct_ctf_rfloat_recon,
    processed_image_half,
    relion_norm_high_shell,
    scale,
    group_ids,
    translation_angles,
    rect_indices,
    exact_positions,
    *,
    image_shape,
):
    """One program per (half size, capacity class): the chunk's whole operand set.

    Every array is taken by image id and the padded slots are zeroed, which is
    what the per-chunk preparation's capacity mask did. ``scale`` keeps 1 on a
    padded slot so the Wavg kernel never divides by zero, and ``group_ids``
    keeps -1 so the scale accumulators drop it; both are the per-chunk path's
    own padding values, and neither is observable because a padded slot's
    posterior is zero.

    A padded slot reads the chunk's *first* image rather than image zero, which
    is what ``_pad_batch_to_capacity`` fed the per-chunk preparation. Nothing
    downstream can see the difference, since every padded row is zeroed here,
    but it keeps the two paths' intermediate batches identical.
    """

    image_slots = jnp.asarray(image_slots, dtype=jnp.int32)
    valid = image_slots >= 0
    safe_slots = jnp.where(valid, image_slots, image_slots[0])

    processed_chunk = _gather_rows(processed_image_half, safe_slots, valid)
    raw_translated_wavg_rectangle = _relion_cuda_translate_wavg_norm_images(
        processed_chunk,
        translation_angles,
        rect_indices,
        image_shape,
    )
    return (
        _gather_rows(score_input, safe_slots, valid),
        _gather_rows(corr_img_score, safe_slots, valid),
        _gather_rows(highres_xi2_half, safe_slots, valid),
        _gather_rows(translation_prior, safe_slots, valid),
        _gather_rows(recon_image, safe_slots, valid),
        _gather_rows(recon_weight, safe_slots, valid),
        _gather_rows(noise_image, safe_slots, valid),
        _gather_rows(ctf2_over_nv_recon, safe_slots, valid),
        _gather_rows(direct_ctf_rfloat_recon, safe_slots, valid),
        processed_chunk,
        _gather_rows(relion_norm_high_shell, safe_slots, valid),
        _gather_rows(scale, safe_slots, valid, fill=1.0),
        _gather_rows(group_ids, safe_slots, valid, fill=-1),
        raw_translated_wavg_rectangle,
        raw_translated_wavg_rectangle[:, :, jnp.asarray(exact_positions, dtype=jnp.int32)],
    )


def gather_resident_chunk_operands(
    operands: ResidentHalfOperands,
    image_slots,
    *,
    translation_angles,
    rect_indices,
    exact_positions,
    image_shape,
) -> dict:
    """Gather one chunk's operands out of the half's resident arrays.

    ``image_slots`` is the chunk's capacity-shaped image id vector, ``-1`` on a
    padded slot. The returned keys are the ones the driver's chunk stage
    operands take, so the per-chunk preparation and this gather are
    interchangeable at the call site.
    """

    (
        score_input,
        corr_img_score,
        highres_xi2_half,
        translation_prior,
        recon_image,
        recon_weight,
        noise_image,
        ctf2_over_nv_recon,
        direct_ctf_rfloat_recon,
        processed_image_half,
        relion_norm_high_shell,
        scale,
        group_ids,
        raw_translated_wavg_rectangle,
        raw_translated_wavg_for_atomic,
    ) = _gather_chunk_arrays(
        jnp.asarray(image_slots, dtype=jnp.int32),
        operands.score_input,
        operands.corr_img_score,
        operands.highres_xi2_half,
        operands.translation_prior,
        operands.recon_image,
        operands.recon_weight,
        operands.noise_image,
        operands.ctf2_over_nv_recon,
        operands.direct_ctf_rfloat_recon,
        operands.processed_image_half,
        operands.relion_norm_high_shell,
        operands.scale,
        operands.group_ids,
        jnp.asarray(translation_angles, dtype=jnp.float32),
        jnp.asarray(rect_indices, dtype=jnp.int32),
        jnp.asarray(exact_positions, dtype=jnp.int32),
        image_shape=tuple(int(size) for size in image_shape),
    )
    return {
        "score_input": score_input,
        "corr_img_score": corr_img_score,
        "highres_xi2_half": highres_xi2_half,
        "translation_prior": translation_prior,
        "recon_image": recon_image,
        "recon_weight": recon_weight,
        "noise_image": noise_image,
        "ctf2_over_nv_recon": ctf2_over_nv_recon,
        "direct_ctf_rfloat_recon": direct_ctf_rfloat_recon,
        "processed_image_half": processed_image_half,
        "relion_norm_high_shell": relion_norm_high_shell,
        "raw_translated_wavg_rectangle": raw_translated_wavg_rectangle,
        "raw_translated_wavg_for_atomic": raw_translated_wavg_for_atomic,
        "scale": scale,
        "group_ids": group_ids,
    }
