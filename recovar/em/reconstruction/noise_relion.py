"""EM-only symbols relocated from ``recovar.reconstruction.noise`` (relax split P1, pure move).

Function bodies are AST-identical to their originals at Q (tag q-reconcile-20260923); only the module
path changed.  See pr179_coordination/relax_split_plan_20260923/PLAN.md.
"""

import jax.numpy as jnp
import numpy as np


def normalize_wsum_to_sigma2_noise(wsum_sigma2_noise, wsum_img_power, sumw, image_shape):
    """Convert posterior-weighted noise accumulators to per-shell noise variance.

    Implements RELION's M-step noise update from ``maximizationOtherParameters``
    (ml_optimiser.cpp:5246-5285)::

        sigma2_noise[s] = wsum_total[s] / (2 * sumw * Npix_per_shell[s])

    where ``wsum_total = wsum_sigma2_noise + wsum_img_power`` is the total
    posterior-weighted squared residual (A2 - 2*XA + P_img), ``sumw`` is the
    total number of images processed, and ``Npix_per_shell`` counts
    half-spectrum pixels per shell.

    The factor of 2 accounts for the real and imaginary components of each
    complex Fourier coefficient (RELION convention: sigma2 is per-component
    variance).

    Parameters
    ----------
    wsum_sigma2_noise : array, shape (n_shells,)
        Accumulated A2 - 2*XA per shell from ``_compute_noise_block``.
    wsum_img_power : array, shape (n_shells,)
        Accumulated |img|^2 per shell (P_img term).
    sumw : float
        Total posterior weight (= number of images when posteriors sum to 1).
    image_shape : tuple of int
        2-D image dimensions, e.g. ``(128, 128)``.

    Returns
    -------
    sigma2_noise : jnp.ndarray, shape (n_shells,)
        Per-shell noise variance in **recovar's native FFT units**, ready to
        be fed back into the engine. Same scale as ``|process_images|^2``.
    """
    from recovar.em.helpers.half_spectrum import bin_shell_values_jax, make_relion_noise_shell_indices_half

    # Preserve whatever real dtype the caller's accumulators already carry
    # (float64 end to end when use_float64_scoring/use_float64_projections
    # are set) instead of forcing float32 here. RELION's own
    # wsum_model.sigma2_noise stays RFLOAT (double under double-precision
    # builds) through this exact division; the unconditional float32 casts
    # this replaced discarded that precision right before the final divide,
    # on top of whatever precision compute_noise_block's callers already
    # accumulated it at.
    wsum_sigma2_noise = jnp.asarray(wsum_sigma2_noise)
    wsum_img_power = jnp.asarray(wsum_img_power)
    n_shells = image_shape[0] // 2 + 1

    shell_indices = make_relion_noise_shell_indices_half(image_shape)
    Npix_per_shell = bin_shell_values_jax(
        jnp.ones_like(shell_indices, dtype=jnp.float32),
        shell_indices,
        n_shells,
    )

    total_wsum = wsum_sigma2_noise + wsum_img_power
    sigma2 = total_wsum / (2.0 * sumw * jnp.maximum(Npix_per_shell, 1.0))

    # NOTE: The output is in **recovar's native FFT units**, not "RELION
    # units". Both ``wsum_sigma2_noise`` and ``wsum_img_power`` are accumulated
    # from ``|process_images(batch)|^2``-scaled quantities (the engine never
    # converts), so the resulting sigma2 already lives in the same units as
    # the engine's image power. A previous version divided by ``(H*W)^2`` to
    # "convert to RELION units" — that was wrong: the engine then re-uses this
    # sigma2 in ``processed * CTF / sigma2``, and the divide blows up chi² by
    # ``(H*W)^2``, collapsing every posterior to a single rotation
    # (``Pmax → 1.0``). See ``estimate_initial_noise_spectrum_from_unaligned_images``
    # docstring and ``tmp/diagnose_pmax_gap.py``.

    # Floor at 1e-15 (RELION ml_optimiser.cpp:5279)
    sigma2 = jnp.maximum(sigma2, 1e-15)

    # Fill zeros from previous shell (RELION ml_optimiser.cpp:5281-5284)
    sigma2_np = np.asarray(sigma2)
    for i in range(1, len(sigma2_np)):
        if sigma2_np[i] < 1e-14 and sigma2_np[i - 1] > 1e-14:
            sigma2_np[i] = sigma2_np[i - 1]
    return jnp.asarray(sigma2_np)
