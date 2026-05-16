#!/usr/bin/env python
"""Build a PPCA refinement init NPZ from a recovar pipeline output (v2).

Differences vs ``prepare_ppca_init_from_pipeline_output.py``:

* **Mask applied to PCs** — multiplies each eigenvector by the same
  ``dilated_volume_mask`` the pipeline uses inside its projected-covariance
  solver. The pipeline solves for the basis in masked-image space; reusing
  the unmasked eigenvector breaks that representational consistency. The
  mean is also saved in masked form (alongside the unmasked mean) so the
  EM refinement starts from a mask-restricted model.

* **Pipeline-side W prior captured** — copies the per-Fourier-voxel
  signal-variance prior the pipeline computed (``params.pkl['variance_est']['prior']``
  from ``compute_fsc_prior_gpu_v2``) into the init NPZ together with its
  per-shell average. Downstream EM refinement scripts can opt into
  ``--prior-from-init pipeline-variance`` to use this prior directly
  instead of deriving one from the init W row norms.

The pipeline already saves ``u['rescaled']`` and ``s['rescaled']`` in
their final post-projected-covariance form (after the ``basis @ eigh``
recomposition and any contrast-knockout). This script reuses those
saved values; it does not attempt to undo internal pipeline rescalings.
"""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np

from recovar.core import fourier_transform_utils as ftu
from recovar.em.ppca_refinement.initialization import _average_half_values_over_shells
from recovar.output.output import PipelineOutput
from recovar.utils import helpers


def _load_rescaled_eigenvalues(po: PipelineOutput) -> np.ndarray:
    s = po.get("s")
    if isinstance(s, dict):
        if "rescaled" in s:
            return np.asarray(s["rescaled"], dtype=np.float32).reshape(-1)
        if "s" in s:
            return np.asarray(s["s"], dtype=np.float32).reshape(-1)
    return np.asarray(s, dtype=np.float32).reshape(-1)


def _full_volume_to_half(vol_full_flat: np.ndarray, volume_shape) -> np.ndarray:
    """Convert a flat full-volume Fourier-space array to flat half-spectrum."""
    arr = np.asarray(vol_full_flat).reshape(volume_shape)
    return np.asarray(
        ftu.full_volume_to_half_volume(arr, volume_shape), dtype=np.complex64
    ).reshape(-1)


def _half_size(volume_shape) -> int:
    return int(np.prod(ftu.volume_shape_to_half_volume_shape(tuple(volume_shape))))


def _pipeline_variance_prior(
    params: dict,
    volume_shape,
    *,
    field: str = "prior",
    floor: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull the pipeline-side per-voxel signal variance prior and shell-average.

    ``field='prior'`` matches the prior the pipeline uses to regularize its
    variance Wiener solver (from ``compute_fsc_prior_gpu_v2``). ``field='combined'``
    pulls the corrected signal variance estimate. Both are stored full-Fourier
    flat by the pipeline. Returned arrays are in flat half-spectrum format,
    real-valued (magnitudes squared / variances are inherently nonneg).
    """
    variance_est = params.get("variance_est")
    if variance_est is None or not isinstance(variance_est, dict):
        raise KeyError("pipeline params.pkl missing 'variance_est' dict — cannot pull W prior")
    if field not in variance_est:
        raise KeyError(f"variance_est does not contain '{field}'; available keys: {sorted(variance_est)}")
    full_flat = np.asarray(variance_est[field], dtype=np.float32).reshape(-1)
    expected_full = int(np.prod(volume_shape))
    if full_flat.size != expected_full:
        raise ValueError(
            f"variance_est['{field}'] has {full_flat.size} entries, expected {expected_full} (= prod(volume_shape))"
        )
    half_flat = np.abs(_full_volume_to_half(full_flat.astype(np.complex64), volume_shape))
    half_flat = np.asarray(half_flat, dtype=np.float64)
    if floor > 0.0:
        half_flat = np.maximum(half_flat, float(floor))
    shell_avg = _average_half_values_over_shells(half_flat, volume_shape).astype(np.float32)
    return half_flat.astype(np.float32), shell_avg


def _recompute_mean_prior_from_saved(
    po: "PipelineOutput",
    params: dict,
    *,
    volume_shape,
) -> np.ndarray:
    """Recompute ``mean_prior`` from saved half-maps + noise_var_used.

    Calls ``regularization.compute_fsc_prior_gpu_v2`` with the saved half-maps
    (Fourier-domain) and a uniform-infinity ``lhs`` placeholder, matching the
    shape the pipeline gets from ``compute_relion_prior``. This is a faithful
    approximation when the actual ``lhs = ft_ctf`` accumulator isn't on disk —
    sufficient as a per-shell signal-variance prior for the W M-step Wiener form.
    """
    import jax.numpy as jnp
    from recovar.core import fourier_transform_utils as _ftu
    from recovar.reconstruction import regularization as _regularization

    half1 = np.asarray(po.get("mean_halfmaps")[0], dtype=np.complex64)
    half2 = np.asarray(po.get("mean_halfmaps")[1], dtype=np.complex64)
    noise_var_used = np.asarray(params["noise_var_used"], dtype=np.float32).reshape(-1)
    # Make a per-voxel noise array from the per-shell noise (radial expansion)
    radial_distances = (
        np.asarray(
            _ftu.get_grid_of_radial_distances(volume_shape, scaled=False, frequency_shift=jnp.array([0, 0, 0])),
            dtype=np.int64,
        )
        .reshape(-1)
    )
    # noise_var_used is per-shell; expand to per-voxel via radial lookup.
    n_shells = noise_var_used.shape[0]
    safe_idx = np.clip(radial_distances, 0, n_shells - 1)
    cov_noise_full = noise_var_used[safe_idx].astype(np.float32)

    # Compute FSC-prior with lhs = 1/cov_noise (so prior shape comes from FSC and noise);
    # placeholder prior=inf disables the previous-iter prior term.
    lhs = jnp.asarray(1.0 / np.maximum(cov_noise_full, 1e-30), dtype=jnp.float32)
    prior, fsc_raw, prior_avg = _regularization.compute_fsc_prior_gpu_v2(
        volume_shape,
        jnp.asarray(half1),
        jnp.asarray(half2),
        lhs,
        jnp.ones_like(lhs) * jnp.float32(np.inf),
        frequency_shift=jnp.array([0, 0, 0]),
    )
    return np.asarray(prior).reshape(-1).real


def prepare_ppca_init_v2(
    pipeline_output: str | Path,
    output_dir: str | Path,
    *,
    q: int,
    scale_by_sqrt_eigenvalue: bool = True,
    apply_mask: bool = True,
    mask_kind: str = "dilated",
    save_pipeline_variance_prior: bool = True,
    variance_prior_field: str = "prior",
) -> Path:
    po = PipelineOutput(str(Path(pipeline_output).expanduser().resolve()))
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    q = int(q)
    if q <= 0:
        raise ValueError(f"q must be positive, got {q}")

    mu = np.asarray(helpers.load_mrc(po.paths.mean_volume), dtype=np.float32)
    u_real = np.asarray(po.get_u_real(q), dtype=np.float32)
    if u_real.shape[0] < q:
        raise ValueError(f"pipeline output only has {u_real.shape[0]} saved eigenvectors, requested q={q}")
    s_rescaled = _load_rescaled_eigenvalues(po)
    if s_rescaled.shape[0] < q:
        raise ValueError(f"pipeline output only has {s_rescaled.shape[0]} eigenvalues, requested q={q}")

    volume_shape = tuple(int(x) for x in mu.shape)

    # Volume mask: the pipeline's dilated mask is what its PCA solver uses inside
    # ``model.volume_mask`` (see recovar/heterogeneity/covariance_estimation.py::compute_projected_covariance).
    # We replicate that on the eigenvectors and the mean. The non-dilated mask
    # is also saved for downstream postprocessing if needed.
    dilated_mask = np.asarray(po.get("dilated_volume_mask"), dtype=np.float32)
    plain_mask = np.asarray(po.get("volume_mask"), dtype=np.float32)
    if dilated_mask.shape != volume_shape:
        raise ValueError(f"dilated mask shape {dilated_mask.shape} != volume_shape {volume_shape}")

    chosen_mask = dilated_mask if str(mask_kind) == "dilated" else plain_mask
    W_unmasked = u_real[:q].copy()
    if scale_by_sqrt_eigenvalue:
        W_unmasked *= np.sqrt(np.maximum(s_rescaled[:q], 0.0)).astype(np.float32)[:, None, None, None]

    if apply_mask:
        W = W_unmasked * chosen_mask[None, :, :, :]
        mu_masked = mu * chosen_mask
    else:
        W = W_unmasked
        mu_masked = mu

    payload: dict = {
        "mu": mu_masked.astype(np.float32),
        "mu_unmasked": mu.astype(np.float32),
        "W": W.astype(np.float32),
        "W_unmasked": W_unmasked.astype(np.float32),
        "s_rescaled": s_rescaled[:q].astype(np.float32),
        "volume_shape": np.asarray(volume_shape, dtype=np.int32),
        "voxel_size": np.float32(po.params.get("voxel_size", np.nan)),
        "pipeline_output": np.asarray(str(Path(pipeline_output).expanduser().resolve())),
        "W_scaling": np.asarray("sqrt_s_rescaled" if scale_by_sqrt_eigenvalue else "unit_eigenvectors"),
        "mask_applied": np.asarray(str(mask_kind) if apply_mask else "none"),
        "volume_mask": chosen_mask.astype(np.float32),
        "volume_mask_dilated": dilated_mask.astype(np.float32),
        "volume_mask_plain": plain_mask.astype(np.float32),
    }
    if "noise_var_used" in po.params:
        payload["noise_var_used"] = np.asarray(po.params["noise_var_used"], dtype=np.float32)

    if save_pipeline_variance_prior:
        try:
            params_pkl_path = Path(po.paths.params_file) if hasattr(po.paths, "params_file") else None
        except Exception:
            params_pkl_path = None
        if params_pkl_path is None or not Path(params_pkl_path).exists():
            # Fall back to the default schema location
            params_pkl_path = Path(pipeline_output) / "model" / "params.pkl"
        with open(params_pkl_path, "rb") as f:
            params = pickle.load(f)
        try:
            prior_voxel, prior_shell = _pipeline_variance_prior(
                params, volume_shape, field=variance_prior_field
            )
            payload["pipeline_variance_prior_field"] = np.asarray(str(variance_prior_field))
            payload["pipeline_variance_prior_half_voxel"] = prior_voxel
            payload["pipeline_variance_prior_half_shell"] = prior_shell
            # Also save the 'combined' field if available — useful as a higher-fidelity prior
            # since it includes the data evidence in addition to the Wiener prior.
            if "combined" in params.get("variance_est", {}):
                prior_voxel_c, prior_shell_c = _pipeline_variance_prior(
                    params, volume_shape, field="combined"
                )
                payload["pipeline_variance_combined_half_voxel"] = prior_voxel_c
                payload["pipeline_variance_combined_half_shell"] = prior_shell_c
        except (KeyError, ValueError) as exc:
            payload["pipeline_variance_prior_error"] = np.asarray(str(exc))

        # ``mean_prior`` is the FSC-derived per-voxel signal-variance prior the
        # pipeline computes from the mean half-maps via ``compute_relion_prior``.
        # The pipeline feeds this through ``regularization_init = (mean_prior +
        # epsilon) * REG_INIT_MULTIPLIER / cov_noise`` to build its covariance
        # Wiener prior. For PPCA W M-step replication we save it as-is so the
        # refinement can apply the same shape (``prior_W = mean_prior / n_pcs``).
        mean_prior_full = params.get("mean_prior")
        if mean_prior_full is None:
            # Older pipeline runs (pre v0.7 + mean_prior save) didn't store it. Re-derive
            # from the saved half-maps + noise_var using compute_relion_prior.
            try:
                mean_prior_full = _recompute_mean_prior_from_saved(
                    po, params, volume_shape=volume_shape
                )
                payload["mean_prior_source"] = np.asarray("recomputed_from_halfmaps")
            except Exception as exc:
                payload["mean_prior_error"] = np.asarray(str(exc))
                mean_prior_full = None
        else:
            payload["mean_prior_source"] = np.asarray("pipeline_params_mean_prior")

        if mean_prior_full is not None:
            mean_prior_full = np.asarray(mean_prior_full, dtype=np.float32).reshape(-1)
            if mean_prior_full.size != int(np.prod(volume_shape)):
                payload["mean_prior_error"] = np.asarray(
                    f"mean_prior size {mean_prior_full.size} != {int(np.prod(volume_shape))}"
                )
            else:
                # mean_prior from compute_relion_prior is stored full-Fourier flat,
                # real-valued (it's a variance), in the same centered Fourier
                # layout as the means. Convert to half-spectrum + shell-averaged.
                mp_half = np.abs(_full_volume_to_half(mean_prior_full.astype(np.complex64), volume_shape)).astype(np.float32)
                payload["pipeline_mean_prior_half_voxel"] = mp_half
                payload["pipeline_mean_prior_half_shell"] = _average_half_values_over_shells(
                    mp_half.astype(np.float64), volume_shape
                ).astype(np.float32)
                payload["pipeline_reg_init_multiplier"] = np.float32(1e-2)  # jax_config.REG_INIT_MULTIPLIER

    npz_path = output_dir / "ppca_init.npz"
    np.savez_compressed(npz_path, **payload)

    summary = {
        "pipeline_output": str(Path(pipeline_output).expanduser().resolve()),
        "ppca_init": str(npz_path),
        "q": q,
        "volume_shape": list(volume_shape),
        "scale_by_sqrt_eigenvalue": bool(scale_by_sqrt_eigenvalue),
        "apply_mask": bool(apply_mask),
        "mask_kind": str(mask_kind) if apply_mask else "none",
        "save_pipeline_variance_prior": bool(save_pipeline_variance_prior),
        "variance_prior_field": str(variance_prior_field),
        "s_rescaled": [float(x) for x in s_rescaled[:q]],
        "mu_rms": float(np.sqrt(np.mean(np.asarray(mu_masked, dtype=np.float64) ** 2))),
        "mu_unmasked_rms": float(np.sqrt(np.mean(np.asarray(mu, dtype=np.float64) ** 2))),
        "W_rms": [float(np.sqrt(np.mean(np.asarray(W_i, dtype=np.float64) ** 2))) for W_i in W],
        "W_unmasked_rms": [float(np.sqrt(np.mean(np.asarray(W_i, dtype=np.float64) ** 2))) for W_i in W_unmasked],
        "mask_voxels_inside": float(np.sum(chosen_mask > 0.5)),
        "mask_total_voxels": int(chosen_mask.size),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return npz_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline-output", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--q", type=int, default=4)
    parser.add_argument(
        "--scale-by-sqrt-eigenvalue",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--apply-mask",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--mask-kind",
        choices=("dilated", "plain"),
        default="dilated",
        help="Which pipeline mask to apply. 'dilated' matches what the pipeline PCA solver uses internally.",
    )
    parser.add_argument(
        "--save-pipeline-variance-prior",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--variance-prior-field",
        choices=("prior", "combined"),
        default="prior",
        help="Which variance_est field to save as the pipeline-side W prior.",
    )
    args = parser.parse_args()
    out = prepare_ppca_init_v2(
        args.pipeline_output,
        args.output_dir,
        q=int(args.q),
        scale_by_sqrt_eigenvalue=bool(args.scale_by_sqrt_eigenvalue),
        apply_mask=bool(args.apply_mask),
        mask_kind=str(args.mask_kind),
        save_pipeline_variance_prior=bool(args.save_pipeline_variance_prior),
        variance_prior_field=str(args.variance_prior_field),
    )
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
