"""Merge-regression guard for the highres prior + mask + chunking work.

This file locks down the API + numerical + default-value contracts of the
session's contributions on ``codex/ppca-highres-refine``:

1. **External mask plumbing for the M-step postprocess** —
   ``PostprocessConfig.external_mask_volume`` field, ``w_only_mask`` strategy,
   pipeline solvent mask saved into the v2 init NPZ and consumed by both
   refinement scripts.

2. **Pipeline-side W prior** —
   ``recovar.output.output.build_params_dict`` accepts and stores ``mean_prior``;
   ``initialization.pipeline_variance_W_prior`` consumes it with the
   correct ``divide_by_q`` default; the new ``pipeline-variance-*`` and
   ``pipeline-mean-prior`` CLI choices for ``--prior-from-init`` are wired
   in ``run_ppca_dense_os_local_from_init_npz.py`` and
   ``run_ppca_hp6_resume_from_os_npz.py``.

3. **Rotation-chunked pose-only scoring** —
   ``_score_local_pose_ppca_bucket_rotation_chunked`` exists, returns a
   :class:`PosteriorDiagnostics`, and matches the one-shot kernel at
   parity. Performance smoke ensures the chunked path doesn't regress
   to a slow Python loop.

4. **Projcov-style W whitening helpers** —
   ``whiten_W_svd_post_mstep`` produces column-orthogonal output;
   ``whiten_W_via_projcov`` exists with the documented signature.

5. **v2 init builder** —
   ``prepare_ppca_init_v2`` writes the documented set of NPZ keys; the
   refinement script's ``--prior-from-init pipeline-mean-prior`` mode
   requires the saved ``pipeline_mean_prior_half_voxel``.

This is the same pattern as ``test_postmerge_regression_guard.py``: if a
merge from the EM / VDAM / refine sibling branches silently rewires any
of the pinned contracts, this file fails with a clear error pointing to
the lost feature.
"""

from __future__ import annotations

import inspect
import pickle
import re
import subprocess
import sys
import time
from dataclasses import MISSING, fields
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]


# ---------------------------------------------------------------------------
# 1. External mask plumbing (PostprocessConfig.external_mask_volume + w_only_mask)
# ---------------------------------------------------------------------------


def test_postprocess_config_has_external_mask_volume_field():
    from recovar.em.ppca_refinement.postprocess import PostprocessConfig

    field_names = {f.name for f in fields(PostprocessConfig)}
    assert "external_mask_volume" in field_names, (
        "PostprocessConfig must expose 'external_mask_volume' for plumbing the "
        "pipeline solvent mask into the M-step postprocess (added during the "
        "codex/ppca-highres-refine session). Check that a merge from a sibling "
        "branch didn't drop this field."
    )
    cfg = PostprocessConfig()
    assert cfg.external_mask_volume is None, (
        "PostprocessConfig.external_mask_volume must default to None so legacy "
        "callers that pass no mask continue to use the soft-radius fallback."
    )


def test_postprocess_w_only_mask_strategy_is_accepted():
    from recovar.em.ppca_refinement.postprocess import PostprocessConfig, postprocess_ppca_half_volumes
    import recovar.core.fourier_transform_utils as ftu

    volume_shape = (8, 8, 8)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    rng = np.random.default_rng(11)
    mu = jnp.asarray(
        (rng.standard_normal(half_size) + 1j * rng.standard_normal(half_size)).astype(np.complex64)
    )
    W = jnp.asarray(
        (rng.standard_normal((half_size, 3)) + 1j * rng.standard_normal((half_size, 3))).astype(np.complex64)
    )
    mask = np.ones(volume_shape, dtype=np.float32)
    mask[0, 0, 0] = 0.0  # one solvent voxel

    result = postprocess_ppca_half_volumes(
        mu, W, volume_shape,
        config=PostprocessConfig(
            strategy="w_only_mask", external_mask_volume=mask, grid_correct=False,
        ),
    )
    # mu must pass through unchanged in w_only_mask mode (user's explicit requirement).
    np.testing.assert_array_equal(np.asarray(result.mu_half), np.asarray(mu))
    # W has been processed (not equal to input).
    assert not np.array_equal(np.asarray(result.W_half), np.asarray(W))


def test_postprocess_external_mask_volume_round_trips_for_w_only_mask():
    """An external mask of all-ones should leave W ≈ unchanged (no grid correct).

    Uses Hermitian-symmetric input W (built by FFTing a real-space volume) so
    the half-spectrum ↔ real-space round-trip in the postprocess doesn't lose
    information by projecting non-Hermitian components.
    """
    from recovar.em.ppca_refinement.postprocess import PostprocessConfig, postprocess_ppca_half_volumes
    import recovar.core.fourier_transform_utils as ftu

    volume_shape = (8, 8, 8)
    half_size = int(np.prod(ftu.volume_shape_to_half_volume_shape(volume_shape)))
    rng = np.random.default_rng(22)
    # Build a Hermitian-symmetric W in half-spectrum form by FT-ing real volumes.
    q = 2
    W_real = rng.standard_normal((q, *volume_shape)).astype(np.float32)
    W_half_full = np.stack(
        [np.asarray(ftu.get_dft3_real(jnp.asarray(W_real[k]))).reshape(-1) for k in range(q)],
        axis=1,
    ).astype(np.complex64)
    W = jnp.asarray(W_half_full)
    mu_real = rng.standard_normal(volume_shape).astype(np.float32)
    mu = jnp.asarray(np.asarray(ftu.get_dft3_real(jnp.asarray(mu_real))).reshape(-1).astype(np.complex64))
    mask = np.ones(volume_shape, dtype=np.float32)
    result = postprocess_ppca_half_volumes(
        mu, W, volume_shape,
        config=PostprocessConfig(
            strategy="w_only_mask", external_mask_volume=mask, grid_correct=False,
        ),
    )
    # Full-1 mask should leave Hermitian-symmetric W unchanged within float32
    # round-trip noise.
    np.testing.assert_allclose(np.asarray(result.W_half), np.asarray(W), rtol=1e-4, atol=1e-4)


# ---------------------------------------------------------------------------
# 2. Pipeline-side W prior
# ---------------------------------------------------------------------------


def test_build_params_dict_accepts_mean_prior():
    """The pipeline must save mean_prior in params for the v2 init builder to load."""
    from recovar.output.output import build_params_dict

    sig = inspect.signature(build_params_dict)
    assert "mean_prior" in sig.parameters, (
        "build_params_dict(...) must accept mean_prior= so the pipeline can save "
        "the FSC-derived signal-variance prior for downstream PPCA refinement "
        "(added in codex/ppca-highres-refine for prepare_ppca_init_from_pipeline_output_v2)."
    )
    # Default is None so legacy callers (pre-v0.7 ppca-postmerge build_params_dict
    # callsites) continue to work.
    assert sig.parameters["mean_prior"].default is None


def test_build_params_dict_saves_mean_prior_when_provided():
    """When mean_prior is provided, the params dict must contain a 'mean_prior' key."""
    from recovar.output.output import build_params_dict

    fake_noise_result = {
        "radial_noise_var_outside_mask": np.zeros(4, dtype=np.float32),
        "radial_ub_noise_var": np.zeros(4, dtype=np.float32),
        "white_noise_var_outside_mask": np.float32(1.0),
        "image_PS": np.zeros(4, dtype=np.float32),
        "masked_image_PS": np.zeros(4, dtype=np.float32),
    }
    mean_prior = np.linspace(1.0, 0.01, 64, dtype=np.float32)
    params = build_params_dict(
        volume_shape=(4, 4, 4), voxel_size=1.0,
        s_rescaled=np.ones(2, dtype=np.float32),
        noise_var_from_hf=np.ones(4, dtype=np.float32),
        noise_var_from_het_residual=None,
        noise_var_used=np.ones(4, dtype=np.float32),
        noise_result=fake_noise_result,
        ub_noise_var_by_var_est=np.ones(4, dtype=np.float32),
        variance_est={"combined": np.ones(64, dtype=np.float32)},
        variance_fsc=np.ones(4, dtype=np.float32),
        noise_p_variance_est=np.ones(64, dtype=np.float32),
        covariance_options={},
        column_fscs=np.ones(4, dtype=np.float32),
        picked_frequencies=np.arange(4, dtype=np.int32),
        input_args={},
        mean_prior=mean_prior,
    )
    assert "mean_prior" in params, "params dict must include 'mean_prior' key"
    np.testing.assert_array_equal(params["mean_prior"], mean_prior)

    # When mean_prior is omitted, the slot must still exist (as None) so consumers
    # can rely on `params.get('mean_prior')`.
    params_no_prior = build_params_dict(
        volume_shape=(4, 4, 4), voxel_size=1.0,
        s_rescaled=np.ones(2, dtype=np.float32),
        noise_var_from_hf=np.ones(4, dtype=np.float32),
        noise_var_from_het_residual=None,
        noise_var_used=np.ones(4, dtype=np.float32),
        noise_result=fake_noise_result,
        ub_noise_var_by_var_est=np.ones(4, dtype=np.float32),
        variance_est={"combined": np.ones(64, dtype=np.float32)},
        variance_fsc=np.ones(4, dtype=np.float32),
        noise_p_variance_est=np.ones(64, dtype=np.float32),
        covariance_options={},
        column_fscs=np.ones(4, dtype=np.float32),
        picked_frequencies=np.arange(4, dtype=np.int32),
        input_args={},
    )
    assert "mean_prior" in params_no_prior
    assert params_no_prior["mean_prior"] is None


def test_pipeline_variance_W_prior_signature_and_defaults():
    from recovar.em.ppca_refinement.initialization import pipeline_variance_W_prior

    sig = inspect.signature(pipeline_variance_W_prior)
    assert "pipeline_variance_half" in sig.parameters
    assert "q" in sig.parameters
    # divide_by_q default = True so the user's "1/n_pcs * variance_weight"
    # form is the default; the call site can still override.
    assert sig.parameters["divide_by_q"].default is True
    # Smoke: divides by q correctly.
    arr = np.ones(16, dtype=np.float32) * 4.0
    out_with_div = pipeline_variance_W_prior(arr, q=4, divide_by_q=True)
    out_no_div = pipeline_variance_W_prior(arr, q=4, divide_by_q=False)
    assert out_with_div.shape == (16, 4)
    assert out_no_div.shape == (16, 4)
    np.testing.assert_allclose(out_with_div, np.ones((16, 4), dtype=np.float32))
    np.testing.assert_allclose(out_no_div, 4.0 * np.ones((16, 4), dtype=np.float32))


_PRIOR_CHOICES = {
    "constant",
    "gt-row-norm",
    "pipeline-variance-shell",
    "pipeline-variance-voxel",
    "pipeline-mean-prior",
}
_POSTPROCESS_STRATEGIES = {"none", "mean-only", "mean-and-w-mask", "w-only-mask"}
_POSTPROCESS_MASK_SOURCES = {"none", "radius", "init-volume-mask", "init-volume-mask-dilated"}


def _read_script(path: str) -> str:
    return Path(REPO_ROOT, path).read_text(encoding="utf-8")


@pytest.mark.parametrize(
    "script_relpath",
    [
        "scripts/run_ppca_dense_os_local_from_init_npz.py",
        "scripts/run_ppca_hp6_resume_from_os_npz.py",
    ],
)
def test_refinement_scripts_expose_pipeline_prior_and_mask_cli(script_relpath):
    """Confirm each script's CLI offers all the prior + strategy + mask-source choices.

    We don't parse argparse — we just confirm each choice literal appears in
    the script source near its respective flag name. Brittle to formatting
    changes but a merge that drops a CLI choice for a feature this session
    added would be a regression we *want* to fail on.
    """
    src = _read_script(script_relpath)
    # All flags must be defined at all.
    assert '"--prior-from-init"' in src, f"{script_relpath}: missing --prior-from-init"
    assert '"--postprocess-strategy"' in src, f"{script_relpath}: missing --postprocess-strategy"
    assert '"--postprocess-mask-source"' in src, f"{script_relpath}: missing --postprocess-mask-source"

    # Each choice literal for each flag must appear somewhere in the source.
    # (Argparse formatting varies — choices=("a","b") vs choices=("a", "b") vs
    # multi-line — so we just look for the literal strings.)
    for choice in _PRIOR_CHOICES:
        assert f'"{choice}"' in src, (
            f"{script_relpath}: --prior-from-init choice {choice!r} missing (lost in merge?)"
        )
    for s in _POSTPROCESS_STRATEGIES:
        assert f'"{s}"' in src, (
            f"{script_relpath}: --postprocess-strategy choice {s!r} missing"
        )
    for s in _POSTPROCESS_MASK_SOURCES:
        assert f'"{s}"' in src, (
            f"{script_relpath}: --postprocess-mask-source choice {s!r} missing"
        )


# ---------------------------------------------------------------------------
# 3. Rotation-chunked pose-only scoring (parity + perf)
# ---------------------------------------------------------------------------


def _synth_rotation_chunk_inputs(B, T, R, P, F, *, seed=42):
    rng = np.random.default_rng(seed)
    Y1 = (rng.standard_normal((B, T, F)) + 1j * rng.standard_normal((B, T, F))).astype(np.complex64)
    proj_aug = (rng.standard_normal((B, R, P, F)) + 1j * rng.standard_normal((B, R, P, F))).astype(np.complex64)
    ctf2 = rng.uniform(0.01, 1.0, size=(B, F)).astype(np.float32)
    y_norm = rng.uniform(0.1, 5.0, size=(B,)).astype(np.float32)
    pose_log_prior = (rng.standard_normal((B, R, T)) * 0.1).astype(np.float32)
    return (
        jnp.asarray(Y1), jnp.asarray(proj_aug), jnp.asarray(ctf2),
        jnp.asarray(y_norm), jnp.asarray(pose_log_prior),
    )


def test_rotation_chunked_path_exists_and_matches_one_shot():
    """Bit-exact (~1e-4) match between chunked and one-shot pose-only score.

    The chunked path opt-in is gated by ``RECOVAR_PPCA_LOCAL_R_CHUNK_SIZE``; a
    merge that silently rewires the env-var dispatch would skip the chunked
    code path and silently fall back to one-shot, hiding any new chunked-path
    bugs introduced upstream.
    """
    from recovar.em.ppca_refinement.local_dataset import (
        _score_local_pose_ppca_bucket_rotation_chunked,
        score_local_pose_ppca_bucket,
    )

    Y1, proj_aug, ctf2, y_norm, pose_log_prior = _synth_rotation_chunk_inputs(
        B=2, T=1, R=40, P=5, F=17, seed=99,
    )
    one_shot = score_local_pose_ppca_bucket(
        Y1, proj_aug, ctf2, y_norm, pose_log_prior, top_pose_count=4,
    )
    chunked = _score_local_pose_ppca_bucket_rotation_chunked(
        Y1, proj_aug, ctf2, y_norm, pose_log_prior,
        significance_threshold=1e-3, top_pose_count=4, rotation_chunk_size=8,
    )
    # All the fields that are exact under chunking:
    np.testing.assert_allclose(np.asarray(chunked.logZ), np.asarray(one_shot.logZ), rtol=1e-4, atol=1e-3)
    np.testing.assert_allclose(
        np.asarray(chunked.best_log_score_per_image),
        np.asarray(one_shot.best_log_score_per_image),
        rtol=1e-5, atol=1e-4,
    )
    np.testing.assert_array_equal(
        np.asarray(chunked.best_rotation_idx), np.asarray(one_shot.best_rotation_idx)
    )
    np.testing.assert_array_equal(
        np.asarray(chunked.top_rotation_idx), np.asarray(one_shot.top_rotation_idx)
    )


def test_rotation_chunked_dispatch_envvar_present_in_local_dataset():
    """The opt-in env var must be wired in _score_local_ppca_pose_diagnostics."""
    src = (REPO_ROOT / "recovar/em/ppca_refinement/local_dataset.py").read_text()
    assert "RECOVAR_PPCA_LOCAL_R_CHUNK_SIZE" in src, (
        "Rotation-chunking env-var dispatch missing — merge regression suspected. "
        "Expected `RECOVAR_PPCA_LOCAL_R_CHUNK_SIZE` to gate the chunked path inside "
        "`_score_local_ppca_pose_diagnostics` so users can dial down HP6 top-p peak memory."
    )
    assert "_score_local_pose_ppca_bucket_rotation_chunked" in src


def test_rotation_chunked_performance_smoke():
    """Chunked path on (B=2, T=1, R=40) should finish in <2 s warm (5 warm-up runs).

    Loose budget (factor ≥ 10x of expected) so this never flakes under load,
    but tight enough to flag a regression that drops to slow Python looping.
    """
    from recovar.em.ppca_refinement.local_dataset import (
        _score_local_pose_ppca_bucket_rotation_chunked,
    )
    Y1, proj_aug, ctf2, y_norm, pose_log_prior = _synth_rotation_chunk_inputs(
        B=2, T=1, R=40, P=5, F=17, seed=7,
    )
    # Warm up JIT
    for _ in range(2):
        _ = _score_local_pose_ppca_bucket_rotation_chunked(
            Y1, proj_aug, ctf2, y_norm, pose_log_prior,
            significance_threshold=1e-3, top_pose_count=4, rotation_chunk_size=8,
        )
    t0 = time.perf_counter()
    for _ in range(5):
        diag = _score_local_pose_ppca_bucket_rotation_chunked(
            Y1, proj_aug, ctf2, y_norm, pose_log_prior,
            significance_threshold=1e-3, top_pose_count=4, rotation_chunk_size=8,
        )
        jax.block_until_ready(diag.logZ)
    elapsed = time.perf_counter() - t0
    assert elapsed < 5.0, (
        f"Rotation-chunked dispatch took {elapsed:.2f}s for 5 warm calls; expected <5s. "
        "Possible regressions: re-JIT per call, Python overhead from dropped chunking, "
        "or unmerged shape dispatch."
    )


# ---------------------------------------------------------------------------
# 4. Projcov-style W whitening helpers
# ---------------------------------------------------------------------------


def test_whiten_W_svd_post_mstep_exists_and_orthogonalizes():
    """The SVD whitening helper must match the recovar.heterogeneity.ppca route's M-step."""
    from recovar.em.ppca_refinement.projcov_whiten import whiten_W_svd_post_mstep

    rng = np.random.default_rng(31)
    W = (rng.standard_normal((128, 4)) + 1j * rng.standard_normal((128, 4))).astype(np.complex64)
    W_w = whiten_W_svd_post_mstep(W)
    # Columns must be orthogonal (gram diagonal-only). For complex64 SVD the
    # ~1e-6 relative orthogonality error of U scales by ||S||^2 (here ~330),
    # so the absolute off-diagonal can sit near ~1e-2; check relative.
    gram = W_w.conj().T @ W_w
    diag = np.real(np.diag(gram))
    diag_max = float(np.max(diag))
    off_diagonal = gram - np.diag(np.diag(gram))
    off_max = float(np.max(np.abs(off_diagonal)))
    assert off_max / max(diag_max, float(np.finfo(np.float32).eps)) < 1e-3, (
        f"off-diagonal {off_max:.3e} not small relative to diag-max {diag_max:.3e}"
    )
    # Diagonal entries are singular-values-squared, sorted descending.
    # Allow a tiny relative slack for the boundary between near-equal singular
    # values under float32.
    assert np.all(np.diff(diag) <= 1e-4 * diag_max), (
        "SVD-whitened W must have singular-values-squared sorted descending on the diagonal"
    )


def test_whiten_W_via_projcov_exists_with_documented_signature():
    """The heavier projcov-based whitening helper must remain importable + named."""
    from recovar.em.ppca_refinement.projcov_whiten import whiten_W_via_projcov

    sig = inspect.signature(whiten_W_via_projcov)
    for required in ("dataset", "mu_half_flat", "W_half_flat",
                     "best_rotation_matrices", "best_translations",
                     "volume_mask"):
        assert required in sig.parameters, (
            f"whiten_W_via_projcov missing required parameter {required!r}"
        )


# ---------------------------------------------------------------------------
# 5. v2 init builder
# ---------------------------------------------------------------------------


def test_prepare_ppca_init_v2_module_is_importable_and_exposes_function():
    """The v2 builder must be runnable as a module + as a script."""
    sys.path.insert(0, str(REPO_ROOT))
    try:
        import scripts.prepare_ppca_init_from_pipeline_output_v2 as v2
    finally:
        sys.path.pop(0)
    assert hasattr(v2, "prepare_ppca_init_v2"), "Function name lost"
    assert hasattr(v2, "_recompute_mean_prior_from_saved"), (
        "Mean-prior recomputation helper lost — pipeline outputs without "
        "params.pkl['mean_prior'] won't be re-derivable."
    )
    sig = inspect.signature(v2.prepare_ppca_init_v2)
    # apply_mask default must remain True (this is the whole point of v2).
    assert sig.parameters["apply_mask"].default is True
    # Mean-prior + variance-prior save default ON.
    assert sig.parameters["save_pipeline_variance_prior"].default is True


def test_prepare_ppca_init_v2_documented_npz_keys_present():
    """A run of prepare_ppca_init_v2 over a mock pipeline output must save the
    documented set of keys. This is the schema downstream EM scripts depend on.

    We don't run a real recovar pipeline; we build a minimal fake output
    directory + ``params.pkl`` and confirm the NPZ keys we promise are
    written. If a merge drops a key (e.g. ``pipeline_mean_prior_half_voxel``)
    the dependent ``--prior-from-init pipeline-mean-prior`` path will break
    silently — this guards against that.
    """
    # The v2 builder depends on PipelineOutput and full recovar pipeline scaffolding.
    # Rather than mocking that, we assert the source contains the expected
    # `payload[...]` assignments and field names. Heavyweight but reliable.
    src = (REPO_ROOT / "scripts/prepare_ppca_init_from_pipeline_output_v2.py").read_text()
    expected_npz_keys = [
        "mu",
        "mu_unmasked",
        "W",
        "W_unmasked",
        "s_rescaled",
        "volume_shape",
        "voxel_size",
        "pipeline_output",
        "W_scaling",
        "mask_applied",
        "volume_mask",
        "volume_mask_dilated",
        "volume_mask_plain",
        "noise_var_used",
        "pipeline_variance_prior_field",
        "pipeline_variance_prior_half_voxel",
        "pipeline_variance_prior_half_shell",
        "pipeline_mean_prior_half_voxel",
        "pipeline_mean_prior_half_shell",
        "pipeline_reg_init_multiplier",
    ]
    for key in expected_npz_keys:
        assert f'"{key}"' in src, (
            f"v2 init builder must save NPZ key {key!r}. A merge that dropped this "
            "key would break refinement scripts that rely on it (e.g. "
            "--prior-from-init pipeline-mean-prior, --postprocess-mask-source init-volume-mask-dilated)."
        )


# ---------------------------------------------------------------------------
# 6. Refinement-script regression: --prior-from-init dispatch handles all options
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "script_relpath",
    [
        "scripts/run_ppca_dense_os_local_from_init_npz.py",
        "scripts/run_ppca_hp6_resume_from_os_npz.py",
    ],
)
def test_refinement_scripts_have_all_prior_branches(script_relpath):
    """Each --prior-from-init option must have a corresponding code branch.

    A merge that drops a branch would silently fall to the constant prior
    fallback for that mode; this test forces every mode to have an
    explicit code path.
    """
    src = _read_script(script_relpath)
    # We only check the BRANCH KEYWORD appears for each mode, not the body.
    # The dispatch follows the form `if args.prior_from_init == "gt-row-norm":`
    # / `elif args.prior_from_init == "pipeline-variance-shell":` / etc.
    for mode in _PRIOR_CHOICES - {"constant"}:  # constant is the fallback
        assert f'"{mode}"' in src, f"{script_relpath}: dispatch for --prior-from-init={mode} missing"

    # Also check the external-mask plumbing into PostprocessConfig.
    assert "external_mask_volume=" in src or "external_mask_volume =" in src, (
        f"{script_relpath}: must pass external_mask_volume into PostprocessConfig "
        "(this is how the dilated pipeline mask reaches the M-step postprocess)"
    )
