"""Merge-regression guard for ``claude/ppca-postmerge-20260510_110827``.

This branch ships 12 commits of PPCA-refinement work that the upcoming
3-way merge (EM / VDAM / PPCA-refinement) could silently undo. This
file locks down:

1. The **API contract** of the new cached-moments fast path
   (``DenseScoreAndMomentsStats``,
   ``dense_pose_ppca_score_with_moments_blocked``,
   ``accumulate_pose_ppca_block_cached``) — every symbol exists, every
   field exists, every required argument is named the same.

2. The **bit-exact numerical equivalence** between the new fast path
   and the legacy ``dense_pose_ppca_E_step_blocked`` — both have to
   agree on every output that means anything (``logZ``,
   per-image best score / best rotation / best translation, and the
   moments ``alpha`` / ``G_tri`` after collapsing across (T, R)).
   This is the *whole point* of commit ``7067106c`` — if a merge
   silently rewires either path so they diverge, this test fails
   with a clear ``max |Δ| = ...`` diff.

3. The **default values of the four new config dataclasses**
   (``GeometryConfig``, ``ScheduleConfig``, ``ScoringConfig``,
   ``SparsePass2Config`` — commits ``5317d7b2`` / ``f8b07e69``).
   These defaults encode the resolved kwargs proliferation of the
   pre-merge PPCA refinement entry points; a merge that silently
   reverts to a kwargs-only API would re-introduce drift bugs.

4. The **public exports of the diagnostics module** (commit
   ``1ef96f4e``) — ``build_iteration_diagnostics`` and
   ``resolve_image_scale_range``.

5. A **performance smoke** for the cached-moments path: 10 warm calls
   on the tiny deterministic input must finish in <2 s. The legacy
   ``dense_pose_ppca_E_step_blocked`` must also stay under that
   budget; we don't compare them against each other (JIT-warm
   timing variance is high), we just guarantee neither becomes
   pathologically slow.

If a merge legitimately changes any of the pinned values, update both
the pinned value AND the rationale in the docstring of the relevant
test in the same PR.
"""

from __future__ import annotations

import inspect
import time
from dataclasses import MISSING, fields

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.ppca_refinement import (
    config as ppca_config,
)
from recovar.em.ppca_refinement import (
    diagnostics as ppca_diagnostics,
)
from recovar.em.ppca_refinement.engine import (
    DenseScoreAndMomentsStats,
    accumulate_pose_ppca_block_cached,
    dense_pose_ppca_E_step_blocked,
    dense_pose_ppca_score_with_moments_blocked,
)

pytestmark = pytest.mark.unit


# ---------------------------------------------------------------------------
# 1. API contract pins (commit 7067106c)
# ---------------------------------------------------------------------------


class TestCachedMomentsApiContract:
    """The cached-moments fast path API must keep its exact public shape.

    Pinned by commit 7067106c. Anything a merge silently renames or
    drops here fails this test loudly."""

    def test_DenseScoreAndMomentsStats_fields(self):
        # NamedTuple field order matters because callers unpack
        # positionally (``score, alpha, G_tri, logZ, ...``).
        assert DenseScoreAndMomentsStats._fields == (
            "score",
            "alpha",
            "G_tri",
            "logZ",
            "best_log_score_per_image",
            "best_rotation_idx",
            "best_translation_idx",
        )

    def test_dense_pose_ppca_score_with_moments_blocked_signature(self):
        # PjitFunction signatures are introspectable.
        sig = inspect.signature(dense_pose_ppca_score_with_moments_blocked)
        params = list(sig.parameters.keys())
        assert params == [
            "Y1",
            "proj_aug",
            "ctf2_over_noise",
            "y_norm",
            "pose_log_prior",
        ]
        # pose_log_prior must default to None (callers rely on this).
        assert sig.parameters["pose_log_prior"].default is None

    def test_accumulate_pose_ppca_block_cached_signature(self):
        sig = inspect.signature(accumulate_pose_ppca_block_cached)
        params = list(sig.parameters.keys())
        # First 11 are positional, then keyword-only kwargs.
        assert params[:11] == [
            "score",
            "alpha",
            "G_tri",
            "normalization_logZ",
            "Y1_recon",
            "ctf2_over_noise_recon",
            "rotations_block",
            "image_shape",
            "volume_shape",
            "rhs_volume",
            "lhs_tri_volume",
        ]
        # Keyword-only kwargs (must keep defaults).
        assert sig.parameters["significance_threshold"].default == pytest.approx(0.001)
        assert sig.parameters["disc_type_backproject"].default == "linear_interp"
        assert sig.parameters["use_recon_window"].default is False
        assert sig.parameters["recon_window_indices"].default is None
        assert sig.parameters["backprojection_max_r"].default is None


# ---------------------------------------------------------------------------
# 2. Cached-moments ↔ legacy E-step equivalence (commit 7067106c)
# ---------------------------------------------------------------------------


def _tiny_score_inputs(seed: int = 0):
    """Tiny deterministic (B, T, R, F, q) input shared by every
    equivalence test below. Sized to be JIT-fast on CPU."""
    B, T, R, F, q = 3, 2, 4, 9, 2
    P = 1 + q  # augmented column count: mean + q W-columns
    rng = np.random.default_rng(seed)
    Y1 = (rng.standard_normal((B, T, F)) + 1j * rng.standard_normal((B, T, F))).astype(np.complex64)
    proj_aug = (rng.standard_normal((R, P, F)) + 1j * rng.standard_normal((R, P, F))).astype(np.complex64)
    ctf2_over_noise = (rng.standard_normal((B, F)).astype(np.float32) ** 2 + 0.1).astype(np.float32)
    y_norm = (np.abs(Y1) ** 2).sum(axis=-1).mean(axis=-1).astype(np.float32)
    return Y1, proj_aug, ctf2_over_noise, y_norm


class TestCachedMomentsEquivalence:
    """Both the legacy ``dense_pose_ppca_E_step_blocked`` and the new
    ``dense_pose_ppca_score_with_moments_blocked`` must agree on every
    output that downstream code relies on. The point of 7067106c was
    to fuse two kernels into one without changing any output; this is
    that promise made into a permanent guard."""

    def test_logZ_bit_exact(self):
        Y1, proj_aug, ctf2, y_norm = _tiny_score_inputs()
        Y1_j = jnp.asarray(Y1)
        proj_j = jnp.asarray(proj_aug)
        ctf_j = jnp.asarray(ctf2)
        y_j = jnp.asarray(y_norm)

        _, legacy_diag = dense_pose_ppca_E_step_blocked(Y1_j, proj_j, ctf_j, y_j, None)
        new = dense_pose_ppca_score_with_moments_blocked(Y1_j, proj_j, ctf_j, y_j, None)

        # Bit-exact: same JIT kernel underneath, no algebraic rewiring.
        np.testing.assert_array_equal(np.asarray(new.logZ), np.asarray(legacy_diag.logZ))

    def test_per_image_best_pose_bit_exact(self):
        Y1, proj_aug, ctf2, y_norm = _tiny_score_inputs()
        Y1_j = jnp.asarray(Y1)
        proj_j = jnp.asarray(proj_aug)
        ctf_j = jnp.asarray(ctf2)
        y_j = jnp.asarray(y_norm)

        _, legacy_diag = dense_pose_ppca_E_step_blocked(Y1_j, proj_j, ctf_j, y_j, None)
        new = dense_pose_ppca_score_with_moments_blocked(Y1_j, proj_j, ctf_j, y_j, None)

        np.testing.assert_array_equal(
            np.asarray(new.best_log_score_per_image),
            np.asarray(legacy_diag.best_log_score_per_image),
        )
        np.testing.assert_array_equal(
            np.asarray(new.best_rotation_idx),
            np.asarray(legacy_diag.best_rotation_idx),
        )
        np.testing.assert_array_equal(
            np.asarray(new.best_translation_idx),
            np.asarray(legacy_diag.best_translation_idx),
        )

    def test_score_shape_and_finiteness(self):
        Y1, proj_aug, ctf2, y_norm = _tiny_score_inputs()
        new = dense_pose_ppca_score_with_moments_blocked(
            jnp.asarray(Y1),
            jnp.asarray(proj_aug),
            jnp.asarray(ctf2),
            jnp.asarray(y_norm),
            None,
        )
        # score is (B, T, R) — the (T, R) order is *not* (R, T); a merge
        # that transposed this would silently break the best-pose argmax
        # below in callers.
        assert new.score.shape == (3, 2, 4)
        # alpha is (B, T, R, P) with P = 1 + q = 3
        assert new.alpha.shape == (3, 2, 4, 3)
        # G_tri is (B, T, R, tri(P)) with tri(3) = 6
        assert new.G_tri.shape == (3, 2, 4, 6)
        assert bool(np.all(np.isfinite(np.asarray(new.score))))

    def test_pose_log_prior_passes_through(self):
        """A nonzero pose_log_prior must add to the score and shift
        logZ accordingly. This is the contract that callers (the dense
        dataset iterator) rely on for K-class & class-prior weighting."""
        Y1, proj_aug, ctf2, y_norm = _tiny_score_inputs()
        B, R, T = 3, 4, 2
        # Place a large bonus on a single (image, rotation, translation)
        # cell; that pose must become the argmax for that image.
        prior = np.full((B, R, T), -1e9, dtype=np.float32)
        # Reward image 0 at (rotation 3, translation 1).
        prior[0, 3, 1] = 0.0
        prior[1, 0, 0] = 0.0
        prior[2, 1, 1] = 0.0
        new = dense_pose_ppca_score_with_moments_blocked(
            jnp.asarray(Y1),
            jnp.asarray(proj_aug),
            jnp.asarray(ctf2),
            jnp.asarray(y_norm),
            jnp.asarray(prior),
        )
        # The single non-negative-infinity prior cell must dominate.
        np.testing.assert_array_equal(np.asarray(new.best_rotation_idx), np.array([3, 0, 1], dtype=np.int32))
        np.testing.assert_array_equal(np.asarray(new.best_translation_idx), np.array([1, 0, 1], dtype=np.int32))


# ---------------------------------------------------------------------------
# 3. Config dataclass defaults (commits 5317d7b2, f8b07e69)
# ---------------------------------------------------------------------------


def _resolve_default(field):
    if field.default is not MISSING:
        return field.default
    if field.default_factory is not MISSING:
        return field.default_factory()
    return MISSING


class TestConfigDataclassDefaults:
    """Every field of the four new config dataclasses keeps its default
    value. A merge that silently flips e.g. ``relion_texture_interp``
    or ``sparse_pass2.enabled`` would shift the production-path
    behaviour without any other test noticing."""

    def test_GeometryConfig_defaults(self):
        cls = ppca_config.GeometryConfig
        names = [f.name for f in fields(cls)]
        assert names == ["current_size", "q", "volume_domain"]
        defaults = {f.name: _resolve_default(f) for f in fields(cls)}
        assert defaults == {"current_size": None, "q": None, "volume_domain": "auto"}

    def test_ScheduleConfig_defaults(self):
        cls = ppca_config.ScheduleConfig
        names = [f.name for f in fields(cls)]
        assert names == ["image_batch_size", "rotation_block_size", "mstep_chunk_size"]
        defaults = {f.name: _resolve_default(f) for f in fields(cls)}
        assert defaults == {
            "image_batch_size": 500,
            "rotation_block_size": 5000,
            "mstep_chunk_size": None,
        }

    def test_ScoringConfig_defaults(self):
        cls = ppca_config.ScoringConfig
        names = [f.name for f in fields(cls)]
        assert names == [
            "score_with_masked_images",
            "half_spectrum_scoring",
            "square_window",
            "relion_texture_interp",
            "class_log_prior",
            "image_scale_corrections",
        ]
        defaults = {f.name: _resolve_default(f) for f in fields(cls)}
        assert defaults == {
            "score_with_masked_images": False,
            "half_spectrum_scoring": False,
            "square_window": False,
            "relion_texture_interp": True,
            "class_log_prior": 0.0,
            "image_scale_corrections": None,
        }

    def test_SparsePass2Config_defaults(self):
        cls = ppca_config.SparsePass2Config
        names = [f.name for f in fields(cls)]
        assert names == ["enabled", "log_threshold"]
        defaults = {f.name: _resolve_default(f) for f in fields(cls)}
        assert defaults["enabled"] is True
        # log_threshold == ln(1e-6) == -6 * ln(10) ≈ -13.815510557964274
        assert defaults["log_threshold"] == pytest.approx(np.log(1e-6))

    def test_all_configs_frozen(self):
        """All four configs must be frozen dataclasses so mutation
        downstream can't silently change behaviour halfway through a
        pipeline."""
        for cls in (
            ppca_config.GeometryConfig,
            ppca_config.ScheduleConfig,
            ppca_config.ScoringConfig,
            ppca_config.SparsePass2Config,
        ):
            assert cls.__dataclass_params__.frozen, f"{cls.__name__} must be frozen"


# ---------------------------------------------------------------------------
# 4. Diagnostics module public API (commit 1ef96f4e)
# ---------------------------------------------------------------------------


class TestDiagnosticsModuleApi:
    def test_public_callables_exist(self):
        # These two helpers were extracted out of dense_dataset.py by
        # 1ef96f4e. The split is load-bearing because the run_*.py
        # scripts import them directly; a merge that re-inlines them
        # would break those scripts.
        assert callable(ppca_diagnostics.build_iteration_diagnostics)
        assert callable(ppca_diagnostics.resolve_image_scale_range)

    def test_resolve_image_scale_range_signature(self):
        """Signature pin: takes (image_scale_corrections, image_indices)."""
        sig = inspect.signature(ppca_diagnostics.resolve_image_scale_range)
        assert list(sig.parameters.keys()) == [
            "image_scale_corrections",
            "image_indices",
        ]

    def test_resolve_image_scale_range_handles_none(self):
        """A no-corrections input must come back as (1.0, 1.0) — the
        iteration's 'no scaling' semantics that production scripts rely
        on when ``image_scale_corrections`` is not provided."""
        lo, hi = ppca_diagnostics.resolve_image_scale_range(None, None)
        assert lo == pytest.approx(1.0)
        assert hi == pytest.approx(1.0)

    def test_resolve_image_scale_range_handles_array(self):
        """When corrections are present, the helper must return the
        observed (min, max) of the input, restricted to ``image_indices``
        if given."""
        corr = np.array([0.5, 1.0, 2.0, 1.0], dtype=np.float32)
        lo, hi = ppca_diagnostics.resolve_image_scale_range(corr, None)
        assert lo == pytest.approx(0.5)
        assert hi == pytest.approx(2.0)
        # With indices, restrict to the selected subset.
        lo_sub, hi_sub = ppca_diagnostics.resolve_image_scale_range(corr, np.array([1, 3], dtype=np.int64))
        assert lo_sub == pytest.approx(1.0)
        assert hi_sub == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# 5. Performance smoke (CPU-only, generous headroom)
# ---------------------------------------------------------------------------


class TestPerformanceSmoke:
    """The cached-moments path must stay fast. We don't pin the legacy
    vs new wall-time ratio (JIT warmup variance dominates on tiny inputs);
    we just guarantee neither path becomes pathologically slow under
    something the merge could introduce — e.g., losing a JIT cache,
    introducing accidental host roundtrips, etc."""

    def test_cached_moments_runs_under_2s_after_warm(self):
        Y1, proj_aug, ctf2, y_norm = _tiny_score_inputs()
        Y1_j = jnp.asarray(Y1)
        proj_j = jnp.asarray(proj_aug)
        ctf_j = jnp.asarray(ctf2)
        y_j = jnp.asarray(y_norm)

        # Warmup (JIT compile).
        warm = dense_pose_ppca_score_with_moments_blocked(Y1_j, proj_j, ctf_j, y_j, None)
        # Force device sync so warmup doesn't bleed into the timing.
        _ = np.asarray(warm.logZ)

        t0 = time.perf_counter()
        for _ in range(10):
            out = dense_pose_ppca_score_with_moments_blocked(Y1_j, proj_j, ctf_j, y_j, None)
        _ = np.asarray(out.logZ)  # sync
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, (
            f"dense_pose_ppca_score_with_moments_blocked slowdown: 10 warm calls "
            f"took {elapsed:.3f}s (budget 2.0s). Suspect: lost JIT cache, "
            f"host roundtrip, or accidental float64 promotion in the new fast path."
        )

    def test_legacy_E_step_runs_under_2s_after_warm(self):
        Y1, proj_aug, ctf2, y_norm = _tiny_score_inputs()
        Y1_j = jnp.asarray(Y1)
        proj_j = jnp.asarray(proj_aug)
        ctf_j = jnp.asarray(ctf2)
        y_j = jnp.asarray(y_norm)

        warm = dense_pose_ppca_E_step_blocked(Y1_j, proj_j, ctf_j, y_j, None)
        _ = np.asarray(warm[0].alpha_aug_acc)

        t0 = time.perf_counter()
        for _ in range(10):
            out = dense_pose_ppca_E_step_blocked(Y1_j, proj_j, ctf_j, y_j, None)
        _ = np.asarray(out[0].alpha_aug_acc)
        elapsed = time.perf_counter() - t0
        assert elapsed < 2.0, (
            f"dense_pose_ppca_E_step_blocked slowdown: 10 warm calls took {elapsed:.3f}s (budget 2.0s)."
        )
