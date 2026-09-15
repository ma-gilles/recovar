"""Precision policy helpers for dense single-volume EM."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class DensePrecisionPolicy:
    """Centralized dtype choices for dense EM scoring/projection paths."""

    use_float64_scoring: bool = False
    use_float64_projections: bool = False
    use_float64_normalization: bool = True

    @property
    def score_complex_dtype(self):
        return jnp.complex128 if self.use_float64_scoring else jnp.complex64

    @property
    def score_real_dtype(self):
        return jnp.float64 if self.use_float64_scoring else jnp.float32

    @property
    def projection_complex_dtype(self):
        return jnp.complex128 if self.use_float64_projections else None

    @property
    def normalization_real_dtype(self):
        return jnp.float64 if self.use_float64_normalization else self.score_real_dtype

    def cast_projection_volume(self, volume):
        dtype = self.projection_complex_dtype
        return volume if dtype is None else jnp.asarray(volume, dtype=dtype)

    def cast_scoring_inputs(self, shifted_score_half, score_weight_half, shifted_recon_half):
        shifted_score_half = shifted_score_half.astype(self.score_complex_dtype)
        score_weight_half = score_weight_half.astype(self.score_real_dtype)
        if self.use_float64_scoring:
            shifted_recon_half = shifted_recon_half.astype(self.score_complex_dtype)
        return shifted_score_half, score_weight_half, shifted_recon_half

    def cast_local_preprocessed_inputs(
        self,
        shifted_score,
        shifted_recon,
        shifted_noise,
        score_weight,
        recon_weight,
    ):
        shifted_score = shifted_score.astype(self.score_complex_dtype)
        score_weight = score_weight.astype(self.score_real_dtype)
        if self.use_float64_scoring:
            shifted_recon = shifted_recon.astype(self.score_complex_dtype)
            shifted_noise = shifted_noise.astype(self.score_complex_dtype)
            recon_weight = recon_weight.astype(self.score_real_dtype)
        return shifted_score, shifted_recon, shifted_noise, score_weight, recon_weight

    def cast_projection_scores(self, proj_half, proj_abs2_half):
        if self.use_float64_scoring:
            return proj_half, proj_abs2_half
        return proj_half.astype(jnp.complex64), proj_abs2_half.astype(jnp.float32)

    def cast_local_projection_scores(
        self,
        proj_weighted,
        proj_for_noise,
        proj_abs2_weighted=None,
        proj_abs2_for_noise=None,
    ):
        proj_weighted = proj_weighted.astype(self.score_complex_dtype)
        if proj_for_noise is not None and self.use_float64_scoring:
            proj_for_noise = proj_for_noise.astype(self.score_complex_dtype)
        if proj_abs2_weighted is not None:
            proj_abs2_weighted = proj_abs2_weighted.astype(self.score_real_dtype)
        if proj_abs2_for_noise is not None and self.use_float64_scoring:
            proj_abs2_for_noise = proj_abs2_for_noise.astype(self.score_real_dtype)
        return proj_weighted, proj_for_noise, proj_abs2_weighted, proj_abs2_for_noise

    def cast_local_noise_projection_scores(self, proj_for_noise, proj_abs2_for_noise):
        if not self.use_float64_scoring:
            return proj_for_noise, proj_abs2_for_noise
        if proj_abs2_for_noise is None:
            return proj_for_noise.astype(self.score_complex_dtype), None
        return (
            proj_for_noise.astype(self.score_complex_dtype),
            proj_abs2_for_noise.astype(self.score_real_dtype),
        )

    def cast_local_big_jit_inputs(
        self,
        shifted_score,
        shifted_recon,
        shifted_noise,
        score_weight,
        recon_weight,
        proj_weighted,
        proj_for_noise,
    ):
        shifted_score = shifted_score.astype(self.score_complex_dtype)
        shifted_recon = shifted_recon.astype(self.score_complex_dtype)
        shifted_noise = shifted_noise.astype(self.score_complex_dtype)
        score_weight = score_weight.astype(self.score_real_dtype)
        recon_weight = recon_weight.astype(self.score_real_dtype)
        proj_weighted = proj_weighted.astype(self.score_complex_dtype)
        proj_for_noise = proj_for_noise.astype(self.score_complex_dtype)
        return (
            shifted_score,
            shifted_recon,
            shifted_noise,
            score_weight,
            recon_weight,
            proj_weighted,
            proj_for_noise,
        )

    def cast_dense_big_jit_inputs(
        self,
        shifted_score,
        shifted_recon,
        score_weight,
        recon_weight,
        score_half_weights,
        proj_score,
    ):
        shifted_score = shifted_score.astype(self.score_complex_dtype)
        shifted_recon = shifted_recon.astype(self.score_complex_dtype)
        score_weight = score_weight.astype(self.score_real_dtype)
        recon_weight = recon_weight.astype(self.score_real_dtype)
        score_half_weights = score_half_weights.astype(self.score_real_dtype)
        proj_score = proj_score.astype(self.score_complex_dtype)
        return (
            shifted_score,
            shifted_recon,
            score_weight,
            recon_weight,
            score_half_weights,
            proj_score,
        )


def _diagnostic_float64_pass2_matches(debug_iteration: int | None) -> bool:
    """Select genuine-f64 pass 2 without perturbing an earlier f32 boundary."""

    raw = os.environ.get("RECOVAR_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS", "")
    if debug_iteration is None or not raw.strip():
        return False
    try:
        requested = {int(token.strip()) for token in raw.split(",") if token.strip()}
    except ValueError as exc:
        raise ValueError(
            "RECOVAR_DIAGNOSTIC_FLOAT64_PASS2_ITERATIONS must be comma-separated integers"
        ) from exc
    return int(debug_iteration) in requested

def _local_search_precision_flags(
    debug_iteration: int | None,
    *,
    static_em_kwargs: Mapping[str, object],
    pass_index: int,
) -> tuple[bool, bool]:
    """Resolve local-search precision without changing production defaults.

    The supplied float64 switches apply to both local passes.  The targeted
    diagnostic selector upgrades only pass 2 so it remains useful for
    classifying the fine-score/posterior boundary independently of pass 1.
    """

    if int(pass_index) not in (1, 2):
        raise ValueError(f"local-search pass_index must be 1 or 2, got {pass_index}")
    use_float64_scoring = bool(static_em_kwargs["use_float64_scoring"])
    use_float64_projections = bool(static_em_kwargs["use_float64_projections"])
    if int(pass_index) == 2 and _diagnostic_float64_pass2_matches(debug_iteration):
        use_float64_scoring = True
        use_float64_projections = True
    return use_float64_scoring, use_float64_projections
