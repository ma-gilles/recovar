"""Compatibility of the shared PPCA statistics container."""

from dataclasses import fields

import jax.numpy as jnp

from recovar.ppca.pose_accumulators import AugmentedPPCAStats


def test_legacy_positional_constructor_and_optional_fields():
    rhs = jnp.zeros((2, 3), dtype=jnp.complex64)
    lhs = jnp.zeros((2, 6), dtype=jnp.float32)
    legacy = AugmentedPPCAStats(rhs, lhs, None, None, 1.25, 7, {"source": "legacy"})
    assert legacy.rhs is rhs
    assert legacy.lhs_tri is lhs
    assert legacy.log_likelihood == 1.25
    assert legacy.n_images == 7
    assert legacy.diagnostics == {"source": "legacy"}
    assert legacy.residual_gradient is None
    assert legacy.embeddings is None
    assert legacy.original_image_ids is None

    extended = AugmentedPPCAStats(
        rhs, lhs, residual_gradient=rhs, embeddings=lhs, original_image_ids=jnp.arange(2)
    )
    assert extended.residual_gradient is rhs
    assert extended.embeddings is lhs
    assert jnp.array_equal(extended.original_image_ids, jnp.arange(2))
    assert [field.name for field in fields(AugmentedPPCAStats)][:7] == [
        "rhs", "lhs_tri", "residual_num", "residual_den", "log_likelihood", "n_images", "diagnostics"
    ]
