"""The coarse Gaussian GEMM backend, its diagnostics and RELION's exact coarse operands have their own owners."""

import inspect

from recovar.em.dense_single_volume.helpers import (
    coarse_gaussian_diagnostics,
    coarse_gaussian_gemm,
    relion_coarse_operands,
    significance,
)


def test_owners_hold_the_definitions_and_significance_routes_to_them():
    sig_src = inspect.getsource(significance)
    for mod, names in (
        (coarse_gaussian_gemm, ("_compute_coarse_gaussian_gemm_hybrid_batch", "_coarse_gaussian_gemm_resources", "_resolve_coarse_gaussian_gemm_hybrid_image_batch_size", "_score_relion_coarse_gaussian_gemm_macro")),
        (coarse_gaussian_diagnostics, ("_seal_coarse_gaussian_gemm_diagnostic_scope", "_write_coarse_gaussian_gemm_diagnostic", "_maybe_dump_k_class_significance_batch", "SignificanceDumpComplete")),
        (relion_coarse_operands, ("_assemble_relion_exact_coarse_gaussian_operands", "_relion_coarse_gaussian_square_operands", "_relion_coarse_pose_tie_break_keys", "RelionExactCoarseGaussianOperands")),
    ):
        for name in names:
            assert inspect.getmodule(getattr(mod, name)) is mod and f"\ndef {name}(" not in sig_src and f"\nclass {name}(" not in sig_src
    assert significance._compute_coarse_gaussian_gemm_hybrid_batch is coarse_gaussian_gemm._compute_coarse_gaussian_gemm_hybrid_batch
    assert significance._assemble_relion_exact_coarse_gaussian_operands is relion_coarse_operands._assemble_relion_exact_coarse_gaussian_operands
    for mod in (coarse_gaussian_gemm, coarse_gaussian_diagnostics, relion_coarse_operands):
        assert "helpers.significance import" not in inspect.getsource(mod)
    assert "helpers.coarse_gaussian_diagnostics import" not in inspect.getsource(coarse_gaussian_gemm)
