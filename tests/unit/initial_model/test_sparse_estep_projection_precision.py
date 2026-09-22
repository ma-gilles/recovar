"""VDAM sparse E-step scorer calls request an explicit projection precision."""

import ast
import inspect
import textwrap

import pytest

pytestmark = pytest.mark.unit


def test_every_coarse_scorer_call_passes_projection_precision():
    """Without it the complex128 host PPref bypasses the complex64 texture projector.

    The long-tier native InitialModel case then failed with "a runtime current-image
    projection mask requires the RELION texture projector" once consolidation
    41128dcd0 routed the runtime image mask through this projection.
    """
    from recovar.em.vdam import sparse_pass2_estep

    tree = ast.parse(textwrap.dedent(inspect.getsource(sparse_pass2_estep._run_sparse_pass2_initial_model_estep)))
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "_compute_k_class_significance_batched"
    ]
    assert calls, "expected the coarse significance call"
    for call in calls:
        assert "use_float64_projections" in {kw.arg for kw in call.keywords}, ast.unparse(call)[:200]
