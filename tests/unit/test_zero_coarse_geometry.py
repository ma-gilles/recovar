"""Execute production routing expressions without a full refinement fixture."""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.unit
OWNERS = Path(__file__).resolve().parents[2] / "recovar/em"


def tree(name):
    owner = "refinement" if name == "iteration_loop.py" else "dense"
    return ast.parse((OWNERS / owner / name).read_text())


def evaluate(node, **scope):
    return eval(compile(ast.Expression(node), "<production-route>", "eval"), scope)


@pytest.mark.parametrize(
    "os,local,k,mode,hard,double,expected",
    [
        (0, False, 1, "gaussian", False, False, True),
        (0, True, 1, "gaussian", False, False, False),
        (0, False, 4, "gaussian", False, False, False),
        (0, False, 1, "normalized_cc", False, False, False),
        (0, False, 1, "gaussian", True, False, False),
        (0, False, 1, "gaussian", False, True, False),
        (1, False, 1, "gaussian", False, False, True),
        (1, False, 4, "normalized_cc", True, True, True),
        (1, True, 4, "gaussian", False, False, False),
    ],
)
def test_device_matrix_generation_gate(os, local, k, mode, hard, double, expected):
    gates = [
        n
        for n in ast.walk(tree("iteration_loop.py"))
        if isinstance(n, ast.If)
        and any(
            isinstance(s, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == "adaptive_pass1_rotations" for t in s.targets)
            and isinstance(s.value, ast.Call)
            and isinstance(s.value.func, ast.Name)
            and s.value.func.id == "_relion_adaptive_pass1_rotations"
            for s in n.body
        )
    ]
    assert len(gates) == 1
    got = evaluate(
        gates[0].test,
        use_local=local,
        state=SimpleNamespace(adaptive_oversampling=os),
        n_classes=k,
        firstiter_score_mode_this_iter=mode,
        firstiter_winner_take_all_this_iter=hard,
        _DENSE_EM_STATIC_KWARGS={"use_float64_scoring": double},
    )
    assert bool(got) is expected


@pytest.mark.parametrize(
    "os,sparse,xhalf,mode,double,override,expected",
    [
        (0, True, True, "gaussian", False, True, True),
        (0, True, True, "gaussian", False, False, False),
        (1, True, True, "gaussian", False, True, False),
        (0, False, True, "gaussian", False, True, False),
        (0, True, False, "gaussian", False, True, False),
        (0, True, True, "normalized_cc", False, True, False),
        (0, True, True, "gaussian", True, True, False),
    ],
)
def test_only_coarse_engine_operand_changes(os, sparse, xhalf, mode, double, override, expected):
    calls = [
        n.value
        for n in ast.walk(tree("half_scoring.py"))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "k1_adaptive_result" for t in n.targets)
        and isinstance(n.value, ast.Call)
        and isinstance(n.value.func, ast.Name)
        and n.value.func.id == "run_dense_k_class_em_adaptive"
    ]
    assert len(calls) == 1
    coarse, fine, native = object(), object(), object()
    scope = dict(
        pass2_grids=SimpleNamespace(coarse_rotations=coarse, fine_rotations=fine),
        coarse_scoring_rotations=native if override else None,
        adaptive_os_local=os,
        k1_sparse_pass2=sparse,
        k1_relion_x_half_mstep=xhalf,
        firstiter_score_mode_this_iter=mode,
        diagnostic_float64_pass2=double,
    )
    assert evaluate(calls[0].args[4], **scope) is (native if expected else coarse)
    assert evaluate(calls[0].args[6], **scope) is fine


def test_loop_transports_geometry_separately_from_effective_rotations():
    calls = [
        n.value
        for n in ast.walk(tree("iteration_loop.py"))
        if isinstance(n, ast.Assign)
        and any(isinstance(t, ast.Name) and t.id == "dense_half_kwargs" for t in n.targets)
        and isinstance(n.value, ast.Call)
    ]
    assert len(calls) == 1
    keywords = {k.arg: k.value for k in calls[0].keywords}
    marker = object()
    assert (
        evaluate(
            keywords["coarse_scoring_rotations"],
            adaptive_pass1_rotations=marker,
            state=SimpleNamespace(adaptive_oversampling=0),
        )
        is marker
    )
    assert (
        evaluate(
            keywords["coarse_scoring_rotations"],
            adaptive_pass1_rotations=marker,
            state=SimpleNamespace(adaptive_oversampling=1),
        )
        is None
    )
