"""CLI contract for selecting RELION particle-image Fourier preprocessing."""

from __future__ import annotations

import ast
from pathlib import Path

RUNNER = Path(__file__).resolve().parents[2] / "scripts" / "run_full_refinement.py"


def _runner_tree() -> ast.Module:
    return ast.parse(RUNNER.read_text())


def _image_backend_argument() -> ast.Call:
    for node in ast.walk(_runner_tree()):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        if node.func.attr != "add_argument" or not node.args:
            continue
        if isinstance(node.args[0], ast.Constant) and node.args[0].value == "--image-fourier-backend":
            return node
    raise AssertionError("missing --image-fourier-backend parser option")


def test_image_fourier_backend_cli_preserves_host_default_and_typed_choices():
    argument = _image_backend_argument()
    keywords = {keyword.arg: keyword.value for keyword in argument.keywords}

    assert ast.literal_eval(keywords["default"]) == "host_numpy"
    assert ast.literal_eval(keywords["choices"]) == ("host_numpy", "jax_gpu", "relion_cuda")


def test_image_fourier_backend_cli_is_forwarded_to_refinement():
    refine_calls = [
        node
        for node in ast.walk(_runner_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "refine_single_volume"
    ]
    assert len(refine_calls) == 1

    keywords = {keyword.arg: keyword.value for keyword in refine_calls[0].keywords}
    forwarded = keywords["image_fourier_backend"]
    assert isinstance(forwarded, ast.Attribute)
    assert isinstance(forwarded.value, ast.Name)
    assert (forwarded.value.id, forwarded.attr) == ("args", "image_fourier_backend")


def test_relion_softmask_reduction_cli_has_sealed_diagnostic_choices():
    arguments = [
        node
        for node in ast.walk(_runner_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and node.args[0].value == "--relion-softmask-reduction"
    ]
    assert len(arguments) == 1
    keywords = {keyword.arg: keyword.value for keyword in arguments[0].keywords}
    assert ast.literal_eval(keywords["default"]) == "control"
    assert ast.literal_eval(keywords["choices"]) == (
        "control",
        "native_lane",
        "native_atomic",
    )


def test_relion_softmask_reduction_routes_both_native_diagnostic_modes():
    source = RUNNER.read_text()
    assert 'backend.set_relion_native_lane_reduction(True)' in source
    assert 'os.environ["RECOVAR_RELION_NATIVE_ATOMIC_SOFTMASK_REDUCTION"] = "1"' in source
    assert 'args.image_fourier_backend != "relion_cuda"' in source
