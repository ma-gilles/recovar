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
    # image_fourier_backend is forwarded via the RelionParityOptions group
    # inside refine_single_volume's options= bundle (commit cd6661f2), not as
    # a top-level refine_single_volume keyword.
    parity_calls = [
        node
        for node in ast.walk(_runner_tree())
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "RelionParityOptions"
    ]
    assert len(parity_calls) == 1

    keywords = {keyword.arg: keyword.value for keyword in parity_calls[0].keywords}
    forwarded = keywords["image_fourier_backend"]
    assert isinstance(forwarded, ast.Attribute)
    assert isinstance(forwarded.value, ast.Name)
    assert (forwarded.value.id, forwarded.attr) == ("args", "image_fourier_backend")
