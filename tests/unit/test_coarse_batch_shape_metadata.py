"""The coarse batch-size query must not transfer image values to the host."""

import ast
from pathlib import Path

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.unit


def _production_batch_size(batch_data):
    # Execute the actual production assignment without starting the E-step.
    source = Path(__file__).parents[2] / "recovar/em/scoring/significance.py"
    tree = ast.parse(source.read_text())
    owner = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "_compute_k_class_significance_batched"
    )
    assignments = [
        n
        for n in ast.walk(owner)
        if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "batch_size" for t in n.targets)
    ]
    assert len(assignments) == 1
    namespace = {"batch_data": batch_data, "np": np}
    exec(compile(ast.Module(body=assignments, type_ignores=[]), str(source), "exec"), namespace)
    return namespace["batch_size"]


@pytest.mark.parametrize("rows", [0, 1, 250, 256])
@pytest.mark.parametrize("pixels", [32, 380])
def test_batch_size_does_not_materialize_device_values(rows, pixels):
    class ShapeOnly:
        shape = (rows, pixels, pixels)

        def __array__(self, *args, **kwargs):
            raise AssertionError("Unnecessary device-to-host image transfer")

    assert _production_batch_size(ShapeOnly()) == rows


@pytest.mark.parametrize("rows", [0, 1, 250, 256])
def test_batch_size_matches_numpy_reference(rows):
    batch = np.empty((rows, 3, 3), dtype=np.float32)
    assert _production_batch_size(batch) == int(np.asarray(batch).shape[0])


def test_batch_size_uses_static_jax_shape_without_readback():
    result = jax.eval_shape(_production_batch_size, jax.ShapeDtypeStruct((250, 380, 380), np.float32))
    assert result.shape == ()
