"""The final all-data pass sizes a local pass 1 only for parent-expanded (adaptive) local search.

RELION (``ml_optimiser.cpp``, ``updateImageSizeAndResolutionPointers``) keeps
``coarse_size == current_size`` when ``adaptive_oversampling`` is 0; the
controller therefore assigns ``final_local_pass1_current_size`` only at its
initialization and under ``if use_parent_expanded_final_local``.  The previous
code reached the clamp outside that branch and raised ``UnboundLocalError`` for
lazy (large-order) non-expanded final local searches.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import pytest

import recovar.em.refinement.iteration_loop as iteration_loop

pytestmark = pytest.mark.unit


def _assignments_to(tree, name):
    parents = {}
    for node in ast.walk(tree):
        for child in ast.iter_child_nodes(node):
            parents[child] = node
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
            yield node, parents


def _guarded_by(node, parents, condition_name):
    while node in parents:
        node = parents[node]
        if isinstance(node, ast.If) and isinstance(node.test, ast.Name) and node.test.id == condition_name:
            return True
    return False


def test_final_local_pass1_size_is_assigned_only_under_parent_expansion():
    source = textwrap.dedent(inspect.getsource(iteration_loop._run_relion_iteration_loop))
    tree = ast.parse(source)
    assignments = list(_assignments_to(tree, "final_local_pass1_current_size"))
    assert len(assignments) == 2
    initialization, sized = assignments
    assert isinstance(initialization[0].value, ast.Name) and initialization[0].value.id == "final_current_size"
    assert _guarded_by(sized[0], sized[1], "use_parent_expanded_final_local")
