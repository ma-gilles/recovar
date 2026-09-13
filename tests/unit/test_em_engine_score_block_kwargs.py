"""The dense engine binds its per-batch score-block keywords once for both passes."""

import ast
import inspect

from recovar.em.dense import em_engine


def test_score_block_calls_share_the_per_batch_keywords():
    source = inspect.getsource(em_engine.run_em)
    assert source.count("scores = _score_rotation_block(") == 2
    assert source.count("**score_block_kwargs,") == 2
    assert source.count("score_block_kwargs = dict(") == 1
    assert "shifted_score=shifted_windowed,\n            batch_norm=batch_norm," in source
    tree = ast.parse(source)
    binding = next(n for n in ast.walk(tree) if isinstance(n, ast.Assign) and isinstance(n.targets[0], ast.Name) and n.targets[0].id == "score_block_kwargs")
    keys = {k.arg for k in binding.value.keywords}
    assert keys == {"shifted_score", "batch_norm", "score_weight", "half_weights", "n_images", "n_trans", "image_shape", "volume_shape", "score_mode", "precision_policy"}
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == "_score_rotation_block"]
    assert len(calls) == 2
    for call in calls:
        explicit = {k.arg for k in call.keywords if k.arg is not None}
        assert explicit == {"proj_half", "proj_abs2_half"}
        assert binding.lineno < call.lineno
    # the shared operands are not rebound between the binding and the pass-2 call
    operands = {"shifted_windowed", "batch_norm", "ctf2_over_nv_windowed", "half_weights", "batch_size", "n_trans", "image_shape", "volume_shape", "precision_policy"}
    for node in ast.walk(tree):
        targets = node.targets if isinstance(node, ast.Assign) else ([node.target] if isinstance(node, (ast.AugAssign, ast.AnnAssign)) else [])
        for t in targets:
            for s in ast.walk(t):
                if isinstance(s, ast.Name) and s.id in operands:
                    assert node.lineno < binding.lineno
