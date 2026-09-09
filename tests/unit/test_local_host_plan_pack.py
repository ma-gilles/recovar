"""Routing evidence for opt-in unchanged-host-plan packing; no CUDA arithmetic claim."""

import ast
import copy
import hashlib
import inspect
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cuda
from recovar.em.dense_single_volume import local_em_engine as engine
from recovar.em.dense_single_volume.helpers import deferred_vdam_host_pack as helper
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_binary_flag

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True)])
def test_selector_default_and_explicit_values(monkeypatch, token, expected):
    monkeypatch.delenv(engine.EXACT_LOCAL_HOST_PLAN_PACK_ENV, raising=False)
    if token is not None:
        monkeypatch.setenv(engine.EXACT_LOCAL_HOST_PLAN_PACK_ENV, token)
    assert parse_env_binary_flag(engine.EXACT_LOCAL_HOST_PLAN_PACK_ENV) is expected


@pytest.mark.parametrize("token", ["", "true", "false", "yes", "2", "-1", "typo"])
def test_selector_rejects_unknown_tokens(monkeypatch, token):
    monkeypatch.setenv(engine.EXACT_LOCAL_HOST_PLAN_PACK_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parse_env_binary_flag(engine.EXACT_LOCAL_HOST_PLAN_PACK_ENV)


@pytest.mark.parametrize("deferred,noise", [(False, False), (False, True), (True, False)])
def test_unsupported_request_fails_before_dataset_access(monkeypatch, deferred, noise):
    monkeypatch.setenv(engine.EXACT_LOCAL_HOST_PLAN_PACK_ENV, "1")
    with pytest.raises(ValueError, match="requires deferred packed final-noise"):
        engine.run_local_em_exact(
            None,
            None,
            None,
            None,
            None,
            "linear_interp",
            image_batch_size=1,
            rotation_block_size=1,
            current_size=8,
            _defer_packed_vdam_enabled=deferred,
            _packed_final_noise_enabled=noise,
        )


def _tree():
    return ast.parse(inspect.getsource(engine.run_local_em_exact)).body[0]


def _packing_body():
    blocks = [
        node
        for node in ast.walk(_tree())
        if isinstance(node, ast.If)
        and isinstance(node.test, ast.Name)
        and node.test.id == "return_big_jit_deferred_mstep_inputs"
        and any(
            isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "packed_summed" for t in n.targets)
            for n in node.body
        )
    ]
    assert len(blocks) == 1
    return copy.deepcopy(blocks[0].body)


@pytest.mark.parametrize(
    "deferred,source,noise", [(False, True, True), (True, False, True), (True, True, False), (True, True, True)]
)
def test_actual_bucket_eligibility(deferred, source, noise):
    guards = [
        n
        for n in ast.walk(_tree())
        if isinstance(n, ast.If)
        and ast.unparse(n.test).startswith("host_plan_pack_enabled and")
        and "return_big_jit_deferred_mstep_inputs" in ast.unparse(n.test)
    ]
    assert len(guards) == 1
    code = compile(ast.fix_missing_locations(ast.Module(body=guards, type_ignores=[])), "<bucket eligibility>", "exec")
    env = dict(
        host_plan_pack_enabled=True,
        return_big_jit_deferred_mstep_inputs=deferred,
        return_deferred_source_vdam_operands=source,
        packed_final_noise_enabled=noise,
    )
    if deferred and source and noise:
        exec(code, env)
    else:
        with pytest.raises(ValueError, match="did not reach the deferred source-noise lane"):
            exec(code, env)
    env["host_plan_pack_enabled"] = False
    exec(code, env)


def test_disabled_packing_statement_order_matches_frozen_parent():
    class Disable(ast.NodeTransformer):
        def visit_IfExp(self, node):
            # The separately qualified host-publication option was added after
            # this fixture. Compare its disabled path to the original parent.
            if ast.unparse(node.test) == "host_publication_enabled":
                return self.visit(node.orelse)
            return self.generic_visit(node)

        def visit_If(self, node):
            if ast.unparse(node.test) == "host_plan_pack_enabled":
                return [self.visit(n) for n in node.orelse]
            if ast.unparse(node.test) == "not host_plan_pack_enabled":
                return [self.visit(n) for n in node.body]
            return self.generic_visit(node)

    body = ast.Module(body=_packing_body(), type_ignores=[])
    # The precision integration removed one narrowing cast. Require exactly
    # that expression, then restore its old AST only for the frozen comparison.
    # The live packing test below independently checks dtype preservation.
    rotation_inputs = [
        node.value.args[0] for node in ast.walk(body)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "packed_rotations_np" for target in node.targets)
    ]
    assert len(rotation_inputs) == 1
    assert ast.unparse(rotation_inputs[0]) == "np.asarray(bucket.local_rotations[:unpadded_batch_size])"
    rotation_inputs[0].keywords.append(
        ast.keyword(arg="dtype", value=ast.Attribute(value=ast.Name(id="np", ctx=ast.Load()), attr="float32", ctx=ast.Load()))
    )
    normalized = Disable().visit(body)
    # Entire original packing body from immutable 6d11f325, not just selected expressions.
    digest = hashlib.sha256(ast.dump(normalized, include_attributes=False).encode()).hexdigest()
    assert digest == "2aad4cc3e721ba61736b47a031e252d2b2c1097abda8e653812adfb58f967e4f"


def _execute(
    monkeypatch, enabled, batch=2, use_window=True, bad_shape=False, fail_helper=False,
    cuda_enabled=False, rotation_dtype=np.float32,
):
    b, r, t, p = 3, 4, 3, 2
    probabilities = jnp.asarray(np.arange(b * r * t, dtype=np.float32).reshape(b, r, t) / 16)
    sums = jnp.sum(probabilities, axis=2)
    images = jnp.asarray(np.arange(b * p, dtype=np.float32).reshape(b, p).astype(np.complex64) + 1j)
    ctf = jnp.ones((b, p), jnp.float32)
    inv = jnp.full((b, p), 2, jnp.float32)
    flat = jnp.asarray(np.arange(b * r * p, dtype=np.float32).reshape(b * r, p).astype(np.complex64) + 2j)
    plan = np.asarray([(i, j, 1) for i in range(b) for j in range(r)], np.int32)[::-1].copy()
    take = np.asarray([[3, 1], [2, 0], [1, 3]], np.int32)[:batch]
    mask = np.asarray([[True, False], [True, True], [False, True]])[:batch]
    rotations = np.arange(b * r * 9, dtype=rotation_dtype).reshape(b, r, 3, 3)
    if rotation_dtype == np.float64:
        rotations += 2.0 ** -30  # Values that an unintended float32 cast loses.
    events = []
    calls = []
    returned = []

    def host_plan(*args, **kwargs):
        events.append("host_plan")
        return take, mask, None, int(mask.sum())

    def mstep_rotations(bucket):
        events.append("host_mstep_rotations")
        return bucket.local_rotations + 1

    def lookup(*args, **kwargs):
        events.append("flat_lookup")
        return engine.build_dense_to_flat_local_row_lookup(*args, **kwargs)

    def mapping(*args, **kwargs):
        events.append("flat_mapping")
        return engine.map_dense_local_rows_to_flat_rows(*args, **kwargs)

    def denominator(c, iv, prob):
        events.append("denominator_double")
        # Deliberate routing double: CUDA arithmetic was qualified separately.
        return jnp.full((prob.shape[0], prob.shape[1], c.shape[1]), 3, jnp.float32)

    original = helper.pack_deferred_vdam_host_plan

    def compiled_helper(*args):
        events.append("compiled_helper")
        calls.append(args)
        if fail_helper:
            raise RuntimeError("deliberate helper failure")
        result = original.__wrapped__(*args)
        returned.append(result)
        return result

    monkeypatch.setattr(cuda, "relion_vdam_mstep_denominator_f32", denominator)

    def wrong_route(*args):
        raise AssertionError("Packing selected the wrong backend")

    monkeypatch.setattr(helper, "pack_deferred_vdam_host_plan", wrong_route if cuda_enabled else compiled_helper)
    monkeypatch.setattr(helper, "pack_deferred_vdam_host_plan_cuda", compiled_helper if cuda_enabled else wrong_route)
    env = dict(
        np=np,
        jnp=jnp,
        host_plan_pack_enabled=enabled,
        host_plan_cuda_enabled=cuda_enabled,
        host_publication_enabled=False,
        unpadded_batch_size=batch,
        batch_size=b,
        probs_sum_t=sums,
        reconstruction_probs=probabilities,
        reconstruction_probs_sum_t=sums,
        reconstruction_rotation_mask_np=np.ones((batch, r), bool),
        local_mask_np=np.ones((batch, r), bool),
        rotation_block_size=2,
        resolved_exact_local_bucket_radix=4,
        collect_profile_stats=False,
        packed_final_noise_enabled=True,
        return_deferred_source_vdam_operands=True,
        _build_nonzero_reconstruction_pack_indices=host_plan,
        _local_mstep_rotations=mstep_rotations,
        bucket=SimpleNamespace(local_rotations=rotations, bucket_rotation_count=r),
        deferred_source_vdam_images=images,
        deferred_source_vdam_ctf=ctf,
        deferred_source_vdam_minvsigma2=inv,
        deferred_flat_proj_for_noise=flat[:1] if bad_shape else flat,
        flat_local_row_argument=plan,
        window_spec=SimpleNamespace(n_recon=p, use_window=use_window),
        n_half=p,
        build_dense_to_flat_local_row_lookup=lookup,
        map_dense_local_rows_to_flat_rows=mapping,
    )
    code = compile(
        ast.fix_missing_locations(ast.Module(body=_packing_body(), type_ignores=[])), "<actual packing body>", "exec"
    )
    if bad_shape or fail_helper:
        with pytest.raises(
            RuntimeError, match="packed reconstruction layout" if bad_shape else "deliberate helper failure"
        ):
            exec(code, env)
        assert events.count("denominator_double") == 0
        assert len(calls) == (0 if bad_shape else 1)
        return
    exec(code, env)
    assert events[:3] == ["host_plan", "host_mstep_rotations", "flat_lookup"]
    assert events[3] == "flat_mapping"
    assert events.count("denominator_double") == 1
    assert events.count("compiled_helper") == int(enabled)
    assert env["reconstruction_take_indices"] is take and env["reconstruction_pack_mask_np"] is mask
    assert env["packed_rotations_np"].dtype == rotations.dtype
    np.testing.assert_array_equal(
        env["packed_rotations_np"], np.take_along_axis(rotations[:batch], take[:, :, None, None], axis=1)
    )
    np.testing.assert_array_equal(
        env["packed_mstep_rotations_np"], np.take_along_axis((rotations + 1)[:batch], take[:, :, None, None], axis=1)
    )
    names = (
        "packed_reconstruction_probs",
        "packed_reconstruction_probs_sum_t",
        "packed_source_vdam_images",
        "packed_source_vdam_ctf",
        "packed_source_vdam_minvsigma2",
        "packed_source_vdam_noise_projection",
        "packed_source_vdam_ctf_probs",
    )
    assert env["packed_source_vdam_posterior"] is env["packed_reconstruction_probs"]
    if enabled:
        args = calls[0]
        for actual, expected in zip(args[:6], (probabilities, sums, images, ctf, inv, flat), strict=True):
            assert actual is expected
        assert args[6] is env["reconstruction_take_indices_jnp"]
        assert args[7] is env["reconstruction_pack_mask_jnp"]
        assert args[8].dtype == jnp.int32
        np.testing.assert_array_equal(args[8], env["packed_flat_take_indices"])
        assert all(env[n] is out for n, out in zip(names, returned[0], strict=True))
    return tuple(np.asarray(env[n]) for n in names)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("rotation_dtype", [np.float32, np.float64])
def test_actual_pack_preserves_rotation_precision(monkeypatch, enabled, rotation_dtype):
    _execute(monkeypatch, enabled, rotation_dtype=rotation_dtype)


@pytest.mark.parametrize("batch,use_window", [(2, True), (3, True), (2, False), (3, False)])
@pytest.mark.parametrize("cuda_enabled", [False, True])
def test_actual_pack_route_preserves_host_plan_and_forwards_every_operand(monkeypatch, batch, use_window, cuda_enabled):
    with monkeypatch.context() as m:
        old = _execute(m, False, batch, use_window)
    with monkeypatch.context() as m:
        new = _execute(m, True, batch, use_window, cuda_enabled=cuda_enabled)
    for a, b in zip(old, new, strict=True):
        assert a.shape == b.shape and a.dtype == b.dtype
        np.testing.assert_array_equal(a.view(np.uint8), b.view(np.uint8))


@pytest.mark.parametrize("enabled", [False, True])
def test_projection_shape_check_precedes_cuda_or_helper(monkeypatch, enabled):
    _execute(monkeypatch, enabled, bad_shape=True)


@pytest.mark.parametrize("cuda_enabled", [False, True])
def test_compiled_helper_failure_is_not_silently_replayed(monkeypatch, cuda_enabled):
    _execute(monkeypatch, True, fail_helper=True, cuda_enabled=cuda_enabled)
