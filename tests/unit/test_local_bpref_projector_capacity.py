"""Engine routing contracts for the explicitly enabled BPref capacity ABI."""

import ast
import inspect

import jax.numpy as jnp
import numpy as np
import pytest

from recovar import cuda_backproject as cb
from recovar.em.dense_single_volume import local_em_engine as engine
from recovar.em.dense_single_volume.helpers import projection

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True), (" 1 ", True)])
def test_bpref_selector_is_strict_and_default_off(monkeypatch, token, expected):
    monkeypatch.delenv(engine.EXACT_LOCAL_BPREF_PROJECTOR_CAPACITY_ENV, raising=False)
    if token is not None:
        monkeypatch.setenv(engine.EXACT_LOCAL_BPREF_PROJECTOR_CAPACITY_ENV, token)
    assert engine._local_bpref_projector_capacity_requested() is expected


@pytest.mark.parametrize("token", ["", "true", "yes", "2", "-1", "typo"])
def test_bpref_selector_rejects_unknown_values(monkeypatch, token):
    monkeypatch.setenv(engine.EXACT_LOCAL_BPREF_PROJECTOR_CAPACITY_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        engine._local_bpref_projector_capacity_requested()


def test_optin_requires_shared_capacity_before_dataset_access(monkeypatch):
    monkeypatch.setenv(engine.EXACT_LOCAL_BPREF_PROJECTOR_CAPACITY_ENV, "1")
    monkeypatch.setenv(engine.EXACT_LOCAL_PROJECTOR_CAPACITY_ENV, "0")
    with pytest.raises(ValueError, match="requires the shared local projector capacity"):
        engine.run_local_em_exact(None, None, None, None, None, "linear_interp", image_batch_size=1,
                                  rotation_block_size=1, current_size=8, stable_fourier_window_shapes=True)


def _engine_tree():
    return ast.parse(inspect.getsource(engine.run_local_em_exact)).body[0]


@pytest.mark.parametrize("source_faithful,score_only", [(False, False), (False, True), (True, True), (True, False)])
def test_actual_late_eligibility_guard(source_faithful, score_only):
    # Execute the actual pure eligibility guard without constructing/scoring a
    # dataset. This establishes routing only, not an end-to-end EM execution.
    guard = next(node for node in _engine_tree().body if isinstance(node, ast.If)
                 and isinstance(node.test, ast.BoolOp)
                 and ast.unparse(node.test).startswith("bpref_projector_capacity_enabled and")
                 and "source_faithful_bpref" in ast.unparse(node.test))
    code = compile(ast.fix_missing_locations(ast.Module(body=[guard], type_ignores=[])), "<engine eligibility>", "exec")
    environment = dict(bpref_projector_capacity_enabled=True, source_faithful_bpref=source_faithful, score_only=score_only)
    if source_faithful and not score_only:
        exec(code, environment)
    else:
        with pytest.raises(ValueError, match="source-faithful VDAM accumulator"):
            exec(code, environment)
    environment["bpref_projector_capacity_enabled"] = False
    exec(code, environment)


def _run_actual_projector_selection(monkeypatch, enabled):
    """Execute only the engine's projector-preparation statements, unchanged."""
    function = _engine_tree()
    materialize = next(node for node in ast.walk(function) if isinstance(node, ast.If)
                       and any(isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)
                               and isinstance(n.value.func, ast.Name)
                               and n.value.func.id == "relion_projector_half_to_texture_full"
                               for n in node.body))
    start = next(i for i, node in enumerate(function.body) if isinstance(node, ast.Assign)
                 and any(isinstance(t, ast.Name) and t.id == "local_projection_half_arg" for t in node.targets))
    finish = next(i for i in range(start, len(function.body)) if isinstance(function.body[i], ast.If)
                  and ast.unparse(function.body[i].test) == "bpref_projector_capacity_enabled")
    statements = [materialize, *function.body[start:finish + 1]]
    logical = jnp.ones((7, 7, 4), jnp.complex64)
    full = jnp.ones((7, 7, 7), jnp.complex64) * (2 + 1j)
    half = jnp.ones((11, 11, 6), jnp.complex64) * (3 + 2j)
    runtime = jnp.asarray(2, jnp.int32)
    calls = []

    def materialize_full(value):
        assert value is logical
        calls.append("full")
        return full

    def prepare(value, *, r_max, physical_size, padding_factor):
        assert value is logical and (r_max, physical_size, padding_factor) == (2, 8, 1)
        calls.append("prepare")
        return half, runtime

    monkeypatch.setattr(projection, "relion_projector_half_to_texture_full", materialize_full)
    monkeypatch.setattr(projection, "prepare_relion_projector_capacity", prepare)
    namespace = dict(source_faithful_bpref=True, score_only=False, bpref_projector_capacity_enabled=enabled,
                     projector_capacity_enabled=True, relion_projector_half_big_jit=logical,
                     relion_projector_r_max_big_jit=2, physical_current_size=8, projection_padding_factor=1,
                     source_vdam_projector_full=None)
    exec(compile(ast.fix_missing_locations(ast.Module(body=statements, type_ignores=[])), "<engine projector selection>", "exec"), namespace)
    assert calls == (["prepare"] if enabled else ["full", "prepare"])
    assert namespace["local_projection_half_arg"] is half
    assert namespace["local_projection_runtime_radius"] is runtime
    assert namespace["source_vdam_projector_full"] is (half if enabled else full)
    assert namespace["source_vdam_projector_static_radius"] == (0 if enabled else 2)
    assert namespace["source_vdam_projector_runtime_radius"] is (runtime if enabled else None)
    return namespace


@pytest.mark.parametrize("serial_particles", [False, True])
def test_actual_selection_and_cuda_forwarding_preserve_all_other_operands(monkeypatch, serial_particles):
    captured = []
    result_y, result_ctf = object(), object()

    def cuda(*args, **kwargs):
        captured.append((args, kwargs))
        return result_y, result_ctf, object()

    monkeypatch.setattr(cb, "relion_vdam_mstep_fused_projector_x_half", cuda)
    b, r, t, p = 2, 2, 3, 2
    images = jnp.arange(b * p, dtype=jnp.float32).reshape(b, p).astype(jnp.complex64) + 1j
    ctf, invnoise = jnp.ones((b, p), jnp.float32) * 0.75, jnp.ones((b, p), jnp.float32) * 2
    probabilities = jnp.arange(1, b * r * t + 1, dtype=jnp.float32).reshape(b, r, t) / 16
    rotations = jnp.broadcast_to(jnp.eye(3, dtype=jnp.float32), (b, r, 3, 3))
    mask = jnp.asarray([[True, False], [True, True]])
    fy, fc = jnp.ones((2, 726), jnp.complex64), jnp.ones((2, 726), jnp.float32)
    angles = jnp.asarray([[0, 0], [0.1, 0.2], [-0.1, -0.2]], jnp.float32)
    stable_positions, logical_size = jnp.asarray([1, 2], jnp.int32), jnp.asarray(8, jnp.int32)
    group_ids, lanes = jnp.asarray([0, 1], jnp.int32), jnp.asarray([2, 3], jnp.int32)
    prepared = []
    for enabled in (False, True):
        selected = _run_actual_projector_selection(monkeypatch, enabled)
        prepared.append(selected)
        output = engine._accumulate_relion_vdam_physical_particle_grid(
            images, ctf, invnoise, probabilities, angles, None, rotations, mask, fy, fc,
            projector_full=selected["source_vdam_projector_full"], scoring_rotations=rotations,
            projector_r_max=selected["source_vdam_projector_static_radius"],
            runtime_projector_radius=selected["source_vdam_projector_runtime_radius"],
            pixel_indices=jnp.asarray([273, 274], jnp.int32), image_shape=(32, 32), volume_shape=(11, 11, 11), max_r=4.0,
            stable_dense_positions=stable_positions, logical_current_size=logical_size,
            reconstruction_group_ids=group_ids, worker_lane_ids=lanes, serial_particle_accumulation=serial_particles,
        )
        assert output[0] is result_y and output[1] is result_ctf
    old_args, old_kw = captured[0]
    new_args, new_kw = captured[1]
    assert old_args[0] is new_args[0] is fy and old_args[1] is new_args[1] is fc
    for index, (old, new) in enumerate(zip(old_args, new_args, strict=True)):
        if index not in (8, 13):
            np.testing.assert_array_equal(np.asarray(old), np.asarray(new))
    assert old_args[8] is prepared[0]["source_vdam_projector_full"]
    assert new_args[8] is prepared[1]["local_projection_half_arg"]
    assert (old_args[13], new_args[13]) == (2, 0)
    assert old_kw["runtime_projector_radius"] is None
    assert new_kw["runtime_projector_radius"] is prepared[1]["local_projection_runtime_radius"]
    assert old_kw.keys() == new_kw.keys()
    for key in old_kw.keys() - {"runtime_projector_radius"}:
        if old_kw[key] is None:
            assert new_kw[key] is None
        else:
            np.testing.assert_array_equal(np.asarray(old_kw[key]), np.asarray(new_kw[key]))
    expected = np.asarray(probabilities).copy()
    expected[0, 1] = 0
    np.testing.assert_array_equal(np.asarray(new_args[5]), expected)
    assert new_kw["stable_dense_positions"] is stable_positions and new_kw["logical_current_size"] is logical_size
    np.testing.assert_array_equal(np.asarray(new_kw["worker_lane_ids"]), [0, 0] if serial_particles else [2, 3])


def test_runtime_radius_cannot_escape_to_preprojected_helper(monkeypatch):
    monkeypatch.setattr(cb, "relion_vdam_mstep_fused_x_half", lambda *a, **k: pytest.fail("preprojected CUDA called"))
    with pytest.raises(ValueError, match="requires the inline projector"):
        engine._accumulate_relion_vdam_physical_particle_grid(
            *([None] * 10), pixel_indices=None, image_shape=(32, 32), volume_shape=(11, 11, 11), max_r=4,
            runtime_projector_radius=jnp.asarray(2, jnp.int32),
        )


def test_both_engine_accumulator_calls_bind_the_selected_projector():
    calls = [node for node in ast.walk(_engine_tree()) if isinstance(node, ast.Call)
             and isinstance(node.func, ast.Name) and node.func.id == "_accumulate_relion_vdam_physical_particle_grid"]
    assert len(calls) == 2
    for call in calls:
        options = {k.arg: ast.unparse(k.value) for k in call.keywords}
        assert options["projector_full"] == "source_vdam_projector_full"
        assert options["projector_r_max"] == "source_vdam_projector_static_radius"
        assert options["runtime_projector_radius"] == "source_vdam_projector_runtime_radius"
