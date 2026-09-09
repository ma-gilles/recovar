"""Opt-in local projector capacity: logical ownership, compact outputs and ABI."""

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from recovar import cuda_backproject as cb
from recovar.core import slicing
from recovar.em.dense_single_volume import local_big_jit as big
from recovar.em.dense_single_volume import local_em_engine as engine
from recovar.em.dense_single_volume.helpers import projection as proj
from recovar.em.dense_single_volume.helpers.env_flags import parse_env_binary_flag

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("token,expected", [(None, False), ("0", False), ("1", True)])
def test_selector_default_off(monkeypatch, token, expected):
    monkeypatch.delenv(engine.EXACT_LOCAL_PROJECTOR_CAPACITY_ENV, raising=False)
    if token is not None:
        monkeypatch.setenv(engine.EXACT_LOCAL_PROJECTOR_CAPACITY_ENV, token)
    assert parse_env_binary_flag(engine.EXACT_LOCAL_PROJECTOR_CAPACITY_ENV) is expected


@pytest.mark.parametrize("token", ["", "2", "true", "typo"])
def test_selector_rejects_unknown(monkeypatch, token):
    monkeypatch.setenv(engine.EXACT_LOCAL_PROJECTOR_CAPACITY_ENV, token)
    with pytest.raises(ValueError, match="must be 0 or 1"):
        parse_env_binary_flag(engine.EXACT_LOCAL_PROJECTOR_CAPACITY_ENV)


def _logical(radius, pf):
    n = 2 * pf * radius + 3
    values = np.arange(n * n * (n // 2 + 1), dtype=np.float32).reshape(n, n, n // 2 + 1)
    return (values + 1j * (values + 1)).astype(np.complex64)


@pytest.mark.parametrize("q,radius,pf", [(32, 15, 1), (32, 16, 2), (64, 17, 1), (96, 42, 2)])
def test_padding_preserves_logical_ghosts_and_inputs(monkeypatch, q, radius, pf):
    monkeypatch.setattr(proj, "_cuda_projection_available", lambda: True)
    monkeypatch.setattr(slicing, "_use_cuda", lambda order: order == 1)
    monkeypatch.setenv(proj._RELION_PROJECTOR_TEXTURE_ENV, "1")
    logical = _logical(radius, pf)
    original = logical.copy()
    padded, runtime = proj.prepare_relion_projector_capacity(logical, r_max=radius, physical_size=q, padding_factor=pf)
    n = pf * q + 3
    offset = (n - logical.shape[0]) // 2
    expected = np.zeros((n, n, n // 2 + 1), np.complex64)
    expected[offset : offset + logical.shape[0], offset : offset + logical.shape[1], : logical.shape[2]] = logical
    np.testing.assert_array_equal(np.asarray(padded).view(np.uint32), expected.view(np.uint32))
    np.testing.assert_array_equal(logical.view(np.uint32), original.view(np.uint32))
    assert runtime.shape == () and runtime.dtype == jnp.int32 and int(runtime) == radius


@pytest.mark.parametrize("failure", ["texture_disabled", "cuda_disabled", "shape", "dtype", "radius", "padding"])
def test_preparation_rejects_unsupported_route(monkeypatch, failure):
    monkeypatch.setattr(proj, "_cuda_projection_available", lambda: True)
    monkeypatch.setattr(slicing, "_use_cuda", lambda order: failure != "cuda_disabled")
    monkeypatch.setenv(proj._RELION_PROJECTOR_TEXTURE_ENV, "0" if failure == "texture_disabled" else "1")
    logical = _logical(15, 1)
    if failure == "shape":
        logical = logical[:-2, :-2, :-1]
    if failure == "dtype":
        logical = logical.astype(np.complex128)
    with pytest.raises(ValueError):
        proj.prepare_relion_projector_capacity(
            logical,
            r_max=17 if failure == "radius" else 15,
            physical_size=32,
            padding_factor=3 if failure == "padding" else 1,
        )


def _local_kwargs(q, pf, radius):
    n = 128
    # Include crop +Nyquist, negative y, DC, positive x and a corner outside disk.
    coords = [(q // 2, 0), (-q // 2 + 1, 1), (0, 0), (0, q // 2), (q // 2, q // 2)]
    indices = jnp.asarray([(y + n // 2) * (n // 2 + 1) + x for y, x in coords], jnp.int32)
    return dict(
        projection_pixel_indices=indices,
        image_shape=(n, n),
        proj_volume_shape=(n,) * 3,
        disc_type="linear_interp",
        projection_half_volume=False,
        projection_max_r=q // 2,
        relion_projector_output_size=q,
        projection_relion_texture_interp=False,
        projection_force_jax=False,
        projection_mask_current_image_disk=False,
        use_relion_projector=True,
        relion_projector_r_max=radius,
        projection_padding_factor=pf,
    )


def test_compact_mapping_scaling_and_runtime_trace(monkeypatch):
    q = 32
    crop = jnp.arange(q * (q // 2 + 1), dtype=jnp.float32)[None].astype(jnp.complex64) * (1 + 2j)
    records = []
    monkeypatch.setattr(proj, "_cuda_projection_available", lambda: True)
    monkeypatch.setattr(proj, "project_half_spectrum", lambda *a, **kw: crop)
    monkeypatch.setenv(proj._RELION_PROJECTOR_TEXTURE_ENV, "1")
    monkeypatch.setenv("RECOVAR_DENSE_MEANS_SCALE", "-N2")

    def capacity(half, rotations, radius, **kw):
        records.append((half.shape, radius.aval, kw))
        return crop + radius.astype(jnp.complex64) * 0

    monkeypatch.setattr(cb, "project_relion_half_capacity", capacity)
    kwargs = _local_kwargs(q, 1, 15)
    rotations = jnp.eye(3, dtype=jnp.float32)[None]
    old = big._project_local_half_spectrum(None, jnp.asarray(_logical(15, 1)), rotations, **kwargs)
    new_kwargs = dict(kwargs, relion_projector_r_max=0, projector_capacity=True)

    @jax.jit
    def projected(half, radius):
        return big._project_local_half_spectrum(None, half, rotations, runtime_projector_r_max=radius, **new_kwargs)

    physical = jnp.zeros((35, 35, 18), jnp.complex64)
    for radius in [15, 16]:
        result = projected(physical, jnp.asarray(radius, jnp.int32))
        np.testing.assert_array_equal(np.asarray(result).view(np.uint32), np.asarray(old).view(np.uint32))
    assert len(records) == 1 and records[0][1].shape == () and records[0][1].dtype == jnp.int32
    assert records[0][2] == {"image_shape": (32, 32), "padding_factor": 1}
    assert projected._cache_size() == 1
    expected = np.asarray(crop)[0, [0, 18, 272, 288, 16]] * -(128**2)
    np.testing.assert_array_equal(np.asarray(old)[0].view(np.uint32), expected.view(np.uint32))


def test_positional_radius_preserves_fixed_capacity_carry():
    positional, keyword = big._local_bucket_big_jit_signature_parts()
    names = [p.name for p in positional]
    assert names[-2:] == ["config", "runtime_projector_r_max"]
    assert "runtime_projector_r_max" not in [p.name for p in keyword]
    assert names[7:17] == [
        "Ft_y",
        "Ft_ctf",
        "noise_wsum",
        "noise_img_power",
        "noise_a2",
        "noise_xa",
        "noise_scale_xa",
        "noise_scale_aa",
        "noise_sigma2_offset",
        "noise_sumw",
    ]
    args = tuple(object() for _ in positional)
    prepared = big._prepare_fixed_capacity_local_call(*args)
    assert prepared.leading_arguments + args[7:17] + prepared.trailing_arguments == args
    assert prepared.trailing_arguments[-1] is args[-1]


def test_engine_projection_operand_is_separate_from_bpref():
    # Verify the actual caller binding and preserve the scientific consumers.
    tree = ast.parse(inspect.getsource(engine))
    assignments = [n for n in ast.walk(tree) if isinstance(n, ast.Assign)]
    args = next(
        n.value for n in assignments if any(isinstance(t, ast.Name) and t.id == "big_jit_arguments" for t in n.targets)
    )
    assert isinstance(args, ast.Tuple) and ast.unparse(args.elts[6]) == "local_projection_half_arg"
    assert ast.unparse(args.elts[-1]) == "local_projection_runtime_radius"
    options = next(
        n.value
        for n in assignments
        if any(isinstance(t, ast.Name) and t.id == "big_jit_static_options" for t in n.targets)
    )
    assert {k.arg: ast.unparse(k.value) for k in options.keywords}[
        "relion_projector_r_max"
    ] == "local_projection_static_radius"
    full = next(
        n.value
        for n in assignments
        if any(isinstance(t, ast.Name) and t.id == "source_vdam_projector_full" for t in n.targets)
        and isinstance(n.value, ast.Call)
    )
    assert ast.unparse(full.args[0]) == "relion_projector_half_big_jit"
    for node in ast.walk(tree):
        if isinstance(node, ast.keyword) and node.arg == "projector_r_max":
            assert ast.unparse(node.value) != "local_projection_static_radius"


@pytest.mark.gpu
@pytest.mark.parametrize("q,radius,pf", [(32, 15, 1), (64, 17, 1), (96, 42, 1), (32, 16, 2), (64, 17, 2), (96, 42, 2)])
def test_gpu_compact_local_projection_matches_logical_words(q, radius, pf):
    assert jax.default_backend() == "gpu"
    logical = _logical(radius, pf)
    physical, runtime = proj.prepare_relion_projector_capacity(
        logical, r_max=radius, physical_size=q, padding_factor=pf
    )
    rotations = jnp.asarray(Rotation.random(5, random_state=29).as_matrix(), jnp.float32)
    kwargs = _local_kwargs(q, pf, radius)
    old = big._project_local_half_spectrum(None, jnp.asarray(logical), rotations, **kwargs)
    new = big._project_local_half_spectrum(
        None,
        physical,
        rotations,
        **dict(kwargs, relion_projector_r_max=0, projector_capacity=True, runtime_projector_r_max=runtime),
    )
    np.testing.assert_array_equal(np.asarray(new).view(np.uint32), np.asarray(old).view(np.uint32))


def test_engine_rejects_optin_before_nonstable_dataset_access(monkeypatch):
    monkeypatch.setenv(engine.EXACT_LOCAL_PROJECTOR_CAPACITY_ENV, "1")
    with pytest.raises(ValueError, match="requires stable exact-local"):
        engine.run_local_em_exact(
            None, None, None, None, None, "linear_interp", image_batch_size=1, rotation_block_size=1, current_size=16
        )


@pytest.mark.parametrize(
    "override",
    [
        {"r_max": 15},
        {"runtime_r_max": None},
        {"centered_rows": False},
        {"pixel_indices": None},
        {"projector_output_size": 0},
        {"projector_output_size": 17},
        {"relion_texture_interp": False},
    ],
)
def test_capacity_helper_rejects_unsupported_contract(monkeypatch, override):
    monkeypatch.setattr(cb, "project_relion_half_capacity", lambda *a, **k: pytest.fail("entered CUDA"))
    kwargs = dict(
        r_max=0,
        padding_factor=1,
        return_abs2=False,
        centered_rows=True,
        projector_output_size=32,
        pixel_indices=jnp.asarray([0], jnp.int32),
        projector_capacity=True,
        runtime_r_max=jnp.asarray(15, jnp.int32),
    )
    kwargs.update(override)
    with pytest.raises(ValueError):
        proj.compute_relion_projector_projections_block(
            jnp.zeros((35, 35, 18), jnp.complex64), jnp.eye(3, dtype=jnp.float32)[None], (128, 128), **kwargs
        )
