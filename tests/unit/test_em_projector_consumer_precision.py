"""EM must round only the consumer slab, preserving explicit diagnostics."""

import ast
from pathlib import Path
from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np
import pytest

from recovar.em.refinement.projector_preparation import prepare_scoring_projector

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("classes", [1, 4])
@pytest.mark.parametrize("score_double,project_double", [(False, False), (True, False), (False, True), (True, True)])
def test_consumer_precision_preserves_native_source(classes, score_double, project_double):
    values = np.arange(classes * 5 * 5 * 3, dtype=np.float64).reshape(classes, 5, 5, 3)
    slab = (values + 1.234567890123 + 1j * (values + 0.987654321)).astype(np.complex128)
    before = slab.copy()
    result = prepare_scoring_projector(slab, use_float64_scoring=score_double, use_float64_projections=project_double)
    expected_dtype = np.complex128 if score_double or project_double else np.complex64
    assert result.dtype == expected_dtype
    np.testing.assert_array_equal(result, before.astype(expected_dtype))
    np.testing.assert_array_equal(slab, before)


def test_captured_single_precision_stays_single_for_score_only_double():
    slab = jnp.ones((5, 5, 3), dtype=jnp.complex64)
    assert prepare_scoring_projector(slab) is slab
    assert prepare_scoring_projector(slab, use_float64_scoring=True) is slab
    assert prepare_scoring_projector(slab, use_float64_projections=True).dtype == jnp.complex128
    assert prepare_scoring_projector(None) is None


def test_texture_gate_receives_production_dtype(monkeypatch):
    from recovar.em.helpers import projection

    monkeypatch.setattr(projection, "_cuda_projection_available", lambda: True)
    source = np.ones((7, 7, 4), dtype=np.complex128)
    assert not projection._relion_projector_texture_enabled(source, r_max=2, padding_factor=1)
    assert projection._relion_projector_texture_enabled(prepare_scoring_projector(source), r_max=2, padding_factor=1)
    assert not projection._relion_projector_texture_enabled(
        prepare_scoring_projector(source, use_float64_projections=True),
        r_max=2,
        padding_factor=1,
    )


def test_all_em_phase_owners_wire_both_precision_flags():
    import recovar.em.classification.k_class as module

    tree = ast.parse(Path(module.__file__).read_text())
    expected = {
        "_run_sparse_k_class_adaptive_pass2",
        "_run_sparse_firstiter_global_winner_subset_pass2",
        "run_local_k_class_em",
        "run_dense_k_class_em_adaptive",
    }
    found = set()
    calls = 0
    for function in tree.body:
        if not isinstance(function, ast.FunctionDef):
            continue
        for node in ast.walk(function):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id == "prepare_scoring_projector"
            ):
                assert {k.arg for k in node.keywords} == {"use_float64_scoring", "use_float64_projections"}
                found.add(function.name)
                calls += 1
    assert expected <= found
    assert calls == 5  # Includes the firstiter-CC coarse score probe.


@pytest.mark.parametrize("score_only,texture", [(True, False), (False, True)])
@pytest.mark.parametrize("score_double,project_double", [(False, False), (True, False), (False, True), (True, True)])
def test_k1_local_dispatch_selects_projector_precision(
    monkeypatch,
    score_only,
    texture,
    score_double,
    project_double,
):
    from recovar.em.local import local_search_iteration as module

    class Captured(Exception):
        pass

    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        raise Captured

    monkeypatch.setattr(module, "run_local_em_exact", capture)
    monkeypatch.setattr(module, "_local_search_engine_rotation_block_size", lambda value: value)
    monkeypatch.setattr(
        module,
        "_estimate_relion_em_batch_sizes",
        lambda **kw: SimpleNamespace(
            image_batch_size=kw["requested_image_batch_size"],
            rotation_block_size=kw["requested_rotation_block_size"],
        ),
    )
    dataset = SimpleNamespace(image_shape=(8, 8), volume_shape=(8, 8, 8), voxel_size=1.0)
    translations = np.zeros((1, 2), dtype=np.float32)
    layout = SimpleNamespace(rotation_counts=np.ones(1, dtype=np.int32), translation_grid=translations)
    source = np.full((7, 7, 4), 1.234567890123 + 0.987654321j, dtype=np.complex128)
    before = source.copy()
    with pytest.raises(Captured):
        module._run_local_search_iteration(
            dataset,
            np.zeros(512, dtype=np.complex64),
            np.ones(512),
            np.ones(64),
            np.zeros((1, 3)),
            None,
            healpix_order=0,
            sigma_rot=0.0,
            sigma_psi=0.0,
            translations=translations,
            prior_translations=translations,
            sigma_offset_angstrom=1.0,
            disc_type="linear_interp",
            image_batch_size=1,
            rotation_block_size=16,
            current_size=4,
            pass2_layout=layout,
            relion_projector_half=source,
            relion_projector_r_max=2,
            use_float64_scoring=score_double,
            use_float64_projections=project_double,
            score_only=score_only,
            projection_relion_texture_interp=texture,
        )
    expected = np.complex128 if score_double or project_double else np.complex64
    assert captured["relion_projector_half"].dtype == expected
    np.testing.assert_array_equal(captured["relion_projector_half"], before.astype(expected))
    np.testing.assert_array_equal(source, before)
    assert captured["score_only"] is score_only
    assert captured["projection_relion_texture_interp"] is texture
    assert captured["use_float64_normalization"] is True


@pytest.mark.parametrize("classes", [1, 4])
@pytest.mark.parametrize("score_double,project_double", [(False, False), (True, False), (False, True), (True, True)])
def test_local_class_dispatch_receives_consumer_dtype(monkeypatch, classes, score_double, project_double):
    from recovar.em.classification import k_class

    class Captured(Exception):
        pass

    captured = {}

    def capture(*args, **kwargs):
        captured.update(kwargs)
        raise Captured

    monkeypatch.setattr(k_class, "run_local_em_exact", capture)
    source = np.full((classes, 7, 7, 4), 1.234567890123 + 0.987654321j, dtype=np.complex128)
    before = source.copy()
    with pytest.raises(Captured):
        k_class.run_local_k_class_em(
            SimpleNamespace(n_units=1),
            np.zeros((classes, 512), dtype=np.complex64),
            np.ones(512, dtype=np.float32),
            np.ones(64, dtype=np.float32),
            SimpleNamespace(n_images=1),
            "linear_interp",
            relion_projector_half=source,
            use_float64_scoring=score_double,
            use_float64_projections=project_double,
        )
    expected = np.complex128 if score_double or project_double else np.complex64
    assert captured["relion_projector_half"].dtype == expected
    np.testing.assert_array_equal(captured["relion_projector_half"], before[0].astype(expected))
    np.testing.assert_array_equal(source, before)
