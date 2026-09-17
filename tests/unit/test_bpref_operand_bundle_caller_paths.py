"""The two capture sites must describe the preprocessing they actually ran.

Capture 14034523 aborted at the big-JIT site because the bundle asked the dataset how it
was CONFIGURED. The first repair then labelled both sites from
``relion_exact_bpref_operands``, which inverts the big-JIT case: on the split site that
flag selects ``local_preprocessing._big_jit_preprocess_half`` and bypasses backend
preprocessing, while on the big-JIT site the same flag, with a positive mask radius,
selects ``cuda_backproject.relion_preprocess_real_f32`` -- real RELION CUDA preprocessing
with ``image_only_corrections`` and a parametric radius/cosine-width mask.

Source-string assertions cannot catch that, because both sites satisfied the helper's
contract while one described the wrong preprocessing. These tests EVALUATE each real call
site's argument expressions against controlled operands and then run the resulting bundle
through the real writer.
"""
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from recovar.em.diagnostics import bpref_diagnostics, local_bpref_capture
from recovar.em.local import local_big_jit, local_em_engine

BUNDLE_ENV = local_bpref_capture._HIGH_PRECISION_OPERAND_BUNDLE_ENV
SHAPE = (4, 4)
ROWS = 2
RADIUS, WIDTH = 37.5, 3.25
NORM = np.array([0.75, 1.25], dtype=np.float32)


def _site_expressions():
    """The argument expressions each real capture site passes, keyed by call order."""
    tree = ast.parse(inspect.getsource(local_em_engine))
    sites = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "_exact_local_bpref_operand_bundle"):
            sites.append({kw.arg: kw.value for kw in node.keywords if kw.arg})
    assert len(sites) == 2, f"expected two capture sites, found {len(sites)}"
    return sites


def _evaluate(expr, env):
    """Evaluate one call-site expression against controlled operand values."""
    return eval(compile(ast.Expression(expr), "<site>", "eval"), {}, dict(env))


BIG_JIT_ENV = {"relion_exact_bpref_operands": True, "relion_cuda_preprocess_radius": RADIUS,
               "relion_cuda_preprocess_cosine_width": WIDTH,
               "image_only_corrections_arg": NORM, "big_jit_image_mask_arg": None,
               "big_jit_mask_mode": None}


def _is_big_jit(site):
    """The big-JIT site is the one that forwards the CUDA kernel's own operands."""
    return "relion_preprocess_normalization" in site


def _big_jit_site():
    return next(s for s in _site_expressions() if _is_big_jit(s))


def _split_site():
    return next(s for s in _site_expressions() if not _is_big_jit(s))


# --- the executed label, not the source text ----------------------------------------

def test_big_jit_site_evaluates_to_the_cuda_path_when_the_kernel_would_run():
    """local_big_jit gates relion_preprocess_real_f32 on exact operands AND radius > 0."""
    assert "relion_exact_bpref_operands and relion_cuda_preprocess_radius > 0.0" in \
        inspect.getsource(local_big_jit)
    label = _evaluate(_big_jit_site()["preprocess_path"], BIG_JIT_ENV)
    assert label == "big_jit_relion_cuda"


def test_big_jit_site_evaluates_to_the_jax_path_when_the_radius_is_zero():
    label = _evaluate(_big_jit_site()["preprocess_path"], {**BIG_JIT_ENV,
                                                           "relion_cuda_preprocess_radius": 0.0})
    assert label == "big_jit_jax"


@pytest.mark.parametrize("exact,expected", [(True, "split_exact"), (False, "split_backend")])
def test_split_site_evaluates_to_its_own_labels(exact, expected):
    label = _evaluate(_split_site()["preprocess_path"],
                      {"relion_exact_bpref_operands": exact})
    assert label == expected


def test_the_two_sites_never_evaluate_to_the_same_label():
    """The defect was both sites reducing to one flag, so exercise the shared value."""
    big = _evaluate(_big_jit_site()["preprocess_path"], BIG_JIT_ENV)
    split = _evaluate(_split_site()["preprocess_path"], {"relion_exact_bpref_operands": True})
    assert big != split
    assert {big, split} <= local_bpref_capture._PREPROCESS_PATHS


def test_big_jit_site_forwards_the_kernels_own_operands():
    site = _big_jit_site()
    for arg in ("relion_preprocess_normalization", "relion_cuda_preprocess_radius",
                "relion_cuda_preprocess_cosine_width"):
        assert arg in site, f"big-JIT site does not forward {arg}"
    assert np.array_equal(_evaluate(site["relion_preprocess_normalization"], BIG_JIT_ENV), NORM)
    assert _evaluate(site["relion_cuda_preprocess_radius"], BIG_JIT_ENV) == RADIUS
    assert _evaluate(site["relion_cuda_preprocess_cosine_width"], BIG_JIT_ENV) == WIDTH


# --- real writer round-trip for the new scalars and the normalization ----------------

class _DS:
    image_mask = np.ones(SHAPE, dtype=np.float32)

    def original_image_indices_from_local(self, local):
        return np.asarray(local, dtype=np.int64) + 500


def _static(**over):
    base = dict(
        high_precision_operand_bundle=False, raw_batch_data=None, ctf_params=None,
        noise_variance_half=None, integer_pre_shifts=None, batch_image_corrections=None,
        batch_scale_corrections=None, relion_preprocess_normalization_factors=None,
        relion_cuda_preprocess=False, score_with_masked_images=True, image_mask=None,
        image_mask_mode="not-captured", voxel_size=1.0, ctf_mode="not-captured",
        ctf_dose_per_tilt=0.0, ctf_angle_per_tilt=0.0, disc_type="linear_interp",
        projection_padding_factor=1, reconstruction_padding_factor=1,
        use_relion_x_half_mstep=True, winner_take_all=False, max_r=2,
        window_indices=np.arange(6), image_shape=SHAPE, volume_shape=(4, 4, 4),
        shadow_only_mode=False, shadow_score_bitwise_equal=None,
        shadow_reduction_agreement=None)
    base.update(over)
    return base


def _dump(tmp_path, bundle):
    out = Path(tmp_path)
    mapping = out / "image_names.npy"
    np.save(mapping, np.array([f"{i + 1}@{out / 'stack.mrcs'}" for i in range(600)]),
            allow_pickle=False)
    n = 4
    summed = np.ones((ROWS, n, 1), dtype=np.complex64)
    scores = -np.ones((ROWS, n, 1), dtype=np.float32)
    probs = np.full((ROWS, n, 1), 0.25, dtype=np.float32)
    env = {"RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR": str(out),
           "RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION": "8",
           "RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF": "1",
           "RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE": "4",
           "RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256": "0" * 64,
           "RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY": str(mapping)}
    with mock.patch.dict(os.environ, env, clear=False):
        bpref_diagnostics.set_bpref_contribution_dump_context(iteration=8, half=1)
        try:
            bpref_diagnostics._maybe_dump_bpref_contribution_rows(
                experiment_dataset=_DS(), image_indices=np.arange(ROWS), current_size=4,
                summed=summed, ctf_probs=np.ones((ROWS, n, 1), dtype=np.float32),
                rotations=np.broadcast_to(np.eye(3), (ROWS, n, 3, 3)),
                actual_counts=np.full(ROWS, 4),
                class_actual_rotation_counts=np.ones((ROWS, 4), dtype=np.int64),
                class_segment_rotation_count=1,
                rotation_indices=np.arange(n, dtype=np.int64),
                fine_translations=np.asarray([[0.0, 0.0]]),
                scores=scores, preprior_scores=scores, probs=probs,
                rotation_log_prior=np.zeros((ROWS, n)),
                translation_log_prior=np.zeros((ROWS, 1)),
                log_z=np.zeros(ROWS), best_log_score=np.zeros(ROWS),
                reconstruction_probs=np.zeros_like(probs),
                reconstruction_mask=np.zeros_like(probs, dtype=bool),
                reconstruction_sum_weight=np.zeros(ROWS),
                reconstruction_threshold=np.zeros(ROWS),
                candidate_mask=np.ones_like(probs, dtype=bool), **bundle)
        finally:
            bpref_diagnostics.clear_bpref_contribution_dump_context()
    path = next(q for q in out.glob("*.npz") if "image_names" not in q.name)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _cuda_bundle():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        return local_bpref_capture._exact_local_bpref_operand_bundle(
            _static(), experiment_dataset=_DS(), image_shape=SHAPE,
            preprocess_path="big_jit_relion_cuda",
            relion_preprocess_normalization=NORM,
            relion_cuda_preprocess_radius=RADIUS,
            relion_cuda_preprocess_cosine_width=WIDTH,
            applied_image_mask=None, applied_image_mask_mode=None,
            raw_batch_data=np.ones((ROWS,) + SHAPE, dtype=np.float32),
            ctf_params=np.ones((ROWS, 9), dtype=np.float32),
            noise_variance_half=np.ones(12, dtype=np.float64),
            image_pre_shifts=None, integer_pre_shifts=np.zeros((ROWS, 2), dtype=np.int32),
            real_space_pre_shift_applied=False,
            image_corrections=None, scale_corrections=None,
            image_indices=np.arange(ROWS), unpadded_batch_size=ROWS)


def test_writer_round_trips_the_cuda_mask_scalars_and_normalization(tmp_path):
    payload = _dump(tmp_path, _cuda_bundle())
    assert float(payload["relion_cuda_preprocess_radius"]) == RADIUS
    assert float(payload["relion_cuda_preprocess_cosine_width"]) == WIDTH
    assert np.array_equal(payload["relion_preprocess_normalization_factors"], NORM)
    assert bool(payload["relion_cuda_preprocess"]) is True
    assert bool(payload["high_precision_operand_bundle"]) is True
    # The mode stays a plain mask mode; the engine path is its own field.
    assert str(payload["image_mask_mode"]) == "relion_cuda_parametric"
    assert "|" not in str(payload["image_mask_mode"])
    assert str(payload["preprocess_path"]) == "big_jit_relion_cuda"


def test_unrequested_capture_keeps_the_new_fields_absent(tmp_path):
    """Default-off payload behaviour is unchanged: the scalars record NaN."""
    payload = _dump(tmp_path, _static())
    assert np.isnan(float(payload["relion_cuda_preprocess_radius"]))
    assert np.isnan(float(payload["relion_cuda_preprocess_cosine_width"]))
    assert bool(payload["high_precision_operand_bundle"]) is False
    assert str(payload["preprocess_path"]) == ""
    assert str(payload["image_mask_mode"]) == "not-captured"


def test_cuda_path_without_the_mask_scalars_is_refused():
    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        with pytest.raises(RuntimeError, match="mask radius and cosine width"):
            local_bpref_capture._exact_local_bpref_operand_bundle(
                _static(), experiment_dataset=_DS(), image_shape=SHAPE,
                preprocess_path="big_jit_relion_cuda",
                relion_preprocess_normalization=NORM,
                applied_image_mask=None, applied_image_mask_mode=None,
                raw_batch_data=np.ones((ROWS,) + SHAPE, dtype=np.float32),
                ctf_params=np.ones((ROWS, 9), dtype=np.float32),
                noise_variance_half=np.ones(12), image_pre_shifts=None,
                integer_pre_shifts=np.zeros((ROWS, 2), dtype=np.int32),
                real_space_pre_shift_applied=False, image_corrections=None,
                scale_corrections=None, image_indices=np.arange(ROWS),
                unpadded_batch_size=ROWS)
