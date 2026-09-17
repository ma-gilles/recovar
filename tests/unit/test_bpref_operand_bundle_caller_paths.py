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
from recovar.em.local import local_em_engine

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


# --- every branch combination, evaluated from the real call sites ------------------
#
# Capture 14036320 died on the masked NON-exact big-JIT path: the kernel call at
# local_em_engine.py:2406 always receives big_jit_image_mask_arg, but the capture site
# gated it on relion_exact_bpref_operands and passed None. Earlier coverage evaluated only
# the path label, and only for exact=True, so it could not see this. At K>1
# use_exact_local_relion_operands is False by construction
# (sparse_pass2_estep.py:826-830 requires state.K == 1), so the non-exact branch is the
# one every K=4 capture actually takes.

MASK = _np_mask = None  # set below


def _big_jit_env(exact, radius):
    return {"relion_exact_bpref_operands": exact,
            "relion_cuda_preprocess_radius": radius,
            "relion_cuda_preprocess_cosine_width": WIDTH,
            "image_only_corrections_arg": NORM,
            "big_jit_image_mask_arg": MASK_ARRAY,
            "big_jit_mask_mode": "relion_background_fill"}


MASK_ARRAY = np.ones((4, 4), dtype=np.float32)


@pytest.mark.parametrize("exact,radius,expected", [
    (True, RADIUS, "big_jit_relion_cuda"),
    (True, 0.0, "big_jit_jax"),
    (False, RADIUS, "big_jit_jax"),   # the K>1 capture path, radius set but exact off
    (False, 0.0, "big_jit_jax"),
])
def test_big_jit_path_label_over_every_branch(exact, radius, expected):
    """local_big_jit gates relion_preprocess_real_f32 on exact AND radius > 0."""
    assert _evaluate(_big_jit_site()["preprocess_path"], _big_jit_env(exact, radius)) == expected


@pytest.mark.parametrize("exact,radius", [(True, RADIUS), (True, 0.0),
                                          (False, RADIUS), (False, 0.0)])
def test_big_jit_site_always_forwards_the_mask_the_kernel_receives(exact, radius):
    """local_em_engine.py:1529 resolves the mask unconditionally and :2406 always passes
    it, so no branch may hand the capture None."""
    site = _big_jit_site()
    env = _big_jit_env(exact, radius)
    mask = _evaluate(site["applied_image_mask"], env)
    mode = _evaluate(site["applied_image_mask_mode"], env)
    assert mask is not None and np.asarray(mask).size > 0, (
        "the masked non-exact big-JIT path must not receive a None mask"
    )
    assert mode == "relion_background_fill"


@pytest.mark.parametrize("exact,radius", [(True, RADIUS), (True, 0.0),
                                          (False, RADIUS), (False, 0.0)])
def test_big_jit_site_reports_the_actual_ctf_construction(exact, radius):
    """local_big_jit.py:2224 takes the source-STAR CTF only when exact is set; the
    preprocessing route cannot stand in for it, since big_jit_jax covers both."""
    site = _big_jit_site()
    assert "exact_source_star_ctf" in site
    assert _evaluate(site["exact_source_star_ctf"], _big_jit_env(exact, radius)) is exact


@pytest.mark.parametrize("exact,expected", [(True, "split_exact"), (False, "split_backend")])
def test_split_site_label_and_ctf_over_both_branches(exact, expected):
    site = _split_site()
    env = {"relion_exact_bpref_operands": exact,
           "big_jit_image_mask_arg": MASK_ARRAY, "big_jit_mask_mode": "relion_background_fill"}
    assert _evaluate(site["preprocess_path"], env) == expected
    assert _evaluate(site["exact_source_star_ctf"], env) is exact
    # local_preprocessing resolves its own mask only on the exact branch.
    assert (_evaluate(site["applied_image_mask"], env) is None) is (not exact)


def test_the_two_sites_never_evaluate_to_the_same_label():
    big = _evaluate(_big_jit_site()["preprocess_path"], _big_jit_env(True, RADIUS))
    split = _evaluate(_split_site()["preprocess_path"], {"relion_exact_bpref_operands": True})
    assert big != split
    assert {big, split} <= local_bpref_capture._PREPROCESS_PATHS


def test_big_jit_site_forwards_the_kernels_own_operands():
    site = _big_jit_site()
    for arg in ("relion_preprocess_normalization", "relion_cuda_preprocess_radius",
                "relion_cuda_preprocess_cosine_width"):
        assert arg in site, f"big-JIT site does not forward {arg}"
    env = _big_jit_env(True, RADIUS)
    assert np.array_equal(_evaluate(site["relion_preprocess_normalization"], env), NORM)
    assert _evaluate(site["relion_cuda_preprocess_radius"], env) == RADIUS
    assert _evaluate(site["relion_cuda_preprocess_cosine_width"], env) == WIDTH


FIXTURE = "/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_pdb_k4_5k_128/particles.star"


def _production_ctf_evaluator():
    """config.ctf for the real K=4 fixture, i.e. what compute_ctf_half evaluates.

    Built from the production ForwardModelConfig rather than a stand-in, so the captured
    mode is checked against the evaluator this workload actually uses.
    """
    from recovar.core.configs import ForwardModelConfig
    from recovar.data_io.cryoem_dataset import load_dataset

    ds = load_dataset(FIXTURE, lazy=True, datadir=None, strip_prefix=None)
    return ForwardModelConfig.from_dataset(ds).ctf


requires_fixture = pytest.mark.skipif(
    not Path(FIXTURE).is_file(), reason="K=4 parity fixture not present"
)


def _production_exact_flag(*, n_classes, exact_projector=True, relion_cuda=True):
    """Evaluate sparse_pass2_estep's own use_exact_local_relion_operands expression.

    Executed, not mirrored: the expression is lifted from the real assignment and run
    against controlled values, so a change to that gate breaks this test.
    """
    from recovar.em.vdam import sparse_pass2_estep

    tree = ast.parse(inspect.getsource(sparse_pass2_estep))
    expr = next(
        node.value for node in ast.walk(tree)
        if isinstance(node, ast.Assign)
        and any(getattr(t, "id", None) == "use_exact_local_relion_operands" for t in node.targets)
    )

    class _State:
        K = n_classes

    return bool(eval(compile(ast.Expression(expr), "<gate>", "eval"), {}, {
        "state": _State(), "bool": bool,
        "use_exact_relion_projector": exact_projector,
        "uses_relion_cuda_image_preprocessing": lambda ds: relion_cuda,
        "group_dataset": object(),
    }))


def test_the_k4_workload_cannot_reach_the_exact_operand_branch():
    """sparse_pass2_estep requires state.K == 1, so every K=4 capture runs non-exact."""
    assert _production_exact_flag(n_classes=1) is True
    assert _production_exact_flag(n_classes=4) is False


@requires_fixture
def test_k4_production_configuration_gets_a_masked_non_exact_big_jit_capture(tmp_path):
    """The configuration capture 14036320 actually ran, end to end through the writer.

    This is the branch that failed: exact flag false, mask radius set, masked scoring on.
    """
    exact = _production_exact_flag(n_classes=4)
    site = _big_jit_site()
    env = _big_jit_env(exact, RADIUS)
    assert _evaluate(site["preprocess_path"], env) == "big_jit_jax"
    assert _evaluate(site["exact_source_star_ctf"], env) is False
    mask = _evaluate(site["applied_image_mask"], env)
    assert mask is not None and np.asarray(mask).size > 0

    with mock.patch.dict(os.environ, {BUNDLE_ENV: "1"}, clear=False):
        bundle = local_bpref_capture._exact_local_bpref_operand_bundle(
            _static(), experiment_dataset=_DS(), image_shape=SHAPE,
            preprocess_path="big_jit_jax", exact_source_star_ctf=exact,
            production_ctf=_production_ctf_evaluator(),
            relion_preprocess_normalization=NORM,
            relion_cuda_preprocess_radius=RADIUS, relion_cuda_preprocess_cosine_width=WIDTH,
            applied_image_mask=np.asarray(mask, dtype=np.float32),
            applied_image_mask_mode=_evaluate(site["applied_image_mask_mode"], env),
            raw_batch_data=np.ones((ROWS,) + SHAPE, dtype=np.float32),
            ctf_params=np.ones((ROWS, 9), dtype=np.float32),
            noise_variance_half=np.ones(12, dtype=np.float64),
            image_pre_shifts=None, integer_pre_shifts=np.zeros((ROWS, 2), dtype=np.int32),
            real_space_pre_shift_applied=False, image_corrections=None,
            scale_corrections=None, image_indices=np.arange(ROWS), unpadded_batch_size=ROWS)
    payload = _dump(tmp_path, bundle)
    assert bool(payload["high_precision_operand_bundle"]) is True
    assert str(payload["preprocess_path"]) == "big_jit_jax"
    assert str(payload["image_mask_mode"]) == "relion_background_fill"
    assert np.asarray(payload["image_mask"]).size > 0
    # The CTF the non-exact path actually builds, read off the production config rather
    # than a placeholder. "not-captured" here would describe nothing.
    ctf = _production_ctf_evaluator()
    assert str(payload["ctf_mode"]) == str(getattr(getattr(ctf, "mode", "legacy"), "name", "legacy"))
    assert str(payload["ctf_mode"]) != "not-captured"
    assert float(payload["ctf_dose_per_tilt"]) == float(getattr(ctf, "dose_per_tilt", 0.0))
    assert float(payload["ctf_angle_per_tilt"]) == float(getattr(ctf, "angle_per_tilt", 0.0))
    assert bool(payload["relion_cuda_preprocess"]) is False
    assert np.isnan(float(payload["relion_cuda_preprocess_radius"]))
    assert np.array_equal(payload["relion_preprocess_normalization_factors"],
                          np.ones(ROWS, dtype=np.float32))


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
            preprocess_path="big_jit_relion_cuda", exact_source_star_ctf=True,
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
                preprocess_path="big_jit_relion_cuda", exact_source_star_ctf=True,
                relion_preprocess_normalization=NORM,
                applied_image_mask=None, applied_image_mask_mode=None,
                raw_batch_data=np.ones((ROWS,) + SHAPE, dtype=np.float32),
                ctf_params=np.ones((ROWS, 9), dtype=np.float32),
                noise_variance_half=np.ones(12), image_pre_shifts=None,
                integer_pre_shifts=np.zeros((ROWS, 2), dtype=np.int32),
                real_space_pre_shift_applied=False, image_corrections=None,
                scale_corrections=None, image_indices=np.arange(ROWS),
                unpadded_batch_size=ROWS)
