"""The capture must record the rotation matrices the engine actually scored.

Gate 4 of the own-score replay stalled because the payload stored
``oversampled_rotation_indices`` but no matrices. They cannot be recovered by indexing a
global table: ``local_layout.py:930-937`` appends ``rotations_parts`` and
``rotation_ids_parts`` in the same loop, so the layout emits ids and matrices together and
the ids are meaningful only alongside them. Reconstructing the candidate set instead would
need ``sigma_rot``/``sigma_psi`` (``local_layout.py:264``), which no run artifact records.

Translations need no new field: ``fine_translations`` already stores the executed grid.
"""
from __future__ import annotations

import ast
import inspect
import os
from pathlib import Path
from unittest import mock

import numpy as np
import pytest

from recovar.em.diagnostics import bpref_diagnostics
from recovar.em.local import local_em_engine

B, R, T = 2, 6, 3


class _DS:
    def original_image_indices_from_local(self, local):
        return np.asarray(local, dtype=np.int64) + 40


def _rotations(b=B, r=R):
    """Distinct per-row matrices, so a misaligned write cannot pass by coincidence."""
    base = np.broadcast_to(np.eye(3), (b, r, 3, 3)).astype(np.float32).copy()
    for i in range(b):
        for j in range(r):
            base[i, j, 0, 0] = 1.0 + i * 10 + j
    return base


def _dump(tmp_path, **over):
    out = Path(tmp_path)
    mapping = out / "image_names.npy"
    np.save(mapping, np.array([f"{i + 1}@{out / 'stack.mrcs'}" for i in range(64)]),
            allow_pickle=False)
    summed = np.ones((B, R, 1), dtype=np.complex64)
    scores = -np.ones((B, R, T), dtype=np.float32)
    probs = np.full((B, R, T), 1.0 / (R * T), dtype=np.float32)
    kw = dict(
        experiment_dataset=_DS(), image_indices=np.arange(B), current_size=4,
        summed=summed, ctf_probs=np.ones((B, R, 1), dtype=np.float32),
        rotations=np.broadcast_to(np.eye(3), (B, R, 3, 3)),
        actual_counts=np.full(B, R),
        rotation_indices=np.arange(R, dtype=np.int64),
        fine_translations=np.zeros((T, 2), dtype=np.float32),
        scores=scores, preprior_scores=scores, probs=probs,
        rotation_log_prior=np.zeros((B, R)), translation_log_prior=np.zeros((B, T)),
        log_z=np.zeros(B), best_log_score=np.zeros(B),
        reconstruction_probs=np.zeros_like(probs),
        reconstruction_mask=np.zeros_like(probs, dtype=bool),
        reconstruction_sum_weight=np.zeros(B), reconstruction_threshold=np.zeros(B),
        candidate_mask=np.ones_like(probs, dtype=bool),
        high_precision_operand_bundle=False, raw_batch_data=None, ctf_params=None,
        noise_variance_half=None, integer_pre_shifts=None, batch_image_corrections=None,
        batch_scale_corrections=None, relion_preprocess_normalization_factors=None,
        relion_cuda_preprocess=False, score_with_masked_images=False, image_mask=None,
        image_mask_mode="not-captured", voxel_size=1.0, ctf_mode="not-captured",
        ctf_dose_per_tilt=0.0, ctf_angle_per_tilt=0.0, disc_type="linear_interp",
        projection_padding_factor=1, reconstruction_padding_factor=1,
        use_relion_x_half_mstep=True, winner_take_all=False, max_r=2,
        window_indices=np.arange(6), image_shape=(4, 4), volume_shape=(4, 4, 4),
        shadow_only_mode=False, shadow_score_bitwise_equal=None,
        shadow_reduction_agreement=None,
    )
    kw.update(over)
    env = {"RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR": str(out),
           "RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION": "8",
           "RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF": "1",
           "RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE": "4",
           "RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256": "0" * 64,
           "RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY": str(mapping)}
    with mock.patch.dict(os.environ, env, clear=False):
        bpref_diagnostics.set_bpref_contribution_dump_context(iteration=8, half=1)
        try:
            bpref_diagnostics._maybe_dump_bpref_contribution_rows(**kw)
        finally:
            bpref_diagnostics.clear_bpref_contribution_dump_context()
    path = next(q for q in out.glob("*.npz") if "image_names" not in q.name)
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}, path.stat().st_size


# --- schema ------------------------------------------------------------------------

def test_absent_candidate_rotations_keep_the_field_empty(tmp_path):
    """Default-off payload behaviour is unchanged."""
    payload, _ = _dump(tmp_path)
    assert np.asarray(payload["candidate_rotations"]).size == 0


def test_candidate_rotations_round_trip_row_aligned_with_the_ids(tmp_path):
    rots = _rotations()
    payload, _ = _dump(tmp_path, candidate_rotations=rots)
    stored = np.asarray(payload["candidate_rotations"])
    assert stored.shape == (B, R, 3, 3)
    assert stored.dtype == np.float32
    assert np.array_equal(stored, rots)
    # Alignment is what makes the ids usable: same rotation axis length.
    assert stored.shape[1] == np.asarray(payload["oversampled_rotation_indices"]).shape[-1]


def test_executed_translations_need_no_new_field(tmp_path):
    """fine_translations already records the grid the engine scored."""
    payload, _ = _dump(tmp_path, candidate_rotations=_rotations())
    assert np.asarray(payload["fine_translations"]).shape == (T, 2)


# --- fail closed --------------------------------------------------------------------

@pytest.mark.parametrize("bad,why", [
    (np.zeros((B, R + 1, 3, 3), dtype=np.float32), "wrong rotation axis"),
    (np.zeros((B + 1, R, 3, 3), dtype=np.float32), "wrong particle axis"),
    (np.zeros((B, R, 3), dtype=np.float32), "not 3x3"),
])
def test_misaligned_candidate_rotations_are_refused(tmp_path, bad, why):
    with pytest.raises(ValueError, match="row-aligned with the candidate axis"):
        _dump(tmp_path, candidate_rotations=bad)


def test_non_finite_candidate_rotations_are_refused(tmp_path):
    rots = _rotations()
    rots[0, 0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="must be finite"):
        _dump(tmp_path, candidate_rotations=rots)


# --- engine call sites --------------------------------------------------------------

def _sites():
    tree = ast.parse(inspect.getsource(local_em_engine))
    return [{kw.arg: kw.value for kw in n.keywords if kw.arg}
            for n in ast.walk(tree)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
            and n.func.attr == "_maybe_dump_exact_local_bpref_contribution_rows"] or [
        {kw.arg: kw.value for kw in n.keywords if kw.arg}
        for n in ast.walk(tree)
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        and n.func.id == "_maybe_dump_exact_local_bpref_contribution_rows"]


def test_both_capture_sites_forward_the_buckets_own_candidate_rotations():
    sites = _sites()
    assert len(sites) == 2, f"expected two capture sites, found {len(sites)}"
    for kwargs in sites:
        assert "candidate_rotations" in kwargs, "a capture site does not record the matrices"
        expr = ast.unparse(kwargs["candidate_rotations"])
        # The bucket's own array, not a table lookup or a regenerated plan.
        assert expr.endswith(".local_rotations"), expr
        assert "plan" not in expr and "sampling" not in expr


def test_capture_sites_take_rotations_and_ids_from_the_same_bucket():
    """local_layout emits ids and matrices together; the capture must not split them."""
    for kwargs in _sites():
        rot_src = ast.unparse(kwargs["candidate_rotations"]).split(".local_rotations")[0]
        id_src = ast.unparse(kwargs["rotation_indices"]).split(".local_rotation_ids")[0]
        assert rot_src == id_src, (rot_src, id_src)
