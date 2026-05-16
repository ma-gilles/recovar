"""Merge guards for the 174b4c09 K-class firstiter coarse-scoring work.

Companion to ``test_em_parity_lowpass_and_tau2_fudge.py`` (which locks
down the e767ec50 LP-filter + tau2_fudge fixes and the completion
baselines). This file is scoped to the K-class scoring path and the
significance-dump operand-recording schema added by 174b4c09 "Speed up
K-class firstiter coarse scoring":

  * ``_compute_k_class_significance_batched`` API additions
    (``relion_projector_half``, ``score_mode``, ``collect_significance``,
    ``return_class_best``) and the inner ``_score_block(class_index, ...)``
    first-argument convention.
  * ``use_fused_pass1`` guard set: fused env, gaussian score, no
    relion projector, no dump targets.
  * ``_maybe_dump_k_class_significance_batch`` operand kwargs
    (``shifted_data``, ``ctf2_data``, ``window_indices``,
    ``half_weights_used``) and the resulting npz schema.
  * ``RECOVAR_PASS1_FUSED`` env-var contract.
  * ``iteration_loop`` plumbing of ``RELION_WIDTH_FMASK_EDGE`` through
    to ``_reconstruct_and_postprocess_means`` (value=2 is asserted by
    the sibling lowpass-and-tau2 guard file; here we only assert it is
    actually threaded through).

Run on CPU in seconds. Their job is to fail loudly if a future EM /
VDAM / PPCA branch merge silently drops a load-bearing K-class kwarg,
swaps the fused-pass1 guard set, or breaks the dump-operand schema.

Quality of the underlying numerics is covered by the integration tests
in ``tests/integration/test_em_parity_fast.py`` and the 3 completion
baselines locked down by the sibling guard file. Don't duplicate
behavioral coverage here — these are structural merge guards.
"""

from __future__ import annotations

import inspect
import os
import re
from types import SimpleNamespace

import numpy as np
import pytest

import recovar.em.dense_single_volume.helpers.significance as sig_mod
import recovar.em.dense_single_volume.iteration_loop as iteration_loop

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# 174b4c09: K-class firstiter coarse-scoring API additions
# ----------------------------------------------------------------------


def test_kclass_significance_batched_keeps_174b4c09_api():
    """``_compute_k_class_significance_batched`` must keep the
    relion-projector / score-mode / collect-significance / return-class-best
    parameters added by 174b4c09. A merge that drops any of these would
    silently disable the K-class firstiter speedup or the parity hooks.
    """
    sig = inspect.signature(sig_mod._compute_k_class_significance_batched)
    required = {
        "relion_projector_half",
        "relion_projector_r_max",
        "score_mode",
        "collect_significance",
        "return_class_best",
    }
    missing = required - set(sig.parameters)
    assert not missing, (
        f"_compute_k_class_significance_batched is missing 174b4c09 params: {sorted(missing)}"
    )


def test_kclass_score_block_takes_class_index_first():
    """The inner ``_score_block`` closure in
    ``_compute_k_class_significance_batched`` must accept ``class_index``
    as its first positional argument (174b4c09). Without it, the
    relion-projector path indexes into the wrong volume.
    """
    source = inspect.getsource(sig_mod._compute_k_class_significance_batched)
    match = re.search(
        r"def _score_block\(\s*([^,)]+)\s*,",
        source,
    )
    assert match is not None, "Could not locate _score_block definition in K-class function"
    first_arg = match.group(1).strip()
    assert first_arg == "class_index", (
        f"_score_block first arg must be 'class_index' (174b4c09), got {first_arg!r}"
    )


def test_kclass_use_fused_pass1_gates_remain_in_place():
    """The K-class ``use_fused_pass1`` gate must keep all four guards:
    fused env, gaussian score_mode, no relion projector, no dump
    targets. Drop any of them and 174b4c09's speedup either fires under
    wrong conditions or silently disables itself.
    """
    source = inspect.getsource(sig_mod._compute_k_class_significance_batched)
    fused_idx = source.find("use_fused_pass1 = (")
    assert fused_idx >= 0, "use_fused_pass1 gate is missing from K-class function"
    # Look at the next ~400 chars for the guard clauses.
    window = source[fused_idx : fused_idx + 400]
    for needle in (
        "_pass1_fused_enabled()",
        'score_mode == "gaussian"',
        "not use_relion_projector",
        "dump_target_pre_prior_blocks_per_class is None",
        "dump_target_with_prior_blocks_per_class is None",
    ):
        assert needle in window, f"K-class use_fused_pass1 lost guard: {needle!r}"


# ----------------------------------------------------------------------
# K-class significance dump operand-recording schema
# ----------------------------------------------------------------------


def test_kclass_dump_helper_accepts_operand_kwargs():
    """``_maybe_dump_k_class_significance_batch`` must accept the
    operand-recording kwargs (``shifted_data``, ``ctf2_data``,
    ``window_indices``, ``half_weights_used``). These let the
    significance-dump diagnostic compare RECOVAR's pass-0 operands
    against RELION's pass-0 ``Fimg`` / ``corr_img`` byte-for-byte.
    Removing them would silently collapse the dump back to the
    pre-instrumentation schema and break the RELION-parity diagnostic.
    """
    sig = inspect.signature(sig_mod._maybe_dump_k_class_significance_batch)
    required = {"shifted_data", "ctf2_data", "window_indices", "half_weights_used"}
    missing = required - set(sig.parameters)
    assert not missing, (
        f"_maybe_dump_k_class_significance_batch is missing operand kwargs: {sorted(missing)}"
    )
    for name in required:
        assert sig.parameters[name].default is None, (
            f"{name} default must stay None so callers without operands still work"
        )


def test_kclass_dump_call_site_passes_operand_kwargs():
    """The K-class dump-emission call site must keep passing the operand
    kwargs. AST-level safety net against merges that strip the kwargs at
    the call site while keeping them on the helper signature.
    """
    source = inspect.getsource(sig_mod._compute_k_class_significance_batched)
    call_idx = source.find("_maybe_dump_k_class_significance_batch(")
    assert call_idx >= 0, "K-class function lost its dump-emission call"
    window = source[call_idx : call_idx + 4000]
    for needle in (
        "shifted_data=shifted_data",
        "ctf2_data=ctf2_data",
        "window_indices=window_indices",
        "half_weights_used=",
    ):
        assert needle in window, f"K-class dump call site lost kwarg: {needle!r}"
    # The half_weights_used branch must distinguish windowed vs
    # unwindowed weights — that's how the dump records what the score
    # actually used.
    assert "half_weights_windowed if use_window else half_weights" in window, (
        "Dump call site lost the windowed/unwindowed half_weights selection"
    )


def test_kclass_dump_writes_operand_arrays_to_npz(monkeypatch, tmp_path):
    """End-to-end behavioral test: a one-particle invocation of
    ``_maybe_dump_k_class_significance_batch`` with operands set writes
    them to the npz with sensible shapes/dtypes.
    """

    n_images = 1
    n_classes = 2
    n_rot = 3
    n_trans = 4
    n_pix = 5

    indices = np.array([0], dtype=np.int64)
    experiment_dataset = SimpleNamespace(
        dataset_indices=np.array([42], dtype=np.int64),
    )

    rotations = np.tile(np.eye(3, dtype=np.float32), (n_rot, 1, 1))
    translations = np.zeros((n_trans, 2), dtype=np.float32)
    class_weight_mats = [
        np.ones((n_images, n_rot * n_trans), dtype=np.float64) / (n_rot * n_trans)
        for _ in range(n_classes)
    ]
    batch_sig_mask = np.ones(
        (n_images, n_classes * n_rot * n_trans), dtype=bool
    )
    batch_n_sig = np.array([n_classes * n_rot * n_trans], dtype=np.int64)
    hard_assignment_batch = np.array([0], dtype=np.int64)
    class_assignment_batch = np.array([0], dtype=np.int64)
    global_log_z = np.array([0.0], dtype=np.float64)
    class_log_z_values = [np.array([-0.69], dtype=np.float64) for _ in range(n_classes)]
    best_score = np.array([0.0], dtype=np.float64)
    max_posterior = np.array([0.5], dtype=np.float64)
    class_log_priors = np.zeros(n_classes, dtype=np.float64)

    shifted_data = np.zeros(
        (n_images * n_trans, n_pix), dtype=np.complex128
    )
    ctf2_data = np.zeros((n_images, n_pix), dtype=np.float64)
    window_indices = np.arange(n_pix, dtype=np.int32)
    half_weights_used = np.ones(n_pix, dtype=np.float64)

    dump_dir = tmp_path / "dump"
    dump_dir.mkdir()
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_DIR", str(dump_dir))
    monkeypatch.setenv("RECOVAR_SIGNIFICANCE_DUMP_ORIGINAL_INDICES", "42")
    sig_mod._maybe_dump_k_class_significance_batch(
        experiment_dataset=experiment_dataset,
        indices=indices,
        n_classes=n_classes,
        rotations=rotations,
        translations=translations,
        class_weight_mats=class_weight_mats,
        batch_sig_mask=batch_sig_mask,
        batch_n_sig=batch_n_sig,
        hard_assignment_batch=hard_assignment_batch,
        class_assignment_batch=class_assignment_batch,
        global_log_z=global_log_z,
        class_log_z_values=class_log_z_values,
        best_score=best_score,
        max_posterior=max_posterior,
        rotation_log_prior_padded=None,
        batch_translation_log_prior=None,
        class_log_priors=class_log_priors,
        current_size=14,
        adaptive_fraction=0.999,
        max_significants=1_000_000,
        shifted_data=shifted_data,
        ctf2_data=ctf2_data,
        window_indices=window_indices,
        half_weights_used=half_weights_used,
    )
    files = sorted(os.listdir(dump_dir))
    assert files, "Dump helper failed to write any npz files"
    payload = np.load(dump_dir / files[0])
    for name in ("shifted_data", "ctf2_data", "window_indices", "half_weights"):
        assert name in payload.files, f"Dump npz is missing schema field {name!r}"
    assert payload["shifted_data"].dtype == np.complex128
    assert payload["ctf2_data"].dtype == np.float64
    assert payload["window_indices"].dtype == np.int32
    assert payload["half_weights"].dtype == np.float64
    assert payload["window_indices"].shape == (n_pix,)
    assert payload["half_weights"].shape == (n_pix,)
    assert int(payload["n_classes"]) == n_classes
    assert int(payload["n_rot"]) == n_rot
    assert int(payload["n_trans"]) == n_trans


# ----------------------------------------------------------------------
# Pass1 fused gate (env-var contract)
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),         # unset → off (174b4c09 ships with default off while validated)
        ("", False),
        ("0", False),
        ("no", False),
        ("false", False),
        ("1", True),
        ("true", True),
        ("TRUE", True),        # case-insensitive
        ("yes", True),
        ("YES", True),
        ("on", True),
        ("On", True),
    ],
)
def test_pass1_fused_enabled_env_var_contract(monkeypatch, value, expected):
    """``RECOVAR_PASS1_FUSED`` is the public opt-in for the fused pass1
    path; the K-class call site at ``use_fused_pass1 = ...`` reads it.
    The string contract is stable: 1/true/yes/on are truthy
    (case-insensitive), anything else is off, unset is off.
    """
    if value is None:
        monkeypatch.delenv("RECOVAR_PASS1_FUSED", raising=False)
    else:
        monkeypatch.setenv("RECOVAR_PASS1_FUSED", value)
    assert sig_mod._pass1_fused_enabled() is expected


# ----------------------------------------------------------------------
# WIDTH_FMASK_EDGE plumbing through iteration_loop
# (Constant-value assertion lives in test_em_parity_lowpass_and_tau2_fudge.py;
#  here we only assert the constant is actually threaded through to the
#  postprocess call. Plumbing is what breaks in merges.)
# ----------------------------------------------------------------------


def test_iteration_loop_threads_fmask_edge_through_to_postprocess():
    """``_run_relion_iteration_loop`` must forward
    ``RELION_WIDTH_FMASK_EDGE`` to ``_reconstruct_and_postprocess_means``
    via the ``relion_fmask_edge`` kwarg. A merge that defines the
    constant but stops threading it leaves the LP filter using the
    real-space mask edge (RELION ``WIDTH_FMASK_EDGE`` vs
    ``--maskedge`` are different units).
    """
    source = inspect.getsource(iteration_loop._run_relion_iteration_loop)
    assert "relion_fmask_edge=RELION_WIDTH_FMASK_EDGE" in source, (
        "iteration_loop must forward RELION_WIDTH_FMASK_EDGE to _reconstruct_and_postprocess_means"
    )
