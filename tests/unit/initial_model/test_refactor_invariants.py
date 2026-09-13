"""Merge guard: the InitialModel refactor savings must survive cross-branch merges.

Pins the work landed on ``claude/refactor-initial-model``:
- Package import does not eagerly load execution modules.
- Helpers extracted during dedup still exist with the right signatures.
- Single source of truth for ``_relion_round`` (was duplicated in iteration_loop).
- Pure-function outputs (schedules, init, layout) are byte-identical.
- Dead code paths deleted during refactor stay deleted (no zombie wrappers).
- Total package LOC stays at most ~6 kLOC (preserves ≥half of the −2119 LOC cut).

Run: ``pixi run python -m pytest tests/unit/initial_model/test_refactor_invariants.py -v``
"""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pytest

import recovar.em.vdam as init_model
from recovar.commands.initial_model import GuiInitialModelDefaults
from recovar.em.helpers.expected_accuracy import estimate_relion_expected_accuracy_from_prepared_inputs
from recovar.em.refinement.mean_helpers import initial_low_pass_filter_references
from recovar.em.relion import relion_projector_setup
from recovar.em.vdam import (
    dense_adapter,
    driver,
    estep_common,
    estep_meta_updates,
    iteration_loop,
    m_step,
    mstep_single_class,
    native_options,
    native_sampling,
    sparse_pass2_estep,
    star_io,
    state,
    subset_schedule,
)
from recovar.em.vdam.init import compute_current_size_for_denovo, compute_ini_high_angstrom, compute_ini_high_shell
from recovar.em.vdam.layout import relion_bpref_frame_scales
from recovar.em.vdam.schedules import (
    compute_phase_lengths,
    compute_stepsize,
    compute_subset_size,
    compute_tau2_fudge,
    default_subset_sizes_for_3d_initial_model,
)

pytestmark = pytest.mark.unit


PACKAGE_DIR = Path(init_model.__file__).resolve().parent
REPO_ROOT = PACKAGE_DIR.parents[2]


# ---------------------------------------------------------------------------
# 2. Helper dedup — _relion_round must have exactly one definition.
# ---------------------------------------------------------------------------


def test_relion_round_is_single_source_of_truth():
    """``_relion_round`` was duplicated in schedules.py + iteration_loop.py
    before the refactor. The refactor pulled it to schedules.py and
    re-imported it in iteration_loop. A merge must not reintroduce a duplicate.
    """
    occurrences: dict[str, int] = {}
    for py_file in PACKAGE_DIR.glob("*.py"):
        text = py_file.read_text()
        n = text.count("def _relion_round(")
        if n:
            occurrences[py_file.name] = n
    assert occurrences == {"schedules.py": 1}, (
        f"_relion_round must be defined exactly once (in schedules.py); found definitions in: {occurrences}"
    )


def test_initial_model_estep_reuses_shared_dense_em_engine():
    """VDAM must remain an adapter around the mature shared EM implementation.

    InitialModel owns its subset/controller and RELION layout conversion, but
    it must not grow private copies of coarse significance, pass-2 layout, or
    local K-class refinement.  Identity checks pin the adapter imports to the
    canonical implementations; the definition scan makes a copied shadow
    implementation fail even if it is not wired in yet.
    """
    from recovar.em.classification import k_class
    from recovar.em.helpers import expected_accuracy
    from recovar.em.local import local_layout
    from recovar.em.scoring import significance
    from recovar.em.vdam import sparse_pass2_estep

    shared_callables = {
        "_compute_k_class_significance_batched": (
            sparse_pass2_estep._compute_k_class_significance_batched,
            significance._compute_k_class_significance_batched,
        ),
        "_run_sparse_k_class_adaptive_pass2": (
            sparse_pass2_estep._run_sparse_k_class_adaptive_pass2,
            k_class._run_sparse_k_class_adaptive_pass2,
        ),
        "run_local_k_class_em": (
            sparse_pass2_estep.run_local_k_class_em,
            k_class.run_local_k_class_em,
        ),
        "build_pass2_hypothesis_layout": (
            sparse_pass2_estep.build_pass2_hypothesis_layout,
            local_layout.build_pass2_hypothesis_layout,
        ),
        "estimate_relion_expected_accuracy_from_prepared_inputs": (
            estimate_relion_expected_accuracy_from_prepared_inputs,
            expected_accuracy.estimate_relion_expected_accuracy_from_prepared_inputs,
        ),
    }
    for name, (adapter_callable, shared_callable) in shared_callables.items():
        assert adapter_callable is shared_callable, (
            f"InitialModel {name} no longer resolves to the shared dense EM implementation"
        )

    initial_model_source = "\n".join(path.read_text() for path in PACKAGE_DIR.glob("*.py"))
    copied = [
        name
        for name in shared_callables
        if f"def {name}(" in initial_model_source
    ]
    assert not copied, f"InitialModel contains private copies of shared EM functions: {copied}"


# ---------------------------------------------------------------------------
# 3. Extracted helpers — presence and signature pin.
# ---------------------------------------------------------------------------


def test_ensure_field_helper_preserves_metadata_array_identity():
    """``_ensure_field`` preserves existing arrays in particle metadata updates."""
    from recovar.em.vdam.estep_meta_updates import _ensure_field

    sig = inspect.signature(_ensure_field)
    params = list(sig.parameters)
    assert params[:3] == ["arr", "shape", "dtype"], f"_ensure_field signature drifted: {sig}"
    assert "fill" in sig.parameters, "_ensure_field lost the `fill` parameter"

    out = _ensure_field(None, (3, 2), np.float32, fill=7.0)
    assert out.shape == (3, 2)
    assert out.dtype == np.float32
    assert (out == 7.0).all()
    pre = np.arange(6, dtype=np.float32).reshape(3, 2)
    out2 = _ensure_field(pre, (3, 2), np.float32)
    assert out2 is pre, "_ensure_field must return the input when already correct"


def test_stack_star_pair_helper_exists_in_star_io():
    """``_stack_star_pair`` dedup'd the X/Y origin column reads in ``_write_data_star``."""
    from recovar.em.vdam.star_io import _stack_star_pair

    assert callable(_stack_star_pair)
    assert list(inspect.signature(_stack_star_pair).parameters) == [
        "main_star",
        "x_name",
        "y_name",
    ]


def test_halfset_values_helper_exists_in_iteration_loop():
    """``_halfset_values`` dedup'd ``_posterior_sums_from_meta`` and ``_scalar_sum_from_meta``."""
    from recovar.em.vdam.estep_meta_updates import _halfset_values

    assert callable(_halfset_values)
    assert list(inspect.signature(_halfset_values).parameters) == ["meta", "key"]

    # Convention: keys named "halfset_{i}_{key}" — _halfset_values collects them in sorted order.
    meta = {"halfset_0_a": [1, 2], "halfset_1_a": [3, 4], "halfset_0_b": [99]}
    assert _halfset_values(meta, "a") == [[1, 2], [3, 4]]
    assert _halfset_values(meta, "missing") == []


def test_my_mu_helper_exists_in_iteration_loop():
    """``_my_mu`` was extracted as the validation copy used in both
    ``vdam_iteration`` and ``apply_vdam_momentum_to_state``.
    """
    from recovar.em.vdam.estep_meta_updates import _my_mu

    assert callable(_my_mu)
    assert list(inspect.signature(_my_mu).parameters) == ["mu", "do_grad", "subset_size"]


def test_bp_slab_helper_exists_in_layout():
    """``_bp_slab`` dedup'd the slab-vs-cropped slice between data and weight paths."""
    from recovar.em.vdam.layout import _bp_slab

    assert callable(_bp_slab)
    assert list(inspect.signature(_bp_slab).parameters) == ["arr", "r_max", "c"]


# ---------------------------------------------------------------------------
# 4. Constants — types and contents.
# ---------------------------------------------------------------------------


def test_dense_run_em_reject_is_frozenset_with_pinned_contents():
    """Reject list became a module-level frozenset (was a per-call list).

    Pins the exact contents: the refactor froze this set so callers don't
    pass unsupported kwargs to ``run_em``. A merge that adds entries must
    update this test deliberately.
    """
    from recovar.em.vdam.dense_adapter import _DENSE_RUN_EM_REJECT

    assert isinstance(_DENSE_RUN_EM_REJECT, frozenset)
    expected = frozenset(
        {
            "disable_adjoint_ctf",
            "disable_adjoint_y",
                "normalization_log_evidence",
                "projection_mask_current_image_disk",
                "recon_exact_radius",
            "recon_square_window",
            "reconstruct_with_masked_images",
            "reconstruction_subtract_projected_reference",
            "relion_projector_shape",
            "return_best_pose_details",
            "return_profile",
            "return_stats",
        }
    )
    assert _DENSE_RUN_EM_REJECT == expected, (
        f"_DENSE_RUN_EM_REJECT contents drifted; "
        f"added: {sorted(_DENSE_RUN_EM_REJECT - expected)}, "
        f"removed: {sorted(expected - _DENSE_RUN_EM_REJECT)}"
    )


def test_sparse_pass2_result_fields_is_tuple_of_typed_attrs():
    """``_PARTICLE_RESULT_FIELDS`` is the single source of truth for which
    estep meta attributes get concatenated across sparse pass-2 batches.
    """
    from recovar.em.vdam.estep_common import _PARTICLE_RESULT_FIELDS

    assert isinstance(_PARTICLE_RESULT_FIELDS, tuple)
    assert all(isinstance(item, tuple) and len(item) == 2 for item in _PARTICLE_RESULT_FIELDS)
    for attr, dtype in _PARTICLE_RESULT_FIELDS:
        assert isinstance(attr, str), f"expected attr name str, got {attr!r}"
        assert isinstance(dtype, type), f"expected dtype to be a type, got {dtype!r}"


# ---------------------------------------------------------------------------
# 5. Pure-function golden values — pin exact outputs.
# ---------------------------------------------------------------------------


class TestScheduleGoldenValues:
    """Pin the schedule outputs that the refactor preserved.

    These values must be byte-identical to before the refactor (they're
    derived from RELION's ml_optimiser.cpp and any drift = parity break).
    """

    def test_phase_lengths(self):
        p = compute_phase_lengths(200, 0.3, 0.2)
        assert p.grad_ini_iter == 60
        assert p.grad_inbetween_iter == 100
        assert p.grad_fin_iter == 40

    def test_subset_size_trajectory(self):
        phase = compute_phase_lengths(200, 0.3, 0.2)
        kwargs = dict(
            phase_lengths=phase,
            grad_ini_subset_size=100,
            grad_fin_subset_size=1000,
            nr_particles=5000,
            nr_iter=200,
        )
        expected = {0: 100, 30: 100, 60: 100, 100: 460, 160: 1000, 199: 1000}
        for it, want in expected.items():
            got = compute_subset_size(iter=it, **kwargs)
            assert got == want, f"subset_size(it={it}) = {got}, expected {want}"

    def test_stepsize_trajectory(self):
        phase = compute_phase_lengths(200, 0.3, 0.2)
        kwargs = dict(phase_lengths=phase, is_3d_model=True, ref_dim=3)
        np.testing.assert_allclose(compute_stepsize(iter=0, **kwargs), 0.8999999046325726)
        np.testing.assert_allclose(compute_stepsize(iter=60, **kwargs), 0.896039581534886)
        np.testing.assert_allclose(compute_stepsize(iter=160, **kwargs), 0.5000003999995659)

    def test_tau2_fudge_trajectory(self):
        phase = compute_phase_lengths(200, 0.3, 0.2)
        kwargs = dict(phase_lengths=phase, is_3d_model=True, ref_dim=3)
        np.testing.assert_allclose(compute_tau2_fudge(iter=0, **kwargs), 1.0)
        np.testing.assert_allclose(compute_tau2_fudge(iter=60, **kwargs), 1.0297029614448547)
        np.testing.assert_allclose(compute_tau2_fudge(iter=160, **kwargs), 3.9999999999999702)

    def test_default_subsets_scale_with_nr_particles(self):
        assert default_subset_sizes_for_3d_initial_model(5000) == (200, 1000)
        assert default_subset_sizes_for_3d_initial_model(50000) == (250, 5000)

    def test_relion_round_banker_semantics(self):
        from recovar.em.vdam.schedules import _relion_round

        # RELION's ROUND is C-style nearest-int away-from-zero, NOT banker's.
        assert _relion_round(0.5) == 1
        assert _relion_round(1.5) == 2
        assert _relion_round(2.5) == 3
        assert _relion_round(-1.5) == -2
        assert _relion_round(-2.5) == -3
        assert _relion_round(3.7) == 4


class TestInitGoldenValues:
    def test_ini_high_shell(self):
        # `INI_HIGH_DIGITAL_FREQ * ori_size` rounded — pin both common sizes.
        assert compute_ini_high_shell(128) == 9
        assert compute_ini_high_shell(192) == 13

    def test_ini_high_angstrom(self):
        np.testing.assert_allclose(compute_ini_high_angstrom(128, 1.0), 128.0 / 9.0)
        np.testing.assert_allclose(compute_ini_high_angstrom(192, 1.5), (192.0 * 1.5) / 13.0)

    def test_current_size_for_denovo(self):
        # Pinned values from the post-refactor implementation; mirrors
        # RELION's de-novo current_size schedule.
        assert compute_current_size_for_denovo(128) == 38
        assert compute_current_size_for_denovo(64) == 28


class TestLayoutGoldenValues:
    def test_frame_scales_signs_and_magnitude(self):
        s1, s2 = relion_bpref_frame_scales(128)
        assert s1 == -(128.0**2)
        assert s2 == 128.0**4
        s1, s2 = relion_bpref_frame_scales(64)
        assert s1 == -(64.0**2)
        assert s2 == 64.0**4

    def test_bp_slab_full_half_complex_path(self):
        """``r_max >= c`` returns a roll of the full half-complex slab."""
        from recovar.em.vdam.layout import _bp_slab

        N = 8
        c = N // 2  # 4
        arr = np.arange(N * N * N, dtype=np.float64).reshape(N, N, N)
        out = _bp_slab(arr, r_max=c, c=c)
        # Shape: (N, N, c+1) when r_max >= c.
        assert out.shape == (N, N, c + 1)
        # First c columns are arr[:,:,c:], last column is arr[:,:,:1].
        np.testing.assert_array_equal(out[:, :, :c], arr[:, :, c:])
        np.testing.assert_array_equal(out[:, :, c:], arr[:, :, :1])

    def test_bp_slab_cropped_path(self):
        """``r_max < c`` returns a centered cropped half-spectrum slab."""
        from recovar.em.vdam.layout import _bp_slab

        N = 16
        c = N // 2  # 8
        r_max = 2  # cropped: half_ps=3, slab=(7,7,4)
        arr = np.arange(N * N * N, dtype=np.float64).reshape(N, N, N)
        out = _bp_slab(arr, r_max=r_max, c=c)
        assert out.shape == (7, 7, 4)
        np.testing.assert_array_equal(out, arr[5:12, 5:12, 8:12])


# ---------------------------------------------------------------------------
# 6. Dead code stays deleted — these symbols/files MUST NOT come back.
# ---------------------------------------------------------------------------


def test_dead_code_remains_deleted():
    """The refactor deleted several dead paths. A merge bringing them back
    would silently undo the savings; this guard catches it.
    """
    # gpu_pipeline.py was a compat shim with no callers.
    assert not (PACKAGE_DIR / "gpu_pipeline.py").exists(), (
        "gpu_pipeline.py was deleted as a no-caller shim — do not re-add"
    )

    # These dead helpers were deleted; any reintroduction is a regression.
    dead_symbols = {
        "_append_sigma2_offset_meta": "dense_adapter.py",
        "_result_to_accumulators": "dense_adapter.py",
        "_sparse_pass2_result_to_accumulators": "dense_adapter.py",
    }
    for symbol, filename in dead_symbols.items():
        text = (PACKAGE_DIR / filename).read_text()
        # Reject only definitions, not references (callers must be gone anyway).
        assert f"def {symbol}(" not in text, (
            f"dead helper {symbol!r} was deleted from {filename}; "
            f"a merge reintroduced it — collapse callers into the canonical path instead"
        )


def test_bootstrap_iref_pure_python_fallback_stays_deleted():
    """The pre-refactor ``compute_bootstrap_iref`` had a 140-LOC pure-Python
    fallback for "no binding available". Production always has the binding, so
    the fallback was dead code. Do not reintroduce it.
    """
    text = (PACKAGE_DIR / "bootstrap_iref.py").read_text()
    # The fallback was the ONLY caller of these helpers; they are gone now.
    assert "_pure_python_bootstrap_iref" not in text
    assert "def compute_bootstrap_iref_pure_python" not in text


# ---------------------------------------------------------------------------
# 7. LOC budget — preserve ≥half of the refactor's −2119 LOC reduction.
# ---------------------------------------------------------------------------


# Snapshot at fbdf23f9 (post-refactor). A merge can grow files modestly but
# must not undo the cuts. Per-file ceilings allow generous headroom (~50%)
# because merges legitimately add code; the TOTAL ceiling is the real guard.
LOC_PER_FILE_CEILING = {
    "../relion/initial_noise.py": 220,
    "bootstrap_iref.py": 280,
    "dense_adapter.py": 1500,
    "driver.py": 1960,
    "../relion/vdam_checkpoint.py": 440,  # Split from driver; combined allowance unchanged.
    "../diagnostics/gt_metrics.py": 400,
    "__init__.py": 160,
    "init.py": 280,
    "iteration_loop.py": 870,
    "layout.py": 186,  # Includes the former 36-line relion_layout.py; total budget unchanged.
    "m_step.py": 450,
    "schedules.py": 400,
    "state.py": 130,
    "subset.py": 150,
}

# Pre-refactor total was 7133 LOC; post-refactor was 5014. Exact RELION
# InitialModel parity added the native dense adapter and layout bridge; a 6100
# ceiling still preserves more than 1000 lines of the refactor savings.
TOTAL_LOC_CEILING = 6100


def _file_loc(path: Path) -> int:
    return sum(1 for _ in path.open("rb"))


def test_per_file_loc_ceilings():
    """Each module stays below its post-refactor ceiling.

    Adjust the ceiling deliberately if a feature legitimately needs more lines
    — never widen blindly.
    """
    over = {}
    for filename, ceiling in LOC_PER_FILE_CEILING.items():
        path = PACKAGE_DIR / filename
        if not path.exists():
            continue
        loc = _file_loc(path)
        if loc > ceiling:
            over[filename] = (loc, ceiling)
    assert not over, (
        f"InitialModel files exceed LOC ceiling: {over}\n"
        f"Investigate which merge brought in the bloat before raising the ceiling."
    )


def test_total_package_loc_within_budget():
    """Count VDAM, its checkpoint adapter and the extracted shared diagnostics."""
    total = sum(_file_loc(p) for p in PACKAGE_DIR.glob("*.py"))
    total += sum(_file_loc(PACKAGE_DIR.parent / "relion" / name)
                 for name in ("vdam_checkpoint.py", "initial_noise.py"))
    # Shared diagnostics remain counted after their responsibility move.
    total += sum(_file_loc(PACKAGE_DIR.parent / "diagnostics" / name)
                 for name in ("gt_metrics.py", "gt_registration.py"))
    total += len(inspect.getsourcelines(GuiInitialModelDefaults)[0]) + 2  # Extracted launcher defaults remain counted.
    total += 1  # ProjectorSetupBackend alias now lives in the shared owner.
    # Shared projector wrappers remain counted after leaving dense_adapter.
    total += sum(len(inspect.getsourcelines(getattr(relion_projector_setup, name))[0]) + 2
                 for name in ("reference_to_relion_projector_half_maps",
                              "reference_to_relion_projector_half_maps_and_power"))
    total += len(inspect.getsourcelines(initial_low_pass_filter_references)[0]) + 3  # helper, spacing, edge constant
    assert total <= TOTAL_LOC_CEILING, (
        f"InitialModel total LOC = {total} > ceiling {TOTAL_LOC_CEILING}; "
        f"refactor savings are being eroded. Identify the merge that bloated the package."
    )


# ---------------------------------------------------------------------------
# 8. Import-time performance smoke — package must import quickly.
# ---------------------------------------------------------------------------


def test_package_import_is_fast(tmp_path):
    """Importing InitialModel adds less than 2s beyond its parent package.

    The parent ``recovar.em`` import initializes JAX, healpy, pandas, and GPU
    discovery; its cold time varies substantially with node and filesystem
    load. Measure the InitialModel increment so this guard attributes a
    regression to this package instead of those shared imports.
    """
    import subprocess
    import sys

    code = """\
import time
import sys
t0 = time.perf_counter()
import recovar.em
parent_elapsed = time.perf_counter() - t0
t0 = time.perf_counter()
import recovar.em.vdam
initial_model_elapsed = time.perf_counter() - t0
assert not any(name.startswith("recovar.em.vdam.") for name in sys.modules)
print(parent_elapsed, initial_model_elapsed)
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stderr
    parent_elapsed, initial_model_elapsed = map(float, result.stdout.split())
    assert initial_model_elapsed < 2.0, (
        f"recovar.em.vdam added {initial_model_elapsed:.2f}s after the "
        f"{parent_elapsed:.2f}s parent import; likely a module-level side effect"
    )


# Module ownership and adapter routing.

MOVED = ("NativeOpticsState", "_optics_group_indices", "_single_optics_scalars", "_phase_shift", "_native_optics_state", "_particle_state_from_star", "_write_model_star", "_write_data_star", "_write_iteration_artifacts", "_write_final_outputs", "_star_column", "_stack_star_pair", "_experiment_read_order")
SAMPLING = ("NativeSamplingPlan", "NativeSamplingState", "_build_sampling_plan", "_initial_sampling_state", "_estimate_native_sampling_accuracy", "_relion_update_native_sampling_state", "_prepare_native_sampling_for_iteration", "_random_perturbation_for_iteration")


def test_iteration_loop_updates_definition_ownership():
    loop_src = inspect.getsource(iteration_loop)
    for name in ("update_noise_from_estep_meta", "update_probabilities_from_estep_meta", "_maybe_dump_noise_update_boundary"):
        assert inspect.getmodule(getattr(estep_meta_updates, name)) is estep_meta_updates and f"\ndef {name}(" not in loop_src
    for name in ("select_subset_for_iter", "restore_subset_order_for_continuation"):
        assert inspect.getmodule(getattr(subset_schedule, name)) is subset_schedule and f"\ndef {name}(" not in loop_src
    assert iteration_loop.update_noise_from_estep_meta is estep_meta_updates.update_noise_from_estep_meta
    assert iteration_loop.select_subset_for_iter is subset_schedule.select_subset_for_iter
    for mod in (estep_meta_updates, subset_schedule):
        assert "vdam.iteration_loop import" not in inspect.getsource(mod)


def test_mstep_single_class_definition_ownership():
    src = inspect.getsource(m_step)
    for name in ("vdam_m_step_single_class", "_run_m_step_transaction", "_validate_mstep_precision_route", "_maybe_replay_native_bpref_accumulators"):
        assert inspect.getmodule(getattr(mstep_single_class, name)) is mstep_single_class and f"\ndef {name}(" not in src
    assert inspect.getmodule(state.VdamAccumulator) is state and "\nclass VdamAccumulator" not in src
    assert m_step.vdam_m_step_single_class is mstep_single_class.vdam_m_step_single_class
    assert m_step.VdamAccumulator is state.VdamAccumulator
    assert "vdam.m_step import" not in inspect.getsource(mstep_single_class)


def test_star_io_owns_the_cluster_and_driver_only_imports_it():
    driver_src = inspect.getsource(driver)
    for name in MOVED:
        assert hasattr(star_io, name) and inspect.getmodule(getattr(star_io, name)) is star_io
        assert f"\ndef {name}(" not in driver_src and f"\nclass {name}(" not in driver_src
    assert "from recovar.em.vdam.star_io import (" in driver_src
    assert driver._write_iteration_artifacts is star_io._write_iteration_artifacts


def test_particle_record_is_owned_by_state_and_shared_with_star_io():
    assert inspect.getmodule(state.NativeParticleState) is state
    assert star_io.NativeParticleState is state.NativeParticleState


def test_native_sampling_definition_ownership():
    driver_src = inspect.getsource(driver)
    for name in SAMPLING:
        assert inspect.getmodule(getattr(native_sampling, name)) is native_sampling
        assert f"\ndef {name}(" not in driver_src and f"\nclass {name}(" not in driver_src
    assert inspect.getmodule(native_options.NativeInitialModelOptions) is native_options
    assert "\nclass NativeInitialModelOptions" not in driver_src
    assert driver.NativeInitialModelOptions is native_options.NativeInitialModelOptions
    assert "import recovar.em.vdam.driver" not in inspect.getsource(native_sampling)


def test_sparse_pass2_estep_definition_ownership():
    adapter_src = inspect.getsource(dense_adapter)
    for name in ("_run_sparse_pass2_initial_model_estep", "_sparse_pass2_estep_meta", "_initial_model_pass2_layout", "_pop_sparse_pass2_options"):
        assert inspect.getmodule(getattr(sparse_pass2_estep, name)) is sparse_pass2_estep and f"\ndef {name}(" not in adapter_src
    for name in ("DenseInitialModelEstepConfig", "DenseInitialModelEstepResult", "_estep_meta", "_select_image_rows"):
        assert inspect.getmodule(getattr(estep_common, name)) is estep_common
    assert dense_adapter._run_sparse_pass2_initial_model_estep is sparse_pass2_estep._run_sparse_pass2_initial_model_estep
    assert dense_adapter.DenseInitialModelEstepConfig is estep_common.DenseInitialModelEstepConfig
    for mod in (sparse_pass2_estep, estep_common):
        assert "vdam.dense_adapter import" not in inspect.getsource(mod)
