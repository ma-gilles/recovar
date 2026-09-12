#!/usr/bin/env python3
# ruff: noqa: E402, I001
"""Sealed correctness gate for the default-off fixed-capacity local-call seam.

This gate intentionally makes no runtime or default-promotion claim.  The
current seam substitutes byte-validated host operands immediately before the
same shared numeric wrapper used by the mature path.  Consequently, the
current contract is exact equality.  Paired float32/float64 numerical
envelopes are recorded for a future, separately speed-qualified implementation
whose reduction order may legitimately change.
"""

from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import subprocess
import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

import recovar.core.fourier_transform_utils as ftu
from recovar.em.dense_single_volume import local_big_jit, local_em_engine
from recovar.em.dense_single_volume.batch_planning import (
    _plan_fixed_capacity_whole_local,
    _seal_fixed_capacity_physical_order,
)
from recovar.em.dense_single_volume.fixed_capacity_local import (
    _bind_fixed_capacity_local_execution,
)
from recovar.em.dense_single_volume.local_caches import (
    _assemble_fixed_capacity_local_operands_once,
)
from recovar.em.dense_single_volume.local_layout import (
    LocalHypothesisLayout,
    _fixed_capacity_calls_from_local_buckets,
    _pack_fixed_capacity_local_hypothesis_program,
    bucket_local_hypothesis_layout,
)
from recovar.em.dense_single_volume import local_bucket_stages


SCHEMA = "recovar.fixed_capacity_local_score_gate.v7"
IMAGE_SHAPE = (8, 8)
VOLUME_SHAPE = (8, 8, 8)
IMAGE_SIZE = int(np.prod(IMAGE_SHAPE))
VOLUME_SIZE = int(np.prod(VOLUME_SHAPE))

# These are dormant acceptance envelopes for a future implementation that
# changes only floating-point reduction order and separately demonstrates a
# material production speedup.  The current shared-wrapper seam is stricter:
# every delta must be exactly zero.  The float64 companion is at least 10^3
# tighter for every floating metric, as required by the test policy.
FUTURE_SPEED_QUALIFIED_ENVELOPES = {
    "float32": {
        "score_max_abs": 2.0e-4,
        "score_centered_max_abs": 2.0e-5,
        "log_z_max_abs": 2.0e-4,
        "best_score_max_abs": 2.0e-4,
        "posterior_max_abs": 2.0e-5,
        "posterior_mass_max_abs": 2.0e-5,
    },
    "float64": {
        "score_max_abs": 2.0e-8,
        "score_centered_max_abs": 2.0e-9,
        "log_z_max_abs": 2.0e-8,
        "best_score_max_abs": 2.0e-8,
        "posterior_max_abs": 2.0e-9,
        "posterior_mass_max_abs": 2.0e-9,
    },
}

_BASE_RESULT_NAMES = (
    "Ft_y",
    "Ft_ctf",
    "noise_wsum",
    "noise_img_power",
    "noise_a2",
    "noise_xa",
    "noise_scale_xa",
    "noise_scale_aa",
    "bucket_norm_correction",
    "noise_sigma2_offset",
    "noise_sumw",
    "batch_norm",
    "log_z",
    "best_log_score",
    "best_argmax",
    "max_posterior",
    "probs_sum_t",
    "reconstruction_probs_sum_t",
    "n_significant_samples",
    "reconstruction_sample_mask",
    "reconstruction_rotation_mask",
    "reconstruction_row_count",
)

_EXACT_DIAGNOSTIC_FIELDS = (
    "best_argmax",
    "n_significant_samples",
    "reconstruction_sample_mask",
    "reconstruction_rotation_mask",
    "reconstruction_row_count",
    "candidate_mask",
    "finite_score_mask",
    "posterior_support",
)

_CONTINUOUS_DIAGNOSTIC_FIELDS = (
    "log_z",
    "best_log_score",
    "max_posterior",
    "probs_sum_t",
    "reconstruction_probs_sum_t",
    "debug_scores",
    "debug_probs",
)

# Keep this synchronized with ``run_local_bucket_big_jit``'s
# ``donate_argnums``.  Every gate invocation supplies fresh accumulators, and
# the focused source contract verifies that the donated positions still name
# the two loop-carried outputs described by the implementation comment.
CURRENT_DONATED_POSITIONAL_NAMES = (
    "Ft_y",
    "Ft_ctf",
)


def _identity_ctf(params, image_shape=None, voxel_size=None, *, half_image=False):
    del voxel_size
    shape = IMAGE_SHAPE if image_shape is None else tuple(image_shape)
    size = shape[0] * (shape[1] // 2 + 1) if half_image else int(np.prod(shape))
    return jnp.ones((params.shape[0], size), dtype=jnp.float32)


def _raw_real_process(batch, apply_image_mask=False):
    if apply_image_mask:
        raise AssertionError("the sealed score fixture has no image mask")
    images = jnp.asarray(batch)
    return ftu.get_dft2(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


def _raw_real_process_half(batch, apply_image_mask=False):
    if apply_image_mask:
        raise AssertionError("the sealed score fixture has no image mask")
    images = jnp.asarray(batch)
    return ftu.get_dft2_real(images).reshape((images.shape[0], -1)).astype(jnp.complex64)


class _RawRealDataset:
    """Deterministic, indexed, raw-real dataset used by both gate arms."""

    def __init__(self) -> None:
        rng = np.random.default_rng(20260831)
        self.image_shape = IMAGE_SHAPE
        self.image_size = IMAGE_SIZE
        self.grid_size = IMAGE_SHAPE[0]
        self.padding = 0
        self.volume_shape = VOLUME_SHAPE
        self.volume_size = VOLUME_SIZE
        self.n_images = 3
        self.n_units = 3
        self.voxel_size = 1.0
        self.dtype = np.float32
        self.CTF_params = np.linspace(-0.7, 0.9, 27, dtype=np.float32).reshape(3, 9)
        self.ctf_evaluator = staticmethod(_identity_ctf)
        self.process_images = staticmethod(_raw_real_process)
        self.process_images_half = staticmethod(_raw_real_process_half)
        self.premultiplied_ctf = False
        self._images = rng.standard_normal((3, *IMAGE_SHAPE)).astype(np.float32)
        self.rotation_matrices = np.broadcast_to(
            np.eye(3, dtype=np.float32),
            (3, 3, 3),
        ).copy()
        self.translations = np.zeros((3, 2), dtype=np.float32)

        class _Backend:
            image_mask = None
            image_mask_mode = "multiply"

        class _ImageSource:
            process_images = staticmethod(_raw_real_process)
            process_images_half = staticmethod(_raw_real_process_half)
            backend = _Backend()

        self.image_source = _ImageSource()

    @property
    def image_mask(self):
        return None

    @property
    def data_multiplier(self):
        return 1.0

    def iter_batches(self, batch_size, *, indices=None, by_image=False, **kwargs):
        del by_image, kwargs
        if indices is None:
            indices = np.arange(self.n_images, dtype=np.int32)
        indices = np.asarray(indices, dtype=np.int64)
        for start in range(0, len(indices), max(1, int(batch_size))):
            selected = np.asarray(indices[start : start + max(1, int(batch_size))])
            yield (
                jnp.asarray(self._images[selected]),
                self.rotation_matrices[selected],
                self.translations[selected],
                jnp.asarray(self.CTF_params[selected]),
                None,
                selected,
                selected,
            )

    def original_image_indices_from_local(self, indices):
        return np.asarray(indices, dtype=np.int64)


def _make_rotations(n_rotations: int) -> np.ndarray:
    rng = np.random.default_rng(20260829)
    matrices = rng.standard_normal((n_rotations, 3, 3))
    q, r = np.linalg.qr(matrices)
    q = q * np.sign(np.diagonal(r, axis1=1, axis2=2))[:, None, :]
    q[np.linalg.det(q) < 0] *= -1
    return q.astype(np.float32)


def _hermitian_volume() -> np.ndarray:
    rng = np.random.default_rng(20260830)
    real_volume = rng.standard_normal(VOLUME_SHAPE).astype(np.float32)
    fourier_volume = np.fft.fftshift(np.fft.fftn(real_volume)).ravel()
    return np.asarray(0.03 * fourier_volume, dtype=np.complex64)


@dataclass(frozen=True)
class _GateFixture:
    dataset: _RawRealDataset
    mean: np.ndarray
    mean_variance: np.ndarray
    noise_variance: np.ndarray
    local_layout: LocalHypothesisLayout
    image_pre_shifts: np.ndarray
    fixed_bundle: Any
    bucket_image_order: np.ndarray
    bucket_radix: int
    bucket_image_capacity: int


def build_gate_fixture() -> _GateFixture:
    dataset = _RawRealDataset()
    rotations = _make_rotations(5)
    rotation_ids = (
        np.asarray([0, 1, 2], dtype=np.int32),
        np.asarray([1, 3], dtype=np.int32),
        np.asarray([0, 2, 4], dtype=np.int32),
    )
    rotation_counts = np.asarray([len(ids) for ids in rotation_ids], dtype=np.int32)
    rotation_offsets = np.concatenate(([0], np.cumsum(rotation_counts))).astype(np.int64)
    rotation_ids_flat = np.concatenate(rotation_ids)
    local_layout = LocalHypothesisLayout(
        n_global_rotations=5,
        n_pixels=6,
        n_psi=1,
        rotation_offsets=rotation_offsets,
        rotation_ids_flat=rotation_ids_flat,
        rotations_flat=np.asarray(rotations[rotation_ids_flat], dtype=np.float32),
        rotation_log_priors_flat=np.asarray(
            [0.0, -0.17, -0.83, -0.11, -0.62, -0.07, -0.41, -1.03],
            dtype=np.float32,
        ),
        rotation_counts=rotation_counts,
        translation_grid=np.asarray([[0.0, 0.0], [0.5, -0.5]], dtype=np.float32),
        translation_log_priors=np.asarray(
            [[-0.03, -0.71], [-0.29, -0.09], [-0.13, -0.57]],
            dtype=np.float32,
        ),
    )
    buckets = bucket_local_hypothesis_layout(
        local_layout,
        image_batch_size=2,
        rotation_block_size=8,
        max_hypotheses_per_microbatch=64,
        exact_local_bucket_radix=2,
    )
    if len(buckets) != 2:
        raise RuntimeError(f"sealed score fixture must produce exactly two calls, got {len(buckets)}")
    bucket_image_order = np.concatenate(
        [np.asarray(bucket.image_indices, dtype=np.int32) for bucket in buckets]
    )
    sealed_order = _seal_fixed_capacity_physical_order(bucket_image_order)
    calls = _fixed_capacity_calls_from_local_buckets(buckets, expected_order=sealed_order)
    physical_image_capacity = 4
    physical_row_capacity = max(
        16,
        int(
            sum(
                int(value)
                for bucket in buckets
                for value in bucket.actual_rotation_counts
            )
        ),
    )
    image_capacity_palette: dict[int, tuple[int, ...]] = {}
    for radix in sorted({int(bucket.bucket_rotation_count) for bucket in buckets}):
        image_capacity_palette[radix] = tuple(
            sorted(
                {
                    int(bucket.bucket_image_count)
                    for bucket in buckets
                    if int(bucket.bucket_rotation_count) == radix
                }
            )
        )
    plan = _plan_fixed_capacity_whole_local(
        calls,
        expected_image_order=sealed_order,
        physical_image_capacity=physical_image_capacity,
        physical_row_capacity=physical_row_capacity,
        physical_call_capacity=3,
        image_capacity_palette=image_capacity_palette,
        logical_cutoff=6,
        logical_cutoff_capacity=16,
        enabled=True,
    )
    image_pre_shifts = np.asarray(
        [[0.25, -0.5], [-0.75, 0.375], [0.125, 0.625]],
        dtype=np.float32,
    )
    operands = _assemble_fixed_capacity_local_operands_once(
        dataset,
        plan,
        sealed_order,
        metadata_by_image={"image_pre_shifts": image_pre_shifts},
        tail_fill_value=np.float32(-777.0),
        enabled=True,
    )
    hypotheses = _pack_fixed_capacity_local_hypothesis_program(
        buckets,
        plan,
        sealed_order,
        enabled=True,
    )
    fixed_bundle = _bind_fixed_capacity_local_execution(
        plan,
        operands,
        hypotheses,
        enabled=True,
    )
    return _GateFixture(
        dataset=dataset,
        mean=_hermitian_volume(),
        mean_variance=np.linspace(7.0, 13.0, VOLUME_SIZE, dtype=np.float32),
        noise_variance=np.linspace(8.0, 12.0, IMAGE_SIZE, dtype=np.float32),
        local_layout=local_layout,
        image_pre_shifts=image_pre_shifts,
        fixed_bundle=fixed_bundle,
        bucket_image_order=bucket_image_order,
        bucket_radix=max(int(bucket.bucket_rotation_count) for bucket in buckets),
        bucket_image_capacity=max(int(bucket.bucket_image_count) for bucket in buckets),
    )


def _to_host_array(value) -> np.ndarray:
    return np.array(jax.device_get(value), copy=True, order="C")


def _array_digest(value: np.ndarray) -> str:
    value = np.ascontiguousarray(value)
    digest = hashlib.sha256()
    digest.update(value.dtype.str.encode("ascii") + b"\0")
    digest.update(np.asarray(value.shape, dtype="<i8").tobytes())
    digest.update(value.tobytes(order="C"))
    return digest.hexdigest()


@dataclass(frozen=True)
class _CapturedCall:
    diagnostics: dict[str, np.ndarray]
    inputs: dict[str, np.ndarray | None]
    static_arguments: dict[str, Any]
    donated_input_object_ids: dict[str, int]
    prepared_call: Any
    initial_carry: tuple[np.ndarray, ...]
    replay_static_arguments: dict[str, Any]


@contextmanager
def _capture_shared_numeric_call(
    donated_input_objects: list[object] | None = None,
) -> Iterator[list[_CapturedCall]]:
    original_wrapper = local_em_engine._invoke_local_bucket_big_jit
    shared_numeric = local_bucket_stages.run_local_bucket_big_jit
    signature = inspect.signature(shared_numeric)
    captures: list[_CapturedCall] = []
    if donated_input_objects is None:
        donated_input_objects = []

    def capture_wrapper(*args, **kwargs):
        bound = signature.bind(*args, **kwargs)
        bound.apply_defaults()
        if not bool(bound.arguments["score_only"]):
            raise RuntimeError("score gate intercepted a non-score-only call")
        if bool(bound.arguments["return_debug_arrays"]):
            raise RuntimeError("score gate requires production diagnostics to be disabled")
        if bool(bound.arguments["return_debug_scores"]):
            raise RuntimeError("score gate requires production score dumps to be disabled")
        if bool(bound.arguments["return_debug_operands"]):
            raise RuntimeError("score gate requires production operand dumps to be disabled")

        input_values: dict[str, np.ndarray | None] = {}
        for name, parameter in signature.parameters.items():
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                continue
            if name == "config":
                continue
            value = bound.arguments[name]
            input_values[name] = None if value is None else _to_host_array(value)
        static_arguments = {
            name: bound.arguments[name]
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
            and name not in {"return_debug_arrays", "return_debug_scores", "return_debug_operands"}
        }
        static_arguments["config_repr"] = repr(bound.arguments["config"])
        replay_positional = []
        for name, parameter in signature.parameters.items():
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY:
                continue
            value = bound.arguments[name]
            if name == "config" or value is None:
                replay_positional.append(value)
            else:
                replay_positional.append(_to_host_array(value))
        prepared_call = local_big_jit._prepare_fixed_capacity_local_call(
            *replay_positional
        )
        initial_carry = tuple(
            _to_host_array(bound.arguments[name])
            for name in (
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
            )
        )
        donated_input_object_ids = {}
        for name in CURRENT_DONATED_POSITIONAL_NAMES:
            value = bound.arguments[name]
            if any(value is prior for prior in donated_input_objects):
                raise RuntimeError(f"score gate reused donated input object {name}")
            donated_input_objects.append(value)
            donated_input_object_ids[name] = id(value)

        diagnostic_kwargs = dict(kwargs)
        diagnostic_kwargs["return_debug_arrays"] = True
        diagnostic_kwargs["return_debug_scores"] = True
        diagnostic_kwargs["return_debug_operands"] = False
        replay_static_arguments = {
            name: bound.arguments[name]
            for name, parameter in signature.parameters.items()
            if parameter.kind is inspect.Parameter.KEYWORD_ONLY
        }
        replay_static_arguments.update(
            return_debug_arrays=True,
            return_debug_scores=True,
            return_debug_operands=False,
        )
        full_result = shared_numeric(*args, **diagnostic_kwargs)
        host_result = tuple(_to_host_array(value) for value in full_result)
        if len(host_result) != len(_BASE_RESULT_NAMES) + 2:
            raise RuntimeError(
                "shared score primitive returned an unexpected diagnostic topology: "
                f"{len(host_result)} values",
            )
        diagnostics = dict(zip(_BASE_RESULT_NAMES, host_result[: len(_BASE_RESULT_NAMES)], strict=True))
        diagnostics["debug_scores"] = host_result[-2]
        diagnostics["debug_probs"] = host_result[-1]
        rotation_mask = np.asarray(input_values["rotation_mask"], dtype=bool)
        valid_image_mask = np.asarray(input_values["valid_image_mask"], dtype=bool)
        sample_mask = input_values["sample_mask"]
        candidate_mask = np.broadcast_to(
            valid_image_mask[:, None, None] & rotation_mask[:, :, None],
            diagnostics["debug_scores"].shape,
        ).copy()
        if sample_mask is not None:
            candidate_mask &= np.asarray(sample_mask, dtype=bool)
        diagnostics["candidate_mask"] = candidate_mask
        diagnostics["finite_score_mask"] = np.isfinite(diagnostics["debug_scores"])
        diagnostics["posterior_support"] = diagnostics["debug_probs"] > 0
        captures.append(
            _CapturedCall(
                diagnostics=diagnostics,
                inputs=input_values,
                static_arguments=static_arguments,
                donated_input_object_ids=donated_input_object_ids,
                prepared_call=prepared_call,
                initial_carry=initial_carry,
                replay_static_arguments=replay_static_arguments,
            )
        )
        # The outer engine entered the non-diagnostic production topology and
        # must see precisely that topology even though the gate captured the
        # two diagnostic arrays from the same numeric invocation.
        return full_result[:-2]

    local_em_engine._invoke_local_bucket_big_jit = capture_wrapper
    try:
        yield captures
    finally:
        local_em_engine._invoke_local_bucket_big_jit = original_wrapper


def _run_kwargs(fixture: _GateFixture, precision: str) -> dict[str, Any]:
    if precision not in {"float32", "float64"}:
        raise ValueError(f"unknown precision lane: {precision}")
    use_float64 = precision == "float64"
    return {
        "image_batch_size": 2,
        "rotation_block_size": 8,
        "current_size": 6,
        "accumulate_noise": False,
        "score_with_masked_images": False,
        "half_spectrum_scoring": False,
        "use_float64_scoring": use_float64,
        "use_float64_normalization": True,
        "use_float64_projections": use_float64,
        "image_pre_shifts": fixture.image_pre_shifts,
        "max_hypotheses_per_microbatch": 64,
        "reconstruct_significant_only": True,
        "adaptive_fraction": 0.9,
        "max_significants": -1,
        "unify_local_bucket_sizes": False,
        "exact_local_bucket_radix": 2,
        "disable_adjoint_y": True,
        "disable_adjoint_ctf": True,
        "score_only": True,
    }


def _run_outer(
    fixture: _GateFixture,
    precision: str,
    arm: str,
) -> tuple[Any, ...]:
    kwargs = _run_kwargs(fixture, precision)
    if arm == "default":
        pass
    elif arm == "disabled":
        kwargs.update(
            _fixed_capacity_bundle=fixture.fixed_bundle,
            _fixed_capacity_enabled=False,
            _fixed_capacity_class_count=1,
        )
    elif arm == "fixed":
        kwargs.update(
            _fixed_capacity_bundle=fixture.fixed_bundle,
            _fixed_capacity_enabled=True,
            _fixed_capacity_class_count=1,
        )
    elif arm == "whole":
        kwargs.update(
            _fixed_capacity_bundle=fixture.fixed_bundle,
            _fixed_capacity_enabled=True,
            _fixed_capacity_class_count=1,
            _fixed_capacity_whole_boundary_enabled=True,
        )
    else:
        raise ValueError(f"unknown score-gate arm: {arm}")
    mean_dtype = np.complex128 if precision == "float64" else np.complex64
    scalar_dtype = np.float64 if precision == "float64" else np.float32
    return local_em_engine.run_local_em_exact(
        fixture.dataset,
        jnp.asarray(fixture.mean, dtype=mean_dtype),
        jnp.asarray(fixture.mean_variance, dtype=scalar_dtype),
        jnp.asarray(fixture.noise_variance, dtype=scalar_dtype),
        fixture.local_layout,
        "linear_interp",
        **kwargs,
    )


def _outer_snapshot(result: tuple[Any, ...]) -> dict[str, np.ndarray]:
    if len(result) != 4:
        raise RuntimeError(f"score-only outer result changed topology: {len(result)}")
    ft_y, ft_ctf, hard_assignment, stats = result
    return {
        "Ft_y": _to_host_array(ft_y),
        "Ft_ctf": _to_host_array(ft_ctf),
        "hard_assignment": _to_host_array(hard_assignment),
        "log_evidence_per_image": _to_host_array(stats.log_evidence_per_image),
        "best_log_score_per_image": _to_host_array(stats.best_log_score_per_image),
        "max_posterior_per_image": _to_host_array(stats.max_posterior_per_image),
        "rotation_posterior_sums": _to_host_array(stats.rotation_posterior_sums),
    }


def _run_captured_arm(
    fixture: _GateFixture,
    precision: str,
    arm: str,
    *,
    donated_input_objects: list[object] | None = None,
) -> tuple[tuple[_CapturedCall, ...], dict[str, np.ndarray]]:
    with _capture_shared_numeric_call(donated_input_objects) as captures:
        result = _run_outer(fixture, precision, arm)
        outer = _outer_snapshot(result)
    expected_call_count = int(fixture.fixed_bundle.plan.valid_call_count)
    if len(captures) != expected_call_count:
        raise RuntimeError(
            f"{arm}/{precision} must invoke exactly {expected_call_count} shared score calls, "
            f"got {len(captures)}"
        )
    return tuple(captures), outer


def _run_and_compare_whole_boundary(
    captures: tuple[_CapturedCall, ...],
    *,
    label: str,
) -> dict[str, Any]:
    """Replay captured mature calls through one compiled chronological boundary."""

    if not captures:
        raise RuntimeError("whole-boundary replay requires at least one captured call")
    reference_static = captures[0].replay_static_arguments
    for call_index, captured in enumerate(captures[1:], start=1):
        if captured.replay_static_arguments != reference_static:
            raise AssertionError(
                f"{label} call {call_index} changed static options inside one program"
            )
    initial_carry = tuple(
        jnp.asarray(np.array(value, copy=True)) for value in captures[0].initial_carry
    )
    final_carry, call_outputs = local_big_jit.run_fixed_capacity_whole_local(
        tuple(captured.prepared_call for captured in captures),
        *initial_carry,
        **reference_static,
    )
    final_carry = tuple(_to_host_array(value) for value in final_carry)
    call_outputs = tuple(
        tuple(_to_host_array(value) for value in output) for output in call_outputs
    )
    if len(call_outputs) != len(captures):
        raise AssertionError(
            f"{label} whole boundary returned {len(call_outputs)} calls, "
            f"expected {len(captures)}"
        )

    result_names = _BASE_RESULT_NAMES + ("debug_scores", "debug_probs")
    output_digests: list[dict[str, str]] = []
    for call_index, (captured, call_output) in enumerate(
        zip(captures, call_outputs, strict=True)
    ):
        # This gate is score-only, so the ten state values are invariant across
        # calls and the final carry reconstructs every mature result topology.
        reconstructed = local_big_jit._reconstruct_fixed_capacity_score_only_result(
            final_carry,
            call_output,
        )
        if len(reconstructed) != len(result_names):
            raise AssertionError(
                f"{label} call {call_index} returned an unexpected topology: "
                f"{len(reconstructed)} values"
            )
        digests = {}
        for name, actual in zip(result_names, reconstructed, strict=True):
            expected = captured.diagnostics[name]
            _assert_array_exact(
                f"{label} call {call_index:04d} {name}",
                actual,
                expected,
            )
            digests[name] = _array_digest(actual)
        output_digests.append(digests)
    return {
        "label": label,
        "passed": True,
        "one_compiled_boundary": True,
        "call_count": len(captures),
        "current_seam_exact": True,
        "output_sha256_by_call": output_digests,
    }


def _run_and_compare_uniform_scan_boundary(
    captures: tuple[_CapturedCall, ...],
    *,
    label: str,
) -> dict[str, Any]:
    """Replay uniform mature calls through one device-side scan boundary."""

    if not captures:
        raise RuntimeError("uniform-scan replay requires at least one captured call")
    reference_static = captures[0].replay_static_arguments
    initial_carry = tuple(
        jnp.asarray(np.array(value, copy=True)) for value in captures[0].initial_carry
    )
    final_carry, stacked_call_outputs = (
        local_big_jit.run_fixed_capacity_uniform_local_scan(
            tuple(captured.prepared_call for captured in captures),
            *initial_carry,
            **reference_static,
        )
    )
    final_carry = tuple(_to_host_array(value) for value in final_carry)
    result_names = _BASE_RESULT_NAMES + ("debug_scores", "debug_probs")
    output_digests: list[dict[str, str]] = []
    for call_index, captured in enumerate(captures):
        call_output = tuple(
            _to_host_array(value[call_index]) for value in stacked_call_outputs
        )
        reconstructed = local_big_jit._reconstruct_fixed_capacity_score_only_result(
            final_carry,
            call_output,
        )
        if len(reconstructed) != len(result_names):
            raise AssertionError(
                f"{label} call {call_index} returned an unexpected topology: "
                f"{len(reconstructed)} values"
            )
        digests = {}
        for name, actual in zip(result_names, reconstructed, strict=True):
            expected = captured.diagnostics[name]
            _assert_array_exact(
                f"{label} call {call_index:04d} {name}",
                actual,
                expected,
            )
            digests[name] = _array_digest(actual)
        output_digests.append(digests)
    return {
        "label": label,
        "passed": True,
        "one_compiled_scan_boundary": True,
        "call_count": len(captures),
        "current_seam_exact": True,
        "output_sha256_by_call": output_digests,
    }


def _block_tree(value) -> None:
    for leaf in jax.tree_util.tree_leaves(value):
        block_until_ready = getattr(leaf, "block_until_ready", None)
        if block_until_ready is not None:
            block_until_ready()


def _clone_prepared_call(call):
    def clone(value):
        return np.array(value, copy=True) if isinstance(value, np.ndarray) else value

    return local_big_jit._FixedCapacityPreparedLocalCall(
        leading_arguments=tuple(clone(value) for value in call.leading_arguments),
        trailing_arguments=tuple(clone(value) for value in call.trailing_arguments),
    )


def _fresh_carry(captured: _CapturedCall):
    return tuple(
        jnp.asarray(np.array(value, copy=True)) for value in captured.initial_carry
    )


def _run_prepared_calls_individually(
    call_program,
    initial_carry,
    static_options,
    *,
    synchronize_each_call: bool,
):
    carry = tuple(initial_carry)
    call_outputs = []
    for prepared_call in call_program:
        result = local_big_jit.run_local_bucket_big_jit(
            *prepared_call.leading_arguments,
            *carry,
            *prepared_call.trailing_arguments,
            **static_options,
        )
        if synchronize_each_call:
            _block_tree(result)
        carry = tuple(result[:8]) + tuple(result[9:11])
        call_outputs.append((result[8],) + tuple(result[11:]))
    return carry, tuple(call_outputs)


def _time_numeric_call(callable_):
    started = time.perf_counter()
    result = callable_()
    _block_tree(result)
    return time.perf_counter() - started, result


def run_mechanism_microbenchmark(
    captures: tuple[_CapturedCall, ...],
    *,
    call_counts=(2, 8, 16),
    warm_repeats: int = 5,
) -> dict[str, Any]:
    """Measure launch-boundary scaling without making a production claim."""

    if not captures:
        raise ValueError("mechanism microbenchmark requires captured mature calls")
    if warm_repeats < 3:
        raise ValueError("mechanism microbenchmark requires at least three warm repeats")
    static_options = dict(captures[0].replay_static_arguments)
    static_options.update(
        return_debug_arrays=False,
        return_debug_scores=False,
        return_debug_operands=False,
    )
    rows = []
    for raw_call_count in call_counts:
        call_count = int(raw_call_count)
        if call_count <= 0:
            raise ValueError("mechanism microbenchmark call counts must be positive")
        call_program = tuple(
            _clone_prepared_call(captures[index % len(captures)].prepared_call)
            for index in range(call_count)
        )

        def run_individual(*, synchronize_each_call: bool):
            return _run_prepared_calls_individually(
                call_program,
                _fresh_carry(captures[0]),
                static_options,
                synchronize_each_call=synchronize_each_call,
            )

        def run_whole():
            return local_big_jit.run_fixed_capacity_whole_local(
                call_program,
                *_fresh_carry(captures[0]),
                **static_options,
            )

        def run_scan():
            return local_big_jit.run_fixed_capacity_uniform_local_scan(
                call_program,
                *_fresh_carry(captures[0]),
                **static_options,
            )

        def stack_individual_result(result):
            carry, call_outputs = result
            return carry, jax.tree_util.tree_map(
                lambda *values: jnp.stack(values, axis=0),
                *call_outputs,
            )

        individual_first_s, individual_reference = _time_numeric_call(
            lambda: run_individual(synchronize_each_call=True)
        )
        whole_first_s, whole_reference = _time_numeric_call(run_whole)
        scan_first_s, scan_reference = _time_numeric_call(run_scan)
        stacked_individual_reference = stack_individual_result(individual_reference)
        for index, (actual, expected) in enumerate(
            zip(
                jax.tree_util.tree_leaves(whole_reference),
                jax.tree_util.tree_leaves(individual_reference),
                strict=True,
            )
        ):
            _assert_array_exact(
                f"mechanism call-count {call_count} output leaf {index}",
                _to_host_array(actual),
                _to_host_array(expected),
            )
        for index, (actual, expected) in enumerate(
            zip(
                jax.tree_util.tree_leaves(scan_reference),
                jax.tree_util.tree_leaves(stacked_individual_reference),
                strict=True,
            )
        ):
            _assert_array_exact(
                f"mechanism scan call-count {call_count} output leaf {index}",
                _to_host_array(actual),
                _to_host_array(expected),
            )

        synchronized_samples = []
        enqueued_samples = []
        whole_samples = []
        scan_samples = []
        for repeat in range(warm_repeats):
            if repeat % 2 == 0:
                synchronized_s, _ = _time_numeric_call(
                    lambda: run_individual(synchronize_each_call=True)
                )
                whole_s, _ = _time_numeric_call(run_whole)
                scan_s, _ = _time_numeric_call(run_scan)
            else:
                scan_s, _ = _time_numeric_call(run_scan)
                whole_s, _ = _time_numeric_call(run_whole)
                synchronized_s, _ = _time_numeric_call(
                    lambda: run_individual(synchronize_each_call=True)
                )
            enqueued_s, _ = _time_numeric_call(
                lambda: run_individual(synchronize_each_call=False)
            )
            synchronized_samples.append(synchronized_s)
            enqueued_samples.append(enqueued_s)
            whole_samples.append(whole_s)
            scan_samples.append(scan_s)
        synchronized_median_s = float(np.median(synchronized_samples))
        enqueued_median_s = float(np.median(enqueued_samples))
        whole_median_s = float(np.median(whole_samples))
        scan_median_s = float(np.median(scan_samples))
        rows.append(
            {
                "call_count": call_count,
                "individual_first_s": individual_first_s,
                "whole_first_s": whole_first_s,
                "scan_first_s": scan_first_s,
                "individual_synchronized_warm_s": synchronized_samples,
                "individual_enqueued_warm_s": enqueued_samples,
                "whole_warm_s": whole_samples,
                "scan_warm_s": scan_samples,
                "individual_synchronized_median_s": synchronized_median_s,
                "individual_enqueued_median_s": enqueued_median_s,
                "whole_median_s": whole_median_s,
                "scan_median_s": scan_median_s,
                "speedup_vs_synchronized": synchronized_median_s / whole_median_s,
                "speedup_vs_enqueued": enqueued_median_s / whole_median_s,
                "scan_speedup_vs_synchronized": synchronized_median_s
                / scan_median_s,
                "scan_speedup_vs_enqueued": enqueued_median_s / scan_median_s,
                "exact_outputs": True,
                "scan_exact_outputs": True,
            }
        )
    return {
        "classification": "mechanism_only_not_production_speed_claim",
        "call_counts": [int(value) for value in call_counts],
        "warm_repeats": int(warm_repeats),
        "rows": rows,
    }


def _assert_array_exact(label: str, actual: np.ndarray, expected: np.ndarray) -> None:
    if actual.dtype != expected.dtype:
        raise AssertionError(f"{label} dtype changed: {actual.dtype} vs {expected.dtype}")
    if actual.shape != expected.shape:
        raise AssertionError(f"{label} shape changed: {actual.shape} vs {expected.shape}")
    if actual.tobytes(order="C") != expected.tobytes(order="C"):
        delta = _finite_max_abs(actual, expected)
        raise AssertionError(f"{label} is not bitwise equal; finite max_abs={delta:.9g}")


def _finite_max_abs(actual: np.ndarray, expected: np.ndarray, mask=None) -> float:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    selected = np.ones(actual.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    finite = selected & np.isfinite(actual) & np.isfinite(expected)
    if not np.any(finite):
        return 0.0
    return float(np.max(np.abs(actual[finite].astype(np.float64) - expected[finite].astype(np.float64))))


def _centered_score_delta(actual: np.ndarray, expected: np.ndarray, candidate_mask: np.ndarray) -> np.ndarray:
    actual = np.asarray(actual, dtype=np.float64)
    expected = np.asarray(expected, dtype=np.float64)
    centered = np.zeros_like(actual)
    for image_index in range(actual.shape[0]):
        mask = np.asarray(candidate_mask[image_index], dtype=bool)
        if np.any(mask):
            active_delta = actual[image_index][mask] - expected[image_index][mask]
            reference = float(np.median(active_delta))
            centered[image_index][mask] = active_delta - reference
    return centered


def _comparison_metrics(actual: _CapturedCall, expected: _CapturedCall) -> dict[str, float]:
    candidate_mask = expected.diagnostics["candidate_mask"]
    if not np.array_equal(candidate_mask, actual.diagnostics["candidate_mask"]):
        raise AssertionError("candidate topology differs before continuous comparison")
    actual_scores = np.asarray(actual.diagnostics["debug_scores"], dtype=np.float64)
    expected_scores = np.asarray(expected.diagnostics["debug_scores"], dtype=np.float64)
    active_score_delta = np.abs(actual_scores[candidate_mask] - expected_scores[candidate_mask])
    centered = np.abs(
        _centered_score_delta(
            actual.diagnostics["debug_scores"],
            expected.diagnostics["debug_scores"],
            candidate_mask,
        )[candidate_mask]
    )
    return {
        "score_max_abs": float(np.max(active_score_delta, initial=0.0)),
        "score_p95_abs": float(np.percentile(active_score_delta, 95)) if active_score_delta.size else 0.0,
        "score_centered_max_abs": float(np.max(centered, initial=0.0)),
        "score_centered_p95_abs": float(np.percentile(centered, 95)) if centered.size else 0.0,
        "log_z_max_abs": _finite_max_abs(
            actual.diagnostics["log_z"],
            expected.diagnostics["log_z"],
        ),
        "best_score_max_abs": _finite_max_abs(
            actual.diagnostics["best_log_score"],
            expected.diagnostics["best_log_score"],
        ),
        "posterior_max_abs": _finite_max_abs(
            actual.diagnostics["debug_probs"],
            expected.diagnostics["debug_probs"],
            candidate_mask,
        ),
        "posterior_mass_max_abs": max(
            _finite_max_abs(
                actual.diagnostics["probs_sum_t"],
                expected.diagnostics["probs_sum_t"],
            ),
            _finite_max_abs(
                actual.diagnostics["reconstruction_probs_sum_t"],
                expected.diagnostics["reconstruction_probs_sum_t"],
            ),
        ),
        "max_posterior_max_abs": _finite_max_abs(
            actual.diagnostics["max_posterior"],
            expected.diagnostics["max_posterior"],
        ),
    }


def compare_captured_calls(
    actual: _CapturedCall,
    expected: _CapturedCall,
    *,
    label: str,
    require_exact_current_seam: bool,
) -> dict[str, Any]:
    if actual.static_arguments != expected.static_arguments:
        raise AssertionError(f"{label} static arguments changed")
    if actual.inputs.keys() != expected.inputs.keys():
        raise AssertionError(f"{label} prepared-input topology changed")
    input_digests = {}
    for name in actual.inputs:
        actual_value = actual.inputs[name]
        expected_value = expected.inputs[name]
        if actual_value is None or expected_value is None:
            if actual_value is not None or expected_value is not None:
                raise AssertionError(f"{label} input {name} changed None topology")
            input_digests[name] = None
            continue
        _assert_array_exact(f"{label} input {name}", actual_value, expected_value)
        input_digests[name] = _array_digest(actual_value)

    for field in _EXACT_DIAGNOSTIC_FIELDS:
        _assert_array_exact(
            f"{label} diagnostic {field}",
            actual.diagnostics[field],
            expected.diagnostics[field],
        )
    metrics = _comparison_metrics(actual, expected)
    if require_exact_current_seam:
        for field in _CONTINUOUS_DIAGNOSTIC_FIELDS:
            _assert_array_exact(
                f"{label} diagnostic {field}",
                actual.diagnostics[field],
                expected.diagnostics[field],
            )
        nonzero = {name: value for name, value in metrics.items() if value != 0.0}
        if nonzero:
            raise AssertionError(f"{label} current shared seam has nonzero deltas: {nonzero}")
    return {
        "label": label,
        "passed": True,
        "current_seam_exact": bool(require_exact_current_seam),
        "metrics": metrics,
        "input_sha256": input_digests,
        "donated_input_object_ids": actual.donated_input_object_ids,
    }


def compare_outer_snapshots(
    actual: dict[str, np.ndarray],
    expected: dict[str, np.ndarray],
    *,
    label: str,
) -> dict[str, Any]:
    if actual.keys() != expected.keys():
        raise AssertionError(f"{label} outer topology changed")
    digests = {}
    for name in actual:
        _assert_array_exact(f"{label} outer {name}", actual[name], expected[name])
        digests[name] = _array_digest(actual[name])
    return {"label": label, "passed": True, "output_sha256": digests}


def _save_diagnostics(
    output_path: Path,
    snapshots: dict[str, tuple[tuple[_CapturedCall, ...], dict[str, np.ndarray]]],
) -> None:
    arrays: dict[str, np.ndarray] = {}
    for arm_name, (captured_calls, outer) in snapshots.items():
        for call_index, captured in enumerate(captured_calls):
            for field_name, value in captured.diagnostics.items():
                arrays[
                    f"{arm_name}__call{call_index:04d}__diagnostic__{field_name}"
                ] = np.asarray(value)
        for field_name, value in outer.items():
            arrays[f"{arm_name}__outer__{field_name}"] = np.asarray(value)
    np.savez_compressed(output_path, **arrays)


def _git_head(repo_root: Path) -> str:
    return subprocess.check_output(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        text=True,
    ).strip()


def validate_future_envelope_pair() -> None:
    f32 = FUTURE_SPEED_QUALIFIED_ENVELOPES["float32"]
    f64 = FUTURE_SPEED_QUALIFIED_ENVELOPES["float64"]
    if f32.keys() != f64.keys():
        raise AssertionError("float32/float64 envelope metrics differ")
    for metric in f32:
        if not (f32[metric] > 0 and f64[metric] > 0):
            raise AssertionError(f"{metric} envelopes must be positive")
        if f32[metric] / f64[metric] < 1_000:
            raise AssertionError(f"{metric} float64 companion is not three orders tighter")


def run_gate(
    *,
    output_dir: Path,
    repeat_count: int = 2,
    git_head: str | None = None,
    gpu_uuid: str | None = None,
    include_mechanism_microbenchmark: bool = False,
) -> dict[str, Any]:
    if repeat_count < 2:
        raise ValueError("repeat_count must be at least two")
    validate_future_envelope_pair()
    output_dir.mkdir(parents=True, exist_ok=False)
    fixture = build_gate_fixture()

    snapshots: dict[str, tuple[tuple[_CapturedCall, ...], dict[str, np.ndarray]]] = {}
    comparisons: list[dict[str, Any]] = []
    production_comparisons: list[dict[str, Any]] = []
    whole_boundary_comparisons: list[dict[str, Any]] = []
    uniform_scan_comparisons: list[dict[str, Any]] = []
    donated_input_objects: list[object] = []
    for precision in ("float32", "float64"):
        # Alternate arm order after the first pair so repeat stability is not
        # accidentally conflated with a fixed warm-cache order.
        sequence = [("default", 0), ("disabled", 0), ("fixed", 0)]
        for repeat in range(1, repeat_count):
            sequence.extend((("fixed", repeat), ("default", repeat)))
        for arm, repeat in sequence:
            key = f"{precision}_{arm}_{repeat}"
            snapshots[key] = _run_captured_arm(
                fixture,
                precision,
                arm,
                donated_input_objects=donated_input_objects,
            )

        default = snapshots[f"{precision}_default_0"]
        disabled = snapshots[f"{precision}_disabled_0"]
        fixed = snapshots[f"{precision}_fixed_0"]
        whole_boundary_comparisons.append(
            _run_and_compare_whole_boundary(
                fixed[0],
                label=f"{precision}:mature-per-call-vs-fixed-one-boundary",
            )
        )
        uniform_scan_comparisons.append(
            _run_and_compare_uniform_scan_boundary(
                fixed[0],
                label=f"{precision}:mature-per-call-vs-fixed-uniform-scan",
            )
        )
        for call_index, (default_call, disabled_call, fixed_call) in enumerate(
            zip(default[0], disabled[0], fixed[0], strict=True)
        ):
            comparisons.append(
                compare_captured_calls(
                    disabled_call,
                    default_call,
                    label=(
                        f"{precision}:default-vs-disabled-selector:call-{call_index:04d}"
                    ),
                    require_exact_current_seam=True,
                )
            )
            comparisons.append(
                compare_captured_calls(
                    fixed_call,
                    default_call,
                    label=f"{precision}:mature-vs-fixed:call-{call_index:04d}",
                    require_exact_current_seam=True,
                )
            )
        production_comparisons.append(
            compare_outer_snapshots(
                disabled[1],
                default[1],
                label=f"{precision}:default-vs-disabled-selector-outer",
            )
        )
        production_comparisons.append(
            compare_outer_snapshots(
                fixed[1],
                default[1],
                label=f"{precision}:mature-vs-fixed-outer",
            )
        )
        for repeat in range(1, repeat_count):
            for arm in ("default", "fixed"):
                repeated = snapshots[f"{precision}_{arm}_{repeat}"]
                baseline = snapshots[f"{precision}_{arm}_0"]
                for call_index, (repeated_call, baseline_call) in enumerate(
                    zip(repeated[0], baseline[0], strict=True)
                ):
                    comparisons.append(
                        compare_captured_calls(
                            repeated_call,
                            baseline_call,
                            label=f"{precision}:{arm}-repeat-{repeat}:call-{call_index:04d}",
                            require_exact_current_seam=True,
                        )
                    )
                production_comparisons.append(
                    compare_outer_snapshots(
                        repeated[1],
                        baseline[1],
                        label=f"{precision}:{arm}-outer-repeat-{repeat}",
                    )
                )

        # Exercise the uninstrumented production topology independently.  The
        # capture above uses a debug-output static JIT variant and therefore
        # cannot establish that default production behavior is unchanged.
        production_default = _outer_snapshot(_run_outer(fixture, precision, "default"))
        production_disabled = _outer_snapshot(_run_outer(fixture, precision, "disabled"))
        production_fixed = _outer_snapshot(_run_outer(fixture, precision, "fixed"))
        production_whole = _outer_snapshot(_run_outer(fixture, precision, "whole"))
        production_comparisons.extend(
            (
                compare_outer_snapshots(
                    production_disabled,
                    production_default,
                    label=f"{precision}:production-default-vs-disabled-selector",
                ),
                compare_outer_snapshots(
                    production_fixed,
                    production_default,
                    label=f"{precision}:production-mature-vs-fixed",
                ),
                compare_outer_snapshots(
                    production_whole,
                    production_default,
                    label=f"{precision}:production-mature-vs-fixed-one-boundary",
                ),
            )
        )

    mechanism_microbenchmark = (
        run_mechanism_microbenchmark(snapshots["float32_fixed_0"][0])
        if include_mechanism_microbenchmark
        else None
    )
    diagnostics_path = output_dir / "diagnostics.npz"
    _save_diagnostics(diagnostics_path, snapshots)
    reference_calls = snapshots["float32_default_0"][0]
    active_candidate_count = int(
        sum(np.sum(call.diagnostics["candidate_mask"]) for call in reference_calls)
    )
    active_significant_counts = np.concatenate(
        [
            np.asarray(call.diagnostics["n_significant_samples"])[
                np.asarray(call.inputs["valid_image_mask"], dtype=bool)
            ]
            for call in reference_calls
        ]
    )
    padded_significant_counts = [
        np.asarray(call.diagnostics["n_significant_samples"]).tolist()
        for call in reference_calls
    ]
    payload = {
        "schema": SCHEMA,
        "classification": "correctness_only",
        "passed": True,
        "git_head": git_head,
        "gpu_uuid": gpu_uuid,
        "current_seam_contract": "bitwise_exact_same_shared_wrapper_and_operands",
        "current_seam_exact": True,
        "future_numerical_policy": {
            "activation_requires_material_production_speedup": True,
            "mathematical_equivalence_required": True,
            "repeat_bounded_non_growing_drift_required": True,
            "exact_discrete_support_and_decisions_required": True,
            "unstable_or_unexplained_drift_allowed": False,
            "envelopes": FUTURE_SPEED_QUALIFIED_ENVELOPES,
        },
        "speed_claim_allowed": False,
        "default_promotion_allowed": False,
        "production_whole_boundary_enabled": True,
        "production_executor": "chronological_uniform_scan_segments",
        "mechanism_microbenchmark": mechanism_microbenchmark,
        "repeat_count": int(repeat_count),
        "precision_lanes": ["float32", "float64"],
        "fixture": {
            "n_images": fixture.dataset.n_images,
            "image_shape": list(IMAGE_SHAPE),
            "volume_shape": list(VOLUME_SHAPE),
            "bucket_image_order": fixture.bucket_image_order.tolist(),
            "bucket_image_capacity": fixture.bucket_image_capacity,
            "bucket_radix": fixture.bucket_radix,
            "valid_call_count": int(fixture.fixed_bundle.plan.valid_call_count),
            "physical_call_capacity": int(fixture.fixed_bundle.plan.physical_call_capacity),
            "active_candidate_count": active_candidate_count,
            "active_significant_counts": active_significant_counts.tolist(),
            "padded_significant_counts_by_call": padded_significant_counts,
        },
        "runtime": {
            "jax_backend": jax.default_backend(),
            "jax_enable_x64": bool(jax.config.jax_enable_x64),
            "devices": [str(device) for device in jax.devices()],
            "donated_argnums": [7, 8],
            "donated_positional_names": list(CURRENT_DONATED_POSITIONAL_NAMES),
            "fresh_donated_input_objects_per_invocation": True,
        },
        "comparisons": comparisons,
        "production_comparisons": production_comparisons,
        "whole_boundary_comparisons": whole_boundary_comparisons,
        "uniform_scan_comparisons": uniform_scan_comparisons,
        "diagnostics_npz": diagnostics_path.name,
        "diagnostics_sha256": hashlib.sha256(diagnostics_path.read_bytes()).hexdigest(),
    }
    (output_dir / "gate_result.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
    )
    return payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--repeat-count", type=int, default=2)
    parser.add_argument("--expected-repo-head")
    parser.add_argument("--expected-gpu-uuid")
    parser.add_argument("--require-gpu", action="store_true")
    parser.add_argument("--run-mechanism-microbenchmark", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    repo_root = Path(__file__).resolve().parents[1]
    if args.expected_repo_head is not None:
        observed_head = _git_head(repo_root)
        if observed_head != args.expected_repo_head:
            raise SystemExit(
                f"repository head mismatch: expected {args.expected_repo_head}, got {observed_head}",
            )
    require_gpu = bool(args.require_gpu or args.expected_gpu_uuid)
    if require_gpu and jax.default_backend() != "gpu":
        raise SystemExit(f"H100 score gate requires GPU backend, got {jax.default_backend()}")
    observed_gpu_uuid = None
    if require_gpu:
        devices = jax.devices("gpu")
        if len(devices) != 1:
            raise SystemExit(f"H100 score gate requires exactly one visible GPU, got {devices}")
        nvidia_smi_command = ["nvidia-smi"]
        if args.expected_gpu_uuid is not None:
            nvidia_smi_command.append(f"--id={args.expected_gpu_uuid}")
        nvidia_smi_command.extend(("--query-gpu=uuid", "--format=csv,noheader"))
        observed_gpu_uuid = subprocess.check_output(nvidia_smi_command, text=True).strip()
        if not observed_gpu_uuid.startswith("GPU-") or "\n" in observed_gpu_uuid:
            raise SystemExit(f"failed to resolve one visible physical GPU UUID: {observed_gpu_uuid!r}")
        if args.expected_gpu_uuid is not None and observed_gpu_uuid != args.expected_gpu_uuid:
            raise SystemExit(
                f"GPU UUID mismatch: expected {args.expected_gpu_uuid}, got {observed_gpu_uuid}",
            )
    payload = run_gate(
        output_dir=args.output_dir,
        repeat_count=args.repeat_count,
        git_head=_git_head(repo_root),
        gpu_uuid=observed_gpu_uuid,
        include_mechanism_microbenchmark=args.run_mechanism_microbenchmark,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
