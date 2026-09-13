"""Optional native VDAM M-step replay for causal numerical diagnostics.

The M-step owns when overrides are applied. These readers preserve captured
F64/C128 buffers, template selection and validation; no replay runs on import.
"""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

import numpy as np

from recovar.em.vdam.state import VdamAccumulator

VDAM_NATIVE_SECOND_MOMENT_REPLAY_ENV = "RECOVAR_VDAM_NATIVE_SECOND_MOMENT_REPLAY_BIN"


VDAM_NATIVE_SECOND_MOMENT_REPLAY_ITER_ENV = "RECOVAR_VDAM_NATIVE_SECOND_MOMENT_REPLAY_ITER"


VDAM_NATIVE_FIRST_MOMENT_REPLAY_ENV = "RECOVAR_VDAM_NATIVE_FIRST_MOMENT_REPLAY_BIN"


VDAM_NATIVE_FIRST_MOMENT_REPLAY_ITER_ENV = "RECOVAR_VDAM_NATIVE_FIRST_MOMENT_REPLAY_ITER"


VDAM_NATIVE_BPREF_DATA_REPLAY_ENV = "RECOVAR_VDAM_NATIVE_BPREF_DATA_REPLAY_BIN"


VDAM_NATIVE_BPREF_WEIGHT_REPLAY_ENV = "RECOVAR_VDAM_NATIVE_BPREF_WEIGHT_REPLAY_BIN"


VDAM_NATIVE_BPREF_REPLAY_ITER_ENV = "RECOVAR_VDAM_NATIVE_BPREF_REPLAY_ITER"


VDAM_NATIVE_IREF_INPUT_REPLAY_ENV = "RECOVAR_VDAM_NATIVE_IREF_INPUT_REPLAY_BIN"


VDAM_NATIVE_IREF_INPUT_REPLAY_ITER_ENV = "RECOVAR_VDAM_NATIVE_IREF_INPUT_REPLAY_ITER"


def _replay_iteration_selected(env_name: str, iteration: int) -> bool:
    """Return whether an integer/``all`` diagnostic selector matches."""

    replay_iteration_value = os.environ.get(env_name, "1").strip()
    replay_all_iterations = replay_iteration_value.lower() in {"all", "*"}
    try:
        replay_iteration = None if replay_all_iterations else int(replay_iteration_value)
    except ValueError as exc:
        raise ValueError(
            f"{env_name} must be an integer or 'all'"
        ) from exc
    return replay_iteration is None or int(iteration) == replay_iteration


def _read_native_replay(path: Path, *, expected_shape: tuple[int, ...], dtype) -> np.ndarray:
    """Read a native F64 or interleaved C128 replay with its three-int64 header."""
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        shape = np.fromfile(stream, dtype=np.int64, count=3)
        values = np.fromfile(stream, dtype=np.float64)
    if shape.size != 3 or np.any(shape <= 0):
        raise ValueError(f"{path}: invalid three-int64 shape header")
    components = np.dtype(dtype).itemsize // np.dtype(np.float64).itemsize
    value_count = components * int(np.prod(shape, dtype=np.int64))
    if values.size != value_count:
        kind = "components" if components == 2 else "values"
        raise ValueError(f"{path}: expected {value_count} float64 {kind}, got {values.size}")
    replay = values.view(dtype).reshape(tuple(int(value) for value in shape))
    if replay.shape != expected_shape:
        raise ValueError(f"{path}: replay shape {replay.shape} does not match {expected_shape}")
    if not np.all(np.isfinite(replay)):
        raise ValueError(f"{path}: replay contains non-finite values")
    return replay


def _maybe_replay_native_bpref_accumulators(
    accum_h0: VdamAccumulator,
    accum_h1: VdamAccumulator | None,
    *,
    iteration: int,
    class_idx: int,
) -> tuple[VdamAccumulator, VdamAccumulator | None]:
    """Replay paired native raw BPref buffers for causal diagnosis."""

    data_template = os.environ.get(VDAM_NATIVE_BPREF_DATA_REPLAY_ENV, "").strip()
    weight_template = os.environ.get(VDAM_NATIVE_BPREF_WEIGHT_REPLAY_ENV, "").strip()
    if bool(data_template) != bool(weight_template):
        raise ValueError("native BPref replay requires both data and weight templates")
    if not data_template or not _replay_iteration_selected(
        VDAM_NATIVE_BPREF_REPLAY_ITER_ENV, iteration
    ):
        return accum_h0, accum_h1

    outputs = []
    accumulators = (accum_h0,) if accum_h1 is None else (accum_h0, accum_h1)
    for halfset, accumulator in enumerate(accumulators):
        fields = {
            "iteration": int(iteration),
            "class_idx": int(class_idx),
            "halfset": halfset,
            "half_suffix": "" if halfset == 0 else "_h",
        }
        data_path = Path(data_template.format(**fields))
        weight_path = Path(weight_template.format(**fields))
        outputs.append(
            replace(
                accumulator,
                data=_read_native_replay(
                    data_path, expected_shape=np.asarray(accumulator.data).shape, dtype=np.complex128
                ),
                weight=_read_native_replay(
                    weight_path, expected_shape=np.asarray(accumulator.weight).shape, dtype=np.float64
                ),
            )
        )
    replay_h1 = None if accum_h1 is None else outputs[1]
    return outputs[0], replay_h1


def _maybe_replay_native_reference_input(
    computed: np.ndarray,
    *,
    iteration: int,
    class_idx: int,
) -> np.ndarray:
    """Replay native ``Iref_before`` immediately before reconstruction."""

    replay_template = os.environ.get(VDAM_NATIVE_IREF_INPUT_REPLAY_ENV, "").strip()
    if not replay_template or not _replay_iteration_selected(
        VDAM_NATIVE_IREF_INPUT_REPLAY_ITER_ENV, iteration
    ):
        return computed
    replay_path = Path(
        replay_template.format(iteration=int(iteration), class_idx=int(class_idx))
    )
    computed = np.asarray(computed)
    return _read_native_replay(replay_path, expected_shape=computed.shape, dtype=np.float64)


def _maybe_replay_native_first_moments(
    computed_h0: np.ndarray,
    computed_h1: np.ndarray | None,
    *,
    iteration: int,
    class_idx: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Replay paired native ``Igrad1_post`` buffers for causal diagnosis."""

    replay_template = os.environ.get(VDAM_NATIVE_FIRST_MOMENT_REPLAY_ENV, "").strip()
    if not replay_template or not _replay_iteration_selected(
        VDAM_NATIVE_FIRST_MOMENT_REPLAY_ITER_ENV, iteration
    ):
        return computed_h0, computed_h1

    outputs = []
    computed_values = (computed_h0,) if computed_h1 is None else (computed_h0, computed_h1)
    for halfset, computed in enumerate(computed_values):
        path = Path(
            replay_template.format(
                iteration=int(iteration),
                class_idx=int(class_idx),
                halfset=halfset,
                half_suffix="" if halfset == 0 else "_h",
            )
        )
        outputs.append(
            _read_native_replay(path, expected_shape=np.asarray(computed).shape, dtype=np.complex128)
        )
    replay_h1 = None if computed_h1 is None else outputs[1]
    return outputs[0], replay_h1


def _maybe_replay_native_second_moment(
    computed: np.ndarray,
    *,
    iteration: int,
    class_idx: int,
) -> np.ndarray:
    """Replay one paired native ``Igrad2_post`` dump for causal diagnosis.

    This is an explicit, fail-closed oracle discriminator. It is inactive by
    default and is not a production parity mechanism. The path may contain
    ``{iteration}`` and ``{class_idx}`` placeholders. Setting the iteration
    selector to ``all`` replays a templated buffer at every M-step.
    """

    replay_template = os.environ.get(VDAM_NATIVE_SECOND_MOMENT_REPLAY_ENV, "").strip()
    if not replay_template or not _replay_iteration_selected(
        VDAM_NATIVE_SECOND_MOMENT_REPLAY_ITER_ENV, iteration
    ):
        return computed

    replay_path = Path(
        replay_template.format(iteration=int(iteration), class_idx=int(class_idx))
    )
    computed = np.asarray(computed)
    return _read_native_replay(replay_path, expected_shape=computed.shape, dtype=np.complex128)
