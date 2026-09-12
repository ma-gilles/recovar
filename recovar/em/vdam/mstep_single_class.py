"""Single-class VDAM M-step of the native InitialModel driver.

The per-class M-step transaction, its precision-route validation, the RELION
reconstruction-weight and resolution-shell rules and the optional native
(RELION) replay of BPref accumulators, moments and reference inputs live here;
``m_step`` keeps the multi-class driver.
"""

from __future__ import annotations

import os
from dataclasses import replace
from functools import partial
from pathlib import Path
from typing import Literal, Optional

import numpy as np

from recovar.em.vdam.state import InitialModelState, VdamAccumulator, half_slot_index

XMIPP_EQUAL_ACCURACY: float = 1e-6


RELION_DEFAULT_GRAD_MIN_RESOL_ANGSTROM: float = 20.0


VDAM_NATIVE_SECOND_MOMENT_REPLAY_ENV = "RECOVAR_VDAM_NATIVE_SECOND_MOMENT_REPLAY_BIN"


VDAM_NATIVE_SECOND_MOMENT_REPLAY_ITER_ENV = "RECOVAR_VDAM_NATIVE_SECOND_MOMENT_REPLAY_ITER"


VDAM_NATIVE_FIRST_MOMENT_REPLAY_ENV = "RECOVAR_VDAM_NATIVE_FIRST_MOMENT_REPLAY_BIN"


VDAM_NATIVE_FIRST_MOMENT_REPLAY_ITER_ENV = "RECOVAR_VDAM_NATIVE_FIRST_MOMENT_REPLAY_ITER"


VDAM_NATIVE_BPREF_DATA_REPLAY_ENV = "RECOVAR_VDAM_NATIVE_BPREF_DATA_REPLAY_BIN"


VDAM_NATIVE_BPREF_WEIGHT_REPLAY_ENV = "RECOVAR_VDAM_NATIVE_BPREF_WEIGHT_REPLAY_BIN"


VDAM_NATIVE_BPREF_REPLAY_ITER_ENV = "RECOVAR_VDAM_NATIVE_BPREF_REPLAY_ITER"


VDAM_NATIVE_IREF_INPUT_REPLAY_ENV = "RECOVAR_VDAM_NATIVE_IREF_INPUT_REPLAY_BIN"


VDAM_NATIVE_IREF_INPUT_REPLAY_ITER_ENV = "RECOVAR_VDAM_NATIVE_IREF_INPUT_REPLAY_ITER"


_MSTEP_F32_STATE_DTYPES = {
    "Iref": np.float32,
    "Igrad1": np.complex64,
    "Igrad2": np.complex64,
    "sigma2_class": np.float32,
    "data_vs_prior_class": np.float32,
    "fourier_coverage_class": np.float32,
}


def _validate_mstep_precision_route(
    mstep_compute_dtype: Literal["float32", "float64"],
    mstep_backend: Literal["native", "jax"],
    *,
    use_native_transaction: bool = True,
) -> None:
    """Reject precision-changing diagnostic diversions before reading overrides."""
    if mstep_compute_dtype not in {"float32", "float64"}:
        raise ValueError(f"Unknown mstep_compute_dtype: {mstep_compute_dtype!r}")
    if mstep_compute_dtype == "float64":
        return
    if mstep_backend != "jax" or not use_native_transaction:
        raise ValueError("float32 M-step requires the JAX transaction backend")
    for name in (
        "RECOVAR_MSTEP_DUMP_DIR",
        VDAM_NATIVE_SECOND_MOMENT_REPLAY_ENV,
        VDAM_NATIVE_FIRST_MOMENT_REPLAY_ENV,
        VDAM_NATIVE_BPREF_DATA_REPLAY_ENV,
        VDAM_NATIVE_BPREF_WEIGHT_REPLAY_ENV,
        VDAM_NATIVE_IREF_INPUT_REPLAY_ENV,
    ):
        if name in os.environ and (name == "RECOVAR_MSTEP_DUMP_DIR" or os.environ[name].strip()):
            raise ValueError(f"float32 M-step is incompatible with {name}")


def _validate_mstep_state_precision(state: InitialModelState) -> None:
    for name, dtype in _MSTEP_F32_STATE_DTYPES.items():
        if np.asarray(getattr(state, name)).dtype != np.dtype(dtype):
            raise ValueError(f"float32 M-step requires state.{name} dtype {np.dtype(dtype)}")


def _get_bindings():
    """Import the RELION binding module; raise a clear error if unbuilt."""
    try:
        from recovar.relion_bind import _relion_bind_core as m
    except ImportError as e:  # pragma: no cover
        raise RuntimeError(
            "VDAM M-step requires the RELION bindings. Run:\n  pixi run python recovar/relion_bind/build.py"
        ) from e
    return m


def _r_max_from_state(state: InitialModelState, padding_factor: int) -> int:
    """`r_max = current_size / 2` at padding_factor=1; RELION's
    `initZeros(-1)` computes `r_max = ori_size / (padding_factor * 2)` so
    we pass the current-size equivalent explicitly.

    Source: backprojector.cpp::initZeros and ml_optimiser.cpp:5846.
    """
    return state.current_size // 2


def _relion_resolution_shell(ori_size: int, pixel_size: float, resolution_angstrom: float) -> int:
    """RELION ``MlModel::getPixelFromResolution(1./resolution_angstrom)``."""
    if resolution_angstrom <= 0.0:
        raise ValueError(f"resolution_angstrom must be positive, got {resolution_angstrom}")
    shell = float(ori_size) * float(pixel_size) / float(resolution_angstrom)
    return int(np.floor(shell + 0.5)) if shell >= 0.0 else -int(np.floor(-shell + 0.5))


def _grad_min_resol_shell_from_state(
    state: InitialModelState,
    grad_min_resol_shell: float | None,
) -> float:
    """Default ``reconstructGrad`` min-resol shell from RELION InitialModel."""
    if grad_min_resol_shell is not None:
        return float(grad_min_resol_shell)
    return float(
        _relion_resolution_shell(
            int(state.ori_size),
            float(state.pixel_size),
            RELION_DEFAULT_GRAD_MIN_RESOL_ANGSTROM,
        )
    )


def _has_relion_reconstruction_weight(state: InitialModelState, k: int, accum_h0: VdamAccumulator) -> bool:
    """RELION only reconstructs classes with active prior and nonzero BPref weight."""
    if float(np.asarray(state.pdf_class, dtype=np.float64)[k]) <= 0.0:
        return False
    return float(np.sum(np.asarray(accum_h0.weight, dtype=np.float64))) > XMIPP_EQUAL_ACCURACY


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


def _read_native_complex_replay(path: Path, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        shape = np.fromfile(stream, dtype=np.int64, count=3)
        values = np.fromfile(stream, dtype=np.float64)
    if shape.size != 3 or np.any(shape <= 0):
        raise ValueError(f"{path}: invalid three-int64 shape header")
    value_count = int(np.prod(shape, dtype=np.int64))
    if values.size != 2 * value_count:
        raise ValueError(
            f"{path}: expected {2 * value_count} float64 components, got {values.size}"
        )
    replay = values.view(np.complex128).reshape(tuple(int(value) for value in shape))
    if replay.shape != expected_shape:
        raise ValueError(f"{path}: replay shape {replay.shape} does not match {expected_shape}")
    if not np.all(np.isfinite(replay)):
        raise ValueError(f"{path}: replay contains non-finite values")
    return replay


def _read_native_real_replay(path: Path, *, expected_shape: tuple[int, ...]) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        shape = np.fromfile(stream, dtype=np.int64, count=3)
        replay = np.fromfile(stream, dtype=np.float64)
    if shape.size != 3 or np.any(shape <= 0):
        raise ValueError(f"{path}: invalid three-int64 shape header")
    value_count = int(np.prod(shape, dtype=np.int64))
    if replay.size != value_count:
        raise ValueError(f"{path}: expected {value_count} float64 values, got {replay.size}")
    replay = replay.reshape(tuple(int(value) for value in shape))
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
                data=_read_native_complex_replay(
                    data_path, expected_shape=np.asarray(accumulator.data).shape
                ),
                weight=_read_native_real_replay(
                    weight_path, expected_shape=np.asarray(accumulator.weight).shape
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
    return _read_native_real_replay(replay_path, expected_shape=computed.shape)


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
            _read_native_complex_replay(path, expected_shape=np.asarray(computed).shape)
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
    return _read_native_complex_replay(replay_path, expected_shape=computed.shape)


def _copy_mstep_untouched_slots(values: np.ndarray, updated_slots: tuple[int, ...]) -> np.ndarray:
    """Allocate independent C-order state, leaving only replaced slots unwritten.

    The caller must fill every listed slot before publishing the new state.
    In K=1 all large slots are replaced, so none of the previous values need
    copying. In K-class updates the other classes retain their original bytes.
    """
    out = np.empty_like(values, order="C")
    start = 0
    for slot in updated_slots:
        out[start:slot] = values[start:slot]
        start = slot + 1
    out[start:] = values[start:]
    return out


def _run_m_step_transaction(
    transaction,
    state: InitialModelState,
    k: int,
    accum_h0: VdamAccumulator,
    accum_h1: VdamAccumulator | None,
    *,
    grad_current_stepsize: float,
    tau2_fudge_factor: float,
    padding_factor: int,
    r_max: int,
    min_resol_shell: float,
    mstep_compute_dtype: Literal["float32", "float64"] = "float64",
) -> InitialModelState:
    """Apply one shared-layout transaction and preserve state ownership."""
    from recovar.utils.helpers import recovar_volume_to_relion, relion_volume_to_recovar

    if mstep_compute_dtype == "float32":
        _validate_mstep_state_precision(state)
    copy_token = os.environ.get("RECOVAR_VDAM_MSTEP_COPY_UNTOUCHED", "0")
    if copy_token not in {"0", "1"}:
        raise ValueError("RECOVAR_VDAM_MSTEP_COPY_UNTOUCHED must be 0 or 1")
    copy_untouched = copy_token == "1"
    slot_h0 = half_slot_index(k, 0, state.K, state.pseudo_halfsets)
    slot_h1 = half_slot_index(k, 1, state.K, True) if state.pseudo_halfsets else None
    effective_stepsize = float(grad_current_stepsize) * (
        1.0 - np.exp(-float(3 * state.K + 10) * float(np.asarray(state.pdf_class)[k]))
    )
    result = transaction(
        recovar_volume_to_relion(np.asarray(state.Iref[k])),
        accum_h0.data,
        accum_h0.weight,
        accum_h1.data if accum_h1 is not None else None,
        accum_h1.weight if accum_h1 is not None else None,
        state.Igrad1[slot_h0],
        state.Igrad1[slot_h1] if slot_h1 is not None else None,
        state.Igrad2[k],
        state.fsc_halves_class[0],
        state.fsc_halves_class[k],
        state.tau2_class[k],
        effective_stepsize,
        tau2_fudge_factor,
        state.ori_size,
        padding_factor,
        1,
        r_max,
        min_resol_shell,
    )
    if mstep_compute_dtype == "float32":
        expected_dtypes = {
            "iref": np.float32,
            "mom1_h0": np.complex64,
            "mom2": np.complex64,
            "sigma2": np.float32,
            "data_vs_prior": np.float32,
            "fourier_coverage": np.float32,
        }
        if slot_h1 is not None:
            expected_dtypes["mom1_h1"] = np.complex64
        for name, dtype in expected_dtypes.items():
            if np.asarray(result[name]).dtype != np.dtype(dtype):
                raise ValueError(f"float32 M-step output {name} must have dtype {np.dtype(dtype)}")
        prior = np.asarray(state.tau2_class[k])
        returned_prior = np.asarray(result["tau2"])
        if (
            returned_prior.dtype != prior.dtype
            or returned_prior.shape != prior.shape
            or returned_prior.tobytes() != prior.tobytes()
        ):
            raise ValueError("float32 M-step must preserve authoritative tau2")
    out = replace(state)
    out.Iref = _copy_mstep_untouched_slots(state.Iref, (k,)) if copy_untouched else state.Iref.copy()
    out.Iref[k] = relion_volume_to_recovar(np.asarray(result["iref"]))
    moment_slots = (slot_h0,) if slot_h1 is None else (slot_h0, slot_h1)
    out.Igrad1 = (
        _copy_mstep_untouched_slots(state.Igrad1, moment_slots) if copy_untouched else state.Igrad1.copy()
    )
    out.Igrad1[slot_h0] = np.asarray(result["mom1_h0"])
    if slot_h1 is not None:
        out.Igrad1[slot_h1] = np.asarray(result["mom1_h1"])
    out.Igrad2 = _copy_mstep_untouched_slots(state.Igrad2, (k,)) if copy_untouched else state.Igrad2.copy()
    out.Igrad2[k] = np.asarray(result["mom2"])
    for attribute, key in (
        ("tau2_class", "tau2"),
        ("sigma2_class", "sigma2"),
        ("data_vs_prior_class", "data_vs_prior"),
        ("fourier_coverage_class", "fourier_coverage"),
    ):
        values = getattr(state, attribute).copy()
        dtype = values.dtype if mstep_compute_dtype == "float32" else np.float64
        values[k] = np.asarray(result[key], dtype=dtype)
        setattr(out, attribute, values)
    return out


def vdam_m_step_single_class(
    state: InitialModelState,
    k: int,
    accum_h0: VdamAccumulator,
    accum_h1: Optional[VdamAccumulator],
    *,
    grad_current_stepsize: float,
    tau2_fudge_factor: float,
    grad_min_resol_shell: float | None = None,
    padding_factor: int = 1,
    use_native_transaction: bool = True,
    mstep_backend: Literal["native", "jax"] = "native",
    mstep_compute_dtype: Literal["float32", "float64"] = "float64",
) -> InitialModelState:
    """VDAM M-step for one class (per-class loop matches RELION's binding shape).

    Pseudo-halfsets: FSC/noise-power is derived from the halfset-data difference
    in ``applyMomenta``; ``reconstructGrad`` then uses ``mom1_noise_power``.
    """
    _validate_mstep_precision_route(
        mstep_compute_dtype, mstep_backend, use_native_transaction=use_native_transaction
    )
    if mstep_compute_dtype == "float32":
        _validate_mstep_state_precision(state)
    if mstep_backend not in {"native", "jax"}:
        raise ValueError(f"Unknown mstep_backend: {mstep_backend!r}")
    if not (0 <= k < state.K):
        raise ValueError(f"class index {k} out of range")
    if state.pseudo_halfsets and accum_h1 is None:
        raise ValueError("pseudo_halfsets=True requires accum_h1")
    if not state.pseudo_halfsets and accum_h1 is not None:
        raise ValueError("pseudo_halfsets=False must have accum_h1=None")
    accum_h0, accum_h1 = _maybe_replay_native_bpref_accumulators(
        accum_h0,
        accum_h1,
        iteration=int(getattr(state, "iter", 0)),
        class_idx=k,
    )
    if not _has_relion_reconstruction_weight(state, k, accum_h0):
        return state

    bind = _get_bindings()
    if mstep_compute_dtype == "float32" and not hasattr(bind, "vdam_m_step_transaction"):
        raise RuntimeError("float32 M-step requires transaction-capable native certificates")
    ori_size = state.ori_size
    r_max = _r_max_from_state(state, padding_factor)
    min_resol_shell = _grad_min_resol_shell_from_state(state, grad_min_resol_shell)
    # backprojector.h:335/343 EMA defaults
    mu_first, mu_second = 0.9, 0.999

    _dump_dir = os.environ.get("RECOVAR_MSTEP_DUMP_DIR")
    _do_dump = _dump_dir is not None and int(getattr(state, "iter", 0)) == int(
        os.environ.get("RECOVAR_MSTEP_DUMP_ITER", "1")
    )
    _dump_prefix = f"c{k}_" if state.K > 1 else ""

    # Replay and intermediate dump diagnostics retain their original boundaries.
    replay_requested = any(
        os.environ.get(name, "").strip()
        for name in (
            VDAM_NATIVE_SECOND_MOMENT_REPLAY_ENV,
            VDAM_NATIVE_FIRST_MOMENT_REPLAY_ENV,
            VDAM_NATIVE_BPREF_DATA_REPLAY_ENV,
            VDAM_NATIVE_BPREF_WEIGHT_REPLAY_ENV,
            VDAM_NATIVE_IREF_INPUT_REPLAY_ENV,
        )
    )
    if (
        use_native_transaction
        and not _do_dump
        and not replay_requested
        and hasattr(bind, "vdam_m_step_transaction")
    ):
        transaction = bind.vdam_m_step_transaction
        if mstep_backend == "jax":
            if not hasattr(bind, "vdam_first_moment_initializes"):
                raise RuntimeError("JAX M-step requires the native moment initialization binding")
            from recovar.em.relion.relion_vdam_mstep import relion_vdam_m_step_host

            transaction = relion_vdam_m_step_host
            if mstep_compute_dtype == "float32":
                transaction = partial(transaction, compute_dtype=np.float32)
        return _run_m_step_transaction(
            transaction,
            state,
            k,
            accum_h0,
            accum_h1,
            grad_current_stepsize=grad_current_stepsize,
            tau2_fudge_factor=tau2_fudge_factor,
            padding_factor=padding_factor,
            r_max=r_max,
            min_resol_shell=min_resol_shell,
            mstep_compute_dtype=mstep_compute_dtype,
        )

    def _dump(name, arr):
        if not _do_dump:
            return
        from pathlib import Path as _Path

        _Path(_dump_dir).mkdir(parents=True, exist_ok=True)
        np.save(f"{_dump_dir}/{_dump_prefix}{name}.npy", np.asarray(arr))

    _dump("accum_h0_data", accum_h0.data)
    _dump("accum_h0_weight", accum_h0.weight)
    if state.pseudo_halfsets:
        _dump("accum_h1_data", accum_h1.data)
        _dump("accum_h1_weight", accum_h1.weight)
    _dump("iref_in", state.Iref[k])
    _dump("Igrad1_in_h0", state.Igrad1[half_slot_index(k, 0, state.K, state.pseudo_halfsets)])
    if state.pseudo_halfsets:
        _dump("Igrad1_in_h1", state.Igrad1[half_slot_index(k, 1, state.K, state.pseudo_halfsets)])
    _dump("Igrad2_in", state.Igrad2[k])
    _dump("fsc_halves_in", state.fsc_halves_class[k])

    # Step 2. reweightGrad per halfset
    data_h0 = np.asarray(bind.vdam_reweight_grad(accum_h0.data, accum_h0.weight, ori_size, padding_factor, 1, r_max))
    if state.pseudo_halfsets:
        data_h1 = np.asarray(
            bind.vdam_reweight_grad(accum_h1.data, accum_h1.weight, ori_size, padding_factor, 1, r_max)
        )
    else:
        data_h1 = None
    _dump("data_h0_post_reweight", data_h0)
    if data_h1 is not None:
        _dump("data_h1_post_reweight", data_h1)

    # Step 3. getFristMoment per halfset
    slot_h0 = half_slot_index(k, 0, state.K, state.pseudo_halfsets)
    new_Igrad1 = state.Igrad1.copy()

    new_Igrad1[slot_h0] = np.asarray(
        bind.vdam_first_moment(
            data_h0,
            state.Igrad1[slot_h0],
            ori_size,
            padding_factor,
            1,
            r_max,
            **{"lambda": mu_first},
        )
    )
    if state.pseudo_halfsets:
        slot_h1 = half_slot_index(k, 1, state.K, state.pseudo_halfsets)
        new_Igrad1[slot_h1] = np.asarray(
            bind.vdam_first_moment(
                data_h1,
                state.Igrad1[slot_h1],
                ori_size,
                padding_factor,
                1,
                r_max,
                **{"lambda": mu_first},
            )
        )
    replay_h0, replay_h1 = _maybe_replay_native_first_moments(
        new_Igrad1[slot_h0],
        new_Igrad1[slot_h1] if state.pseudo_halfsets else None,
        iteration=int(getattr(state, "iter", 0)),
        class_idx=k,
    )
    new_Igrad1[slot_h0] = replay_h0
    if state.pseudo_halfsets:
        new_Igrad1[slot_h1] = replay_h1
    _dump("m1_h0_post", new_Igrad1[slot_h0])
    if state.pseudo_halfsets:
        _dump("m1_h1_post", new_Igrad1[slot_h1])

    # Step 4. getSecondMoment (uses both halfset accumulators)
    new_Igrad2 = state.Igrad2.copy()
    if state.pseudo_halfsets:
        computed_Igrad2 = np.asarray(
            bind.vdam_second_moment(
                data_h0,
                data_h1,
                state.Igrad2[k],
                ori_size,
                padding_factor,
                1,
                r_max,
                **{"lambda": mu_second},
            )
        )
        new_Igrad2[k] = _maybe_replay_native_second_moment(
            computed_Igrad2,
            iteration=int(getattr(state, "iter", 0)),
            class_idx=k,
        )
        _dump("m2_computed_before_replay", computed_Igrad2)
        _dump("m2_post", new_Igrad2[k])

    # Step 5. applyMomenta. Non-halfset: pass m1 twice to trigger do_half=false.
    m1_h0 = new_Igrad1[slot_h0]
    m1_h1 = new_Igrad1[slot_h1] if state.pseudo_halfsets else m1_h0
    _post_data, mom1_noise_power = bind.vdam_apply_momenta(
        data_h0, m1_h0, m1_h1, new_Igrad2[k], ori_size, padding_factor, 1, r_max
    )
    _post_data = np.asarray(_post_data)
    mom1_noise_power = np.asarray(mom1_noise_power)
    _dump("post_apply_data", _post_data)
    _dump("mom1_noise_power", mom1_noise_power)

    # Step 6. updateSSNRarrays (update_tau2_with_fsc=false in gradient mode);
    # drives updateCurrentResolution for the next expectation step.
    new_tau2_class = state.tau2_class.copy()
    new_sigma2_class = state.sigma2_class.copy()
    new_fourier_coverage_class = state.fourier_coverage_class.copy()
    new_data_vs_prior_class = state.data_vs_prior_class.copy()
    # RELION's gradient InitialModel path routes class 0 FSC into the common
    # updateSSNRarrays call, then passes per-class FSC to reconstructGrad below.
    fsc_for_ssnr = np.asarray(state.fsc_halves_class[0], dtype=np.float64)
    # RELION calls updateSSNRarrays on BPref[iclass], not on an average with
    # the pseudo-halfset BPref[iclass + nr_classes].  The latter contributes
    # to gradient moments only.  Captured native K=1 buffers confirm that
    # BPref[0].weight matches accum_h0.weight shell-by-shell.
    weight_for_ssnr = accum_h0.weight
    tau2, sigma2, data_vs_prior, fourier_coverage = bind.vdam_update_ssnr_arrays_from_bpref(
        weight_for_ssnr,
        fsc_for_ssnr,
        state.tau2_class[k],
        tau2_fudge_factor,
        ori_size,
        padding_factor,
        1,
        r_max,
        False,
        False,
        False,
    )
    new_tau2_class[k] = np.asarray(tau2, dtype=np.float64)
    new_sigma2_class[k] = np.asarray(sigma2, dtype=np.float64)
    new_data_vs_prior_class[k] = np.asarray(data_vs_prior, dtype=np.float64)
    new_fourier_coverage_class[k] = np.asarray(fourier_coverage, dtype=np.float64)
    _dump("tau2_post_ssnr", new_tau2_class[k])
    _dump("sigma2_post_ssnr", new_sigma2_class[k])
    _dump("data_vs_prior_post_ssnr", new_data_vs_prior_class[k])
    _dump("fourier_coverage_post_ssnr", new_fourier_coverage_class[k])

    # Step 7. reconstructGrad updates Iref[k] (RELION pseudo-halfset path —
    # mom1_noise_power required for correct FSC/tau weighting). Convert
    # recovar↔RELION at the boundary so the gradient is in the same frame as the accumulators.
    from recovar.utils.helpers import recovar_volume_to_relion, relion_volume_to_recovar

    iref_relion_in = recovar_volume_to_relion(np.asarray(state.Iref[k]))
    iref_relion_in = _maybe_replay_native_reference_input(
        iref_relion_in,
        iteration=int(getattr(state, "iter", 0)),
        class_idx=k,
    )
    _dump("iref_relion_in", iref_relion_in)
    effective_stepsize = float(grad_current_stepsize) * (
        1.0 - np.exp(-float(3 * state.K + 10) * float(np.asarray(state.pdf_class)[k]))
    )
    _dump("effective_stepsize", np.asarray([effective_stepsize], dtype=np.float64))
    new_Iref = state.Iref.copy()
    new_Iref[k] = relion_volume_to_recovar(
        np.asarray(
            bind.vdam_reconstruct_grad(
                iref_relion_in,
                _post_data,
                accum_h0.weight,
                state.fsc_halves_class[k],
                effective_stepsize,
                tau2_fudge_factor,
                ori_size,
                padding_factor,
                1,
                r_max,
                min_resol_shell,
                False,
                True,
                mom1_noise_power,
            )
        )
    )
    _dump("iref_out_recovar_frame", new_Iref[k])
    _dump("iref_out_relion_frame", recovar_volume_to_relion(new_Iref[k]))

    new_state = replace(state)
    new_state.Iref = new_Iref
    new_state.Igrad1 = new_Igrad1
    new_state.Igrad2 = new_Igrad2
    new_state.tau2_class = new_tau2_class
    new_state.sigma2_class = new_sigma2_class
    new_state.data_vs_prior_class = new_data_vs_prior_class
    new_state.fourier_coverage_class = new_fourier_coverage_class
    return new_state
