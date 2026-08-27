#!/usr/bin/env python3
"""Locate the first K=1 reconstruction-stage mismatch against native RELION."""

from __future__ import annotations

import argparse
import json
import struct
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils, mask, padding
from recovar.reconstruction import regularization, relion_functions
from recovar.utils import helpers


def _load(path: Path, dtype: np.dtype) -> np.ndarray:
    raw = path.read_bytes()
    shape = struct.unpack_from("<3q", raw)
    values = np.frombuffer(raw, dtype=dtype, offset=64).copy()
    if values.size != int(np.prod(shape)):
        raise ValueError(f"payload size mismatch for {path}")
    return values.reshape(shape)


def _stage_path(
    root: Path,
    half: int,
    stage: str,
    call_index: int = 0,
    *,
    allow_any_rank: bool = False,
) -> Path:
    paths = sorted(
        root.glob(
            f"reconstruct_rank{half:02d}_pid*_call{call_index:04d}_{stage}.bin"
        )
    )
    if not paths and allow_any_rank:
        paths = sorted(
            root.glob(f"reconstruct_rank*_pid*_call{call_index:04d}_{stage}.bin")
        )
    if len(paths) != 1:
        raise ValueError(
            f"expected one half-{half} call-{call_index} {stage} dump, found {paths}"
        )
    return paths[0]


def _public_full_to_relion_half(values: np.ndarray, side: int) -> np.ndarray:
    public = np.asarray(values).reshape((side, side, side))
    centered = public.transpose(2, 1, 0)[:, :, side // 2 :]
    return np.ascontiguousarray(np.fft.ifftshift(centered, axes=(0, 1)))


def _json_scalar(value: np.generic | complex | float | int) -> object:
    value = np.asarray(value).item()
    if isinstance(value, complex):
        return {"real": float(value.real), "imag": float(value.imag)}
    return float(value)


def _metrics(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual)
    expected = np.asarray(expected)
    if actual.shape != expected.shape:
        raise ValueError(f"shape mismatch: {actual.shape} != {expected.shape}")
    delta = actual - expected
    mismatch = np.flatnonzero(actual.reshape(-1) != expected.reshape(-1))
    absolute = np.abs(delta).reshape(-1)
    first_flat = None if mismatch.size == 0 else int(mismatch[0])
    max_flat = int(np.argmax(absolute))

    def coordinate(flat_index: int | None) -> list[int] | None:
        if flat_index is None:
            return None
        return [int(value) for value in np.unravel_index(flat_index, actual.shape)]

    denominator = float(np.linalg.norm(expected.reshape(-1)))
    return {
        "count": int(expected.size),
        "bitwise_equal_count": int(np.count_nonzero(actual == expected)),
        "first_mismatch_flat_index": first_flat,
        "first_mismatch_coordinate": coordinate(first_flat),
        "first_mismatch_actual": (
            None if first_flat is None else _json_scalar(actual.reshape(-1)[first_flat])
        ),
        "first_mismatch_expected": (
            None if first_flat is None else _json_scalar(expected.reshape(-1)[first_flat])
        ),
        "max_absolute_flat_index": max_flat,
        "max_absolute_coordinate": coordinate(max_flat),
        "max_absolute_actual": _json_scalar(actual.reshape(-1)[max_flat]),
        "max_absolute_expected": _json_scalar(expected.reshape(-1)[max_flat]),
        "relative_l2": float(np.linalg.norm(delta.reshape(-1)) / denominator),
        "max_absolute": float(absolute[max_flat]),
    }


def _shell_vector_report(actual: np.ndarray, expected: np.ndarray) -> dict[str, object]:
    actual = np.asarray(actual, dtype=np.float64).reshape(-1)
    expected = np.asarray(expected, dtype=np.float64).reshape(-1)
    if actual.shape != expected.shape:
        raise ValueError(f"shell-vector shape mismatch: {actual.shape} != {expected.shape}")
    mismatch = np.flatnonzero(actual != expected)
    absolute = np.abs(actual - expected)
    relative = absolute / np.maximum(np.abs(expected), np.finfo(np.float64).tiny)
    return {
        "metrics": _metrics(actual, expected),
        "first_mismatch_shell": None if mismatch.size == 0 else int(mismatch[0]),
        "max_absolute_shell": int(np.argmax(absolute)),
        "max_relative_shell": int(np.argmax(relative)),
        "actual": actual.tolist(),
        "expected": expected.tolist(),
    }


def _native_floor(regularized: np.ndarray, radial_floor: np.ndarray, padding_factor: int) -> np.ndarray:
    side = regularized.shape[0]
    freq = np.fft.fftfreq(side) * side
    kz, iy, jx = np.meshgrid(freq, freq, np.arange(regularized.shape[2]), indexing="ij")
    shells = np.floor(np.sqrt(kz * kz + iy * iy + jx * jx) / float(padding_factor)).astype(np.int64)
    shells = np.minimum(shells, radial_floor.size - 1)
    return np.maximum(regularized, radial_floor[shells])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-accumulator", type=Path, required=True)
    parser.add_argument("--native-stage-dir", type=Path, required=True)
    parser.add_argument("--native-call-index", type=int, default=0)
    parser.add_argument(
        "--recovar-fsc",
        type=Path,
        help="Optional saved same-iteration FSC used to reconstruct RECOVAR's live tau2 input.",
    )
    parser.add_argument(
        "--native-tau-provenance",
        choices=("unknown", "live-pre-reconstruction", "post-reconstruction-model"),
        default="unknown",
        help=(
            "Lifecycle stage of the native tau2 dump. A live RECOVAR tau replay is "
            "only valid against tau captured at the native reconstruction call site."
        ),
    )
    parser.add_argument("--recovar-full-half-axis", type=int, default=0)
    parser.add_argument("--halves", type=int, choices=(1, 2), nargs="+", default=(1, 2))
    parser.add_argument(
        "--allow-any-native-rank",
        action="store_true",
        help="Accept the single rank emitted by non-MPI relion_external_reconstruct.",
    )
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()
    if args.native_call_index < 0:
        raise ValueError("native call index must be non-negative")
    if args.recovar_fsc is not None and args.native_tau_provenance != "live-pre-reconstruction":
        raise ValueError(
            "--recovar-fsc requires --native-tau-provenance=live-pre-reconstruction; "
            "a model.star tau spectrum may already contain RELION's post-reconstruction taper"
        )

    with np.load(args.recovar_accumulator, allow_pickle=False) as archive:
        grid_size = int(archive["grid_size"])
        current_size = int(archive["current_size"])
        padding_factor = int(archive["padding_factor"])
        volume_shape = tuple(int(value) for value in archive["volume_shape"])
        accumulator_shape = tuple(int(value) for value in archive["mstep_accumulator_shape"])
        numerators = [np.asarray(archive[f"Ft_y_{index}"]) for index in (0, 1)]
        weights = [np.asarray(archive[f"Ft_ctf_{index}"]) for index in (0, 1)]

    accumulator_side = accumulator_shape[0]
    if accumulator_shape != (accumulator_side,) * 3:
        raise ValueError(f"non-cubic accumulator {accumulator_shape}")
    n2 = float(grid_size**2)
    n4 = float(grid_size**4)
    wiener_radius = padding_factor * (current_size // 2)
    valid = relion_functions._relion_current_size_decenter_mask(
        accumulator_shape, wiener_radius, half_volume=False
    ).reshape(-1)
    reconstruction_shape = relion_functions._relion_reconstruction_padded_shape(
        volume_shape, padding_factor
    )
    recovar_fsc = None
    if args.recovar_fsc is not None:
        recovar_fsc = np.asarray(np.load(args.recovar_fsc, allow_pickle=False), dtype=np.float64).reshape(-1)

    halves: dict[str, object] = {}
    for half in args.halves:
        native_tau = _load(
            _stage_path(
                args.native_stage_dir,
                half,
                "tau2",
                args.native_call_index,
                allow_any_rank=args.allow_any_native_rank,
            ),
            np.dtype("<f8"),
        ).reshape(-1)
        native_weight = _load(
            _stage_path(
                args.native_stage_dir,
                half,
                "fweight_decentered",
                args.native_call_index,
                allow_any_rank=args.allow_any_native_rank,
            ),
            np.dtype("<f8"),
        )
        native_regularized = _load(
            _stage_path(
                args.native_stage_dir,
                half,
                "fweight_regularized",
                args.native_call_index,
                allow_any_rank=args.allow_any_native_rank,
            ),
            np.dtype("<f8"),
        )
        native_radial_floor = _load(
            _stage_path(
                args.native_stage_dir,
                half,
                "radavg_weight",
                args.native_call_index,
                allow_any_rank=args.allow_any_native_rank,
            ),
            np.dtype("<f8"),
        ).reshape(-1)
        native_denominator = _native_floor(native_regularized, native_radial_floor, padding_factor)
        native_divided = _load(
            _stage_path(
                args.native_stage_dir,
                half,
                "fconv_divided",
                args.native_call_index,
                allow_any_rank=args.allow_any_native_rank,
            ),
            np.dtype("<c16"),
        )
        native_before = helpers.relion_volume_to_recovar(
            _load(
                _stage_path(
                    args.native_stage_dir,
                    half,
                    "volume_before_gridding",
                    args.native_call_index,
                    allow_any_rank=args.allow_any_native_rank,
                ),
                np.dtype("<f8"),
            )
        )
        native_after = helpers.relion_volume_to_recovar(
            _load(
                _stage_path(
                    args.native_stage_dir,
                    half,
                    "volume_after_gridding",
                    args.native_call_index,
                    allow_any_rank=args.allow_any_native_rank,
                ),
                np.dtype("<f8"),
            )
        )

        weight = jnp.asarray(weights[half - 1])
        numerator = jnp.asarray(numerators[half - 1])

        def replay(
            tau_recovar: jnp.ndarray,
            *,
            tau_is_1d: bool,
        ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
            denominator = relion_functions.adjust_regularization_relion_style(
                weight.real,
                accumulator_shape,
                tau=tau_recovar,
                padding_factor=padding_factor,
                max_res_shell=current_size // 2,
                half_volume=False,
                tau2_fudge=1.0,
                minres_map=5,
                relion_native_shell_floor=True,
                native_volume_shape=volume_shape,
                tau_is_1d=tau_is_1d,
                relion_filter_scale=float(grid_size**4),
            )
            divided = numerator * valid.astype(numerator.real.dtype) / denominator
            divided_half = fourier_transform_utils.full_volume_to_half_volume(
                divided.reshape(accumulator_shape), accumulator_shape
            )
            divided_half = relion_functions._relion_window_centered_half_fourier(
                divided_half, accumulator_shape, reconstruction_shape
            )
            before = fourier_transform_utils.get_idft3_real(
                divided_half,
                volume_shape=reconstruction_shape,
            )
            before = padding.unpad_volume_spatial_domain(
                before, reconstruction_shape[0] - volume_shape[0]
            )
            before, _ = mask.soft_mask_outside_map(before, cosine_width=3)
            after, _ = relion_functions.griddingCorrect(
                before.reshape(volume_shape), volume_shape[0], padding_factor, order=1
            )
            return np.asarray(denominator), np.asarray(divided), np.asarray(before), np.asarray(after)

        tau_recovar_exact = jnp.asarray(native_tau * n4, dtype=jnp.float64)
        denominator, divided, before_np, after_np = replay(
            tau_recovar_exact, tau_is_1d=True
        )
        tau_recovar_f32 = jnp.asarray(
            np.asarray(native_tau * n4, dtype=np.float32), dtype=jnp.float64
        )
        rounded_denominator, rounded_divided, rounded_before, rounded_after = replay(
            tau_recovar_f32, tau_is_1d=True
        )

        rec_weight_native = _public_full_to_relion_half(
            np.asarray(weight * valid.astype(weight.dtype)), accumulator_side
        ) * n4
        rec_denominator_native = _public_full_to_relion_half(
            denominator, accumulator_side
        ) * n4
        rec_divided_native = _public_full_to_relion_half(
            divided, accumulator_side
        ) / (-n2)
        rec_numerator_native = _public_full_to_relion_half(
            np.asarray(numerator * valid.astype(numerator.real.dtype)), accumulator_side
        ) * (-n2)
        native_numerator = native_divided * native_denominator
        native_decenter_support = _public_full_to_relion_half(
            valid.reshape(accumulator_shape), accumulator_side
        ).astype(bool)

        live_tau_replay = None
        live_tau_float64_ablation = None
        if recovar_fsc is not None:
            live_tau, _, live_tau_details = regularization.compute_relion_tau2_from_weights(
                weight,
                weight,
                jnp.asarray(recovar_fsc),
                volume_shape,
                tau2_fudge=1.0,
                padding_factor=padding_factor,
                r_max=current_size // 2,
                return_details=True,
                full_half_axis=args.recovar_full_half_axis,
                accumulator_volume_shape=accumulator_shape,
            )
            live_denominator, live_divided, live_before, live_after = replay(
                jnp.asarray(live_tau, dtype=jnp.float64), tau_is_1d=False
            )
            live_tau_replay = {
                "tau2_prior_shells_scaled_n4_vs_native": _shell_vector_report(
                    np.asarray(live_tau_details["prior_shells"], dtype=np.float64),
                    native_tau * n4,
                ),
                "wiener_denominator_after_floor": _metrics(
                    _public_full_to_relion_half(live_denominator, accumulator_side) * n4,
                    native_denominator,
                ),
                "fconv_divided": _metrics(
                    _public_full_to_relion_half(live_divided, accumulator_side) / (-n2),
                    native_divided,
                ),
                "volume_before_gridding": _metrics(live_before, native_before),
                "volume_after_gridding": _metrics(live_after, native_after),
            }
            shell_sum = np.asarray(live_tau_details["shell_sum"], dtype=np.float64)
            shell_count = np.asarray(live_tau_details["shell_count"], dtype=np.float64)
            average_weight = np.divide(
                shell_sum,
                shell_count,
                out=np.zeros_like(shell_sum),
                where=shell_count > 0.0,
            )
            clamped_fsc = np.asarray(live_tau_details["fsc_shells"], dtype=np.float64)
            ssnr = clamped_fsc / (1.0 - clamped_fsc)
            live_tau_shells_float64 = np.divide(
                ssnr,
                float(padding_factor**3) * average_weight,
                out=np.asarray(live_tau_details["prior_shells"], dtype=np.float64).copy(),
                where=average_weight > 0.0,
            )
            f64_denominator, f64_divided, f64_before, f64_after = replay(
                jnp.asarray(live_tau_shells_float64, dtype=jnp.float64),
                tau_is_1d=True,
            )
            live_tau_float64_ablation = {
                "tau2_prior_shells_scaled_n4_vs_native": _shell_vector_report(
                    live_tau_shells_float64,
                    native_tau * n4,
                ),
                "tau2_prior_shells_vs_production_float32": _shell_vector_report(
                    live_tau_shells_float64,
                    np.asarray(live_tau_details["prior_shells"], dtype=np.float64),
                ),
                "wiener_denominator_after_floor": _metrics(
                    _public_full_to_relion_half(f64_denominator, accumulator_side) * n4,
                    native_denominator,
                ),
                "fconv_divided": _metrics(
                    _public_full_to_relion_half(f64_divided, accumulator_side) / (-n2),
                    native_divided,
                ),
                "volume_before_gridding": _metrics(f64_before, native_before),
                "volume_after_gridding": _metrics(f64_after, native_after),
            }

        halves[str(half)] = {
            "native_tau_head": native_tau[:8].tolist(),
            "native_tau_nonzero_shell_count": int(np.count_nonzero(native_tau)),
            "native_tau_last_nonzero_shell": (
                None if not np.any(native_tau != 0.0) else int(np.flatnonzero(native_tau != 0.0)[-1])
            ),
            "stages": {
                "bpref_numerator_decentered": _metrics(
                    rec_numerator_native, native_numerator
                ),
                "bpref_numerator_decentered_sign_aligned": _metrics(
                    -rec_numerator_native, native_numerator
                ),
                "fweight_decentered": _metrics(rec_weight_native, native_weight),
                "wiener_denominator_after_floor": _metrics(
                    rec_denominator_native, native_denominator
                ),
                "wiener_denominator_on_decenter_support": _metrics(
                    rec_denominator_native[native_decenter_support],
                    native_denominator[native_decenter_support],
                ),
                "fconv_divided": _metrics(rec_divided_native, native_divided),
                "fconv_divided_sign_aligned": _metrics(
                    -rec_divided_native, native_divided
                ),
                "volume_before_gridding": _metrics(before_np, native_before),
                "volume_before_gridding_sign_aligned": _metrics(
                    -before_np, native_before
                ),
                "volume_after_gridding": _metrics(after_np, native_after),
                "volume_after_gridding_sign_aligned": _metrics(
                    -after_np, native_after
                ),
            },
            "float32_rounded_tau_replay": {
                "wiener_denominator_after_floor": _metrics(
                    _public_full_to_relion_half(rounded_denominator, accumulator_side) * n4,
                    native_denominator,
                ),
                "fconv_divided": _metrics(
                    _public_full_to_relion_half(rounded_divided, accumulator_side) / (-n2),
                    native_divided,
                ),
                "volume_before_gridding": _metrics(rounded_before, native_before),
                "volume_after_gridding": _metrics(rounded_after, native_after),
            },
            "recovar_live_tau_replay": live_tau_replay,
            "recovar_live_tau_float64_ablation": live_tau_float64_ablation,
        }

    report = {
        "schema": "recovar.em.k1_reconstruction_stage_boundary.v1",
        "recovar_accumulator": str(args.recovar_accumulator.resolve()),
        "native_stage_dir": str(args.native_stage_dir.resolve()),
        "native_call_index": args.native_call_index,
        "requested_halves": list(args.halves),
        "recovar_fsc": None if args.recovar_fsc is None else str(args.recovar_fsc.resolve()),
        "native_tau_provenance": args.native_tau_provenance,
        "recovar_full_half_axis": args.recovar_full_half_axis,
        "metric_policy": "scale-sensitive relative-L2; no fitted rescaling",
        "halves": halves,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
