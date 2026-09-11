#!/usr/bin/env python
"""Re-evaluate RELION's VDAM expected-accuracy boundary from saved artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import mrcfile
import numpy as np
import starfile

from recovar.data_io.cryoem_dataset import load_dataset
from recovar.em.initial_model.driver import (
    _native_optics_state,
    read_star,
)
from recovar.em.initial_model.star_io import _micrograph_sort_order
from recovar.relion_bind import _relion_bind_core as bind


def _column(table, name: str) -> np.ndarray:
    for candidate in (name, name.removeprefix("_")):
        if candidate in table.columns:
            return np.asarray(table[candidate])
    raise KeyError(f"missing STAR column {name}")


def _scalar(table, name: str) -> float:
    if isinstance(table, dict):
        for candidate in (name, name.removeprefix("rln"), f"_{name}"):
            if candidate in table:
                return float(table[candidate])
    for candidate in (name, name.removeprefix("rln"), f"_{name}"):
        if hasattr(table, "columns") and candidate in table.columns:
            return float(np.asarray(table[candidate]).reshape(-1)[0])
    raise KeyError(f"missing STAR scalar {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-star", type=Path, required=True)
    parser.add_argument("--previous-prefix", type=Path, required=True)
    parser.add_argument("--target-prefix", type=Path, required=True)
    parser.add_argument(
        "--candidate-meta",
        type=Path,
        help="optional live iteration metadata to compare with the serialized replay",
    )
    parser.add_argument(
        "--live-inputs",
        type=Path,
        help="optional exact live binding operands captured before the iteration",
    )
    parser.add_argument("--random-seed", type=int, default=0)
    parser.add_argument("--padding-factor", type=int, default=1)
    parser.add_argument("--sigma2-fudge", type=float, default=1.0)
    parser.add_argument("--max-trials", type=int, default=100)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    input_particles, input_optics = read_star(args.input_star)
    order = _micrograph_sort_order(input_particles)
    trials = np.asarray(order[: min(args.max_trials, order.size)], dtype=np.int64)
    dataset = load_dataset(str(args.input_star), lazy=True)
    optics = _native_optics_state(input_particles, input_optics, dataset)

    previous_data, _ = read_star(f"{args.previous_prefix}_data.star")
    target_model = starfile.read(f"{args.target_prefix}_model.star", always_dict=True)
    previous_model = starfile.read(f"{args.previous_prefix}_model.star", always_dict=True)
    with mrcfile.open(f"{args.previous_prefix}_class001.mrc", permissive=True) as handle:
        reference = np.asarray(handle.data, dtype=np.float64).copy()

    eulers = np.column_stack(
        [
            _column(previous_data, "_rlnAngleRot")[: trials.size],
            _column(previous_data, "_rlnAngleTilt")[: trials.size],
            _column(previous_data, "_rlnAnglePsi")[: trials.size],
        ]
    ).astype(np.float64, copy=False)
    class_ids = _column(previous_data, "_rlnClassNumber")[: trials.size].astype(np.int32) - 1
    noise = np.asarray(
        previous_model["model_optics_group_1"]["rlnSigma2Noise"], dtype=np.float64
    )
    general = target_model["model_general"]
    current_size = int(_scalar(general, "rlnCurrentImageSize"))
    tau2_fudge = _scalar(general, "rlnTau2FudgeFactor")

    out = bind.vdam_expected_angular_errors(
        np.ascontiguousarray(reference[None]),
        np.ascontiguousarray(eulers),
        np.ascontiguousarray(trials),
        np.ascontiguousarray(class_ids),
        np.asarray([1.0], dtype=np.float64),
        np.ascontiguousarray(noise),
        np.ascontiguousarray(optics.defU, dtype=np.float64),
        np.ascontiguousarray(optics.defV, dtype=np.float64),
        np.ascontiguousarray(optics.defAngle, dtype=np.float64),
        np.ascontiguousarray(optics.phase_shift, dtype=np.float64),
        float(optics.voltage),
        float(optics.Cs),
        float(optics.Q0),
        float(optics.pixel_size),
        int(reference.shape[0]),
        current_size,
        int(args.padding_factor),
        1,
        float(args.sigma2_fudge),
        int(args.random_seed),
        True,
        False,
        np.arange(trials.size, dtype=np.int64),
    )
    classes = target_model["model_classes"]
    native_acc_rot = _scalar(classes, "rlnAccuracyRotations")
    native_acc_trans = _scalar(classes, "rlnAccuracyTranslationsAngst")
    payload = {
        "input_star": str(args.input_star),
        "previous_prefix": str(args.previous_prefix),
        "target_prefix": str(args.target_prefix),
        "trials": int(trials.size),
        "current_size": current_size,
        "tau2_fudge": tau2_fudge,
        "sigma2_fudge": float(args.sigma2_fudge),
        "computed_acc_rot": float(out["acc_rot"]),
        "computed_acc_trans_angstrom": float(out["acc_trans"]),
        "native_acc_rot": native_acc_rot,
        "native_acc_trans_angstrom": native_acc_trans,
        "acc_rot_error": float(out["acc_rot"]) - native_acc_rot,
        "acc_trans_error_angstrom": float(out["acc_trans"]) - native_acc_trans,
        "replay_source": "serialized_previous_iteration_artifacts",
    }
    if args.candidate_meta is not None:
        candidate_meta = json.loads(args.candidate_meta.read_text())
        observed_rot = float(candidate_meta["estimated_acc_rot"])
        observed_trans = float(candidate_meta["estimated_acc_trans_angstrom"])
        payload.update(
            candidate_meta=str(args.candidate_meta),
            observed_acc_rot=observed_rot,
            observed_acc_trans_angstrom=observed_trans,
            observed_acc_rot_error=observed_rot - native_acc_rot,
            observed_acc_trans_error_angstrom=observed_trans - native_acc_trans,
            observed_minus_serialized_replay_acc_rot=observed_rot - float(out["acc_rot"]),
            observed_minus_serialized_replay_acc_trans_angstrom=(
                observed_trans - float(out["acc_trans"])
            ),
        )
    if args.live_inputs is not None:
        live = np.load(args.live_inputs)
        live_seed_part_ids = (
            np.asarray(live["random_seed_particle_ids"], dtype=np.int64)
            if "random_seed_particle_ids" in live.files
            else np.arange(np.asarray(live["trial_particle_ids"]).size, dtype=np.int64)
        )
        live_out = bind.vdam_expected_angular_errors(
            np.ascontiguousarray(live["refs_relion"], dtype=np.float64),
            np.ascontiguousarray(live["eulers"], dtype=np.float64),
            np.ascontiguousarray(live["trial_particle_ids"], dtype=np.int64),
            np.ascontiguousarray(live["class_ids"], dtype=np.int32),
            np.ascontiguousarray(live["pdf_class"], dtype=np.float64),
            np.ascontiguousarray(live["sigma2_noise"], dtype=np.float64),
            np.ascontiguousarray(live["defU"], dtype=np.float64),
            np.ascontiguousarray(live["defV"], dtype=np.float64),
            np.ascontiguousarray(live["defAngle"], dtype=np.float64),
            np.ascontiguousarray(live["phase_shift"], dtype=np.float64),
            float(live["voltage"]),
            float(live["Cs"]),
            float(live["Q0"]),
            float(live["pixel_size"]),
            int(live["ori_size"]),
            int(live["current_image_size"]),
            int(live["padding_factor"]),
            1,
            float(live["sigma2_fudge"]),
            int(live["random_seed"]),
            True,
            False,
            np.ascontiguousarray(live_seed_part_ids, dtype=np.int64),
        )

        def _comparison(serialized, live_value) -> dict[str, object]:
            lhs = np.asarray(serialized)
            rhs = np.asarray(live_value)
            same_shape = lhs.shape == rhs.shape
            max_abs = None
            if same_shape and lhs.size:
                max_abs = float(
                    np.max(np.abs(lhs.astype(np.float64) - rhs.astype(np.float64)))
                )
            return {
                "same_shape": same_shape,
                "exact": bool(same_shape and np.array_equal(lhs, rhs)),
                "max_abs_error": max_abs,
            }

        serialized_operands = {
            "refs_relion": reference[None],
            "eulers": eulers,
            "trial_particle_ids": trials,
            "random_seed_particle_ids": np.arange(trials.size, dtype=np.int64),
            "class_ids": class_ids,
            "pdf_class": np.asarray([1.0], dtype=np.float64),
            "sigma2_noise": noise,
            "defU": optics.defU,
            "defV": optics.defV,
            "defAngle": optics.defAngle,
            "phase_shift": optics.phase_shift,
        }
        payload.update(
            live_inputs=str(args.live_inputs),
            live_replay_acc_rot=float(live_out["acc_rot"]),
            live_replay_acc_trans_angstrom=float(live_out["acc_trans"]),
            live_recorded_acc_rot=float(live["acc_rot"]),
            live_recorded_acc_trans_angstrom=float(live["acc_trans"]),
            live_operand_comparison={
                name: _comparison(
                    serialized,
                    live_seed_part_ids if name == "random_seed_particle_ids" else live[name],
                )
                for name, serialized in serialized_operands.items()
            },
        )
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n")
    print(rendered)


if __name__ == "__main__":
    main()
