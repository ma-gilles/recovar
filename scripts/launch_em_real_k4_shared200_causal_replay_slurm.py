#!/usr/bin/env python3
"""Seal or submit the EMPIAR-10076 K=4 shared-200 causal replay.

The launcher is deliberately fail-closed.  Before writing a launch bundle it
verifies the frozen 10k fixture, the exact 200 particles visited by both
engines at iteration 1, their frozen 93/107 half split, the RELION iteration-0
state, the iteration-1 controller state, and the capture-capable RELION
binary/source.  The default action is a dry run that writes a reviewable
manifest and sbatch script but does not submit anything.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from recovar.data_io.starfile import read_star
from recovar.em.sampling import read_relion_sampling_metadata

REPO_ROOT = Path(__file__).resolve().parents[1]
SCHEMA = "recovar.em_real_k4_shared200_causal_replay_launch.v5"
TARGET_SCHEMA = "recovar.em_real_k4_shared200_targets.v2"
SHARED_SET_SCHEMA = "recovar.em_real_kclass_shared_visited_subset.v1"

DEFAULT_RUNTIME_ROOT = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime")
DEFAULT_FIXTURE_DIR = Path("/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_fixture_20260712/data")
DEFAULT_CONTROL_PAIR_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_initialmodel_realgate_3942224f5_20260901/pair"
)
DEFAULT_SHARED_SET = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "real_k4_10076_it1_shared200_3942224f5_20260901/outputs/"
    "shared_visited_particles_it001.json"
)
DEFAULT_RELION_CAPTURE_ROOT = Path(
    "/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/"
    "relion_empty_support_capture_20260901"
)
DEFAULT_RELION_CAPTURE_SOURCE = DEFAULT_RELION_CAPTURE_ROOT / "source"
DEFAULT_RELION_CAPTURE_BINARY = DEFAULT_RELION_CAPTURE_ROOT / "build/bin/relion_refine"
DEFAULT_RELION_BIND_SOURCE = Path("/scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/src")

EXPECTED_FIXTURE_STAR_SHA256 = "2560afeea6839dddbb38b47d26cdf8944535a799d1e6d3e1441535c96043998f"
EXPECTED_SOURCE_INDICES_SHA256 = "b58a6d11fb292a9ed9573ac75c6a0673f4a0e8c216f4dadf11a7f0537b0e2c9d"
EXPECTED_PARTICLE_STACK_SHA256 = "70d0c19995221491d27c9323f21c40df27e153bdf1e783bc78c4d38fe41a9c09"
EXPECTED_PARTICLE_STACK_SIZE = 34_576_532_480
EXPECTED_PARTICLE_STACK_IMAGES = 131_879
EXPECTED_SHARED_SET_SHA256 = "581157ff693aac6f5853d335d9cd0c59aa3fc11e60f54b325feb692ff05a9bd7"
EXPECTED_PAIR_REPORT_SHA256 = "af22573582ea68dc07099686ebb09a282ad3ed82b2998c5bc962454baefdc31d"
EXPECTED_CAPTURE_RELION_BINARY_SHA256 = "2e109e842c93e34410be219db6ab0e978d4d26e52da0964fea0133d0878f0e84"
EXPECTED_CAPTURE_RELION_HEAD = "6697bf85a98297153cd57e4485c63c4381548a1c"
EXPECTED_CAPTURE_RELION_TREE = "3b940fe717ded8e109364ace1b746ab0164a0874"
CAPTURE_CONTRACT_RELATIVE_PATH = Path("src/acc/empty_support_capture_contract.h")
EXPECTED_CAPTURE_CONTRACT_TOKENS = (
    "fine_score_empty_sparse_support_v1",
    "bpref_factor_empty_sparse_support_v2",
    "fine_score_empty_support_is_well_formed_v1",
    "bpref_factor_empty_support_is_well_formed_v2",
)
EXPECTED_BIND_RELION_HEAD = "f2c1a384400aec37dc6805856a5ba645650a44f1"
EXPECTED_BIND_RELION_TREE = "1aa4902144f521ae29834e5acf382ff41cf302d0"
EXPECTED_CONTINUED_ITER0_MARKER = (
    "[RELION_CONTINUE_ITER0_PRESERVE_STATE] iter=0 skipping fresh initialiseFromImages "
    "maps=4 first_moments=8 second_moments=4 pseudo_halfsets=1"
)
EXPECTED_STOP_AFTER_LIVE_ITER_MARKER = (
    "[RELION_STOP_AFTER_LIVE_ITER] iter=1 outputs_written=1"
)


@dataclass(frozen=True)
class FrozenCase:
    dataset: str = "EMPIAR-10076"
    iteration: int = 1
    previous_iteration: int = 0
    K: int = 4
    symmetry: str = "C1"
    particle_count: int = 200
    half1_particle_count: int = 93
    half2_particle_count: int = 107
    current_size: int = 56
    healpix_order: int = 1
    random_seed: int = 0
    random_perturbation: float = -0.07990610599517822
    pixel_size_angstrom: float = 1.6375
    offset_range_pixels: float = 6.0
    offset_step_pixels: float = 2.0
    image_batch_size: int = 50
    rotation_block_size: int = 5000


CASE = FrozenCase()
THRESHOLDS: dict[str, float] = {
    "minimum_candidate_tuple_exact_fraction": 1.0,
    "maximum_centered_raw_score_relative_l2": 5.0e-5,
    "maximum_centered_combined_score_relative_l2": 5.0e-5,
    "maximum_posterior_relative_l2": 1.0e-4,
    "maximum_posterior_row_sum_abs_error": 1.0e-7,
    "minimum_support_jaccard": 1.0,
    "minimum_winner_agreement": 0.995,
    "maximum_pmax_rmse": 1.0e-4,
    "maximum_pmax_abs_error": 1.0e-3,
    "minimum_assignment_accuracy": 0.995,
    "minimum_per_class_map_fsc_auc": 0.999,
    "minimum_native_control_map_fsc_auc": 0.999999,
    "minimum_capture_inertness_map_fsc_auc": 0.999999,
    # A continuation reload is not bitwise identical to the original fresh
    # iteration. The sealed 200-particle discriminator measured 199/200
    # assignments and per-class FSC-AUC >= 0.9995 against that frozen target.
    # Repeat/capture inertness remain strict above.
    "minimum_frozen_target_map_fsc_auc": 0.999,
    "minimum_native_assignment_inertness": 1.0,
    "minimum_frozen_target_assignment_accuracy": 0.995,
    "minimum_class_fraction": 0.01,
}


class PreflightError(RuntimeError):
    """Raised when frozen launch provenance or topology has drifted."""


_SHA256_CACHE: dict[tuple[str, int, int], str] = {}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise PreflightError(message)


def _sha256(path: Path) -> str:
    path = path.resolve()
    stat = path.stat()
    cache_key = (str(path), stat.st_size, stat.st_mtime_ns)
    cached = _SHA256_CACHE.get(cache_key)
    if cached is not None:
        return cached
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    value = digest.hexdigest()
    _SHA256_CACHE[cache_key] = value
    return value


def _git_text(root: Path, *args: str) -> str:
    try:
        return subprocess.check_output(["git", *args], cwd=root, text=True).strip()
    except subprocess.CalledProcessError as exc:
        raise PreflightError(f"cannot resolve Git provenance in {root}") from exc


def _column(table, *names: str) -> str:
    for name in names:
        if name in table.columns:
            return name
        alternate = name[1:] if name.startswith("_") else f"_{name}"
        if alternate in table.columns:
            return alternate
    raise PreflightError(f"STAR table lacks all required columns {names}")


def _stack_index(image_name: object) -> int:
    text = str(image_name)
    prefix, separator, _ = text.partition("@")
    _require(bool(separator), f"invalid RELION image identity: {text!r}")
    try:
        value = int(prefix)
    except ValueError as exc:
        raise PreflightError(f"invalid RELION stack index: {text!r}") from exc
    _require(value > 0, f"RELION stack index must be positive: {text!r}")
    return value


def _ordered_text_sha256(values: Iterable[str]) -> str:
    payload = "".join(f"{value}\n" for value in values).encode()
    return hashlib.sha256(payload).hexdigest()


def _file_record(path: Path, *, role: str) -> dict[str, Any]:
    _require(path.is_file(), f"missing {role}: {path}")
    return {
        "role": role,
        "path": str(path.resolve()),
        "size_bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _validate_record(record: dict[str, Any]) -> None:
    path = Path(str(record.get("path", "")))
    _require(path.is_absolute() and path.is_file(), f"manifest input is missing: {path}")
    _require(path.stat().st_size == int(record.get("size_bytes", -1)), f"input size drift: {path}")
    observed = _sha256(path)
    _require(observed == record.get("sha256"), f"input checksum drift: {path}: {observed}")


def _validate_capture_binary_mode(path: Path) -> None:
    _require(
        path.name == "relion_refine",
        "gradient InitialModel replay requires the non-MPI relion_refine executable",
    )


def _star_scalar(text: str, label: str) -> str:
    match = re.search(rf"(?m)^{re.escape(label)}\s+(\S+)\s*$", text)
    _require(match is not None, f"STAR scalar is missing: {label}")
    return match.group(1)


def _replace_star_scalar(text: str, label: str, value: object) -> str:
    pattern = re.compile(rf"(?m)^({re.escape(label)}\s+)\S+(\s*)$")
    updated, count = pattern.subn(rf"\g<1>{value}\g<2>", text)
    _require(count == 1, f"expected exactly one {label}, observed {count}")
    return updated


def materialize_iteration0_continuation_bundle(
    *,
    source_optimiser: Path,
    source_sampling: Path,
    output_dir: Path,
) -> tuple[Path, Path, dict[str, Any]]:
    """Create the missing, pre-initialisation RELION iteration-0 sampling state.

    RELION deliberately writes the iteration-0 optimiser without its referenced
    sampling STAR.  Continuing from that optimiser is therefore unsupported
    unless the pre-initialisation sampling state is reconstructed.  The saved
    iteration-1 sampling file proves the exact topology and Angstrom-valued
    geometry.  For an iteration-0 continuation, however, ``initialiseGeneral``
    multiplies the current offset range and step by the model pixel size.  We
    therefore seal the corresponding command-line pixel values here.  RELION's
    ``HealpixSampling::read`` does not restore ``random_perturbation``; the live
    iteration-1 value is independently forced and checked by the capture binary.
    """

    source_metadata = read_relion_sampling_metadata(source_sampling)
    expected_range_angstrom = CASE.offset_range_pixels * CASE.pixel_size_angstrom
    expected_step_angstrom = CASE.offset_step_pixels * CASE.pixel_size_angstrom
    _require(source_metadata["healpix_order"] == CASE.healpix_order, "source sampling order drift")
    _require(
        np.isclose(source_metadata["offset_range"], expected_range_angstrom, rtol=0.0, atol=1.0e-7),
        "source sampling offset range drift",
    )
    _require(
        np.isclose(source_metadata["offset_step"], expected_step_angstrom, rtol=0.0, atol=1.0e-7),
        "source sampling offset step drift",
    )
    _require(
        np.isclose(source_metadata["perturbation_factor"], 0.5, rtol=0.0, atol=1.0e-12),
        "source sampling perturbation factor drift",
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    derived_sampling = output_dir / "run_it000_sampling_replay.star"
    sampling_text = source_sampling.read_text()
    _require(_star_scalar(sampling_text, "_rlnSymmetryGroup").upper() == CASE.symmetry, "sampling symmetry drift")
    replacements = {
        "_rlnOffsetRange": f"{CASE.offset_range_pixels:.6f}",
        "_rlnOffsetStep": f"{CASE.offset_step_pixels:.6f}",
        "_rlnOffsetRangeOriginal": f"{CASE.offset_range_pixels:.6f}",
        "_rlnOffsetStepOriginal": f"{CASE.offset_step_pixels:.6f}",
        "_rlnSamplingPerturbInstance": "0.000000",
    }
    for label, value in replacements.items():
        sampling_text = _replace_star_scalar(sampling_text, label, value)
    derived_sampling.write_text(sampling_text)
    derived_metadata = read_relion_sampling_metadata(derived_sampling)
    _require(derived_metadata["offset_range"] == CASE.offset_range_pixels, "derived offset range drift")
    _require(derived_metadata["offset_step"] == CASE.offset_step_pixels, "derived offset step drift")
    _require(derived_metadata["random_perturbation"] == 0.0, "derived perturbation must be clear")

    optimiser_text = source_optimiser.read_text()
    _require(
        _star_scalar(optimiser_text, "_rlnDoGradientRefine") == "1",
        "iteration-0 optimiser is no longer a gradient InitialModel state",
    )
    source_sampling_reference = Path(_star_scalar(optimiser_text, "_rlnOrientSamplingStarFile"))
    _require(
        source_sampling_reference.name == "run_it000_sampling.star" and not source_sampling_reference.exists(),
        "iteration-0 optimiser no longer has the expected missing sampling boundary",
    )
    expected_pair_dir = source_optimiser.parent.resolve()
    for label, filename in (
        ("_rlnModelStarFile", "run_it000_model.star"),
        ("_rlnExperimentalDataStarFile", "run_it000_data.star"),
    ):
        dependency = Path(_star_scalar(optimiser_text, label))
        _require(dependency.resolve() == expected_pair_dir / filename, f"iteration-0 dependency drift: {label}")
        _require(dependency.is_file(), f"iteration-0 dependency is missing: {dependency}")
    derived_optimiser = output_dir / "run_it000_optimiser_replay.star"
    optimiser_text = _replace_star_scalar(
        optimiser_text,
        "_rlnOrientSamplingStarFile",
        derived_sampling.resolve(),
    )
    derived_optimiser.write_text(optimiser_text)
    _require(
        Path(_star_scalar(derived_optimiser.read_text(), "_rlnOrientSamplingStarFile")).resolve()
        == derived_sampling.resolve(),
        "derived optimiser sampling reference drift",
    )
    provenance = {
        "method": "iteration-1 topology with iteration-0 pre-initialisation pixel offsets",
        "source_sampling": _file_record(source_sampling, role="RELION iteration-1 sampling topology source"),
        "source_optimiser": _file_record(source_optimiser, role="RELION iteration-0 optimiser source"),
        "derived_sampling": _file_record(derived_sampling, role="sealed RELION iteration-0 replay sampling"),
        "derived_optimiser": _file_record(derived_optimiser, role="sealed RELION iteration-0 replay optimiser"),
        "source_offset_range_angstrom": source_metadata["offset_range"],
        "source_offset_step_angstrom": source_metadata["offset_step"],
        "derived_offset_range_pixels": derived_metadata["offset_range"],
        "derived_offset_step_pixels": derived_metadata["offset_step"],
        "live_iteration_1_perturbation_override": CASE.random_perturbation,
    }
    return derived_optimiser, derived_sampling, provenance


def _assigned_stack_indices(path: Path) -> set[int]:
    particles, _ = read_star(str(path))
    image_column = _column(particles, "rlnImageName")
    class_column = _column(particles, "rlnClassNumber")
    images = [_stack_index(value) for value in particles[image_column]]
    _require(len(images) == len(set(images)), f"duplicate image identities in {path}")
    labels = np.asarray(particles[class_column], dtype=np.int64)
    _require(np.all((labels >= 0) & (labels <= CASE.K)), f"invalid class labels in {path}")
    return {image for image, label in zip(images, labels, strict=True) if label > 0}


def _shared_target_rows(
    *,
    fixture_star: Path,
    shared_set_path: Path,
    relion_it1_data: Path,
    recovar_it1_data: Path,
) -> tuple[Any, Any, dict[str, Any]]:
    shared = json.loads(shared_set_path.read_text())
    _require(shared.get("schema") == SHARED_SET_SCHEMA, "shared-set schema drift")
    _require(shared.get("iteration") == CASE.iteration, "shared-set iteration drift")
    _require(shared.get("same_visited_particle_ids") is True, "engines did not visit the same particles")
    _require(shared.get("relion_assigned_count") == CASE.particle_count, "RELION shared count drift")
    _require(shared.get("recovar_assigned_count") == CASE.particle_count, "RECOVAR shared count drift")
    identities = shared.get("visited_particle_ids")
    _require(isinstance(identities, list), "shared particle identities are not a list")
    target_stack_indices = [_stack_index(value) for value in identities]
    _require(len(target_stack_indices) == CASE.particle_count, "shared target count drift")
    _require(len(set(target_stack_indices)) == CASE.particle_count, "shared targets are not unique")

    particles, optics = read_star(str(fixture_star))
    image_column = _column(particles, "rlnImageName")
    half_column = _column(particles, "rlnRandomSubset")
    source_stack_indices = [_stack_index(value) for value in particles[image_column]]
    _require(len(source_stack_indices) == len(set(source_stack_indices)), "fixture image identities are not unique")
    source_rows = {stack_index: row for row, stack_index in enumerate(source_stack_indices)}
    missing = sorted(set(target_stack_indices) - set(source_rows))
    _require(not missing, f"shared targets are absent from the fixture: {missing[:10]}")
    selected_rows = np.asarray([source_rows[value] for value in target_stack_indices], dtype=np.int64)
    selected = particles.iloc[selected_rows].copy().reset_index(drop=True)
    halves = np.asarray(selected[half_column], dtype=np.int64)
    _require(np.all(np.isin(halves, (1, 2))), "shared target half labels must be 1 or 2")
    half_counts = {str(half): int(np.count_nonzero(halves == half)) for half in (1, 2)}
    _require(
        half_counts == {"1": CASE.half1_particle_count, "2": CASE.half2_particle_count},
        f"shared target half split drift: {half_counts}",
    )

    expected = set(target_stack_indices)
    _require(_assigned_stack_indices(relion_it1_data) == expected, "RELION assigned target set drift")
    _require(_assigned_stack_indices(recovar_it1_data) == expected, "RECOVAR assigned target set drift")
    target_record = {
        "schema": TARGET_SCHEMA,
        "dataset": CASE.dataset,
        "iteration": CASE.iteration,
        "particle_count": CASE.particle_count,
        "half_counts": half_counts,
        "ordering": "lexicographic rlnImageName order from frozen shared-set artifact",
        "image_identities": [str(value) for value in identities],
        "stack_indices_one_based": target_stack_indices,
        "original_indices_zero_based": [value - 1 for value in target_stack_indices],
        "subset_local_indices_zero_based": list(range(CASE.particle_count)),
        "ordered_image_identity_sha256": _ordered_text_sha256(str(value) for value in identities),
        "ordered_identity_half_sha256": _ordered_text_sha256(
            f"{identity}\t{half}" for identity, half in zip(identities, halves.tolist(), strict=True)
        ),
        "half_count_derivation": (
            "count rlnRandomSubset values after exact ordered rlnImageName join from "
            "the frozen shared-set artifact into the frozen fixture STAR"
        ),
        "lineage": {
            "fixture_star_sha256": EXPECTED_FIXTURE_STAR_SHA256,
            "shared_set_sha256": EXPECTED_SHARED_SET_SHA256,
        },
    }
    return selected, optics, target_record


def _write_star_block(stream, name: str, table) -> None:
    stream.write(f"{name}\n\nloop_\n")
    for column in table.columns:
        stream.write(f"{column}\n")
    for values in table.itertuples(index=False, name=None):
        tokens = [str(value) for value in values]
        _require(not any(any(char.isspace() for char in token) for token in tokens), "STAR value contains whitespace")
        stream.write(" ".join(tokens) + "\n")


def write_deterministic_subset_star(
    *,
    output: Path,
    selected,
    optics,
    particle_stack: Path,
) -> None:
    image_column = _column(selected, "rlnImageName")
    stack_path = particle_stack.resolve()
    selected = selected.copy()
    selected[image_column] = [f"{_stack_index(value)}@{stack_path}" for value in selected[image_column]]
    with output.open("w") as stream:
        stream.write("# Frozen EMPIAR-10076 shared-200 causal replay input\n\n")
        if optics is not None:
            _write_star_block(stream, "data_optics", optics)
            stream.write("\n")
            _write_star_block(stream, "data_particles", selected)
        else:
            _write_star_block(stream, "data_", selected)


def write_fixed_image_identity_mapping(
    *,
    output: Path,
    particles,
    particle_stack: Path,
    stack_image_count: int = EXPECTED_PARTICLE_STACK_IMAGES,
) -> None:
    """Seal the full fixture's stack identities without object arrays/pickle."""

    image_column = _column(particles, "rlnImageName")
    stack_indices = np.asarray([_stack_index(value) for value in particles[image_column]], dtype=np.int64)
    _require(
        len(np.unique(stack_indices)) == len(stack_indices),
        "fixture image identities must have unique physical-stack indices",
    )
    _require(
        stack_image_count > 0
        and (stack_indices.size == 0 or (int(stack_indices.min()) > 0 and int(stack_indices.max()) <= stack_image_count)),
        "fixture image identity lies outside the frozen physical-stack extent",
    )
    absolute_stack = particle_stack.resolve()
    selected_identities = [f"{index}@{absolute_stack}" for index in stack_indices.tolist()]
    width = max(map(len, selected_identities), default=1)
    identities = np.full(stack_image_count, b"", dtype=f"S{width}")
    identities[stack_indices - 1] = np.asarray(selected_identities, dtype=f"S{width}")
    np.save(output, identities, allow_pickle=False)


def write_subset_local_image_identity_mapping(
    *,
    output: Path,
    selected,
    particle_stack: Path,
) -> None:
    """Seal ordered subset-local rows to immutable physical-stack identities."""

    image_column = _column(selected, "rlnImageName")
    stack_indices = [_stack_index(value) for value in selected[image_column]]
    _require(len(stack_indices) == len(set(stack_indices)), "subset image identities must be unique")
    absolute_stack = particle_stack.resolve()
    identities = [f"{index}@{absolute_stack}" for index in stack_indices]
    width = max(map(len, identities), default=1)
    np.save(output, np.asarray(identities, dtype=f"S{width}"), allow_pickle=False)


def _fixed_case_record() -> dict[str, Any]:
    return {
        "dataset": CASE.dataset,
        "iteration": CASE.iteration,
        "previous_iteration": CASE.previous_iteration,
        "K": CASE.K,
        "symmetry": CASE.symmetry,
        "particle_count": CASE.particle_count,
        "half1_particle_count": CASE.half1_particle_count,
        "half2_particle_count": CASE.half2_particle_count,
        "current_size": CASE.current_size,
        "healpix_order": CASE.healpix_order,
        "random_seed": CASE.random_seed,
        "random_perturbation": CASE.random_perturbation,
        "pixel_size_angstrom": CASE.pixel_size_angstrom,
        "offset_range_pixels": CASE.offset_range_pixels,
        "offset_step_pixels": CASE.offset_step_pixels,
        "image_batch_size": CASE.image_batch_size,
        "rotation_block_size": CASE.rotation_block_size,
    }


def _validate_controller_state(control_pair_root: Path) -> dict[str, Any]:
    meta_path = control_pair_root / "recovar/run_it001_recovar_meta.json"
    meta = json.loads(meta_path.read_text())
    checks = {
        "current_size": int(meta.get("current_size", -1)) == CASE.current_size,
        "healpix_order": int(meta.get("healpix_order", -1)) == CASE.healpix_order,
        "random_perturbation": float(meta.get("random_perturbation", np.nan)) == CASE.random_perturbation,
    }
    _require(all(checks.values()), f"iteration-1 controller state drift: {checks}")
    return {
        "checks": checks,
        "meta_path": str(meta_path.resolve()),
        "meta_sha256": _sha256(meta_path),
    }


def _source_provenance(source_dir: Path) -> dict[str, Any]:
    _require((source_dir / "src/acc/acc_ml_optimiser_impl.h").is_file(), "capture source lacks scorer")
    contract_header = source_dir / CAPTURE_CONTRACT_RELATIVE_PATH
    _require(contract_header.is_file(), "capture source lacks empty-support contract header")
    _require((source_dir / "src/ml_optimiser.cpp").is_file(), "capture source lacks optimiser")
    _require((source_dir / "src/ml_optimiser.h").is_file(), "capture source lacks optimiser header")
    status = _git_text(source_dir, "status", "--porcelain", "--untracked-files=no")
    _require(not status, f"capture RELION source has tracked changes:\n{status}")
    scorer_text = (source_dir / "src/acc/acc_ml_optimiser_impl.h").read_text(errors="replace")
    contract_text = contract_header.read_text(errors="replace")
    optimiser_text = (source_dir / "src/ml_optimiser.cpp").read_text(errors="replace")
    optimiser_header_text = (source_dir / "src/ml_optimiser.h").read_text(errors="replace")
    for token, text in (
        ("RELION_BPRE_CAPTURE_STACKS", scorer_text),
        ("RELION_FINE_SCORE_CAPTURE_CLASSES", scorer_text),
        ('#include "src/acc/empty_support_capture_contract.h"', scorer_text),
        ("RELION_SAMPLING_PERTURBATION_OVERRIDE", optimiser_text),
        ("RELION_CONTINUE_ITER0_PRESERVE_STATE", optimiser_header_text),
    ):
        _require(token in text, f"capture RELION source lacks {token}")
    for token in EXPECTED_CAPTURE_CONTRACT_TOKENS:
        _require(token in contract_text, f"capture RELION contract lacks {token}")
    provenance = {
        "root": str(Path(_git_text(source_dir, "rev-parse", "--show-toplevel")).resolve()),
        "git_head": _git_text(source_dir, "rev-parse", "HEAD"),
        "git_tree": _git_text(source_dir, "rev-parse", "HEAD^{tree}"),
        "tracked_dirty": False,
        "scorer_sha256": _sha256(source_dir / "src/acc/acc_ml_optimiser_impl.h"),
        "empty_support_contract_sha256": _sha256(contract_header),
        "optimiser_sha256": _sha256(source_dir / "src/ml_optimiser.cpp"),
        "optimiser_header_sha256": _sha256(source_dir / "src/ml_optimiser.h"),
    }
    _require(provenance["git_head"] == EXPECTED_CAPTURE_RELION_HEAD, "capture RELION head drift")
    _require(provenance["git_tree"] == EXPECTED_CAPTURE_RELION_TREE, "capture RELION tree drift")
    return provenance


def _bind_source_provenance(source_dir: Path) -> dict[str, Any]:
    root = Path(_git_text(source_dir, "rev-parse", "--show-toplevel")).resolve()
    status = _git_text(root, "status", "--porcelain", "--untracked-files=no")
    _require(not status, f"binding RELION source has tracked changes:\n{status}")
    provenance = {
        "root": str(root),
        "git_head": _git_text(root, "rev-parse", "HEAD"),
        "git_tree": _git_text(root, "rev-parse", "HEAD^{tree}"),
        "tracked_dirty": False,
    }
    _require(provenance["git_head"] == EXPECTED_BIND_RELION_HEAD, "binding RELION head drift")
    _require(provenance["git_tree"] == EXPECTED_BIND_RELION_TREE, "binding RELION tree drift")
    return provenance


def _input_paths(args: argparse.Namespace) -> list[tuple[str, Path]]:
    pair = args.control_pair_root
    paths: list[tuple[str, Path]] = [
        ("fixture manifest", args.fixture_dir / "fixture_manifest.json"),
        ("fixture STAR", args.fixture_dir / "particles.star"),
        ("fixture indices", args.fixture_dir / "source_indices.npy"),
        ("fixture particle stack", args.fixture_dir / "particles.256.mrcs"),
        ("shared-set artifact", args.shared_set),
        ("control pair report", pair / "pair_report.json"),
        ("RELION iteration-0 optimiser", pair / "relion/run_it000_optimiser.star"),
        ("RELION iteration-0 model", pair / "relion/run_it000_model.star"),
        ("RELION iteration-0 data", pair / "relion/run_it000_data.star"),
        ("RELION iteration-1 model", pair / "relion/run_it001_model.star"),
        ("RELION iteration-1 data", pair / "relion/run_it001_data.star"),
        ("RELION iteration-1 sampling", pair / "relion/run_it001_sampling.star"),
        ("RECOVAR iteration-1 data", pair / "recovar/run_it001_data.star"),
        ("RECOVAR iteration-1 metadata", pair / "recovar/run_it001_recovar_meta.json"),
        ("capture RELION binary", args.relion_capture_binary),
        ("RELION binding projector", args.relion_bind_source / "projector.h"),
        ("pixi Python", args.pixi_python),
    ]
    for class_one_based in range(1, CASE.K + 1):
        paths.extend(
            (
                (
                    f"RELION initial class {class_one_based}",
                    pair / f"relion/run_it000_class{class_one_based:03d}.mrc",
                ),
                (
                    f"RELION target class {class_one_based}",
                    pair / f"relion/run_it001_class{class_one_based:03d}.mrc",
                ),
                (
                    f"RELION initial first moment {class_one_based}",
                    pair / f"relion/run_it000_1moment{class_one_based:03d}.mrc",
                ),
                (
                    f"RELION initial second moment {class_one_based}",
                    pair / f"relion/run_it000_2moment{class_one_based:03d}.mrc",
                ),
            )
        )
    return paths


def _validate_iteration0_dependency_closure(args: argparse.Namespace) -> None:
    pair = args.control_pair_root / "relion"
    model_text = (pair / "run_it000_model.star").read_text()
    observed_model_files = {
        Path(token).resolve()
        for token in re.findall(r"(?<!@)(/\S+\.mrc)(?=\s|$)", model_text)
    }
    expected_model_files = {
        (pair / f"run_it000_{kind}{class_id:03d}.mrc").resolve()
        for class_id in range(1, CASE.K + 1)
        for kind in ("class", "1moment", "2moment")
    }
    _require(
        observed_model_files == expected_model_files,
        "RELION iteration-0 model dependency closure drift",
    )
    _require(all(path.is_file() for path in observed_model_files), "RELION iteration-0 model dependency is missing")

    data_text = (pair / "run_it000_data.star").read_text()
    stack_references = set(re.findall(r"\d+@(\S+\.mrcs)(?=\s|$)", data_text))
    _require(stack_references == {"particles.256.mrcs"}, "RELION iteration-0 particle-stack closure drift")
    _require((args.fixture_dir / "particles.256.mrcs").is_file(), "RELION iteration-0 particle stack is missing")


def preflight(args: argparse.Namespace) -> dict[str, Any]:
    for path in (args.fixture_dir, args.control_pair_root, args.relion_capture_source, args.relion_bind_source):
        _require(path.is_dir(), f"required directory is missing: {path}")
    for _, path in _input_paths(args):
        _require(path.is_file(), f"required input is missing: {path}")
    _require(os.access(args.relion_capture_binary, os.X_OK), "capture RELION binary is not executable")
    _validate_capture_binary_mode(args.relion_capture_binary)
    _require(_sha256(args.fixture_dir / "particles.star") == EXPECTED_FIXTURE_STAR_SHA256, "fixture STAR drift")
    _require(
        _sha256(args.fixture_dir / "source_indices.npy") == EXPECTED_SOURCE_INDICES_SHA256,
        "fixture source indices drift",
    )
    stack = args.fixture_dir / "particles.256.mrcs"
    _require(stack.stat().st_size == EXPECTED_PARTICLE_STACK_SIZE, "fixture particle stack size drift")
    _require(_sha256(stack) == EXPECTED_PARTICLE_STACK_SHA256, "fixture particle stack drift")
    _require(_sha256(args.shared_set) == EXPECTED_SHARED_SET_SHA256, "shared-set artifact drift")
    _require(
        _sha256(args.control_pair_root / "pair_report.json") == EXPECTED_PAIR_REPORT_SHA256,
        "control pair report drift",
    )
    _require(
        _sha256(args.relion_capture_binary) == EXPECTED_CAPTURE_RELION_BINARY_SHA256,
        "capture RELION binary drift",
    )
    _validate_iteration0_dependency_closure(args)

    selected, optics, target_record = _shared_target_rows(
        fixture_star=args.fixture_dir / "particles.star",
        shared_set_path=args.shared_set,
        relion_it1_data=args.control_pair_root / "relion/run_it001_data.star",
        recovar_it1_data=args.control_pair_root / "recovar/run_it001_data.star",
    )
    controller = _validate_controller_state(args.control_pair_root)
    source = _source_provenance(args.relion_capture_source)
    bind_source = _bind_source_provenance(args.relion_bind_source)
    repo_status = _git_text(REPO_ROOT, "status", "--porcelain", "--untracked-files=all")
    _require(not repo_status, f"launcher requires a clean RECOVAR source tree:\n{repo_status}")
    return {
        "selected": selected,
        "optics": optics,
        "target_record": target_record,
        "controller": controller,
        "relion_capture_source": source,
        "relion_bind_source": bind_source,
        "repo_head": _git_text(REPO_ROOT, "rev-parse", "HEAD"),
        "repo_tree": _git_text(REPO_ROOT, "rev-parse", "HEAD^{tree}"),
    }


def _quote(value: object) -> str:
    return shlex.quote(str(value))


def render_sbatch(args: argparse.Namespace, *, expected_head: str, manifest_path: Path) -> str:
    run_root = args.output_root
    runtime_prefix = args.runtime_root / f"real_k4_shared200_{run_root.name}"
    pair = args.control_pair_root
    venv = run_root / "build/venv"
    python = venv / "bin/python"
    bind_dir = run_root / "build/relion_bind"
    cuda_lib = run_root / "build/cuda/libcuda_backproject.so"
    subset_star = run_root / "inputs/particles_shared200.star"
    targets = run_root / "inputs/frozen_targets.json"
    replay_optimiser = run_root / "inputs/continuation/run_it000_optimiser_replay.star"
    image_names_mapping = run_root / "inputs/image_names_subset_local.npy"
    target_loader = (
        "import json,pathlib; p=json.loads(pathlib.Path(" + repr(str(targets)) + ").read_text()); "
        "print(','.join(map(str,p['stack_indices_one_based'])))"
    )
    recovar_local_loader = (
        "import json,pathlib; p=json.loads(pathlib.Path(" + repr(str(targets)) + ").read_text()); "
        "print(','.join(map(str,p['subset_local_indices_zero_based'])))"
    )
    recovar_command = [
        str(python),
        "-m",
        "scripts.run_k_class_parity",
        "--relion-dir",
        str(pair / "relion"),
        "--data-star",
        str(subset_star),
        "--prev-iter",
        "0",
        "--target-iter",
        "1",
        "--output-dir",
        str(run_root / "recovar/output"),
        "--image-batch-size",
        str(CASE.image_batch_size),
        "--rotation-block-size",
        str(CASE.rotation_block_size),
        "--adaptive-2pass",
        "--adaptive-oversampling",
        "1",
        "--sparse-pass2",
        "--accumulate-noise",
        "--radial-window",
        "--relion-x-half-mstep",
        "--image-fourier-backend",
        "relion_cuda",
    ]
    continuation = " \\\n  "
    recovar_text = continuation.join(_quote(value) for value in recovar_command)
    native_smoke_exit = ""
    if args.native_smoke_only:
        native_smoke_exit = f"""test -s "${{ROOT}}/native/control_a/output/run_it001_sampling.star"
{_quote(python)} -m scripts.audit_em_real_kclass_initialmodel \\
  --candidate-dir "${{ROOT}}/native/control_a/output" \\
  --reference-dir {_quote(pair / "relion")} \\
  --K {CASE.K} --checkpoint 1 \\
  --minimum-fsc-auc {THRESHOLDS["minimum_frozen_target_map_fsc_auc"]} \\
  --minimum-assignment-accuracy {THRESHOLDS["minimum_frozen_target_assignment_accuracy"]} \\
  --minimum-class-fraction {THRESHOLDS["minimum_class_fraction"]} \\
  --output-json "${{ROOT}}/analysis/native_smoke_trajectory.json" \\
  --output-shells-npz "${{ROOT}}/analysis/native_smoke_shells.npz"
find "${{ROOT}}/native/control_a" -type f -print0 \\
  | sort -z | xargs -0 sha256sum > "${{ROOT}}/provenance/science_outputs_${{SLURM_JOB_ID}}.sha256"
exit 0
"""
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=real-k4-shared200
#SBATCH --output={_quote(run_root / "logs/replay-%j.out")}
#SBATCH --error={_quote(run_root / "logs/replay-%j.err")}
#SBATCH --partition={args.partition}
#SBATCH --account={args.account}
#SBATCH --constraint={args.constraint}
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem={args.mem}
#SBATCH --time={args.time_limit}

set -euo pipefail
: "${{EXPECTED_MANIFEST_SHA256:?submission must export EXPECTED_MANIFEST_SHA256}}"
ROOT={_quote(run_root)}
MANIFEST={_quote(manifest_path)}
EXPECTED_HEAD={_quote(expected_head)}
RUNTIME_ROOT={_quote(runtime_prefix)}_${{SLURM_JOB_ID}}
export TMPDIR="${{RUNTIME_ROOT}}/tmp"
export PIXI_HOME="${{RUNTIME_ROOT}}/pixi_home"
export RATTLER_CACHE_DIR="${{RUNTIME_ROOT}}/rattler_cache"
mkdir -p "${{TMPDIR}}" "${{PIXI_HOME}}" "${{RATTLER_CACHE_DIR}}"
touch "${{RUNTIME_ROOT}}/SAFE_TO_DELETE"

cd {_quote(REPO_ROOT)}
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV JAX_PLATFORM_NAME
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER CONDA_SHLVL
while IFS='=' read -r variable_name _; do
  case "${{variable_name}}" in RECOVAR_*|RELION_*|JAX_*|XLA_*) unset "${{variable_name}}" ;; esac
done < <(env)
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export OMP_NUM_THREADS=8

test "$(git rev-parse HEAD)" = "${{EXPECTED_HEAD}}"
test -z "$(git status --short --untracked-files=all)"
test "$(sha256sum "${{MANIFEST}}" | awk '{{print $1}}')" = "${{EXPECTED_MANIFEST_SHA256}}"
{_quote(args.pixi_python)} -m scripts.launch_em_real_k4_shared200_causal_replay_slurm \
  --validate-manifest "${{MANIFEST}}"

if [[ -f /etc/profile.d/modules.sh ]]; then source /etc/profile.d/modules.sh; fi
set +u
module purge
module load {_quote(args.cuda_module)}
set -u
export CUDA_HOME=/usr/local/cuda-12.8
export PATH="${{CUDA_HOME}}/bin:${{PATH}}"
CUDA_TARGET_LIB_DIR="${{CUDA_HOME}}/targets/x86_64-linux/lib"
PIXI_ENV_ROOT={_quote(args.pixi_python.parent.parent)}
PIXI_NVIDIA_ROOT="$(find "${{PIXI_ENV_ROOT}}/lib" -maxdepth 4 -type d -path '*/site-packages/nvidia' -print -quit)"
test -d "${{CUDA_TARGET_LIB_DIR}}"
test -n "${{PIXI_NVIDIA_ROOT}}" && test -d "${{PIXI_NVIDIA_ROOT}}"
PIXI_NVIDIA_LIB_DIRS="$(find "${{PIXI_NVIDIA_ROOT}}" -type d -name lib -print | sort | paste -sd: -)"
test -n "${{PIXI_NVIDIA_LIB_DIRS}}"
test -n "$(find "${{PIXI_NVIDIA_ROOT}}" -type f -name 'libcusparse.so*' -print -quit)"
export LD_LIBRARY_PATH="${{PIXI_NVIDIA_LIB_DIRS}}:${{CUDA_TARGET_LIB_DIR}}:${{PIXI_ENV_ROOT}}/lib:${{LD_LIBRARY_PATH:-}}"
test -f "${{PIXI_ENV_ROOT}}/include/fftw/fftw3.h"
export CMAKE_INCLUDE_PATH="${{PIXI_ENV_ROOT}}/include/fftw:${{PIXI_ENV_ROOT}}/include:${{CMAKE_INCLUDE_PATH:-}}"
export CMAKE_LIBRARY_PATH="${{PIXI_ENV_ROOT}}/lib:${{CMAKE_LIBRARY_PATH:-}}"

nvidia-smi --query-gpu=uuid,name,pci.bus_id --format=csv,noheader > "${{ROOT}}/provenance/allocation_gpu_table_${{SLURM_JOB_ID}}.csv"
test "$(wc -l < "${{ROOT}}/provenance/allocation_gpu_table_${{SLURM_JOB_ID}}.csv")" -eq 1
grep -q H100 "${{ROOT}}/provenance/allocation_gpu_table_${{SLURM_JOB_ID}}.csv"
scontrol show job "${{SLURM_JOB_ID}}" -o > "${{ROOT}}/provenance/scontrol_${{SLURM_JOB_ID}}.txt"
REQ_TRES="$(awk '{{for (i=1; i<=NF; i++) if ($i ~ /^ReqTRES=/) {{sub(/^ReqTRES=/, "", $i); print $i}}}}' "${{ROOT}}/provenance/scontrol_${{SLURM_JOB_ID}}.txt")"
ALLOC_TRES="$(awk '{{for (i=1; i<=NF; i++) if ($i ~ /^AllocTRES=/) {{sub(/^AllocTRES=/, "", $i); print $i}}}}' "${{ROOT}}/provenance/scontrol_${{SLURM_JOB_ID}}.txt")"
test -n "${{REQ_TRES}}" && test -n "${{ALLOC_TRES}}"
test "${{REQ_TRES}}" = "${{ALLOC_TRES}}"
[[ "${{ALLOC_TRES}}" == *"gres/gpu=1"* ]]

{_quote(args.pixi_python)} -m venv --system-site-packages {_quote(venv)}
PIP_NO_INDEX=1 PIP_DISABLE_PIP_VERSION_CHECK=1 {_quote(venv / "bin/pip")} install \
  -e {_quote(REPO_ROOT)} --no-deps --no-build-isolation --ignore-installed
export RELION_SRC_DIR={_quote(args.relion_bind_source)}
export RECOVAR_RELION_BIND_BUILD_DIR={_quote(bind_dir)}
export RECOVAR_RELION_BIND_JOBS=6
export RECOVAR_CUDA_LIB={_quote(cuda_lib)}
{_quote(python)} recovar/relion_bind/build.py
env PYTHON={_quote(python)} make -C recovar/cuda LIB={_quote(cuda_lib)} \
  CUDA_ARCH='-gencode arch=compute_90,code=sm_90 -gencode arch=compute_90,code=compute_90' all
{_quote(python)} -c "import pathlib,recovar,jax; r=pathlib.Path.cwd().resolve(); assert pathlib.Path(recovar.__file__).resolve().is_relative_to(r); assert len(jax.devices('gpu')) == 1; assert 'H100' in jax.devices('gpu')[0].device_kind"

TARGET_STACKS="$({_quote(python)} -c {_quote(target_loader)})"
RECOVAR_LOCAL_ORIGINALS="$({_quote(python)} -c {_quote(recovar_local_loader)})"

run_native_arm() {{
  local arm="$1"
  local capture_class="$2"
  local arm_root="${{ROOT}}/native/${{arm}}"
  mkdir -p "${{arm_root}}/output" "${{arm_root}}/factors"
  test -z "$(find "${{arm_root}}/output" "${{arm_root}}/factors" -mindepth 1 -print -quit)"
  while IFS='=' read -r variable_name _; do
    case "${{variable_name}}" in RELION_BPRE_*|RELION_FINE_*|RELION_SAMPLING_PERTURBATION_OVERRIDE*|RELION_STOP_AFTER_LIVE_ITER*) unset "${{variable_name}}" ;; esac
  done < <(env)
  export RELION_SAMPLING_PERTURBATION_OVERRIDE={CASE.random_perturbation}
  export RELION_SAMPLING_PERTURBATION_OVERRIDE_ITER=1
  export RELION_STOP_AFTER_LIVE_ITER=1
  if [[ "${{capture_class}}" != 0 ]]; then
    export RELION_BPRE_CAPTURE_DIR="${{arm_root}}/factors"
    export RELION_BPRE_CAPTURE_SCHEMA=2
    export RELION_BPRE_CAPTURE_ITER=1
    export RELION_BPRE_CAPTURE_CLASS="${{capture_class}}"
    export RELION_BPRE_CAPTURE_EXPECTED_PARTICLES={CASE.particle_count}
    export RELION_BPRE_CAPTURE_MAX_PARTICLES_PER_RANK={CASE.particle_count}
    export RELION_BPRE_CAPTURE_EXPECTED_FOLLOWERS=1
    export RELION_BPRE_CAPTURE_MAX_BYTES=8000000000
    export RELION_BPRE_CAPTURE_STACKS="${{TARGET_STACKS}}"
    export RELION_BPRE_CAPTURE_GEOMETRY_ONLY=1
    export RELION_FINE_SCORE_CAPTURE_CLASSES="${{capture_class}}"
  fi
  local command=(
    srun --ntasks=1 --cpus-per-task=8 --cpu-bind=none {_quote(args.relion_capture_binary)}
    --continue {_quote(replay_optimiser)}
    --o "${{arm_root}}/output/run" --auto_iter_max 1 --pool 3 --gpu 0 --j 8
  )
  printf '%q ' "${{command[@]}}" > "${{ROOT}}/provenance/command_${{arm}}_${{SLURM_JOB_ID}}.sh"
  printf '\n' >> "${{ROOT}}/provenance/command_${{arm}}_${{SLURM_JOB_ID}}.sh"
  (cd {_quote(args.fixture_dir)}; "${{command[@]}}") < /dev/null \
    > "${{arm_root}}/output/runner.stdout" 2> "${{arm_root}}/output/runner.stderr"
  for class_id in 001 002 003 004; do test -s "${{arm_root}}/output/run_it001_class${{class_id}}.mrc"; done
  test -s "${{arm_root}}/output/run_it001_data.star"
  test -s "${{arm_root}}/output/run_it001_sampling.star"
  test ! -e "${{arm_root}}/output/run_it002_optimiser.star"
  test "$(grep -Fxc {_quote(EXPECTED_CONTINUED_ITER0_MARKER)} "${{arm_root}}/output/runner.stdout")" -eq 1
  test "$(grep -Fxc {_quote(EXPECTED_STOP_AFTER_LIVE_ITER_MARKER)} "${{arm_root}}/output/runner.stdout")" -eq 1
  grep -Fq '[RELION_SAMPLING_PERTURBATION_OVERRIDE] iter 1 requested' "${{arm_root}}/output/runner.stdout"
  {_quote(python)} -c "from recovar.em.sampling import read_relion_sampling_metadata as r; m=r('${{arm_root}}/output/run_it001_sampling.star'); assert m['healpix_order']=={CASE.healpix_order}; assert abs(m['offset_range']-{CASE.offset_range_pixels * CASE.pixel_size_angstrom})<1e-7; assert abs(m['offset_step']-{CASE.offset_step_pixels * CASE.pixel_size_angstrom})<1e-7; assert abs(m['random_perturbation']-({CASE.random_perturbation}))<1e-5"
  if [[ "${{capture_class}}" = 0 ]]; then
    test -z "$(find "${{arm_root}}/factors" -mindepth 1 -print -quit)"
  else
    test "$(find "${{arm_root}}/factors" -maxdepth 1 -name '*.bpre-v2.bin' | wc -l)" -eq {CASE.particle_count}
    test "$(find "${{arm_root}}/factors" -maxdepth 1 -name '*.fine-score-v1.bin' | wc -l)" -eq {CASE.particle_count}
  fi
}}

run_native_arm control_a 0
{native_smoke_exit}
run_native_arm control_b 0
for class_id in 1 2 3 4; do run_native_arm "class${{class_id}}" "${{class_id}}"; done

mkdir -p "${{ROOT}}/recovar/output" "${{ROOT}}/recovar/pass2" \
  "${{ROOT}}/recovar/accum" "${{ROOT}}/recovar/contributions"
export RECOVAR_EXPECTED_REPO_ROOT={_quote(REPO_ROOT)}
export RECOVAR_SPARSE_KCLASS_FUSED=1
export RECOVAR_LOCAL_ADAPTIVE_PASS2_FULL_PARENT=0
export RECOVAR_PASS2_DUMP_ORIGINAL_INDICES="${{RECOVAR_LOCAL_ORIGINALS}}"
export RECOVAR_PASS2_DUMP_CURRENT_SIZE={CASE.current_size}
export RECOVAR_PASS2_DUMP_ITERATION=1
export RECOVAR_PASS2_DUMP_DIR="${{ROOT}}/recovar/pass2"
export RECOVAR_INITIAL_MODEL_ACCUM_DUMP_DIR="${{ROOT}}/recovar/accum"
export RECOVAR_BPREF_CONTRIBUTION_DUMP_DIR="${{ROOT}}/recovar/contributions"
export RECOVAR_BPREF_CONTRIBUTION_DUMP_ITERATION=1
export RECOVAR_BPREF_CONTRIBUTION_DUMP_CURRENT_SIZE={CASE.current_size}
export RECOVAR_BPREF_CONTRIBUTION_DUMP_ORIGINAL_INDICES="${{RECOVAR_LOCAL_ORIGINALS}}"
export RECOVAR_BPREF_HIGH_PRECISION_OPERAND_BUNDLE=1
export RECOVAR_BPREF_CONTRIBUTION_IMAGE_NAMES_NPY={_quote(image_names_mapping)}
export RECOVAR_BPREF_CONTRIBUTION_STACK_SHA256={EXPECTED_PARTICLE_STACK_SHA256}
export RECOVAR_BPREF_CONTRIBUTION_DUMP_RUN_ID="empiar10076-k4-shared200-it1-${{SLURM_JOB_ID}}"
unset RECOVAR_PASS2_DUMP_CLASS RECOVAR_PASS2_DUMP_STOP_AFTER_TARGET
unset RECOVAR_BPREF_CONTRIBUTION_DUMP_CLASS RECOVAR_BPREF_CONTRIBUTION_DUMP_HALF
unset RECOVAR_BPREF_CONTRIBUTION_TARGET_ONLY RECOVAR_BPREF_CONTRIBUTION_STOP_AFTER_TARGET
unset RECOVAR_FINAL_ALL_DATA_AFTER_MAX_ITER RECOVAR_FINAL_ALL_DATA_GRID_CORRECT
{recovar_text} > "${{ROOT}}/recovar/output/runner.stdout" 2> "${{ROOT}}/recovar/output/runner.stderr"
test "$(find "${{ROOT}}/recovar/pass2" -maxdepth 1 -name '*.npz' | wc -l)" -eq 800
test -n "$(find "${{ROOT}}/recovar/contributions" -maxdepth 1 -name '*.npz' -print -quit)"
test -s "${{ROOT}}/recovar/output/k_class_parity_arrays.npz"
test -s "${{ROOT}}/recovar/output/summary.json"

set +e
{_quote(python)} -m scripts.audit_em_real_k4_shared200_causal_replay \
  --manifest "${{MANIFEST}}" --output "${{ROOT}}/analysis/causal_replay_report.json"
audit_status=$?
set -e
printf '%s\n' "${{audit_status}}" > "${{ROOT}}/analysis/audit_exit_code.txt"
find "${{ROOT}}/native" "${{ROOT}}/recovar" "${{ROOT}}/analysis" -type f -print0 \
  | sort -z | xargs -0 sha256sum > "${{ROOT}}/provenance/science_outputs_${{SLURM_JOB_ID}}.sha256"
exit "${{audit_status}}"
"""


def _expected_outputs(run_root: Path, *, native_smoke_only: bool) -> dict[str, Any]:
    if native_smoke_only:
        return {
            "mode": "native-smoke",
            "native_control": str(run_root / "native/control_a/output"),
            "native_iteration_1_class_maps": CASE.K,
            "native_iteration_1_sampling": str(run_root / "native/control_a/output/run_it001_sampling.star"),
        }
    return {
        "mode": "full",
        "native_controls": [str(run_root / f"native/control_{label}/output") for label in ("a", "b")],
        "native_capture_arms": {
            str(class_id): str(run_root / f"native/class{class_id}/factors") for class_id in range(1, CASE.K + 1)
        },
        "native_fine_score_files": CASE.particle_count * CASE.K,
        "native_bpref_factor_files": CASE.particle_count * CASE.K,
        "recovar_pass2_files": CASE.particle_count * CASE.K,
        "recovar_output": str(run_root / "recovar/output"),
        "audit_report": str(run_root / "analysis/causal_replay_report.json"),
    }


def validate_manifest(path: Path) -> dict[str, Any]:
    manifest = json.loads(path.read_text())
    _require(manifest.get("schema") == SCHEMA, "launch-manifest schema drift")
    _require(manifest.get("status") == "preflight_complete", "launch manifest is not sealed")
    _require(manifest.get("fixed_case") == _fixed_case_record(), "frozen case changed")
    _require(manifest.get("thresholds") == THRESHOLDS, "frozen thresholds changed")
    _require(Path(manifest.get("run_root", "")).resolve() == path.parent.resolve(), "manifest/run-root mismatch")
    _require(manifest.get("requested_resources", {}).get("exclusive") is False, "exclusive allocation is forbidden")
    _require(manifest.get("requested_resources", {}).get("gpus") == 1, "exactly one GPU is required")
    for record in manifest.get("input_records", []):
        _validate_record(record)
    _require(manifest.get("input_records"), "launch manifest has no input records")
    for key in ("targets", "subset_star", "sbatch_script"):
        _validate_record(manifest[key])
    _validate_record(manifest["image_names_mapping"])
    continuation = manifest.get("continuation_bundle", {})
    for key in ("source_sampling", "source_optimiser", "derived_sampling", "derived_optimiser"):
        _validate_record(continuation[key])
    derived_optimiser = Path(continuation["derived_optimiser"]["path"])
    derived_sampling = Path(continuation["derived_sampling"]["path"])
    _require(
        Path(_star_scalar(derived_optimiser.read_text(), "_rlnOrientSamplingStarFile")).resolve()
        == derived_sampling.resolve(),
        "sealed continuation sampling reference drift",
    )
    derived_metadata = read_relion_sampling_metadata(derived_sampling)
    _require(derived_metadata["offset_range"] == CASE.offset_range_pixels, "sealed continuation range drift")
    _require(derived_metadata["offset_step"] == CASE.offset_step_pixels, "sealed continuation step drift")
    _require(derived_metadata["random_perturbation"] == 0.0, "sealed continuation perturbation drift")
    source_root = Path(manifest["source"]["root"])
    _require(_git_text(source_root, "rev-parse", "HEAD") == manifest["source"]["git_head"], "RECOVAR head drift")
    _require(_git_text(source_root, "rev-parse", "HEAD^{tree}") == manifest["source"]["git_tree"], "RECOVAR tree drift")
    _require(not _git_text(source_root, "status", "--porcelain", "--untracked-files=all"), "RECOVAR source is dirty")
    capture_manifest = manifest["relion_capture_source"]
    capture_source = Path(capture_manifest["root"])
    _require(
        _source_provenance(capture_source) == capture_manifest,
        "capture RELION source provenance drift",
    )
    capture_binary_records = [
        record
        for record in manifest["input_records"]
        if record.get("role") == "capture RELION binary"
    ]
    _require(
        len(capture_binary_records) == 1
        and capture_binary_records[0]["sha256"]
        == EXPECTED_CAPTURE_RELION_BINARY_SHA256,
        "capture RELION binary provenance drift",
    )
    bind_source = Path(manifest["relion_bind_source"]["root"])
    _require(
        _git_text(bind_source, "rev-parse", "HEAD") == manifest["relion_bind_source"]["git_head"],
        "binding RELION head drift",
    )
    _require(
        _git_text(bind_source, "rev-parse", "HEAD^{tree}") == manifest["relion_bind_source"]["git_tree"],
        "binding RELION tree drift",
    )
    _require(
        not _git_text(bind_source, "status", "--porcelain", "--untracked-files=no"), "binding RELION source is dirty"
    )
    targets = json.loads(Path(manifest["targets"]["path"]).read_text())
    _require(targets.get("schema") == TARGET_SCHEMA, "target manifest schema drift")
    _require(targets.get("particle_count") == CASE.particle_count, "target manifest count drift")
    _require(
        targets.get("half_counts") == {"1": CASE.half1_particle_count, "2": CASE.half2_particle_count},
        "target manifest half split drift",
    )
    identities = targets.get("image_identities")
    stacks = targets.get("stack_indices_one_based")
    subset_local_indices = targets.get("subset_local_indices_zero_based")
    _require(isinstance(identities, list) and len(identities) == CASE.particle_count, "target identity count drift")
    _require(isinstance(stacks, list) and len(stacks) == CASE.particle_count, "target stack count drift")
    _require(
        subset_local_indices == list(range(CASE.particle_count)),
        "target subset-local row topology drift",
    )
    _require([_stack_index(value) for value in identities] == stacks, "target identity order drift")
    _require(
        targets.get("ordered_image_identity_sha256") == _ordered_text_sha256(str(value) for value in identities),
        "target ordered-identity digest drift",
    )
    subset, _ = read_star(manifest["subset_star"]["path"])
    image_column = _column(subset, "rlnImageName")
    half_column = _column(subset, "rlnRandomSubset")
    subset_stacks = [_stack_index(value) for value in subset[image_column]]
    subset_halves = np.asarray(subset[half_column], dtype=np.int64)
    _require(subset_stacks == stacks, "materialized subset order drift")
    _require(
        targets.get("ordered_identity_half_sha256")
        == _ordered_text_sha256(
            f"{identity}\t{half}" for identity, half in zip(identities, subset_halves.tolist(), strict=True)
        ),
        "target ordered identity/half digest drift",
    )
    identity_mapping = np.load(manifest["image_names_mapping"]["path"], allow_pickle=False)
    _require(
        identity_mapping.ndim == 1 and identity_mapping.dtype.kind in {"U", "S"},
        "sealed image identity mapping must be a fixed-width rank-1 string array",
    )
    _require(
        len(identity_mapping) == CASE.particle_count,
        "sealed image identity mapping count drift",
    )
    selected_mapping = identity_mapping.astype(str).tolist()
    stack_records = [
        record for record in manifest["input_records"] if record.get("role") == "fixture particle stack"
    ]
    _require(len(stack_records) == 1, "manifest must seal exactly one fixture particle stack")
    sealed_stack = Path(stack_records[0]["path"]).resolve()
    expected_mapping = [f"{stack}@{sealed_stack}" for stack in stacks]
    _require(selected_mapping == expected_mapping, "sealed image identity mapping target join drift")
    return manifest


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--validate-manifest", type=Path)
    parser.add_argument("--output-root", type=Path)
    parser.add_argument("--fixture-dir", type=Path, default=DEFAULT_FIXTURE_DIR)
    parser.add_argument("--control-pair-root", type=Path, default=DEFAULT_CONTROL_PAIR_ROOT)
    parser.add_argument("--shared-set", type=Path, default=DEFAULT_SHARED_SET)
    parser.add_argument("--relion-capture-source", type=Path, default=DEFAULT_RELION_CAPTURE_SOURCE)
    parser.add_argument("--relion-capture-binary", type=Path, default=DEFAULT_RELION_CAPTURE_BINARY)
    parser.add_argument("--relion-bind-source", type=Path, default=DEFAULT_RELION_BIND_SOURCE)
    parser.add_argument("--pixi-python", type=Path, default=Path(sys.executable))
    parser.add_argument("--runtime-root", type=Path, default=DEFAULT_RUNTIME_ROOT)
    parser.add_argument("--partition", default=os.environ.get("SBATCH_PARTITION", "cryoem"))
    parser.add_argument("--account", default=os.environ.get("SBATCH_ACCOUNT", "gilles"))
    parser.add_argument("--constraint", default=os.environ.get("SBATCH_CONSTRAINT", "h100"))
    parser.add_argument("--mem", default="192G")
    parser.add_argument("--time-limit", default="02:00:00")
    parser.add_argument("--cuda-module", default="cudatoolkit/12.8")
    parser.add_argument("--native-smoke-only", action="store_true")
    parser.add_argument("--submit", action="store_true")
    args = parser.parse_args(argv)
    for name in (
        "validate_manifest",
        "output_root",
        "fixture_dir",
        "control_pair_root",
        "shared_set",
        "relion_capture_source",
        "relion_capture_binary",
        "relion_bind_source",
        "pixi_python",
        "runtime_root",
    ):
        value = getattr(args, name)
        if value is not None:
            setattr(args, name, value.expanduser().resolve())
    if args.validate_manifest is None and args.output_root is None:
        parser.error("--output-root is required unless --validate-manifest is used")
    if args.validate_manifest is not None and args.submit:
        parser.error("--submit cannot be combined with --validate-manifest")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.validate_manifest is not None:
        validate_manifest(args.validate_manifest)
        print(f"Manifest preflight passed: {args.validate_manifest}")
        return 0

    assert args.output_root is not None
    _require(
        not args.output_root.exists() or not any(args.output_root.iterdir()),
        f"refusing non-empty root: {args.output_root}",
    )
    state = preflight(args)
    args.output_root.mkdir(parents=True, exist_ok=True)
    for directory in (
        "inputs",
        "scripts",
        "logs",
        "analysis",
        "provenance",
        "build",
    ):
        (args.output_root / directory).mkdir()
    (args.output_root / "SAFE_TO_DELETE").touch()
    targets_path = args.output_root / "inputs/frozen_targets.json"
    targets_path.write_text(json.dumps(state["target_record"], indent=2, sort_keys=True) + "\n")
    subset_path = args.output_root / "inputs/particles_shared200.star"
    write_deterministic_subset_star(
        output=subset_path,
        selected=state["selected"],
        optics=state["optics"],
        particle_stack=args.fixture_dir / "particles.256.mrcs",
    )
    reread, _ = read_star(str(subset_path))
    _require(len(reread) == CASE.particle_count, "materialized subset count drift")
    image_names_mapping_path = args.output_root / "inputs/image_names_subset_local.npy"
    write_subset_local_image_identity_mapping(
        output=image_names_mapping_path,
        selected=state["selected"],
        particle_stack=args.fixture_dir / "particles.256.mrcs",
    )
    _, _, continuation_bundle = materialize_iteration0_continuation_bundle(
        source_optimiser=args.control_pair_root / "relion/run_it000_optimiser.star",
        source_sampling=args.control_pair_root / "relion/run_it001_sampling.star",
        output_dir=args.output_root / "inputs/continuation",
    )

    script_path = args.output_root / "scripts/run_shared200_causal_replay.sbatch"
    script_path.write_text(
        render_sbatch(
            args,
            expected_head=state["repo_head"],
            manifest_path=args.output_root / "launch_manifest.json",
        )
    )
    script_path.chmod(0o755)
    input_records = [_file_record(path, role=role) for role, path in _input_paths(args)]
    manifest = {
        "schema": SCHEMA,
        "status": "preflight_complete",
        "scientific_scope": "EMPIAR-10076 K=4 iteration-1 shared-200 fixed-state causal replay",
        "scientific_limitations": (
            "diagnostic boundary evidence only; this is not a gold-standard half-map or biological-class claim"
        ),
        "run_root": str(args.output_root),
        "source": {
            "root": str(REPO_ROOT),
            "git_head": state["repo_head"],
            "git_tree": state["repo_tree"],
            "tracked_dirty": False,
        },
        "relion_capture_source": state["relion_capture_source"],
        "relion_bind_source": state["relion_bind_source"],
        "fixed_case": _fixed_case_record(),
        "thresholds": THRESHOLDS,
        "controller_state": state["controller"],
        "continuation_bundle": continuation_bundle,
        "input_records": input_records,
        "targets": _file_record(targets_path, role="frozen shared-200 target manifest"),
        "subset_star": _file_record(subset_path, role="deterministic shared-200 STAR"),
        "image_names_mapping": _file_record(
            image_names_mapping_path,
            role="fixed-width subset-local rlnImageName mapping",
        ),
        "sbatch_script": _file_record(script_path, role="sealed Slurm launcher"),
        "expected_outputs": _expected_outputs(args.output_root, native_smoke_only=args.native_smoke_only),
        "requested_resources": {
            "partition": args.partition,
            "account": args.account,
            "constraint": args.constraint,
            "gpus": 1,
            "gpu_type": "H100 80GB",
            "nodes": 1,
            "ntasks": 1,
            "cpus_per_task": 8,
            "total_cpus": 8,
            "mem": args.mem,
            "time_limit": args.time_limit,
            "exclusive": False,
        },
        "capture_policy": {
            "mode": "native-smoke" if args.native_smoke_only else "full",
            "native_repeat_controls": 2,
            "native_class_capture_arms": 4,
            "native_capture_geometry_only": True,
            "native_gradient_execution": "non-MPI relion_refine under one Slurm task with eight threads",
            "native_capture_rank_count": 1,
            "native_gradient_pseudo_half_rule": "op.part_id modulo 2",
            "native_gradient_pseudo_half_counts": [100, 100],
            "recovar_all_classes_single_replay": True,
            "recovar_high_precision_contribution_bundle": True,
            "recovar_subset_particle_count": CASE.particle_count,
            "correlation_used": False,
        },
        "scorecard_change_admissible": False,
        "safe_to_delete_marker": str(args.output_root / "SAFE_TO_DELETE"),
    }
    manifest_path = args.output_root / "launch_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    validate_manifest(manifest_path)
    manifest_sha256 = _sha256(manifest_path)
    submit_command = [
        "sbatch",
        "--parsable",
        f"--export=ALL,EXPECTED_MANIFEST_SHA256={manifest_sha256}",
        str(script_path),
    ]
    print(f"Run root: {args.output_root}")
    print(f"Manifest: {manifest_path}")
    print(f"Manifest SHA256: {manifest_sha256}")
    print(f"Sbatch: {script_path}")
    print(f"Submit: {shlex.join(submit_command)}")
    if not args.submit:
        print("Dry run only; no Slurm job submitted.")
        return 0
    result = subprocess.run(submit_command, check=True, capture_output=True, text=True)
    job_id = result.stdout.strip().split(";", 1)[0]
    allocation = subprocess.check_output(["scontrol", "show", "job", job_id], text=True)
    print(f"Submitted Slurm job {job_id}")
    print(allocation.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
