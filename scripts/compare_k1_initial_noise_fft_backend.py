#!/usr/bin/env python3
"""Compare RECOVAR and source-faithful RELION startup-noise FFT operands."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import mrcfile
import numpy as np
import starfile

from recovar.em.initial_model.avg_unaligned import (
    _fix_negative_sigma2,
    _radial_power_spectrum,
    _softmask_outside_map,
)
from scripts.run_full_refinement import _relion_fresh_initial_noise_layout


def _particles(document):
    return document["particles"] if isinstance(document, dict) else document


def _relion_radial_power(relion_bind, image: np.ndarray, shells: np.ndarray) -> np.ndarray:
    transform = np.asarray(
        relion_bind.fourier_transform_window_center_2d(image, image.shape[0]),
        dtype=np.complex128,
    )
    power = np.abs(transform) ** 2
    sums = np.bincount(shells.ravel(), weights=power.ravel(), minlength=image.shape[0] // 2 + 1)
    counts = np.bincount(shells.ravel(), minlength=image.shape[0] // 2 + 1)
    return sums[: counts.size] / np.maximum(counts, 1)


def _sigma(sum_power: np.ndarray, average_image: np.ndarray, n_images: int, spectrum_fn) -> np.ndarray:
    average_spectrum = spectrum_fn(average_image) / 2.0
    return _fix_negative_sigma2(sum_power / (2.0 * float(n_images)) - average_spectrum)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-star", type=Path, required=True)
    parser.add_argument("--relion-data-star", type=Path, required=True)
    parser.add_argument("--relion-bind-directory", type=Path, required=True)
    parser.add_argument("--particle-diameter-ang", type=float, required=True)
    parser.add_argument("--width-mask-edge-px", type=int, required=True)
    parser.add_argument("--max-particles", type=int, default=1000)
    parser.add_argument("--output-directory", type=Path, required=True)
    args = parser.parse_args()

    if args.output_directory.exists():
        raise FileExistsError(f"refusing to overwrite {args.output_directory}")
    args.output_directory.mkdir(parents=True)
    os.environ["RECOVAR_RELION_BIND_BUILD_DIR"] = str(args.relion_bind_directory.resolve())
    from recovar.relion_bind import _relion_bind_core as relion_bind

    source_document = starfile.read(args.source_star)
    relion_document = starfile.read(args.relion_data_star)
    source_particles = _particles(source_document)
    relion_particles = _particles(relion_document)
    source_rows, optics_groups = _relion_fresh_initial_noise_layout(
        source_particles,
        relion_particles,
    )
    if np.unique(optics_groups).size != 1:
        raise NotImplementedError("diagnostic currently requires one optics group")
    source_rows = source_rows[: int(args.max_particles)]
    if source_rows.size != int(args.max_particles):
        raise ValueError("insufficient particles for the requested startup-noise panel")

    optics = source_document["optics"]
    if len(optics) != 1:
        raise NotImplementedError("diagnostic currently requires one optics row")
    pixel_size = float(optics.iloc[0]["rlnImagePixelSize"])
    radius = float(args.particle_diameter_ang) / (2.0 * pixel_size)

    first_identity = str(source_particles.iloc[int(source_rows[0])]["rlnImageName"])
    _first_index, stack_text = first_identity.split("@", 1)
    stack_path = Path(stack_text)
    if not stack_path.is_absolute():
        stack_path = (args.source_star.parent / stack_path).resolve()

    with mrcfile.mmap(stack_path, permissive=True) as stack:
        image_size = int(stack.data.shape[-1])
        if stack.data.shape[-2:] != (image_size, image_size):
            raise ValueError("particle images must be square")
        n_shells = image_size // 2 + 1
        ky = np.fft.fftfreq(image_size) * image_size
        kx = np.arange(n_shells, dtype=np.float64)
        shells = np.rint(np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)).astype(np.int64)
        shell_mask = shells < n_shells
        shells = np.where(shell_mask, shells, n_shells)

        sums = {
            "recovar_numpy": np.zeros(n_shells, dtype=np.float64),
            "relion_mask_numpy_fft": np.zeros(n_shells, dtype=np.float64),
            "relion_mask_fftw": np.zeros(n_shells, dtype=np.float64),
        }
        averages = {name: np.zeros((image_size, image_size), dtype=np.float64) for name in sums}
        for source_row in source_rows:
            identity = str(source_particles.iloc[int(source_row)]["rlnImageName"])
            stack_index_text, identity_stack_text = identity.split("@", 1)
            identity_stack = Path(identity_stack_text)
            if not identity_stack.is_absolute():
                identity_stack = (args.source_star.parent / identity_stack).resolve()
            if identity_stack != stack_path:
                raise NotImplementedError("diagnostic currently requires one particle stack")
            raw = np.asarray(stack.data[int(stack_index_text) - 1], dtype=np.float64)
            recovar_masked = _softmask_outside_map(
                raw,
                radius,
                float(args.width_mask_edge_px),
            ).astype(np.float64)
            relion_masked = np.asarray(
                relion_bind.soft_mask_outside_map_2d(
                    raw,
                    radius,
                    float(args.width_mask_edge_px),
                ),
                dtype=np.float64,
            )
            averages["recovar_numpy"] += recovar_masked
            averages["relion_mask_numpy_fft"] += relion_masked
            averages["relion_mask_fftw"] += relion_masked
            sums["recovar_numpy"] += _radial_power_spectrum(recovar_masked, n_shells)
            sums["relion_mask_numpy_fft"] += _radial_power_spectrum(relion_masked, n_shells)
            sums["relion_mask_fftw"] += _relion_radial_power(
                relion_bind,
                relion_masked,
                shells,
            )[:n_shells]

    spectra = {}
    for name in sums:
        average = averages[name] / float(source_rows.size)
        if name == "relion_mask_fftw":
            spectrum_fn = lambda image: _relion_radial_power(relion_bind, image, shells)[:n_shells]
        else:
            spectrum_fn = lambda image: _radial_power_spectrum(image, n_shells)
        spectra[name] = _sigma(sums[name], average, int(source_rows.size), spectrum_fn)
        np.save(args.output_directory / f"sigma2_{name}.npy", spectra[name])

    reference = spectra["recovar_numpy"]
    report = {
        "schema": "recovar.em.k1_initial_noise_fft_backend.v1",
        "particle_count": int(source_rows.size),
        "source_rows_head": source_rows[:8].tolist(),
        "pixel_size": pixel_size,
        "image_size": image_size,
        "comparisons_vs_recovar_numpy": {},
    }
    for name, values in spectra.items():
        delta = values - reference
        report["comparisons_vs_recovar_numpy"][name] = {
            "relative_l2": float(np.linalg.norm(delta) / np.linalg.norm(reference)),
            "max_abs": float(np.max(np.abs(delta))),
            "inverse_float32_mismatch_count": int(
                np.count_nonzero(
                    (1.0 / values).astype(np.float32)
                    != (1.0 / reference).astype(np.float32)
                )
            ),
        }
    (args.output_directory / "REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n",
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
