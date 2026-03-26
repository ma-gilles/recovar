#!/usr/bin/env python3

"""Generate RELION5 tomography inputs for CryoET Data Portal dataset DS-10468.

This script queries the CryoET Data Portal metadata directly and writes:

- ``tilt_series/aligned_tilt_series.star``
- one per-run RELION5 tilt-series STAR under ``tilt_series/``
- ``full_picks.star`` containing RELION5 particle picks
- ``download_manifest.json`` / ``download_manifest.tsv`` with aligned tilt stacks
- ``summary.json`` with per-run counts and paths

The validated target for issue 35 is the ``cytosolic ribosome`` object in
dataset ``10468``. The script keeps that as the default, but allows changing
``--object-name`` for experimentation.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import sys
import urllib.request
from pathlib import Path

from cryoet_data_portal import Client, Dataset
from scipy.spatial.transform import Rotation as R


BASE_HTTPS = "https://files.cryoetdataportal.cziscience.com/"
ORIENTED_POINT_SHAPES = {"OrientedPoint", "orientedPoint"}
POINT_SHAPES = {"Point", "point"}
VOXEL_SPACING_RE = re.compile(r"VoxelSpacing([0-9.]+)")


def load_json(url: str):
    with urllib.request.urlopen(url) as fh:
        return json.load(fh)


def iter_ndjson(url: str):
    with urllib.request.urlopen(url) as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if line:
                yield json.loads(line)


def abs_https(path_or_url: str) -> str:
    if path_or_url.startswith(("http://", "https://")):
        return path_or_url
    return BASE_HTTPS + path_or_url.lstrip("/")


def format_value(value):
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        if math.isfinite(value) and value.is_integer():
            return f"{value:.1f}"
        return f"{value:.12g}"
    return str(value)


def write_star(path: Path, header_lines: list[str], rows: list[list[object]]) -> None:
    with path.open("w", encoding="utf-8") as fh:
        for line in header_lines:
            fh.write(line)
            fh.write("\n")
        for row in rows:
            fh.write(" ".join(format_value(value) for value in row))
            fh.write("\n")


def choose_annotation(run, object_name: str):
    annotations = [
        ann
        for ann in run.annotations
        if ann.object_name == object_name and int(getattr(ann, "object_count", 0) or 0) > 0
    ]
    if not annotations:
        return None
    return max(annotations, key=lambda ann: int(getattr(ann, "object_count", 0) or 0))


def parse_voxel_spacing(annotation_url: str, run) -> float:
    match = VOXEL_SPACING_RE.search(annotation_url)
    if match:
        return float(match.group(1))
    for tomogram in run.tomograms:
        value = tomogram.to_dict().get("voxel_spacing")
        if value is not None:
            return float(value)
    raise RuntimeError(f"Could not determine voxel spacing for run {run.name}")


def choose_point_url(annotation_metadata: dict) -> tuple[str | None, bool]:
    files = annotation_metadata.get("files") or []
    for entry in files:
        if entry.get("shape") in ORIENTED_POINT_SHAPES:
            return abs_https(entry["path"]), True
    for entry in files:
        if entry.get("shape") in POINT_SHAPES:
            return abs_https(entry["path"]), False
    ndjson_files = [entry for entry in files if str(entry.get("path", "")).endswith(".ndjson")]
    if len(ndjson_files) == 1:
        return abs_https(ndjson_files[0]["path"]), False
    return None, False


def zrot_from_matrix(matrix_2x2) -> float:
    return math.degrees(math.atan2(matrix_2x2[1][0], matrix_2x2[0][0]))


def build_tilt_rows(run, tiltseries_path: str):
    if not run.tiltseries:
        raise RuntimeError(f"Run {run.name} has no tilt series")
    tiltseries = run.tiltseries[0]
    tilt_dict = tiltseries.to_dict()
    pixel_size = float(tilt_dict["pixel_spacing"])

    if not run.alignments:
        raise RuntimeError(f"Run {run.name} has no alignments")
    alignment = run.alignments[0]
    alignment_dict = alignment.to_dict()
    alignment_meta = load_json(alignment_dict["https_alignment_metadata"])
    per_section_alignment = {
        int(entry["z_index"]): entry for entry in alignment_meta["per_section_alignment_parameters"]
    }

    per_section_parameters = sorted(run.per_section_parameters, key=lambda entry: int(entry.z_index))
    frames_by_id = {int(frame.id): frame for frame in run.frames}

    tilt_rows = []
    for section in per_section_parameters:
        frame = frames_by_id[int(section.frame_id)]
        align_entry = per_section_alignment[int(section.z_index)]
        tilt_rows.append(
            [
                f"{int(section.z_index) + 1}@{tiltseries_path}",
                1.0,
                float(align_entry["volume_x_rotation"]),
                float(align_entry["tilt_angle"]),
                zrot_from_matrix(align_entry["in_plane_rotation"]),
                float(align_entry["x_offset"]) * pixel_size,
                float(align_entry["y_offset"]) * pixel_size,
                float(section.major_defocus),
                float(section.minor_defocus),
                float(section.astigmatic_angle),
                float(section.phase_shift),
                float(frame.accumulated_dose),
                1,
                float(section.raw_angle),
                0.0,
                float(section.major_defocus),
                0.0,
                float(section.max_resolution),
                0.0,
            ]
        )

    volume_dimension = alignment_meta["volume_dimension"]
    size_x = int(round(float(volume_dimension["x"]) / pixel_size))
    size_y = int(round(float(volume_dimension["y"]) / pixel_size))
    size_z = int(round(float(volume_dimension["z"]) / pixel_size))

    return {
        "tilt_dict": tilt_dict,
        "pixel_size": pixel_size,
        "size_x": size_x,
        "size_y": size_y,
        "size_z": size_z,
        "rows": tilt_rows,
    }


def build_particle_rows(run, annotation, tilt_info):
    annotation_url = annotation.to_dict()["https_metadata_path"]
    annotation_metadata = load_json(annotation_url)
    point_url, has_orientation = choose_point_url(annotation_metadata)
    if not point_url:
        files = annotation_metadata.get("files") or []
        raise RuntimeError(f"Run {run.name} annotation {annotation.id} has no point file: {files}")

    tilt_pixel_size = tilt_info["pixel_size"]
    tomo_voxel_spacing = parse_voxel_spacing(annotation_url, run)
    coord_scale = tomo_voxel_spacing / tilt_pixel_size
    center_x = tilt_info["size_x"] / 2.0
    center_y = tilt_info["size_y"] / 2.0
    center_z = tilt_info["size_z"] / 2.0

    particle_rows = []
    for idx, point in enumerate(iter_ndjson(point_url), start=1):
        location = point["location"]
        coord_x = float(location["x"]) * coord_scale
        coord_y = float(location["y"]) * coord_scale
        coord_z = float(location["z"]) * coord_scale

        if has_orientation and "xyz_rotation_matrix" in point:
            rot, tilt, psi = R.from_matrix(point["xyz_rotation_matrix"]).inv().as_euler(
                "ZYZ", degrees=True
            )
        else:
            rot, tilt, psi = 0.0, 0.0, 0.0

        particle_rows.append(
            [
                run.name,
                f"{run.name}/{idx}",
                coord_x,
                coord_y,
                coord_z,
                (coord_x - center_x) * tilt_pixel_size,
                (coord_y - center_y) * tilt_pixel_size,
                (coord_z - center_z) * tilt_pixel_size,
                float(rot),
                float(tilt),
                float(psi),
                1 if idx % 2 else 2,
                1,
            ]
        )

    expected = int(getattr(annotation, "object_count", 0) or 0)
    if expected and expected != len(particle_rows):
        raise RuntimeError(
            f"Run {run.name} annotation count mismatch: metadata={expected}, parsed={len(particle_rows)}"
        )

    return {
        "annotation_url": annotation_url,
        "point_url": point_url,
        "has_orientation": has_orientation,
        "voxel_spacing": tomo_voxel_spacing,
        "rows": particle_rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-id", type=int, default=10468)
    parser.add_argument("--object-name", default="cytosolic ribosome")
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()

    output_root = args.output_root.resolve()
    tilt_series_dir = output_root / "tilt_series"
    tiltseries_dir = output_root / "tiltseries"
    logs_dir = output_root / "logs"
    for path in (output_root, tilt_series_dir, tiltseries_dir, logs_dir):
        path.mkdir(parents=True, exist_ok=True)

    client = Client()
    dataset = Dataset.get_by_id(client, args.dataset_id)

    global_tilt_rows = []
    global_particle_rows = []
    download_entries = []
    run_summaries = []
    skipped_runs = []
    optics_pixel_size = None
    optics_voltage_kv = None
    optics_spherical_aberration = None

    runs = sorted(dataset.runs, key=lambda run: run.name)
    for run in runs:
        annotation = choose_annotation(run, args.object_name)
        if not annotation:
            continue

        try:
            tilt_info = build_tilt_rows(run, str((tiltseries_dir / f"{run.name}.mrcs").resolve()))
            tiltseries_star_path = (tilt_series_dir / f"{run.name}.star").resolve()
            particle_info = build_particle_rows(run, annotation, tilt_info)
        except Exception as exc:
            skipped_runs.append(
                {
                    "run_name": run.name,
                    "annotation_id": int(annotation.id),
                    "object_count": int(getattr(annotation, "object_count", 0) or 0),
                    "reason": str(exc),
                }
            )
            print(f"SKIP {run.name}: {exc}", file=sys.stderr)
            continue

        tilt_dict = tilt_info["tilt_dict"]
        if optics_pixel_size is None:
            optics_pixel_size = float(tilt_dict["pixel_spacing"])
            optics_voltage_kv = float(tilt_dict["acceleration_voltage"]) / 1000.0
            optics_spherical_aberration = float(tilt_dict["spherical_aberration_constant"])

        write_star(
            tiltseries_star_path,
            [
                "# Created by generate_relion5_ds10468_manifest.py",
                "",
                f"data_{run.name}",
                "",
                "loop_",
                "_rlnMicrographName",
                "_rlnCtfScalefactor",
                "_rlnTomoXTilt",
                "_rlnTomoYTilt",
                "_rlnTomoZRot",
                "_rlnTomoXShiftAngst",
                "_rlnTomoYShiftAngst",
                "_rlnDefocusU",
                "_rlnDefocusV",
                "_rlnDefocusAngle",
                "_rlnPhaseShift",
                "_rlnMicrographPreExposure",
                "_rlnTomoTiltMovieFrameCount",
                "_rlnTomoNominalStageTiltAngle",
                "_rlnTomoNominalTiltAxisAngle",
                "_rlnTomoNominalDefocus",
                "_rlnAccumMotionTotal",
                "_rlnCtfMaxResolution",
                "_rlnCtfFigureOfMerit",
            ],
            tilt_info["rows"],
        )

        global_tilt_rows.append(
            [
                run.name,
                float(tilt_dict["acceleration_voltage"]) / 1000.0,
                float(tilt_dict["spherical_aberration_constant"]),
                0.1,
                float(tilt_dict["pixel_spacing"]),
                -1,
                "opticsGroup1",
                float(tilt_dict["pixel_spacing"]),
                str(tiltseries_star_path),
                tilt_info["size_x"],
                tilt_info["size_y"],
                tilt_info["size_z"],
            ]
        )
        global_particle_rows.extend(particle_info["rows"])

        download_path = (tiltseries_dir / f"{run.name}.mrcs").resolve()
        download_entries.append(
            {
                "run_name": run.name,
                "url": tilt_dict["https_mrc_file"],
                "output_path": str(download_path),
                "file_size_bytes": int(float(tilt_dict["file_size_mrc"])),
            }
        )

        run_summaries.append(
            {
                "run_name": run.name,
                "object_name": args.object_name,
                "annotation_id": int(annotation.id),
                "particle_count": len(particle_info["rows"]),
                "tilt_count": len(tilt_info["rows"]),
                "tilt_image_count": len(particle_info["rows"]) * len(tilt_info["rows"]),
                "tiltseries_url": tilt_dict["https_mrc_file"],
                "tiltseries_output_path": str(download_path),
                "tiltseries_size_bytes": int(float(tilt_dict["file_size_mrc"])),
                "annotation_metadata_url": particle_info["annotation_url"],
                "point_url": particle_info["point_url"],
                "has_orientation": particle_info["has_orientation"],
                "tomo_voxel_spacing_angstrom": particle_info["voxel_spacing"],
                "tilt_series_pixel_size_angstrom": float(tilt_dict["pixel_spacing"]),
                "tomo_size_tilt_series_pixels": {
                    "x": tilt_info["size_x"],
                    "y": tilt_info["size_y"],
                    "z": tilt_info["size_z"],
                },
            }
        )
        print(
            f"KEEP {run.name}: particles={len(particle_info['rows'])} tilts={len(tilt_info['rows'])}",
            file=sys.stderr,
        )

    if not run_summaries:
        raise RuntimeError(
            f"No runs were converted for dataset {args.dataset_id} object {args.object_name!r}"
        )

    aligned_tilt_series_path = (tilt_series_dir / "aligned_tilt_series.star").resolve()
    write_star(
        aligned_tilt_series_path,
        [
            "# Created by generate_relion5_ds10468_manifest.py",
            "",
            "data_global",
            "",
            "loop_",
            "_rlnTomoName",
            "_rlnVoltage",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
            "_rlnMicrographOriginalPixelSize",
            "_rlnTomoHand",
            "_rlnOpticsGroupName",
            "_rlnTomoTiltSeriesPixelSize",
            "_rlnTomoTiltSeriesStarFile",
            "_rlnTomoSizeX",
            "_rlnTomoSizeY",
            "_rlnTomoSizeZ",
        ],
        global_tilt_rows,
    )

    full_picks_path = (output_root / "full_picks.star").resolve()
    write_star(
        full_picks_path,
        [
            "# Created by generate_relion5_ds10468_manifest.py",
            "",
            "data_optics",
            "",
            "loop_",
            "_rlnOpticsGroup",
            "_rlnOpticsGroupName",
            "_rlnTomoTiltSeriesPixelSize",
            "_rlnVoltage",
            "_rlnSphericalAberration",
            "_rlnAmplitudeContrast",
            (
                "1 opticsGroup1 "
                f"{format_value(optics_pixel_size)} "
                f"{format_value(optics_voltage_kv)} "
                f"{format_value(optics_spherical_aberration)} 0.1"
            ),
            "",
            "",
            "data_particles",
            "",
            "loop_",
            "_rlnTomoName",
            "_rlnTomoParticleName",
            "_rlnCoordinateX",
            "_rlnCoordinateY",
            "_rlnCoordinateZ",
            "_rlnCenteredCoordinateXAngst",
            "_rlnCenteredCoordinateYAngst",
            "_rlnCenteredCoordinateZAngst",
            "_rlnAngleRot",
            "_rlnAngleTilt",
            "_rlnAnglePsi",
            "_rlnRandomSubset",
            "_rlnOpticsGroup",
        ],
        global_particle_rows,
    )

    manifest_path = (output_root / "download_manifest.json").resolve()
    with manifest_path.open("w", encoding="utf-8") as fh:
        json.dump(download_entries, fh, indent=2)
        fh.write("\n")

    download_tsv_path = (output_root / "download_manifest.tsv").resolve()
    with download_tsv_path.open("w", encoding="utf-8") as fh:
        fh.write("run_name\turl\toutput_path\tfile_size_bytes\n")
        for entry in download_entries:
            fh.write(
                f"{entry['run_name']}\t{entry['url']}\t{entry['output_path']}\t{entry['file_size_bytes']}\n"
            )

    summary = {
        "dataset_id": args.dataset_id,
        "object_name": args.object_name,
        "runs_seen": len(runs),
        "runs_kept": len(run_summaries),
        "runs_skipped": len(skipped_runs),
        "total_particles": sum(item["particle_count"] for item in run_summaries),
        "total_tilt_images": sum(item["tilt_image_count"] for item in run_summaries),
        "total_tiltseries_bytes": sum(item["tiltseries_size_bytes"] for item in run_summaries),
        "aligned_tilt_series_star": str(aligned_tilt_series_path),
        "full_picks_star": str(full_picks_path),
        "download_manifest_json": str(manifest_path),
        "download_manifest_tsv": str(download_tsv_path),
        "runs": run_summaries,
        "skipped": skipped_runs,
    }
    summary_path = (output_root / "summary.json").resolve()
    with summary_path.open("w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2)
        fh.write("\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
