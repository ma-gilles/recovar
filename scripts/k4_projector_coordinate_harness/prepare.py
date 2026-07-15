#!/usr/bin/env python3
"""Prepare the frozen K=4 particle-3591 texture-coordinate microharness."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
from pathlib import Path

import numpy as np

N_ROTATIONS = 8
IMAGE_SIZE = 40
HALF_WIDTH = 21
N_PIXELS = IMAGE_SIZE * HALF_WIDTH
PPREF_SHAPE = (83, 83, 42)  # z, y, x-half; x is the fastest stored axis.
SPECIAL_ROTATION = 4
SPECIAL_RELION_PIXEL = 242
DENSE_SCALE = -(128**2)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_counted(path: Path, dtype: np.dtype) -> np.ndarray:
    with path.open("rb") as stream:
        count = np.fromfile(stream, dtype=np.int64, count=1)
        values = np.fromfile(stream, dtype=dtype)
    if count.shape != (1,) or values.size != int(count[0]):
        raise ValueError(f"invalid counted vector: {path}")
    return values


def write_raw(path: Path, values: np.ndarray, dtype: np.dtype) -> None:
    array = np.ascontiguousarray(values, dtype=dtype)
    path.write_bytes(array.tobytes(order="C"))


def relion_pixel_coordinates() -> np.ndarray:
    coordinates = np.empty((N_PIXELS, 2), dtype=np.int32)
    for pixel in range(N_PIXELS):
        row, x = divmod(pixel, HALF_WIDTH)
        y = row if row <= IMAGE_SIZE // 2 else row - IMAGE_SIZE
        coordinates[pixel] = (y, x)
    return coordinates


def relion_to_recovar_columns(window_indices: np.ndarray) -> np.ndarray:
    indices = np.asarray(window_indices, dtype=np.int64)
    if indices.shape != (N_PIXELS,):
        raise ValueError(f"unexpected RECOVAR window shape: {indices.shape}")
    full_half_width = 128 // 2 + 1
    mapping = {}
    for column, index in enumerate(indices.tolist()):
        row, x = divmod(index, full_half_width)
        key = (row - 128 // 2, x)
        if key in mapping:
            raise ValueError(f"duplicate RECOVAR Fourier coordinate: {key}")
        mapping[key] = column
    result = np.empty(N_PIXELS, dtype=np.int32)
    for pixel, (y, x) in enumerate(relion_pixel_coordinates().tolist()):
        try:
            result[pixel] = mapping[(y, x)]
        except KeyError as error:
            raise ValueError(f"RELION pixel {pixel} coordinate {(y, x)} is absent") from error
    if sorted(result.tolist()) != list(range(N_PIXELS)):
        raise ValueError("RELION-to-RECOVAR pixel mapping is not a permutation")
    return result


def float_bits(values: np.ndarray) -> list[int]:
    return np.asarray(values, dtype=np.float32).reshape(-1).view(np.uint32).astype(np.uint64).tolist()


def build_inputs(artifact_root: Path, operand_root: Path, output: Path) -> dict:
    payload = operand_root / "relion_capture" / "payload"
    score_prefix = payload / "iter1_rank1_part2882_pass1_"
    state_prefix = payload / "state_iter1_rank1_device0_class1_"
    recovar_capture = artifact_root / "audit/job11224544/recovar_capture/pass2_orig003591_cs040.npz"
    required = [
        Path(f"{state_prefix}projector_real.bin"),
        Path(f"{state_prefix}projector_imag.bin"),
        Path(f"{score_prefix}fine_class1_euler_matrices_xfloat.bin"),
        Path(f"{score_prefix}fine_operand_reference_real_xfloat.bin"),
        Path(f"{score_prefix}fine_operand_reference_imag_xfloat.bin"),
        recovar_capture,
    ]
    for path in required:
        if not path.is_file():
            raise FileNotFoundError(path)

    ppref_real = read_counted(required[0], np.float32)
    ppref_imag = read_counted(required[1], np.float32)
    if ppref_real.shape != (int(np.prod(PPREF_SHAPE)),) or ppref_imag.shape != ppref_real.shape:
        raise ValueError("unexpected PPref topology")

    eulers_promoted = read_counted(required[2], np.float64)
    if eulers_promoted.shape != (N_ROTATIONS * 9,):
        raise ValueError(f"unexpected fine-Euler topology: {eulers_promoted.shape}")
    eulers = eulers_promoted.astype(np.float32)
    if not np.array_equal(eulers.astype(np.float64), eulers_promoted):
        raise ValueError("fine Euler dump is not a lossless promotion of float32")
    eulers = eulers.reshape(N_ROTATIONS, 3, 3)

    relion_real_all = read_counted(required[3], np.float64).astype(np.float32)
    relion_imag_all = read_counted(required[4], np.float64).astype(np.float32)
    expected_all = 32 * N_PIXELS
    if relion_real_all.shape != (expected_all,) or relion_imag_all.shape != (expected_all,):
        raise ValueError("unexpected RELION fine-reference topology")
    relion_all = (relion_real_all + 1j * relion_imag_all).reshape(32, N_PIXELS)
    for rotation in range(N_ROTATIONS):
        first = relion_all[rotation * 4]
        for translation in range(1, 4):
            if not np.array_equal(first, relion_all[rotation * 4 + translation]):
                raise ValueError("RELION projected references vary across translations")
    relion_reference = np.ascontiguousarray(relion_all[::4], dtype=np.complex64)

    with np.load(recovar_capture, allow_pickle=False) as capture:
        rotations = np.asarray(capture["rotations"], dtype=np.float32)
        if rotations.shape != (N_ROTATIONS, 3, 3):
            raise ValueError("unexpected RECOVAR rotation topology")
        if not np.array_equal(rotations, np.swapaxes(eulers, 1, 2)):
            raise ValueError("RELION device Eulers and RECOVAR rotations are not exactly transposes")
        mapping = relion_to_recovar_columns(capture["window_indices"])
        recovar_projection = np.asarray(capture["proj_half"], dtype=np.complex64)
    if recovar_projection.shape != (N_ROTATIONS, N_PIXELS):
        raise ValueError("unexpected RECOVAR projected-reference topology")
    recovar_reference = np.ascontiguousarray(
        recovar_projection[:, mapping] / np.float32(DENSE_SCALE), dtype=np.complex64
    )

    expected_mapping = int(mapping[SPECIAL_RELION_PIXEL])
    if expected_mapping != 641:
        raise ValueError(f"special identity drifted: RELION pixel {SPECIAL_RELION_PIXEL} maps to {expected_mapping}")
    special_delta = (
        recovar_reference[SPECIAL_ROTATION, SPECIAL_RELION_PIXEL]
        - relion_reference[SPECIAL_ROTATION, SPECIAL_RELION_PIXEL]
    )
    if not 1e-7 < abs(special_delta) < 1e-5:
        raise ValueError(f"special projected-reference gap drifted: {special_delta}")

    inputs = output / "inputs"
    inputs.mkdir(parents=True)
    write_raw(inputs / "ppref_real.f32", ppref_real, np.float32)
    write_raw(inputs / "ppref_imag.f32", ppref_imag, np.float32)
    write_raw(inputs / "eulers.f32", eulers, np.float32)
    write_raw(inputs / "relion_reference.f32x2", relion_reference, np.complex64)
    write_raw(inputs / "recovar_reference.f32x2", recovar_reference, np.complex64)
    write_raw(inputs / "relion_to_recovar_column.i32", mapping, np.int32)

    source_hashes = {str(path.resolve()): sha256(path) for path in required}
    input_hashes = {path.name: sha256(path) for path in sorted(inputs.iterdir())}
    special_relion = relion_reference[SPECIAL_ROTATION, SPECIAL_RELION_PIXEL]
    special_recovar = recovar_reference[SPECIAL_ROTATION, SPECIAL_RELION_PIXEL]
    return {
        "schema": "k4_p3591_projector_coordinate_inputs_v1",
        "source_artifact_root": str(artifact_root.resolve()),
        "source_operand_root": str(operand_root.resolve()),
        "source_hashes": source_hashes,
        "input_hashes": input_hashes,
        "dimensions": {
            "rotations": N_ROTATIONS,
            "image_size": IMAGE_SIZE,
            "half_width": HALF_WIDTH,
            "pixels": N_PIXELS,
            "ppref_shape_zyx": list(PPREF_SHAPE),
            "coordinate_fields": [
                "raw_x",
                "raw_y",
                "raw_z",
                "post_hermitian_x",
                "post_hermitian_y",
                "post_hermitian_z",
                "texture_x",
                "texture_y",
                "texture_z",
            ],
        },
        "frozen_semantics": {
            "ppref_class_zero_based": 1,
            "ppref_rank": 1,
            "ppref_r_max": 20,
            "padding_factor": 2,
            "current_image_size": 40,
            "recovar_dense_scale": DENSE_SCALE,
            "euler_relation": "recovar_rotation == transpose(relion_device_euler), exact float32",
        },
        "special_identity": {
            "canonical_recovar_particle_zero_based": 3591,
            "relion_internal_particle_zero_based": 2882,
            "immutable_rlnImageName": "3592@particles.128.mrcs",
            "fine_candidate_index_zero_based": 18,
            "fine_hidden_id": 899570,
            "rotation_row_zero_based": SPECIAL_ROTATION,
            "translation_index_zero_based": 42,
            "relion_pixel_zero_based": SPECIAL_RELION_PIXEL,
            "relion_pixel_yx": relion_pixel_coordinates()[SPECIAL_RELION_PIXEL].tolist(),
            "recovar_window_column_zero_based": expected_mapping,
            "relion_reference_real_imag": [float(special_relion.real), float(special_relion.imag)],
            "relion_reference_bits": float_bits([special_relion.real, special_relion.imag]),
            "recovar_reference_real_imag": [float(special_recovar.real), float(special_recovar.imag)],
            "recovar_reference_bits": float_bits([special_recovar.real, special_recovar.imag]),
            "complex_abs_gap": float(abs(special_delta)),
        },
    }


def run_checked(command: list[str], *, cwd: Path | None = None) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(command, cwd=cwd, text=True, capture_output=True)
    if result.returncode:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stdout}\n{result.stderr}"
        )
    return result


def render_sbatch(root: Path, python: Path) -> str:
    return f"""#!/usr/bin/env bash
#SBATCH --job-name=k4_proj_coord
#SBATCH --output={root}/logs/%x_%j.out
#SBATCH --error={root}/logs/%x_%j.err
#SBATCH --partition=cryoem
#SBATCH --account=gilles
#SBATCH --constraint=h100
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=32G
#SBATCH --time=00:10:00

set -euo pipefail
ROOT={root}
PY={python}
RUNTIME=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/runtime/k4_projector_coordinate_${{SLURM_JOB_ID}}
mkdir -p "${{RUNTIME}}"/{{tmp,pixi_home,rattler_cache}} "${{ROOT}}/provenance"
touch "${{RUNTIME}}/SAFE_TO_DELETE"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
unset CONDA_DEFAULT_ENV CONDA_EXE CONDA_PYTHON_EXE CONDA_PROMPT_MODIFIER CONDA_SHLVL
export PYTHONNOUSERSITE=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export TMPDIR="${{RUNTIME}}/tmp"
export PIXI_HOME="${{RUNTIME}}/pixi_home"
export RATTLER_CACHE_DIR="${{RUNTIME}}/rattler_cache"

test -f "${{ROOT}}/SAFE_TO_DELETE"
test -x "${{ROOT}}/bin/projector_coordinate_harness"
test -x "${{PY}}"
test -n "${{CUDA_VISIBLE_DEVICES-}}"
[[ "${{CUDA_VISIBLE_DEVICES}}" != *,* ]]
test -z "$(find "${{ROOT}}/results" -mindepth 1 -type f -print -quit)"
(cd "${{ROOT}}" && sha256sum -c prepared_artifacts.sha256)
"${{PY}}" "${{ROOT}}/validate.py" --root "${{ROOT}}" --inputs-only

source /etc/profile.d/modules.sh
export PS1=${{PS1-}}
set +u
module purge
module load cudatoolkit/12.6
set -u
GPU_RECORD=$(nvidia-smi --query-gpu=index,name,uuid,compute_cap,memory.total,driver_version --format=csv,noheader)
grep -q 'NVIDIA H100' <<<"${{GPU_RECORD}}"
grep -Eq '(^|, )9\\.0(,|$)' <<<"${{GPU_RECORD}}"

{{
  date --iso-8601=seconds
  echo "job_id=${{SLURM_JOB_ID}}"
  echo "node=$(hostname)"
  echo "cuda_visible_devices=${{CUDA_VISIBLE_DEVICES}}"
  echo "slurm_job_gpus=${{SLURM_JOB_GPUS-}}"
  echo "${{GPU_RECORD}}"
  cat /proc/self/cgroup
  scontrol show job -dd "${{SLURM_JOB_ID}}"
  sha256sum "${{ROOT}}/bin/projector_coordinate_harness"
}} > "${{ROOT}}/provenance/run_${{SLURM_JOB_ID}}.txt" 2>&1

"${{ROOT}}/bin/projector_coordinate_harness" "${{ROOT}}/inputs" "${{ROOT}}/results"
"${{PY}}" "${{ROOT}}/validate.py" --root "${{ROOT}}" \
  --output "${{ROOT}}/analysis/projector_coordinate_report.json"
grep -q '"status": "pass"' "${{ROOT}}/analysis/projector_coordinate_report.json"
find "${{ROOT}}/results" "${{ROOT}}/analysis" "${{ROOT}}/provenance" \
  -type f -print0 | sort -z | xargs -0 sha256sum \
  > "${{ROOT}}/provenance/run_artifacts_${{SLURM_JOB_ID}}.sha256"
echo "K4_PROJECTOR_COORDINATE_PASS job=${{SLURM_JOB_ID}}"
"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--artifact-root", type=Path, required=True)
    parser.add_argument("--operand-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--nvcc", default="nvcc")
    args = parser.parse_args()

    output = args.output_root.resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"output root is not empty: {output}")
    output.mkdir(parents=True, exist_ok=True)
    for name in ("analysis", "bin", "logs", "provenance", "results", "src"):
        (output / name).mkdir()
    (output / "SAFE_TO_DELETE").write_text("Disposable K4 particle-3591 projector-coordinate diagnostic.\n")

    here = Path(__file__).resolve().parent
    source = here / "projector_coordinate_harness.cu"
    validator = here / "validate.py"
    shutil.copy2(source, output / "src" / source.name)
    shutil.copy2(validator, output / "validate.py")
    manifest = build_inputs(args.artifact_root.resolve(), args.operand_root.resolve(), output)
    (output / "input_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")

    binary = output / "bin" / "projector_coordinate_harness"
    build_command = [
        args.nvcc,
        "-std=c++17",
        "-O3",
        "-lineinfo",
        "--generate-code=arch=compute_90,code=sm_90",
        "-o",
        str(binary),
        str(output / "src" / source.name),
    ]
    build = run_checked(build_command)
    (output / "build.log").write_text("$ " + " ".join(build_command) + "\n" + build.stdout + build.stderr)
    os.chmod(binary, 0o755)
    nvcc_version = run_checked([args.nvcc, "--version"]).stdout
    cuobjdump = shutil.which("cuobjdump")
    cubin_listing = run_checked([cuobjdump, "--list-elf", str(binary)]).stdout if cuobjdump else ""
    if "sm_90" not in cubin_listing:
        raise ValueError("compiled harness does not contain an sm_90 ELF image")

    sbatch = output / "run.sbatch"
    sbatch.write_text(render_sbatch(output, args.python.resolve()))
    os.chmod(sbatch, 0o755)
    (output / "README.md").write_text(
        "# Frozen K=4 particle-3591 projector coordinate microharness\n\n"
        "This target-only H100 diagnostic first compares the exact float32 bytes staged by "
        "the RELION-direct and RECOVAR full-volume-to-compact texture paths. It then evaluates "
        "the same 8 rotations by 840 Fourier pixels using current RECOVAR source order, exact "
        "RELION source order, two explicit FMA orders, explicit noncontracted multiply/add, and "
        "the adjacent 1/256 y interpolation bins. All coordinate variants share one frozen "
        "RECOVAR-staged texture; a separate RELION-direct replay checks the captured oracle.\n\n"
        "The special fail-closed identity is particle 3591, fine candidate 18 (hidden 899570), "
        "rotation row 4, translation 42, RELION pixel 242 (y=11, x=11), mapped to RECOVAR "
        "window column 641. Intermediate comparisons use exact "
        "array metrics; this microharness does not make a map-quality claim.\n\n"
        "Submit exactly:\n\n"
        f"```bash\ncd {output}\nsbatch --parsable {sbatch}\n```\n"
    )
    static_validation = {
        "schema": "k4_p3591_projector_coordinate_prepared_v1",
        "status": "pass",
        "root": str(output),
        "source_worktree_head": run_checked(["git", "-C", str(here), "rev-parse", "HEAD"]).stdout.strip(),
        "source_worktree_diff_sha256": hashlib.sha256(
            run_checked(["git", "-C", str(here), "diff", "--binary"]).stdout.encode()
        ).hexdigest(),
        "nvcc_version": nvcc_version.strip(),
        "binary_sha256": sha256(binary),
        "cubin_listing": cubin_listing.splitlines(),
        "special_identity": manifest["special_identity"],
        "limitations": [
            "The harness replays frozen projector inputs; it does not tap an ephemeral production register.",
            "RELION PPref source values were captured before initMdl and were losslessly serialized as float32.",
            "The RECOVAR stage readback is captured before cudaMemcpy3D, but only the submitted H100 run executes it.",
            "Adjacent-bin samples diagnose the hardware interpolation threshold; they are not production alternatives.",
            "No FSC/FSC-AUC map claim is possible from this one-particle projector microharness.",
        ],
    }
    (output / "static_validation.json").write_text(json.dumps(static_validation, indent=2, sort_keys=True) + "\n")

    prepared_files = sorted(
        path for path in output.rglob("*") if path.is_file() and path.name != "prepared_artifacts.sha256"
    )
    (output / "prepared_artifacts.sha256").write_text(
        "".join(f"{sha256(path)}  {path.relative_to(output)}\n" for path in prepared_files)
    )
    print(json.dumps(static_validation, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
