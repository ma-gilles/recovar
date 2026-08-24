"""Standalone driver for RELION InitialModel / VDAM ab-initio refinement.

Equivalent to the GUI-generated command

    relion_refine --grad --denovo_3dref [...] --pad 1 --auto_sampling [...]
    relion_align_symmetry --i <last_model.star> --o initial_model.mrc [...]

Rejects MPI (RELION's `pipeline_jobs.cpp:3437` does the same).

Use:

    pixi run python scripts/run_ab_initio.py \\
        --i particles.star --o out/run \\
        --nr_iter 200 --K 1 --sym C1 \\
        --particle_diameter 200 --tau2_fudge 4

This script handles argument parsing + RELION command composition. Non-dry
runs execute the native recovar InitialModel path implemented in
`recovar.em.initial_model.driver`, which uses the dense K-class E-step adapter
with the VDAM iteration loop.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

from recovar.em.initial_model.schedules import GuiInitialModelDefaults

INITIAL_MODEL_GUI_DEFAULTS = GuiInitialModelDefaults()

_CONCRETE_RECOVAR_PROVENANCE_MODULES = (
    "recovar.em.initial_model.schedules",
    "recovar.em.initial_model.driver",
    "recovar.em.initial_model.iteration_loop",
    "recovar.em.initial_model.dense_adapter",
)


def _assert_expected_repo_imports() -> dict[str, str]:
    """Fail fast when InitialModel resolves through another editable checkout."""
    expected_root_value = os.environ.get("RECOVAR_EXPECTED_REPO_ROOT")
    if not expected_root_value:
        return {}

    expected_root = Path(expected_root_value).expanduser().resolve()
    imported: dict[str, str] = {}
    failures: list[str] = []
    for module_name in _CONCRETE_RECOVAR_PROVENANCE_MODULES:
        module = importlib.import_module(module_name)
        module_file_value = getattr(module, "__file__", None)
        module_file = Path(module_file_value).resolve() if module_file_value else None
        imported[module_name] = str(module_file)
        print(f"InitialModel import provenance: {module_name}={module_file}", flush=True)
        if module_file is None or not module_file.is_relative_to(expected_root):
            failures.append(f"{module_name}={module_file}")
    if failures:
        raise RuntimeError(
            "RECOVAR InitialModel import provenance failure: expected every concrete module under "
            f"{expected_root}, found " + ", ".join(failures)
        )
    return imported


def _reject_mpi() -> None:
    """Match RELION's pipeline_jobs.cpp:3435-3439 behaviour."""
    # There is no explicit MPI flag in this driver; RELION's check triggers
    # on `nr_mpi > 1`. If a user passes `--nr_mpi N` with N>1 we reject.
    raise SystemExit("ERROR: Gradient refinement is not supported together with MPI.")


@dataclass
class InitialModelJobOptions:
    """One-to-one mapping of the GUI InitialModel job options.

    Defaults mirror pipeline_jobs.cpp:3376-3425.
    """

    fn_img: str = ""
    outputname: str = "ab_initio/run"
    nr_iter: int = INITIAL_MODEL_GUI_DEFAULTS.nr_iter
    grad_write_iter: int = INITIAL_MODEL_GUI_DEFAULTS.grad_write_iter
    nr_classes: int = INITIAL_MODEL_GUI_DEFAULTS.nr_classes
    tau2_fudge: float = INITIAL_MODEL_GUI_DEFAULTS.tau2_fudge
    sym_name: str = INITIAL_MODEL_GUI_DEFAULTS.sym_name
    do_run_C1: bool = INITIAL_MODEL_GUI_DEFAULTS.do_run_C1
    particle_diameter: float = INITIAL_MODEL_GUI_DEFAULTS.particle_diameter
    do_solvent: bool = INITIAL_MODEL_GUI_DEFAULTS.do_solvent  # --flatten_solvent
    do_ctf_correction: bool = INITIAL_MODEL_GUI_DEFAULTS.do_ctf_correction
    ctf_intact_first_peak: bool = False
    do_parallel_discio: bool = True
    nr_pool: int = 3
    do_preread_images: bool = False
    scratch_dir: str = ""
    do_combine_thru_disc: bool = False
    use_gpu: bool = False
    gpu_ids: str = ""
    nr_threads: int = 1
    other_args: str = ""
    nr_mpi: int = 1


def build_command(opts: InitialModelJobOptions) -> List[str]:
    """Compose the RELION command verbatim per pipeline_jobs.cpp:3428-3613.

    Returns the list of tokens (not a shell string) so callers can shlex
    or exec directly.
    """
    if opts.nr_mpi > 1:
        _reject_mpi()
    if not opts.fn_img:
        raise SystemExit("ERROR: empty field for input STAR file (fn_img)")
    if opts.grad_write_iter < 1:
        raise SystemExit("ERROR: grad_write_iter must be >= 1")

    tokens: List[str] = [
        "relion_refine",
        "--o",
        f"{opts.outputname}",
        "--iter",
        str(opts.nr_iter),
        "--grad",
        "--denovo_3dref",
        "--grad_write_iter",
        str(opts.grad_write_iter),
        "--i",
        opts.fn_img,
    ]

    if opts.do_ctf_correction:
        tokens.append("--ctf")
        if opts.ctf_intact_first_peak:
            tokens.append("--ctf_intact_first_peak")

    tokens += ["--K", str(opts.nr_classes)]

    # sym handling
    if opts.do_run_C1:
        tokens += ["--sym", "C1"]
    else:
        tokens += ["--sym", opts.sym_name]

    if opts.do_solvent:
        tokens.append("--flatten_solvent")
    tokens.append("--zero_mask")

    if not opts.do_combine_thru_disc:
        tokens.append("--dont_combine_weights_via_disc")
    if not opts.do_parallel_discio:
        tokens.append("--no_parallel_disc_io")
    if opts.do_preread_images:
        tokens.append("--preread_images")
    elif opts.scratch_dir:
        tokens += ["--scratch_dir", opts.scratch_dir]

    tokens += ["--pool", str(opts.nr_pool)]

    tokens.append("--pad")
    tokens.append("1")

    tokens += ["--particle_diameter", str(opts.particle_diameter)]
    tokens += [
        "--oversampling",
        "1",
        "--healpix_order",
        "1",
        "--offset_range",
        "6",
        "--offset_step",
        "2",
        "--auto_sampling",
    ]
    tokens += ["--tau2_fudge", str(opts.tau2_fudge)]
    tokens += ["--j", str(opts.nr_threads)]

    if opts.use_gpu:
        tokens += ["--gpu", opts.gpu_ids]

    if opts.other_args:
        tokens.append(opts.other_args)

    return tokens


def build_align_symmetry_command(outputname: str, nr_iter: int, sym_name: str, do_run_C1: bool) -> List[str]:
    """Mirror the second command emitted by getCommandsInimodelJob
    (pipeline_jobs.cpp:3573-3588).
    """
    fn_model = f"{outputname}_it{nr_iter:03d}_model.star"
    out_mrc = outputname.rstrip("run") + "initial_model.mrc"
    tokens = [
        "relion_align_symmetry",
        "--i",
        fn_model,
        "--o",
        out_mrc,
    ]
    if do_run_C1 and sym_name not in ("C1", "c1"):
        tokens += ["--sym", sym_name]
    else:
        tokens += ["--sym", "C1"]
    tokens += ["--apply_sym", "--select_largest_class"]
    return tokens


def _parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="recovar RELION-parity InitialModel/VDAM driver",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--i", dest="fn_img", required=True, help="Input images STAR file")
    p.add_argument("--o", dest="outputname", default="ab_initio/run", help="Output name prefix (no trailing slash)")
    p.add_argument("--nr_iter", type=int, default=INITIAL_MODEL_GUI_DEFAULTS.nr_iter)
    p.add_argument(
        "--grad_write_iter",
        type=int,
        default=INITIAL_MODEL_GUI_DEFAULTS.grad_write_iter,
        help="Write trajectory artifacts every N iterations and at the final iteration",
    )
    p.add_argument("--K", dest="nr_classes", type=int, default=INITIAL_MODEL_GUI_DEFAULTS.nr_classes)
    p.add_argument("--tau2_fudge", type=float, default=INITIAL_MODEL_GUI_DEFAULTS.tau2_fudge)
    p.add_argument("--sym", dest="sym_name", default=INITIAL_MODEL_GUI_DEFAULTS.sym_name)
    p.add_argument(
        "--do_run_C1",
        type=int,
        default=int(INITIAL_MODEL_GUI_DEFAULTS.do_run_C1),
        help="1 = run in C1 and apply symmetry later, 0 = run in sym_name",
    )
    p.add_argument(
        "--particle_diameter",
        type=float,
        default=INITIAL_MODEL_GUI_DEFAULTS.particle_diameter,
    )
    p.add_argument("--j", dest="nr_threads", type=int, default=1)
    p.add_argument("--nr_mpi", type=int, default=1, help="Rejected at > 1 (RELION behaviour for --grad).")
    p.add_argument("--gpu", dest="gpu_ids", default="", help="If non-empty, --gpu <gpu_ids> is appended")
    p.add_argument("--scratch_dir", default="")
    p.add_argument("--datadir", default=None, help="Directory used to resolve relative STAR image paths")
    p.add_argument("--strip_prefix", default=None, help="Prefix to strip from STAR image paths before --datadir")
    p.add_argument(
        "--random_seed",
        type=int,
        default=INITIAL_MODEL_GUI_DEFAULTS.random_seed,
        help="Native path seed for bootstrap and VDAM subsets",
    )
    p.add_argument(
        "--healpix_order",
        type=int,
        default=INITIAL_MODEL_GUI_DEFAULTS.healpix_order,
        help="Native path base Healpix order",
    )
    p.add_argument(
        "--oversampling",
        type=int,
        default=INITIAL_MODEL_GUI_DEFAULTS.oversampling,
        help="Native path adaptive oversampling level",
    )
    p.add_argument(
        "--offset_range",
        type=float,
        default=INITIAL_MODEL_GUI_DEFAULTS.offset_range_px,
        help="Native path translation search range in pixels",
    )
    p.add_argument(
        "--offset_step",
        type=float,
        default=INITIAL_MODEL_GUI_DEFAULTS.offset_step_px,
        help="Native path translation search step in pixels",
    )
    p.add_argument(
        "--perturbation_factor",
        type=float,
        default=INITIAL_MODEL_GUI_DEFAULTS.perturbation_factor,
    )
    p.add_argument("--random_perturbation", type=float, default=None)
    p.add_argument("--image_batch_size", type=int, default=INITIAL_MODEL_GUI_DEFAULTS.image_batch_size)
    p.add_argument("--rotation_block_size", type=int, default=INITIAL_MODEL_GUI_DEFAULTS.rotation_block_size)
    p.add_argument(
        "--image_fourier_backend",
        choices=("auto", "host_numpy", "jax_gpu", "relion_cuda"),
        default="auto",
        help=(
            "Image preprocessing backend. 'auto' selects the RELION CUDA path "
            "for GPU runs and the host NumPy path otherwise."
        ),
    )
    p.add_argument(
        "--bootstrap_min_particles",
        type=int,
        default=INITIAL_MODEL_GUI_DEFAULTS.bootstrap_min_particles,
    )
    p.add_argument(
        "--sigma2_min_particles",
        type=int,
        default=INITIAL_MODEL_GUI_DEFAULTS.sigma2_min_particles,
    )
    p.add_argument("--translation_sigma_angstrom", type=float, default=None)
    p.add_argument("--eager_images", action="store_true", help="Load image stack eagerly instead of lazily")
    p.add_argument("--no_iter_artifacts", action="store_true", help="Only write final native output artifacts")
    p.add_argument(
        "--require_custom_cuda",
        action="store_true",
        help="Fail before InitialModel execution unless the GPU custom CUDA FFI path is ready",
    )
    cuda_launch_mode = p.add_mutually_exclusive_group()
    cuda_launch_mode.add_argument(
        "--deterministic_cuda",
        action="store_true",
        help=(
            "Serialize CUDA launches for repeatability diagnostics. This is slower and "
            "does not remove RELION's intra-kernel GPU reduction variability."
        ),
    )
    cuda_launch_mode.add_argument(
        "--allow_async_cuda",
        action="store_true",
        help=(
            "Explicitly select the default asynchronous CUDA launch mode. Retained for "
            "compatibility with earlier parity runners."
        ),
    )
    p.add_argument("--dry_run", action="store_true", help="Only print the assembled command(s)")
    p.add_argument(
        "--diagnostic_stop_after_iteration",
        type=int,
        default=None,
        help=(
            "Stop a diagnostic replay after this numbered iteration while retaining --nr_iter "
            "for every RELION schedule calculation"
        ),
    )
    p.add_argument(
        "--padding_factor",
        type=int,
        default=INITIAL_MODEL_GUI_DEFAULTS.padding_factor,
        help="K-class M-step BPref padding factor. Use 2 to give the M-step the trilinear-interpolation margin RELION's BackProjector expects (closer parity for c2 CC).",
    )
    return p.parse_args(argv)


def _configure_cuda_launch_blocking(*, deterministic_cuda: bool) -> str:
    """Select the InitialModel CUDA launch mode before CUDA starts."""

    value = "1" if bool(deterministic_cuda) else "0"
    os.environ["CUDA_LAUNCH_BLOCKING"] = value
    return value


def _require_custom_cuda_runtime() -> dict:
    import jax

    import recovar.cuda_backproject as cuda_backproject
    from recovar.core import slicing

    slicing._on_gpu.cache_clear()
    devices = [getattr(device, "platform", "") for device in jax.devices()]
    report = {
        "default_backend": jax.default_backend(),
        "device_platforms": devices,
        "slicing_on_gpu": bool(slicing._on_gpu()),
        "custom_cuda_requested": bool(cuda_backproject.custom_cuda_requested()),
        "cuda_available": bool(cuda_backproject.cuda_available()),
    }
    print("RECOVAR InitialModel CUDA runtime gate: " + json.dumps(report, sort_keys=True), flush=True)
    if not report["slicing_on_gpu"]:
        raise RuntimeError(f"InitialModel requires a visible GPU for this run: {report}")
    if not report["custom_cuda_requested"] or not report["cuda_available"]:
        raise cuda_backproject.cuda_unavailable_error()
    return report


def main(argv: Optional[List[str]] = None) -> int:
    _assert_expected_repo_imports()
    args = _parse_args(argv)
    opts = InitialModelJobOptions(
        fn_img=args.fn_img,
        outputname=args.outputname,
        nr_iter=args.nr_iter,
        grad_write_iter=args.grad_write_iter,
        nr_classes=args.nr_classes,
        tau2_fudge=args.tau2_fudge,
        sym_name=args.sym_name,
        do_run_C1=bool(args.do_run_C1),
        particle_diameter=args.particle_diameter,
        nr_threads=args.nr_threads,
        nr_mpi=args.nr_mpi,
        use_gpu=bool(args.gpu_ids),
        gpu_ids=args.gpu_ids,
        scratch_dir=args.scratch_dir,
    )

    cmd = build_command(opts)
    align_cmd = build_align_symmetry_command(opts.outputname, opts.nr_iter, opts.sym_name, opts.do_run_C1)

    if args.dry_run:
        print(" ".join(cmd))
        print(" ".join(align_cmd))
        return 0

    if args.gpu_ids:
        _configure_cuda_launch_blocking(deterministic_cuda=bool(args.deterministic_cuda))

    if args.require_custom_cuda:
        _require_custom_cuda_runtime()

    from recovar.em.initial_model.driver import NativeInitialModelOptions, run_native_initial_model

    native_opts = NativeInitialModelOptions(
        fn_img=opts.fn_img,
        outputname=opts.outputname,
        nr_iter=opts.nr_iter,
        nr_classes=opts.nr_classes,
        tau2_fudge=opts.tau2_fudge,
        sym_name=opts.sym_name,
        do_run_C1=opts.do_run_C1,
        particle_diameter=opts.particle_diameter,
        do_solvent=opts.do_solvent,
        do_zero_mask=True,
        do_ctf_correction=opts.do_ctf_correction,
        random_seed=args.random_seed,
        healpix_order=args.healpix_order,
        oversampling=args.oversampling,
        offset_range_px=args.offset_range,
        offset_step_px=args.offset_step,
        perturbation_factor=args.perturbation_factor,
        random_perturbation=args.random_perturbation,
        image_batch_size=args.image_batch_size,
        rotation_block_size=args.rotation_block_size,
        bootstrap_min_particles=args.bootstrap_min_particles,
        sigma2_min_particles=args.sigma2_min_particles,
        lazy=not args.eager_images,
        datadir=args.datadir,
        strip_prefix=args.strip_prefix,
        translation_sigma_angstrom=args.translation_sigma_angstrom,
        write_iter_artifacts=not args.no_iter_artifacts,
        grad_write_iter=opts.grad_write_iter,
        padding_factor=int(args.padding_factor),
        image_fourier_backend=(
            "relion_cuda"
            if args.image_fourier_backend == "auto" and bool(args.gpu_ids)
            else (
                "host_numpy"
                if args.image_fourier_backend == "auto"
                else args.image_fourier_backend
            )
        ),
        deterministic_cuda=bool(args.deterministic_cuda),
        diagnostic_stop_after_iteration=args.diagnostic_stop_after_iteration,
    )
    result = run_native_initial_model(native_opts)
    print(f"recovar InitialModel complete: {result.final_mrc}")
    print(f"Final model STAR: {result.final_model_star}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
