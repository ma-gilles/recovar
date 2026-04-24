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

This script handles argument parsing + command composition + writing the
assembled command to stdout. Connecting the E-step callback to
`recovar.em.dense_single_volume` kernels is the remaining Phase 5 wiring
against the RELION fixture.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from typing import List, Optional


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
    nr_iter: int = 200
    nr_classes: int = 1
    tau2_fudge: float = 4.0
    sym_name: str = "C1"
    do_run_C1: bool = True
    particle_diameter: float = 200.0
    do_solvent: bool = True  # --flatten_solvent
    do_ctf_correction: bool = True
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

    tokens: List[str] = [
        "relion_refine",
        "--o",
        f"{opts.outputname}",
        "--iter",
        str(opts.nr_iter),
        "--grad",
        "--denovo_3dref",
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
    p.add_argument("--nr_iter", type=int, default=200)
    p.add_argument("--K", dest="nr_classes", type=int, default=1)
    p.add_argument("--tau2_fudge", type=float, default=4.0)
    p.add_argument("--sym", dest="sym_name", default="C1")
    p.add_argument(
        "--do_run_C1", type=int, default=1, help="1 = run in C1 and apply symmetry later, 0 = run in sym_name"
    )
    p.add_argument("--particle_diameter", type=float, default=200.0)
    p.add_argument("--j", dest="nr_threads", type=int, default=1)
    p.add_argument("--nr_mpi", type=int, default=1, help="Rejected at > 1 (RELION behaviour for --grad).")
    p.add_argument("--gpu", dest="gpu_ids", default="", help="If non-empty, --gpu <gpu_ids> is appended")
    p.add_argument("--scratch_dir", default="")
    p.add_argument("--dry_run", action="store_true", help="Only print the assembled command(s)")
    return p.parse_args(argv)


def main(argv: Optional[List[str]] = None) -> int:
    args = _parse_args(argv)
    opts = InitialModelJobOptions(
        fn_img=args.fn_img,
        outputname=args.outputname,
        nr_iter=args.nr_iter,
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

    # Real execution path: the recovar VDAM iteration loop driven by
    # `recovar.em.initial_model.iteration_loop.run_vdam_iterations`. This is
    # connected to the dense-path E-step adapter in a follow-up Phase-5
    # commit once the RELION fixture has been wired to the data loader.
    print(
        "recovar run_ab_initio: real-execution path not yet wired; "
        "use --dry_run to see the RELION-equivalent command assembly.",
        file=sys.stderr,
    )
    print("Assembled relion_refine command:")
    print(" ".join(cmd))
    print("Assembled relion_align_symmetry command:")
    print(" ".join(align_cmd))
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
