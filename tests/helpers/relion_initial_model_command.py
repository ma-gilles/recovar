"""Independent RELION InitialModel command reference for parity checks.

These builders mirror the RELION GUI; RECOVAR executes through
``recovar.commands.initial_model``.
"""

from dataclasses import dataclass
from typing import List

from recovar.commands.initial_model import GuiInitialModelDefaults

INITIAL_MODEL_GUI_DEFAULTS = GuiInitialModelDefaults()


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
