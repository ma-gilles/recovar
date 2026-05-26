"""Post-run `relion_align_symmetry` equivalent.

After VDAM converges, RELION runs `relion_align_symmetry` on the final
`run_itNNN_model.star` to:

  1. Align the reconstructed volume to the user-requested symmetry axes.
  2. Apply the symmetry operators.
  3. Select the largest class and write it as `initial_model.mrc`.

Command assembled at pipeline_jobs.cpp:3574-3588.

This module exposes the post-processing contract; actual symmetry
alignment relies on either:
  - RELION's `relion_align_symmetry` CLI (bind via subprocess), or
  - recovar's own symmetry helpers once wired up.

For Phase-6 the command-composition API is what the parity tests pin;
full-run invocation lands in the Phase-5 fixture-wiring step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class AlignSymmetrySpec:
    """Parameters for the post-run symmetry alignment.

    Matches the tokens assembled at pipeline_jobs.cpp:3574-3588:
      `relion_align_symmetry --i <last_model.star> --o <initial_model.mrc>
        [--sym <G>] --apply_sym --select_largest_class`
    """

    last_model_star: str
    out_mrc: str
    sym_name: str
    do_run_C1: bool = True


def build_align_symmetry_tokens(spec: AlignSymmetrySpec) -> List[str]:
    """Compose the `relion_align_symmetry` command tokens.

    Mirrors pipeline_jobs.cpp:3574-3588 exactly. If `do_run_C1` is True
    and the user-selected sym is non-C1, we pass the real symmetry to
    align to. Otherwise --sym C1 (no-op alignment for C1 runs).
    """
    tokens = [
        "relion_align_symmetry",
        "--i",
        spec.last_model_star,
        "--o",
        spec.out_mrc,
    ]
    if spec.do_run_C1 and spec.sym_name not in ("C1", "c1"):
        tokens += ["--sym", spec.sym_name]
    else:
        tokens += ["--sym", "C1"]
    tokens += ["--apply_sym", "--select_largest_class"]
    return tokens
