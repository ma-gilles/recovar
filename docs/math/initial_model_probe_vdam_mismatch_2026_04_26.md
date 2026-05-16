# InitialModel BPref probe — VDAM vs standard E-M mismatch (2026-04-26)

## Conclusion

`scripts/probe_estep_coherent.py` and the BPref CC gate in
`tests/unit/initial_model/test_estep_fixture.py::test_estep_to_bpref_forward_parity`
compare the wrong things. The +0.74 CC ceiling is structural, not a
backproject-scatter bug.

## What the probe does

1. Loads RELION's iter-1 dumped post-perturbation rotation+translation
   grids and the matched fixture particles.
2. Calls recovar's `run_em(...)` once with those grids → gets standard
   E-step weights × standard M-step accumulator (`Ft_y`, `Ft_ctf`).
3. Crops the centered (N,N,N) accumulator to a (31,31,16) sub-box and
   compares to RELION's `pipe_it1_c0_bp_data_pre_reweight.bin` /
   `pipe_it1_c0_bp_weight.bin`.

## What RELION actually dumped

From `mstep_it1_c0_meta.txt`:
```
grad_current_stepsize=0.500000
effective_stepsize=0.499999
tau2_fudge_factor=4.000000
```

These are **VDAM gradient-update parameters**. RELION's InitialModel
runs with `--grad --denovo_3dref`, which uses the VDAM update path,
not standard E-M. The dumped `pipe_it1_c0_bp_data_pre_reweight.bin` is
the BPref accumulator after VDAM-specific scaling/blending (gradient
step, prior subtraction, tau2-fudge boost) — not the raw standard
M-step accumulator that `run_em` produces.

## Diagnostic evidence

`/scratch/gpfs/GILLES/mg6942/_agent_scratch/converter_sweep/`:

- All 24 axis-permutation × sign-flip × conjugation × half-axis
  variants of the layout converter give CC in [-0.745, +0.745]. No
  permutation reaches +0.99. → Not a layout bug.
- Magnitude ratios: `‖ours‖/‖target‖_data = 8.86e-5`,
  `‖ours‖/‖target‖_weight = 4.35e-8`. Recovar's accumulator is
  10⁴–10⁷× smaller. → Not a scaling-by-N² bug; this is the VDAM
  step-size + tau2-fudge scaling that the standard `run_em` doesn't
  replicate.

## What the probe was conflating

> "M-step is at parity, gap is E-step" (from
> `~/.claude/projects/-scratch-.../memory/project_relion_parity_mstep_at_parity.md`)

That memory was written for the standard E-M iteration loop
(`_run_relion_iteration_loop`), where 5k iter-13→14 reaches gap
-1.07e-4 vs codex gold. It does **not** apply to the InitialModel
VDAM path, which:

- Uses `--grad` gradient-step rather than `M_new = accumulator / weight`
- Applies `grad_current_stepsize` blending against the prior volume
- Uses `tau2_fudge_factor=4` regularization scaling
- Skips gridding correction in the BPref accumulator (`bpref_skip_gridding=1`)

## Recommendation

Either:

1. **Drop the InitialModel BPref probe + parity gate.** The current
   probe is structurally broken: it compares standard-E-M output to
   VDAM output. Resume parity work via `_run_relion_iteration_loop`
   on the canonical 5k codex fixture (where gap is -1.07e-4 already).

2. **Port RELION's VDAM update path** into recovar. This is a
   significant separate engineering effort: the VDAM iteration
   sequence in `ml_optimiser.cpp::doVDAMupdate` and surrounding code
   needs replicating. Not a patch; a new module.

The active iter-1 cold-start gap on the standard E-M path is **-17.6%**
(see `recovar/em/CLAUDE.md` "Active known gaps") — that's the real
unresolved parity work, not the +0.74 InitialModel CC plateau.

## Branch state

This finding is on `claude/initial-model-vdam` rebased onto
`claude/relion-parity-local-search-fix` (rebase tip: `51ef7e58`,
already documenting the round-12 localization claim, which we now
supersede with this VDAM-mismatch diagnosis).
