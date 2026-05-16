# tau2_fudge=-1 sentinel bug in K=1 replay parity (2026-05-15)

## Symptom

On the K=1 100k/256 RELION-parity replay run (job 8255968 against
`pdb_k1_g256_n100000_completion_20260512_171123/relion_autorefine_k1_it015_os0_bayes_clean9d9`),
`ave_Pmax` collapses immediately after iter-1:

| iter | recovar Pmax | RELION Pmax (run_it{NNN}_half1_model.star) |
|------|-------------:|--------------------------------------------:|
| 1    | 0.937        | 0.936                                       |
| 2    | 0.323        | 0.979                                       |
| 3    | 0.156        | 0.999                                       |
| 4    | 0.150        | ≥0.999                                      |

iter-1 is at RELION parity but iter-2+ diverge. The K=1 5k/128 e2e
diagnostic (job 8267612) shows the normal cold-start trajectory (iter1
0.041 → iter3 0.966) — so the collapse is specific to replay mode.

## Root cause

`scripts/run_full_refinement.py::_parse_relion_tau2_fudge` reads
`_rlnTau2FudgeArg` from RELION's `run_it000_optimiser.star`. For K=1
auto-refine where the user did not pass `--tau2_fudge`, RELION writes
`-1.0` to that field as a "use binary default" sentinel
(`ml_optimiser.cpp:881-882`):

```cpp
tau2_fudge_arg = textToFloat(parser.getOption("--tau2_fudge", "...", "-1"));
mymodel.tau2_fudge_factor = tau2_fudge_arg > 0 ? tau2_fudge_arg : 1;
```

Recovar's parser returned `-1.0` verbatim. `_resolve_tau2_fudge` then
took the override path and propagated `-1.0` to the Wiener
reconstruction (`recovar/reconstruction/relion_functions.py:639,673`):

```python
inv_tau = 1 / (oversampling_factor * tau2_fudge * safe_tau)
```

`tau2_fudge = -1` flips the sign of `inv_tau`, so
`regularized_filter = filter_flat + inv_tau` lands at or below zero;
the subsequent `jnp.maximum(.., EPSILON)` clamps it to floor, which
removes regularization entirely. iter-1's reconstructed volume is then
dominated by high-frequency noise. iter-2's E-step on that volume sees
a smeared projector and produces a flat posterior — hence the Pmax
collapse.

K=1 5k/128 sanity runs and the fast EM-parity replay tests pass because
they all set `--tau2_fudge 1.0` (or 4.0 for K=4) on the CLI, bypassing
the optimiser-star override path. K=2/K=4 fixtures pass because their
optimiser.star has `_rlnTau2FudgeArg = 4.0` (from
`pipeline_jobs.cpp::initialiseClass3DJob` which always emits
`--tau2_fudge`).

## Fix

`scripts/run_full_refinement.py`:

1. `_parse_relion_tau2_fudge` now prefers `_rlnTau2FudgeFactor` (the
   actual value RELION used, written in `model.star`) over
   `_rlnTau2FudgeArg`, and treats `_rlnTau2FudgeArg <= 0` as `None` so
   `_resolve_tau2_fudge` falls back to the K-class default. This
   mirrors RELION's `tau2_fudge_arg > 0 ? arg : 1` semantic.
2. The caller now reads `run_it000_half1_model.star` first; the
   optimiser-star branch is only taken when the model-star value is
   absent.

Two new unit tests in `tests/unit/test_run_full_refinement_overrides.py`:

- `test_relion_tau2_fudge_parser_maps_arg_negative_one_to_none`
- `test_relion_tau2_fudge_parser_prefers_factor_over_arg`

## Live job 8255968

Submitted before the fix; its final maps will reflect the bug. Do not
cite its corr/FSC numbers as the K=1 100k/256 completion benchmark.
Re-run with the fix to get the real metrics.
