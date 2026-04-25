# InitialModel E-step BPref parity — attack plan (2026-04-25)

## Where we are

End-to-end iter-1 BPref CC (small fixture, 500/64):
- raw run_em + recovar grid + gaussian: **+0.49**
- + RELION-exact rotations + R_from_relion + halfset-sort + gaussian (coherent): **+0.59**

The M-step chain and layout converter are at machine precision (CC=+1.000000).
Only the E-step accumulator is the gap.

## Per-component diagnosis (Phase B)

Per-cell posterior CC vs RELION's exp_Mweight = ~0 even for the same
particle 0 with RELION-exact rotations + R_from_relion + RELION-sorted
halfsets. argmax rotations are 102° apart. **Image preprocessing /
scoring divergence is the dominant gap, not pruning or perturbation.**

## Insight from existing recovar parity work

The user's key observation: standard EM refinement in recovar achieves
high RELION parity. We've been bypassing it. The existing
`refine_single_volume(mode="relion")` in
`recovar/em/dense_single_volume/iteration_loop.py` already supports:

- `adaptive_oversampling=1`, `adaptive_fraction=0.999`,
  `max_significants=500` — the 2-pass adaptive pruning RELION uses
- `perturb_replay_relion_dir` — replay RELION's per-iter
  SamplingPerturbation values from `run_itNNN_sampling.star`
- `emulate_relion_firstiter_cc=True` — iter-1 binarization
- `first_iteration_score_mode="normalized_cc"` — RELION's iter-1 CC
  scoring path
- `save_intermediates_dir` — dumps per-iter `it{NNN}_Ft_y_{0,1}.npy`

This is the entire parity infrastructure I was hand-rebuilding from
scratch in `gpu_pipeline.run_iter_gpu_vdam`.

## The fixture mismatch

`refine_single_volume(mode="relion")` expects an AUTO-REFINE RELION run
(`--auto_refine --split_random_halves`) with `run_itNNN_half1_model.star`
files. The InitialModel fixture used here has `--grad --denovo_3dref`
output without those files. The replay path doesn't directly apply.

## Right next attack (for next session)

Two parallel paths:

### Path A — auto-refine equivalence test
1. Run RELION in `--auto_refine --split_random_halves` mode on the same
   500-particle box-64 fixture.
2. Add the same instrumentation (`docs/patches/relion_estep_dump.patch`
   already in place; just enable for `--auto_refine` runs).
3. Drive `refine_single_volume(mode="relion", perturb_replay_relion_dir=...,
   adaptive_oversampling=1, save_intermediates_dir=...)` for iter 1.
4. Compare `it000_Ft_y_0.npy` → BPref layout vs RELION's dumped
   `pipe_it1_c0_bp_data_pre_reweight.bin`.
5. Expected: high CC. If yes, the codepath works → migrate gpu_pipeline
   to call refine_single_volume's iter machinery for VDAM (with the
   M-step swapped to the bit-exact chain we already have).

### Path B — InitialModel-specific replay
1. Add a `pseudo_halfsets=True` mode to `refine_single_volume(mode="relion")`
   that mirrors RELION's halfset alternation logic.
2. Add the equivalent of `--grad --denovo_3dref` E-step variations
   (mostly the absence of FSC join and the gradient blend M-step) on
   top of the existing parity machinery.
3. Drive iter 1 against the existing InitialModel fixture and BPref dump.

Path A is faster to validate (proves the codepath works at scale) but
requires a new RELION run. Path B reuses our fixture but extends the
refine_single_volume API.

## What we already have committed

Branch tip `9b88f285` includes:

- `docs/patches/relion_estep_dump.patch` — RELION E-step instrumentation
  (exp_Mweight, oversampled rotations, oversampled translations,
  perturbation params)
- `scripts/run_relion_dump_small.sh` — single-seed RELION driver for
  coherent dumps
- `scripts/probe_estep_coherent.py` — current Phase-B probe driver
- `recovar/em/initial_model/gpu_pipeline.py::run_iter_gpu_vdam` —
  bit-exact M-step path (CC=+1.0) waiting for an E-step that matches
- `recovar/em/initial_model/gpu_pipeline.py::run_em_output_to_bpref` —
  layout converter (byte-exact round-trip)
- `tests/unit/initial_model/test_large_fixture_parity.py` —
  M-step chain at machine precision on 5k/256

Once the E-step gap is closed, `tests/unit/initial_model/test_estep_fixture.py::test_estep_bpref_forward_parity`
asserts `cc_h0 > 0.9999` (currently soft-baseline).
