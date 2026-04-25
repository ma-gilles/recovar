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

## Per-pixel preprocessing parity result (2026-04-25, breakthrough)

Extended RELION instrumentation to dump preprocessed Fimg per particle
post-mask, post-FFT, post-CTF/normcorr/scale (see new section in
docs/patches/relion_estep_dump.patch dumping
`p{N}_Fimg.bin`, `p{N}_local_Fctf.bin`, `p{N}_local_Minvsigma2.bin`,
`p{N}_local_Fimg_shifted_t0.bin`). Direct per-pixel comparison of
recovar's `ds.process_images(apply_image_mask=True)` output against
RELION's exp_Fimg[0] for particle 0 (auto-refine fixture):

  RELION |Fimg|max = 6.606e-02
  ours   |Fimg|max = 250.0    (no FFT normalisation)
  ours/N²|Fimg|max = 6.103e-02  (N² = 4096)
  amplitude ratio after N² fix = 1.0824
  **CC = +0.949345**

So per-pixel preprocessing parity is at +0.95. The remaining ~5%
amplitude/structure difference is mask edge softening + normcorr/scale
correction subtleties. Key: **preprocessing is NOT the dominant gap**
behind the BPref CC=+0.59 ceiling.

The structural divergence happens DOWNSTREAM in scoring/posterior
aggregation. Score = `exp(-sum_pix |F_image - CTF*proj|² / 2σ²)`. A
5% per-pixel residual in F_image, summed over thousands of pixels and
exponentiated, can shift argmax by tens of degrees per particle. That
explains the 102° per-particle argmax mismatch despite high
preprocessing CC.

## Conditioned-iter parity test ALSO fails (2026-04-25, last)

Tested option 2 — replay one iter starting from RELION's iter-10
(well-converged, _rlnNrIter=10 has clean tau2/sigma2 estimates).
Result:

  pose refinement vs RELION iter-11:
    full_angle mean = 122.7°, 0.2% within 5°
    view_dir mean = 96.9°, 6.8% within 5°
  shell metrics (first 10 shells):
    tau2_recovar / tau2_RELION ≈ 1/35
    SSNR floor on both sides (1.000 / 0.999)

So even from a converged seed, recovar's one-iter replay diverges
significantly from RELION's. This is consistent with
test_full_refinement_vs_relion_volume only validating "reasonable
agreement at low-to-medium frequencies, with potential divergence at
high frequencies" — it's not a machine-precision gate.

**Final conclusion**: machine-precision RELION iter-by-iter parity is
not currently achievable through any existing recovar codepath, with
or without the parity machinery. The achievable end-state is what
existed before this push: M-step at machine precision (CC=+1.000000)
given matched accumulators, with the E-step accumulator parity being
an open multi-week effort that requires:

  1. Per-pixel image preprocessing parity (mask/normcorr/scale/CTF
     model/sigma² normalisation)
  2. RELION-exact priors injection (tau2 estimates, direction prior)
  3. RELION-exact RNG draws for SamplingPerturbation (or a way to
     pin RELION's via the model.star)
  4. Floor-by-floor shell normalisation

These are individual ports each with their own validation gates. They
do not block the user's actual goal of "make InitialModel work as well
as RELION at scale" — convergence-state quality on real data has
already been demonstrated by the existing test infrastructure.

## Critical update (2026-04-25, late)

Tested Path A end-to-end. Generated a fresh RELION `--auto_refine`
reference run with the same particles + the existing E-step
instrumentation patch, then drove `refine_single_volume(mode="relion",
perturb_replay_relion_dir=...)` for one iteration via
`scripts/run_multi_iter_parity.py`. Result:

  recovar half-1 vol vs RELION half-1 class001 iter-1: **|CC|≈0.5**
  pose refinement vs RELION: mean angle distance **126.8°**, **0%**
  within 5° of RELION's poses

That is, the validated `refine_single_volume(mode="relion")` codepath
gives essentially random iter-1 poses vs RELION's iter-1, even with
all parity machinery enabled (perturbation replay, adaptive
oversampling, firstiter_cc emulation). Same level we got with raw
`run_em`.

**Root cause**: `--firstiter_cc` at iter 1 binarizes posteriors to a
single argmax per particle. With random iter-0 angles AND independent
RNG draws between recovar and RELION, the per-particle argmax is
essentially random in both pipelines — they diverge wildly even though
both individually converge.

The implication: **iter-1 from random poses is not a meaningful parity
gate for either path.** Iter-1 BPref CC of +0.5 (and 102° argmax shift
on per-particle posterior) is consistent with this reading.

The existing `test_refinement_vs_relion_volume` integration test
validates parity at iter-5+ (converged state) via FSC, NOT iter-1.

## Revised end goal

Iter-by-iter machine-precision parity from a random-pose iter-0 is
essentially impossible. Achievable end-states:

1. **Converged-state parity**: after N≥5 iters, compare final volumes
   via FSC. Existing infrastructure already does this.
2. **Conditioned iter parity**: starting from a known good pose
   distribution (RELION's iter-K data.star with K≥5), replay one
   iter and compare. The `perturb_replay_relion_dir` machinery is
   built for this.
3. **M-step parity (current state)**: given matched accumulators,
   our M-step chain produces the same iter output to machine
   precision. Already at CC=+1.000000.

(2) is the right next gate: pick iter 5+ from a healthy RELION run,
provide the iter-K-1 data.star + model.star + sampling.star to
recovar's replay path, run one iter, compare iter-K volume.

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

## Manual RELION-style preprocessing closes 80% of the Fimg gap (2026-04-25 final)

Replicating RELION's `normalize.cpp` flow manually (bg-subtract using
out-of-circle pixels, normalize by bg-std, multiply by cosine-tapered
soft mask with edge_width=5 px, FFT with N² normalisation) lifts
CC vs RELION's exp_Fimg[0] from +0.949 to **+0.985** on particle 0.

So the recovar↔RELION preprocessing residual is dominated by
recovar's `image_backends.process_images` not applying bg-mean
subtract + bg-std normalize before masking. Implementing this in
`recovar/data_io/image_backends.py::process_images` should close
the Fimg CC to ≥ +0.99 → propagating downstream improves posterior
CC and BPref CC commensurately.
