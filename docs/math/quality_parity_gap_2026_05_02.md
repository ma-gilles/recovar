# K=1 ab-initio quality parity gap — 2026-05-02

Captured during EM-quality-parity Phase 1 on commit `55a60045`
(branch `claude/em-quality-parity`, derived from
`origin/claude/initial-model-vdam`).

## Replay parity is intact (machine precision)

Replay starting from RELION it003 with `--max_iter 1` against
`data_noise1_5k_normalized` matches RELION's it004 essentially exactly:

```
recovar-vs-RELION half1/half2 corr   = 0.999963
recovar Pmax / RELION Pmax           = 0.9735 / 0.9737   (Δ = 2.0e-4)
pose full_angle vs RELION             = 0.000° (exact)
trans_px vs RELION                    = 0.000 (exact)
recovar GT corr / RELION GT corr      = 0.946768 / 0.946794
```

Provenance gate: all 5 load-bearing parity commits (`7834dc0b`,
`5f21574a`, `0650b550`, `b125883f`, `0903a64c`) are in HEAD's ancestry.
Reproducible via `pixi run test-em-parity-fast`.

## Ab-initio (run_full_refinement) diverges catastrophically

8-iter end-to-end run on the SAME 5k 128² fixture, starting from
`reference_init.mrc` and the same RELION half-set assignments
(`particles_with_halfsets.star`). Slurm n/a (local GPU 1, A100 80GB,
total wall 672.9 s). Artifact:
`/scratch/gpfs/GILLES/mg6942/_agent_scratch/qparity_phase1_abinitio_5k128/`.

Per-iter recovar-vs-RELION (RELION values from
`run_it{N}_half1_model.star::_rlnAveragePmax|_rlnCurrentImageSize|_rlnCurrentResolution`):

| Iter | recovar cs | recovar res Å | recovar Pmax | RELION cs | RELION res Å | RELION Pmax | Pmax gap |
|-----:|----------:|--------------:|-------------:|----------:|-------------:|------------:|----------|
|  1   |    56     | 21.76         | 0.0519       |   56      | 21.76        | 0.04359     |  +19% (recovar HIGHER) |
|  2   |    70     | 38.86 ❌      | 0.5075       |   70      | 21.76        | 0.65118     |  −22% |
|  3   |    48 ❌  | 38.86 ❌      | 0.8810       |   82      | 22.67        | 0.96470     |  −9%  |
|  4   |    48     | 38.86 ❌      | 0.7892       |   80      | 22.67        | 0.97374     |  −19% |
|  5   |    48     | 36.27 ❌      | 0.7015       |   80      | 22.67        | 0.97161     |  −28% |
|  6   |    50     | 36.27 ❌      | 0.5023       |   80      | 21.76        | 0.92843     |  −46% |
|  7   |    50     | 36.27 ❌      | 0.6260       |   82      | 21.76        | 0.95024     |  −34% |
|  8   |    50     | 36.27 ❌      | 0.6917       |   82      | 20.92        | 0.88368     |  −22% |

Final reconstruction quality vs ground truth (merged half maps):

| series                     | corr_vs_GT | FSC<0.5 shell | FSC<0.5 res Å |
|----------------------------|-----------:|--------------:|--------------:|
| recovar (this run, iter 8) |  0.495391  |             3 |     181.33    |
| RELION it008 half1         |  0.959559  |            36 |      15.11    |

## What this tells us

1. **Iteration 1 already diverges.** Same `current_size=56`, same
   resolution=21.76 Å, but recovar's `ave_Pmax=0.0519` is **19% HIGHER**
   than RELION's `0.04359`. A higher average max-posterior at iter 1
   means the posterior is sharper. Likely sources:
     * temperature/normalization mismatch in the iter-1 score
     * different sigma2_noise bootstrap (we estimate from masked images,
       but the mask convention or shell-radial mapping may not match
       RELION's exact iter-0 noise prior)
     * different initial reference filtering — RELION's `--ini_high 30`
       does both (a) cap `current_size` AND (b) low-pass the input
       volume; `run_full_refinement.py` only does (a), relying on
       `reference_init.mrc` being pre-filtered (it is — to ~108 Å,
       MORE aggressive than RELION's 30 Å, so this should be OK)

2. **Resolution regresses at iter 2** (21.76 → 38.86 Å) and never
   recovers. The convergence-state machinery in
   `recovar/em/dense_single_volume/iteration_loop.py` and
   `recovar/em/dense_single_volume/helpers/convergence.py` is computing
   the FSC-derived "current resolution" differently than RELION at the
   transition between iteration boundaries. RELION's
   `_rlnCurrentResolution` stays steady at 21.76 from iter 1 → 2.

3. **`current_size` collapses at iter 3** (70 → 48) while RELION grows
   (70 → 82). This is a direct consequence of (2): with the wrong
   resolution estimate, the next-iter `current_size` cap is computed
   incorrectly.

4. **HEALPix advances at iter 4** (3 → 4) prematurely. RELION stays at
   HEALPix order 3 through it008. This is likely a knock-on effect of
   the resolution regression.

## Root-cause hypothesis ranking

In order of suspicion (most likely first):

1. **FSC-to-resolution mapping**. The shell-to-resolution conversion
   may be off by one shell at certain transitions, especially at the
   `--low_resol_join_halves 40` boundary. Compare
   `compute_relion_fsc_from_backprojector` in
   `recovar/reconstruction/regularization.py` line-by-line against
   `BackProjector::calculateDownSampledFourierShellCorrelation` in
   `/scratch/gpfs/GILLES/mg6942/relion/src/backprojector.cpp`.
2. **Iter-1 sigma2_noise / tau2 prior**. RELION's iter-1 uses a
   data-derived initial noise estimate that differs from
   `estimate_initial_noise_spectrum_from_unaligned_images`. Verify
   the noise spectrum at iter 1 matches RELION's
   `_rlnSigma2Noise` from `run_it001_half1_model.star`.
3. **Convergence resolution update timing**. Investigate whether
   `convergence.compute_current_resolution()` is called before or after
   the FSC join, and whether it uses the joined or per-half FSC. The
   regression at iter 2 (21.76 → 38.86 Å) suggests it's reading the
   wrong FSC array.
4. **HEALPix advance threshold**. RELION's
   `MlOptimiser::checkConvergence` uses `acc_rot` against the angular
   sampling step. recovar's advance criterion may be triggering on a
   different metric.

## Replay vs ab-initio

The replay test at iter 3→4 PASSES at corr=0.999963. This means the
E-step / M-step / FSC / tau² / SamplingPerturbation kernels are correct
when fed the right inputs. The ab-initio gap is **upstream of those
kernels** — in the per-iteration state-update code (resolution,
current_size, HEALPix order, noise prior). The right place to look is
NOT the parity-fix commits in `recovar/em/dense_single_volume/em_engine.py`
or the M-step code — it's the iteration-level orchestration in
`recovar/em/dense_single_volume/iteration_loop.py` and
`recovar/em/dense_single_volume/helpers/convergence.py`.

## Fixes landed in this branch (commits 9b2c0727, 8d73a014)

1. **`run_full_refinement.py` mask path resolution** — was silently falling
   back to default 0.85/0.99 window mask because it only searched
   `<data_dir>/relion_ref/`, not `<data_dir>/relion_ref*/` (so the common
   `relion_ref_os0/` fixture was missed). Added explicit
   `--relion_optimiser` CLI arg and broadened the candidate search.
2. **`_maybe_apply_relion_image_mask` mode** — was setting only the mask
   geometry, leaving the dataset in the default `multiply` mode.
   Switched to `backend.set_relion_image_mask(...)` which sets both
   geometry and `relion_background_fill` mode (RELION-exact softMaskOutsideMap).

### Effect on iter-by-iter Pmax (5k 128² normalized, all on commit 8d73a014):

| iter | Phase1a Pmax | Phase1c Pmax (mask + bg-fill) | RELION Pmax | gap |
|-----:|-------------:|-------------------------------:|------------:|-----|
|  1   | 0.0519       | 0.0402                         | 0.04359     | -8% |
|  2   | 0.5075       | 0.3866                         | 0.65118     | -41% |
|  3   | 0.8810       | 0.8758                         | 0.96470     | -9% |
|  8   | 0.6917       | 0.7427                         | 0.88368     | -16% |

Final reconstruction quality vs ground truth (merged):

| series | corr_vs_GT | FSC<0.5 res Å |
|--------|-----------:|--------------:|
| Phase1a (no fixes)         | 0.495 | 181.33 |
| Phase1b (mask geometry)    | 0.684 |  68.00 |
| Phase1c (mask + bg-fill)   | 0.683 |  68.00 |
| Phase1d (no tau2 floor)    | 0.682 |  68.00 |
| RELION it008               | 0.960 |  15.11 |

### Iter-1 sigma2_noise after the mask fix

Was 0.75-0.82× RELION's at every shell (causing sharper posterior, noise
reinforcement). Now within 1% of RELION at every shell ≥ 1, within 3% at
shell 0. Bootstrap noise (from
`estimate_initial_noise_spectrum_from_unaligned_images`) and the iter-1
output noise both match RELION.

## Remaining gap: iter-1 backprojection accumulator divergence

After the two fixes, iter-1 sigma2_noise matches RELION and the iter-1
ave_Pmax is within 8% of RELION. But iter-1 gold-standard FSC (h1 vs h2
within recovar) is still systematically lower than RELION's at shells
18+:

| shell | recovar GS FSC | RELION GS FSC | diff |
|------:|---------------:|--------------:|------|
|   14  | 0.984          | 0.988         | -0.004 |
|   16  | 0.941          | 0.962         | -0.022 |
|   18  | 0.856          | 0.919         | -0.063 |
|   20  | 0.778          | 0.879         | -0.101 |
|   22  | 0.589          | 0.770         | -0.181 |
|   26  | 0.136          | 0.476         | -0.340 |

Cross-FSC of recovar's iter-1 half maps vs RELION's iter-1 half maps:
0.99 at shells 5-14, drops to 0.91 at shell 18, 0.87 at shell 20, 0.33 at
shell 26. Half-map STDs match within 1% (0.00964 vs 0.00976). So both
engines produce similar-energy half maps, but recovar's contain
high-frequency content that doesn't correlate with RELION's — and that
high-frequency content is independent between recovar's two halves
(low GS FSC at shells 18+).

Phase 1d (init_mean_variance floor removed) made NO difference to the
iter-1 FSC — confirming the iter-1 reconstruction's tau2 is computed from
the iter-1 FSC, not from the init_mean_variance bootstrap. So the
divergence is in the **iter-1 E-step posteriors** or **iter-1 M-step
backprojection accumulators (Ft_y, Ft_ctf)**, NOT in the bootstrap prior.

## Next steps (for the iter-1 root cause)

1. Dump iter-1 E-step posteriors per particle for recovar and RELION.
   Compare the per-pose probability distributions; if recovar's are
   sharper (more concentrated on a few poses), the noise model in the
   E-step still differs.
2. Dump iter-1 backprojection accumulators (Ft_y, Ft_ctf) and compare
   shell-by-shell. If the accumulators differ at shells 18+, the
   per-pose backprojection weights or the projection convention differ.
3. Audit `recovar/em/dense_single_volume/em_engine.py` lines 1010-1060
   (image_corrections, image_pre_shifts) — `run_full_refinement.py`
   doesn't pass these but they default to None, which should be the same
   as RELION's iter-1 (no per-image normCorrection yet).
4. Compare projection regularization at iter 1 — recovar uses
   ``init_mean_variance`` for E-step projection regularization, RELION
   uses iter-0 tau2 from model.star. We showed those match within 5%
   for shells 0-5, but they differ wildly at shells 6+ (recovar floor
   4e-6 vs RELION ~1e-20). Even though Phase 1d showed this doesn't
   affect iter-1 FSC, it might affect the E-step likelihoods.

## Reproduction

```bash
WORKDIR=/home/mg6942/myscratch/recovar_wt_qparity_20260502_183314_5307
cd $WORKDIR
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
export TMPDIR=/scratch/gpfs/GILLES/mg6942/tmp/qparity_$(date +%s)
mkdir -p "$TMPDIR"
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_PREALLOCATE=false

# Replay (passes at corr ≥ 0.999):
pixi run python scripts/run_multi_iter_parity.py \
  --relion_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0 \
  --data_star  /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star \
  --iter 3 --max_iter 1 \
  --gt_volume  /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/reference_gt.mrc \
  --output_dir /scratch/gpfs/GILLES/mg6942/_agent_scratch/qparity_baseline

# Ab-initio with all fixes (final corr 0.683):
pixi run python scripts/run_full_refinement.py \
  --data_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized \
  --output   /scratch/gpfs/GILLES/mg6942/_agent_scratch/qparity_phase1c_abinitio_5k128_relion_bgfill \
  --max_iter 8 \
  --healpix_order 3 \
  --offset_range 3.0 --offset_step 1.0 \
  --adaptive_oversampling 0 --tau2_fudge 4.0 \
  --perturb_factor 0.5 --perturb_seed 42 \
  --init_resolution 30.0 \
  --image_batch_size 500 \
  --relion_half_sets /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles_with_halfsets.star \
  --relion_optimiser /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0/run_it001_optimiser.star
```
