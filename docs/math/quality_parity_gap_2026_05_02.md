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

# Ab-initio (catastrophic divergence):
pixi run python scripts/run_full_refinement.py \
  --data_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized \
  --output   /scratch/gpfs/GILLES/mg6942/_agent_scratch/qparity_phase1_abinitio_5k128 \
  --max_iter 8 \
  --healpix_order 3 \
  --offset_range 3.0 --offset_step 1.0 \
  --adaptive_oversampling 0 --tau2_fudge 4.0 \
  --perturb_factor 0.5 --perturb_seed 42 \
  --init_resolution 30.0 \
  --image_batch_size 500 \
  --relion_half_sets /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles_with_halfsets.star
```
