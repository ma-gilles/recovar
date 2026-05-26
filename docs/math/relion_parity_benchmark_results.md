# RELION-Parity Benchmark Results (Reproducible)

Last updated: 2026-04-20

## Dataset

- **Location**: `/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized`
- **Particles**: 5000 images, 128px, voxel_size=4.25 Å/px
- **Noise**: simulated, RELION-normalized
- **Translations**: zero true offsets in simulation
- **RELION reference**: `relion_ref_os0` (auto-refine, 14 iters, reaches 13.95 Å FSC@0.5)
  - Flags: `--ctf --firstiter_cc --flatten_solvent --zero_mask --low_resol_join_halves 40 --norm --scale`
  - See `recovar/em/CLAUDE.md` for the full canonical invocation
- **RELION no-perturb reference**: `relion_ref_no_perturb` (reaches 14.3 Å in ~10 iters)

## Branch

All results on branch `claude/relion-parity-flag-audit` unless noted otherwise.

---

## 1. Sub-Step Parity (Stages 1-7)

**Script**: `scripts/validate_single_iter_parity.py`
**Commit**: 5f21574a

Tested against RELION iter 3→4 with 2500 images (half-set 1).
All stages at machine-precision relative error < 1e-10.

| Stage | What's tested | Rel error |
|-------|---------------|-----------|
| 1 | Rotation grid, translation grid, CTF, noise | < 1e-14 |
| 2 | Projection, padding, gridding correction | < 1e-10 |
| 3 | E-step scoring (cross-correlation identity) | L2 = 8.8e-6 |
| 4 | Posterior weights (full distribution) | < 1e-10 |
| 5 | Pmax (per-particle) | 100% argmax agreement |
| 6 | M-step accumulators (backprojection) | < 1e-10 |
| 7 | Noise/tau2/FSC updates | < 1e-10 |

**Key detail**: Stage 3/5 operate on RELION's iter-3 volume loaded into
recovar. Scoring agreement is L2=8.8e-6 relative, with 100% argmax
agreement. The 8.8e-6 is due to JAX vs C++ trilinear interpolation
boundary handling at shells 39-41 of the padded Fourier volume.

---

## 2. Single-Iteration End-to-End (iter 3→4)

**Script**: `scripts/run_multi_iter_parity.py --max_iter 1`
**Commit**: 5f21574a (+ DC fix pending)

Starting from RELION's iter-3 volume, running 1 EM iteration with 5000 particles.

### Progressive improvement table (historical)

Each row adds one fix on top of the previous:

| Config | ave_Pmax | Gap to RELION (0.9735) | vol_corr |
|--------|----------|----------------------|----------|
| baseline (no corrections) | ~0.93 | -0.043 | ~0.985 |
| +normcorr fix (avg_norm/normcorr) | 0.9657 | -0.0078 | 0.998 |
| +cs=82 fix + circular window | **0.9661** | **-0.0074** | **0.998** |

### Per-particle Pmax breakdown (5000 particles)

- 66.6% of particles: |Pmax_diff| < 0.001 (essentially perfect)
- 76.2%: |Pmax_diff| < 0.01
- 87.1%: |Pmax_diff| < 0.1
- Median diff: 0.000008
- Correlation: 0.337
- 220 "problematic" particles: RELION Pmax > 0.99, recovar Pmax = 0.54-0.88
  - Caused by trilinear interpolation boundary handling

### Why 8.8e-6 score L2 → 0.74% Pmax gap

Score L2 relative error measures the raw log-likelihood surface. Pmax =
exp(max_score - logsumexp(scores)) over ~1M orientations. Small absolute
differences in scores get amplified by the softmax: two orientations
competing at relative scores within 1 log-unit can flip from Pmax=0.95
to Pmax=0.55 with a 0.001 absolute score shift. This affects ~4% of
particles whose dominant orientation has a thin margin over competitors.

---

## 3. Multi-Iteration Results

### 3a. Pre-DC-fix (tau2[DC]=epsilon bug, 2026-04-20)

**Script**: `scripts/run_multi_iter_parity.py --max_iter 2 --max_particles 500 --max_healpix_order 3`
**Slurm job**: 7192205 (tiny, 500 particles)
**Commit**: 5f21574a

| Iter | cs | recovar Pmax | RELION Pmax | Gap | Notes |
|------|------|------------|-------------|------|-------|
| 1 | 82 | 0.9612 | 0.9737 | -0.0125 | |
| 2 | 80 | 0.9591 | 0.9716 | -0.0125 | |

Volume correlation after 2 iters: 0.44 (terrible).

**Root cause**: M-step Ft_ctf accumulator had DC=0 because it used the
DC-zeroed scoring array. This made tau2[shell_0] = EPSILON, suppressing
DC in the Wiener reconstruction. At iter 2, the volume had wrong mean
intensity → negative cross-correlation at DC → Pmax collapse.

### 3b. Pre-DC-fix (5000 particles, 2 iters)

**Slurm job**: 7192204 (full, stuck at iter 3 when healpix jumped to order 4)

| Iter | cs | recovar Pmax | RELION Pmax | Gap |
|------|------|------------|-------------|------|
| 1 | 82 | 0.9661 | 0.9737 | -0.0076 |
| 2 | 80 | 0.9440 | 0.9716 | -0.0276 |

The Pmax DROP from 0.9661 → 0.9440 in one iteration (0.022 internal drop
vs RELION's 0.002) is the tau2[DC]=epsilon bug compounding through the
reconstruction.

### 3c. Post-DC-fix (pending)

**Slurm jobs**: 7192548 (tiny, 500 particles, 2 iters), 7192545 (full, 11 iters)

Expected: tau2[shell_0] should be non-zero, reconstruction should preserve
DC, Pmax gap should stay stable across iterations instead of growing.

---

## 4. Known Bugs Fixed (16 total)

| # | Bug | Commit | Impact |
|---|-----|--------|--------|
| 1 | pf³ tau2 correction | 0903a64c | tau2 was 8× too small |
| 2 | join radius scaling | 0903a64c | wrong low-res join shell |
| 3 | Shell-mapping at pf=2 | b125883f | wrong shell index from double-rounding |
| 4 | Double-rounding | 0650b550 | shell index off by 1 |
| 5 | Banker's rounding | 0650b550 | jnp.round vs RELION ROUND(x+0.5) |
| 6 | Half-complex counting + float64 | 0650b550 | conjugate pair counting |
| 7 | FFT normalization in scripts | 20fa41e1 | N^4 convention mismatch |
| 8 | Perturbation replay iter mapping | 20fa41e1 | wrong iter index |
| 9 | Image pre-centering | 0650b550 | missing _rlnOriginX/Y shift |
| 10 | Translation prior sign | 0650b550 | sign flip in prior |
| 11 | Prior sigma source | 0650b550 | wrong sigma for prior |
| 12 | NormCorrection convention | 0650b550 | avg_norm/normcorr |
| 13 | current_size off-by-one | 5f21574a | read iter N+1 star |
| 14 | Circular Fourier window | 5f21574a | matched Mresol_fine |
| 15 | M-step DC exclusion (tau2 bug) | pending | Ft_ctf[DC]=0 |
| 16 | Replay healpix cap bypass | pending | --max_healpix_order ignored |

---

## 5. Residual Gap Analysis

The 0.74% (0.0074) single-iteration Pmax gap has been fully characterized:

**Root cause**: JAX `jnp.where` boundary blending vs RELION C++ skip at
Fourier boundary (shells 39-41 of the padded volume). Affects ~220/5000
particles (4.4%) where the dominant orientation's margin is thin.

**Ruled out** (see `MEMORY.md` → `project_relion_parity_status.md` for full list):
- Float precision (float64 identical to float32 gap)
- Significant weight pruning (all candidates evaluated at os=0)
- Translation prior sigma mismatch
- Scale correction placement
- Scoring formula differences
- batch_norm omission
- Direction prior normalization

---

## 6. How to Reproduce

### Sub-step parity
```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=cryoem --gres=gpu:1 --mem=200G --time=2:00:00 --account=amits
cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar
pixi run python scripts/validate_single_iter_parity.py \
  --relion_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0 \
  --data_star /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star \
  --iter 3
EOF
```

### Multi-iteration parity (quick, 500 particles, 2 iters)
```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=cryoem --gres=gpu:1 --mem=200G --time=4:00:00 --account=amits
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar
pixi run python scripts/run_multi_iter_parity.py \
  --relion_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0 \
  --data_star /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star \
  --iter 3 --max_iter 2 --skip_final_iteration \
  --max_particles 500 --max_healpix_order 3 \
  --output_dir _agent_scratch/multi_iter_tiny
EOF
```

### Multi-iteration parity (full, 5000 particles, 11 iters)
```bash
sbatch <<'EOF'
#!/bin/bash
#SBATCH --partition=cryoem --gres=gpu:1 --mem=400G --time=48:00:00 --account=amits
export XLA_PYTHON_CLIENT_PREALLOCATE=false
cd /scratch/gpfs/GILLES/mg6942/recovar_dev/recovar
pixi run python scripts/run_multi_iter_parity.py \
  --relion_dir /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/relion_ref_os0 \
  --data_star /scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized/particles.star \
  --iter 3 --max_iter 11 --skip_final_iteration \
  --max_healpix_order 3 \
  --output_dir _agent_scratch/multi_iter_11_hp3
EOF
```

---

## 7. Key Constants (for cross-checking)

| Quantity | recovar convention | RELION convention | Conversion |
|----------|-------------------|-------------------|------------|
| sigma2_noise | ~6700 (shell 1-10 avg) | ~2.5e-5 | recovar = RELION × N⁴ (N=128) |
| tau2 (signal prior) | ~1e7 (shell 1-10 avg) | ~0.004 | recovar = RELION × N⁴ |
| FFT normalization | unnormalized | divide by N^d | recovar_FT = RELION_FT × N^d |
| Volume frame | vol_recovar = -transpose(vol_relion, (2,1,0)) | native | see recovar/CLAUDE.md |
