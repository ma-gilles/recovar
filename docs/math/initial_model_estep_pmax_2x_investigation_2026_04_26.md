# InitialModel E-step Pmax 2× sharpness investigation (2026-04-26)

## Observation

On the small RELION-InitialModel fixture (500/64, iter-1 cold start with
`do_firstiter_cc=0` → gaussian score):

| | recovar | RELION |
|---|---|---|
| Pmax mean | 0.260 | 0.122 |
| ratio | 2.137× | — |

Recovar's posteriors are **2.14× sharper than RELION's**. This means our
E-step diff² has more weight on the top pose than RELION's, which
manifests downstream as a different angular distribution in the M-step
accumulator (per-shell bp_weight CC ≈ 0 even though radial profile
ratio is constant ≈ 6e-8 = 1/N⁴).

## Score formula match (verified)

RELION at `ml_optimiser.cpp:8215`:
```cpp
diff2 += (diff_real² + diff_imag²) * 0.5 * Minvsigma2
```
Then `weight = pdf_orient × pdf_offset × exp(-diff2)`.

Recovar's `_e_step_block_scores` at `em_engine.py:225`:
```python
cross = -2 * Re(<conj(Y), Pref_w>)
norms = ctf2_over_nv @ |Pref|²_w
score = -0.5 * (cross + norms)  # = -0.5 × Σ |Frefctf - Fimg|² / σ²
posterior ∝ exp(score)
```

Both reduce to `exp(-0.5 × Σ |diff|² / σ²)`. The 0.5 factor is
correctly placed in both. **No single 2× factor in the formula.**

## Empirical σ² scaling sweep (does NOT reveal a clean factor)

```
σ² × 0.5: Pmax = 0.542  (3.55× ratio sharper)
σ² × 1.0: Pmax = 0.260  (2.14×)
σ² × 1.34: Pmax ≈ 0.122 (target — interpolated)
σ² × 2.0: Pmax = 0.069  (0.57×)
σ² × 4.0: Pmax = 0.012  (0.10×)
```

If the gap were a simple constant σ² multiplier, the fix would be a
non-physical scale-by-1.34. Suspicious — likely a per-shell convention.

## RELION shell-remap convention (`ml_optimiser.cpp:7493-7532`)

For windowed E-step (`current_size < ori_size`), RELION computes:
```cpp
remap = (ori_size × pix_size) / (my_image_size × my_pixel_size);  // 64/28 = 2.286
ires_remapped = ROUND(remap × ires_windowed);
Minvsigma2 = 1 / (sigma2_fudge × sigma2_noise[ires_remapped]);
```

So at windowed shell r_w=5, RELION looks up `sigma2_noise[round(2.286·5)] = sigma2_noise[11]`,
not `sigma2_noise[5]`. Recovar's windowed pixels use `sigma2_noise[r_full] = sigma2_noise[r_w]`.

### Empirical: RELION remap makes Pmax WORSE

```
recovar (no remap):     Pmax = 0.2601  ratio 2.14
recovar (RELION remap): Pmax = 0.3008  ratio 2.47   ← WORSE
target (RELION):        Pmax = 0.1217
```

So although RELION's source applies this remap, copying it directly
into recovar makes the gap larger. This implies recovar's underlying
code path **already accounts for the remap implicitly** (perhaps via
how the windowed pixels are addressed in the full N=64 layout, vs
RELION which addresses them via a 28-grid Mresol_fine), and applying
the remap on top double-counts the conversion.

## What's NOT the cause (ruled out this session)

- ❌ Score formula 0.5 placement (matches)
- ❌ Constant σ² scaling (no clean factor reaches 0.122)
- ❌ Direct application of RELION's `ires_remapped` (makes it worse)
- ❌ Half-image w=1 vs w=2 (already matches via `half_spectrum_scoring=True`)
- ❌ score_with_masked_images (no change)
- ❌ Sampling perturbation port (verified, +0.001 CC lift)
- ❌ 8× rotation oversampling (verified, +0.001 CC lift)
- ❌ FFT N^d normalization (magnitudes lift, CC unchanged as expected)

## Likely remaining suspects (not yet pinned)

1. **DC exclusion** — RELION sets `Minvsigma2[DC] = 0` (line 7531: `if (ires > 0)`).
   Recovar uses `1/sigma2_noise[0]` ≈ 145 at DC (since `sigma2_noise[0] = 6.89e-3`,
   already a placeholder big value but nonzero). Could shift the score
   non-trivially.

2. **Per-pixel CTF handling at Nyquist** — RELION's CTF computed on
   the windowed grid may differ slightly at the y-Nyquist row when
   the image dimensions are even. Memory has noted a similar gotcha
   in production (`project_nyquist_ctf_inconsistency`).

3. **`exp_local_sqrtXi2` and `exp_highres_Xi2_img`** in RELION's
   `getAllSquaredDifferences` (line 8210): `diff2 = exp_highres_Xi2_img/2 + ...`
   adds a per-image high-res-energy constant. If recovar omits this
   constant (which it does — `batch_norm` is "carried separately"),
   the constant cancels in the softmax ONLY if it's truly per-image
   and constant across poses. Need to verify.

4. **`exp_local_Minvsigma2 *= STmult`** at line 7574-7592 (when
   subtomogram weighting is active). Probably no-op for SPA but worth
   checking.

5. **`mymodel.prior_offset_class`** affecting translation pdfs
   differently between RELION's continuous prior and recovar's
   discrete grid prior.

## Recommendation

Closing this gap is **not a 1-commit fix**. It requires:
- Single-particle, single-pose dump: recovar's diff² and exp(-diff²)
  per pose vs RELION's matching dump (we have RELION's dump in
  `/scratch/gpfs/GILLES/mg6942/_agent_scratch/relion_estep_dump_small/`).
- Per-pose comparison to localize whether the gap is uniform across
  poses (constant calibration) or pose-dependent (formula difference).

Until that diagnosis is done, the +0.74 BPref CC ceiling is the right
reading of the test, and pushing past it requires this E-step
investigation, not the 3-vector plan.
