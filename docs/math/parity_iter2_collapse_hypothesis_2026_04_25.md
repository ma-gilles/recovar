# Parity iter-2 collapse — hypothesis brief
*2026-04-25 · authored by `doc-researcher` teammate (parity-debug team)*

## TL;DR

Late-iter parity (iter 13→14) was solved on `b187e439`, but on the rebase branch `--iter 0` shows **catastrophic ave_Pmax collapse at iter 2** (0.041 → 0.000) accompanied by a **740× volume amplification** (mean_real_ds.norm 0.013 → 9.7). The expected behavior on the 1k/64 tiny benchmark is a smooth iter-1→iter-2 rise (~0.26 → 0.93). The collapse is therefore a **rebase-specific bug**, not algorithmic mismatch.

## 1. Rebase-added code paths — RELION-expected vs recovar-tweak

| Code path | Classification | Notes |
|---|---|---|
| `use_global_significant_support` branch | **recovar-specific (regression candidate)** | Not in RELION source. The per-image Python loop with `image_batch_size=1` is also a perf bug; sparse-pass2 fix agent is rewriting it with shape-bucket batching. |
| `_align_fourier_volume_sign_to_reference` | **recovar-specific (cosmetic parity hack)** | Absorbs the `vol_recovar = -transpose(vol_relion, (2,1,0))` negation. Removable post-parity. |
| `_apply_relion_initial_lowpass_filter` + iter-1 refilter | **mixed** | Iter-1 firstiter_cc handling mirrors RELION; the *refilter* is a recovar approximation. |
| **Pre-Wiener tau2 update from init FSC (1.5e3 → 1.6e6 max)** | **recovar-specific approximation — top suspect** | Not in RELION. RELION derives tau2 from actual backprojected weights post-M-step (`BackProjector::updateSSNRarrays`). Recovar's pre-reconstruction tau2 update is a downstream approximation that is documented in `relion_parity_deep_dive.md` as known to "poison downstream noise/statistics." |

## 2. Expected iter-2 ave_Pmax behavior (per `plan_relion_parity_v3.md` 1k/64 baseline)

| iter | recovar Pmax | RELION Pmax | gap |
|---:|---:|---:|---:|
| 1 | 0.2603 | 0.2445 | 6% |
| **2** | **0.9315** | **0.8725** | **7%** |
| 3 | 0.9617 | 0.9790 | 2% |
| 4 | 0.9904 | 0.9799 | 1% |
| 5 | 0.9986 | 0.9870 | 1% |

The catastrophic collapse to ~0 we observe contradicts the v3 baseline AND late-iter parity (iter 13→14 was stable on `b187e439`). The bug must live in the **iter-1 → iter-2 transition** introduced by the rebase.

## 3. Canonical smoke test (from `plan_relion_parity_v3.md:848-873`)

```bash
# Tiny: 1k particles, 64 box, healpix_order=3
relion_refine_mpi --i particles.star --o smoke_relion/ \
  --ctf --firstiter_cc --flatten_solvent --zero_mask \
  --low_resol_join_halves 40 --norm --scale \
  --healpix_order 3 --oversampling 0 \
  --j 1 --auto_refine --maxiter 5

pixi run python scripts/run_multi_iter_parity.py \
  --relion_dir smoke_relion/ --data_star particles.star \
  --iter 1 --max_iter 5 --max_healpix_order 3 \
  --output_dir _agent_scratch/smoke_recovar

pixi run python scripts/diff_relion_recovar_per_iter.py \
  --relion_dir smoke_relion/ --recovar_dir _agent_scratch/smoke_recovar
```

Assertion table from v3 plan (lines 858-871):
- `current_size`: exact (±1 shell)
- `ave_Pmax`: < 10% rel gap (iter 2+)
- `tau2` per shell: < 5% rel
- `sigma2_noise` per shell: < 10% rel after unit normalization (iter 2+)
- Final volume CC vs RELION: > 0.99

## 4. Prior incident notes touching the iter-1 → iter-2 transition

**Incident #15 (M-step DC exclusion)** — `relion_parity_benchmark_results.md` §3b-3c:
- M-step `Ft_ctf` used DC-zeroed scoring array → tau2[shell_0] = EPSILON → DC suppressed in reconstruction → wrong-mean iter-2 volume → cross-correlation at DC negative → Pmax collapse
- Fix needed: `Ft_ctf[DC]` must accumulate from full CTF²/σ², not from the scoring array.

**Blocker #5 (deep dive)** — Noise update path:
- RELION posterior-weights residuals during E-step AND accumulates into M-step wsum.
- Recovar still hard-assignment-based, run post-EM.
- Iter-2 effect: poisoned chi²/current_size/posterior softness for the next iter.

**Blocker #6 (deep dive)** — Tau2 derived differently:
- RELION: `updateSSNRarrays()` from actual backprojected weights.
- Recovar: pre-reconstruction estimate from unperturbed FSC + gridding surrogate.
- Iter-2 effect: wrong tau² → wrong Wiener → wrong reconstructed amplitude → bad next-iter posterior. **This matches our observed 740× volume amplification.**

## 5. Conclusion + recommended next ablation order

The **pre-Wiener tau2 amplification (1000× spike) is the highest-priority suspect** because:
1. The 740× volume norm jump matches the predicted "wrong tau² → wrong Wiener → wrong amplitude" failure mode.
2. The path is documented as recovar-specific approximation, not RELION algorithm.
3. The deep-dive doc explicitly flags it as broken.

Recommended ablation execution order (override the alphabetical P3/P4/P5):
1. **P4 (skip pre-Wiener tau2 amplification at iter 1)** — most likely root cause
2. **P3 (disable sign alignment)** — independent diagnostic
3. **P5 (disable lowpass refilter)** — likely innocent based on doc reading
