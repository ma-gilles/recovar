# K=2, K=8, and K=16 three-seed RELION comparison

This report records the completed C1 synthetic extension beyond K=4. The
sealed run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_k2_k8_k16_three_seed_0d85b576b_20260902`,
and every disposable external root carries a `SAFE_TO_DELETE` marker. RECOVAR
commit `0d85b576b8864c89327246ff1d44c48ff69e0240` and RELION commit
`d476e6f6a4f1f37627c06ace5227fc374c0c2b05` were used for all nine matched
same-H100 refinements.

The machine-readable records are:

- `campaigns/k2-ribosembly-three-seed-0d85b576b-h100.json`
- `campaigns/k8-ribosembly-three-seed-0d85b576b-h100.json`
- `campaigns/k16-ribosembly-three-seed-0d85b576b-h100.json`

## Result

All nine replicates pass the signed per-class GT FSC-AUC science contract, and
all 468 audited GT class cells pass (390 numbered-iteration cells plus 78 final
cells). K=2 also passes the stricter trajectory
gate for all three seeds. K=8 and K=16 are deliberately classified
`SCIENCE_EQUIVALENT`, not `TRAJECTORY_EXACT`: their reconstructed classes have
near-identical FSC quality, but their final hard assignments fall below the
frozen 0.99 agreement threshold in some seeds.

| Case | Seeds | Strict class cells | GT class cells | Minimum direct FSC-AUC | Worst RECOVAR-minus-RELION GT FSC-AUC | Minimum final assignment | Occupancy |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| K=2, 10k, box 128, white-noise 1, uniform | 41001--41003 | 36/36 | 36/36 | 0.996765918 | -0.000112792 | 0.997700 | clear |
| K=8, 10k, box 128, white-noise 3, Kent head-heavy | 41001--41003 | 140/144 | 144/144 | 0.993459222 | -0.000391835 | 0.980900 | clear |
| K=16, 20k, box 128, white-noise 3, uniform | 41001--41003 | 288/288 | 288/288 | 0.997139525 | -0.000449036 | 0.981200 | clear |

The K=16 strict failures are assignment-only: every one of its 240 numbered
class cells and 48 final class cells passes the direct-map and signed-GT
science thresholds. Final per-seed direct FSC-AUC minima are 0.998021321,
0.997139525, and 0.997167187. No RECOVAR or RELION class has less than 1% of
the particles.

## Matched performance

| K | RECOVAR median wall (s) | RELION median wall (s) | RECOVAR max HBM (MiB) | RELION max HBM (MiB) |
| ---: | ---: | ---: | ---: | ---: |
| 2 | 1,074 | 112 | 18,109 | 79,579 |
| 8 | 1,892 | 152 | 33,501 | 79,583 |
| 16 | 3,423 | 343 | 33,501 | 79,597 |

These are matched-H100 measurements. HBM is sampled every five seconds and is
therefore a lower bound rather than an allocator-exact maximum. The quality
acceptance does not imply performance parity; these workloads remain roughly
an order of magnitude slower in RECOVAR.

## Deterministic parallel audit replay

The K-class trajectory auditor computes a square matrix of independent FSC
pairs at each numbered and final boundary.  Commit
`8069ac01508d57bfd74a7686930ebcea66b6e328` adds an explicit
`--pair-workers` option while preserving row-major result assembly.  CPU job
`13363579` replayed the seed-41001 K=16 audit with four workers and the same
frozen thresholds.  The expected assignment-only audit status remained 2;
the fail-closed wrapper then verified that JSON and Markdown were byte-for-byte
equal to the sealed serial products and that all 288 arrays in the compressed
shellwise archive were elementwise equal.

| Measurement | Serial sealed audit | Four-worker replay |
| --- | ---: | ---: |
| Wall time | 2,618.92 s | 932.30 s |
| Relative throughput | 1.00x | 2.81x |
| Output semantics | assignment-only strict failure | identical |
| Slurm MaxRSS | not separately sealed | 2,656,712 KiB |

Job `13363579` requested and received exactly four CPUs and 64 GB on
`della-h17n8`, used no GPU, was non-exclusive, and completed `0:0` in 15:34
including validation.  Its isolated run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k16_audit_pair_workers_8069ac015_20260902`
and carries `SAFE_TO_DELETE`.  The scientific results above did not change;
this gate qualifies deterministic audit throughput only.

## Reproduction and verification

The case scripts, exact environment, input hashes, source commits, Slurm
requested/allocated resources, executable hashes, map checksums, and audit
logs are embedded in the three campaign records. The aggregate was regenerated
with:

```bash
pixi run python -m scripts.aggregate_em_kclass_multiseed \
  /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_k2_k8_k16_three_seed_0d85b576b_20260902 \
  --matrix-summary /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_k2_k8_k16_three_seed_0d85b576b_20260902/em_kclass_robustness_summary.json \
  --case-table /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_k2_k8_k16_three_seed_0d85b576b_20260902/case_table.tsv \
  --expected-seeds 41001,41002,41003 --require-trajectory-audits \
  --output-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_k2_k8_k16_three_seed_0d85b576b_20260902/em_kclass_multiseed_trajectory_summary_v2.json \
  --output-markdown /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/kclass_k2_k8_k16_three_seed_0d85b576b_20260902/em_kclass_multiseed_trajectory_summary_v2.md
```

The aggregate outcome is `COMPLETE_ALL_CASES_ALL_SEEDS`. The campaign sealer
then independently replays the frozen trajectory thresholds, endpoint
evaluation, assignment topology, occupancy, resource accounting, and source
provenance before writing the compact registry evidence.
