# Exact-input K=4 execution invariance, 2026-09-01

This campaign isolates RECOVAR execution blocking from scientific inputs. For
each seed (41001, 41002, and 41003), case 25 generated one 3,000-particle,
box-128, C1, white-noise-1 dataset and one five-iteration RELION oracle. Cases
26 and 27 consumed those same SHA-256-sealed bytes. They did not regenerate
particles or rerun RELION.

| Case | RECOVAR image batch | requested rotation block | Role | Jobs |
| --- | ---: | ---: | --- | --- |
| 25 | 50 | 8192 | shared-input and shared-RELION producer | 13307356--13307358 |
| 26 | 17 | 8192 | image-batch consumer | 13307359--13307361 |
| 27 | 50 | 257 | rotation-block consumer | 13307363--13307365 |

The baseline value is 8192, not the launcher's unrelated submission default of
2000. The runtime audit requires the declared axis to be the only non-metadata
difference. It also verifies every input/oracle manifest, the shared-data and
shared-RELION links, exact requested versus allocated Slurm resources,
nonexclusive launchers, controller topology, every numbered half and merged
map, final Hungarian class matching, image-level assignments, and per-class GT
FSC-AUC.

## Result

All nine trajectories completed and all six baseline/control comparisons pass
the frozen science gate (direct map FSC-AUC at least 0.995, class-assignment
agreement at least 0.99, and per-class GT FSC-AUC delta at least -0.002).

| Quantity over all six comparisons | Worst observed |
| --- | ---: |
| Numbered half/merged map FSC-AUC | 0.9999999733 |
| Final matched-map FSC-AUC | 0.9999999796 |
| Final image-level class agreement | 1.0000000000 |
| Per-class GT FSC-AUC delta | -0.0000008897 |
| Numbered half-map absolute difference | 0.0000009313 |

The maps are not bitwise identical. The differences are last-bit f32/GPU
reduction effects: controller decisions and every final class assignment are
identical, while scientific map agreement is many orders of magnitude inside
the frozen thresholds. This is accepted scientific execution invariance, not a
bitwise-reproducibility claim.

That result compares case 26 or 27 directly with case 25 on the same seed. It
does not imply that every individual RECOVAR-versus-RELION trajectory passes
the stricter frozen cross-engine gate. The separately sealed trajectory audits
retain their mixed formal pass/fail statuses; formal failures can still have
GT-quality-equivalent endpoints without satisfying the broader frozen
`SCIENCE_EQUIVALENT` policy.

| Seed | Case 25, batch 50/block 8192 | Case 26, batch 17/block 8192 | Case 27, batch 50/block 257 |
| ---: | --- | --- | --- |
| 41001 | fail (task 0, exit 2) | fail (task 3, exit 2) | fail (task 6, exit 2) |
| 41002 | pass (task 1, exit 0) | pass (task 4, exit 0) | pass (task 7, exit 0) |
| 41003 | fail (task 2, exit 2) | fail (task 5, exit 2) | fail (task 8, exit 2) |

The mirrored 3/9 pass pattern is evidence that the strict cross-engine misses
are seed/trajectory effects, not image-batch or rotation-block effects. For
seed 41001 the earliest miss is at iteration 5 and the final assignment
agreement is 0.984333; for seed 41003 the earliest miss is at iteration 4 and
the final agreement is exactly 0.99. Seed 41002 has exact final assignments
and passes all 20 iteration/class cells in every execution control. Every
formal failure still passes the per-class GT FSC-AUC delta floor of -0.002.
They remain `UNRESOLVED_TRAJECTORY_FAILURE`, however: both failing seeds have
a final matched cross-engine FSC-AUC below 0.99, and seed 41001 also has
assignment agreement below 0.99. They are not relabeled
`SCIENCE_EQUIVALENT`.

RECOVAR wall times for cases 25, 26, and 27 were respectively 811--830 s,
780--819 s, and 761--776 s. Sampled peak HBM was 17,087--17,089 MiB for seeds
41001/41002 and 33,471 MiB for seed 41003 in all three controls. The one shared
RELION oracle per seed took 43--47 s and sampled 79,585--79,587 MiB. Missing
per-consumer RELION wall/HBM values are intentional reuse, not missing
execution evidence.

The generic matrix summary scans three shared-input fixture roots in addition
to the nine trajectory roots and therefore prints three `pending` rows. The
validated multi-seed summary reports `COMPLETE_ALL_CASES_ALL_SEEDS` for the
actual 3 cases by 3 seeds. Do not interpret the fixture rows as trajectories.

## Provenance and reproduction

- Source commit: `91e8a30f4ebc9a88f834b1b9220dcfc3b34c31b7`.
- Run root: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_exact_batch_invariance_91e8a30f4_20260901` (`SAFE_TO_DELETE` is present).
- Scientific jobs: 13307356--13307361 and 13307363--13307365, each one H100,
  24 CPUs, and 192 GB; requested and allocated TRES match.
- Setup/summary/audit jobs: 13307355, 13307366, 13308558, and 13309304.
- Durable audit JSON SHA-256:
  `69da433358ded46aa75153ea1dec2d06754e6186d02650043fb3ae7226765d32`.
- Independent audit JSON SHA-256:
  `03fb3f0d92735a131958c6651a791704c3b279f64a78a140f154abef95bf337d`.
- Case table SHA-256:
  `19218adc6441819c8cfb028a9bdef2a80d282f58f7407fc445aa010788ee32cf`.
- Multi-seed summary SHA-256:
  `9b10cc93b9ab4e6822fceb72b7f00c18b6923570cf364ec34667f03732d54f96`.

Trajectory-audit arrays 13309394 and 13309470 are rejected harness attempts.
The first could not import the checkout when invoked as a standalone script;
the second was cancelled before acceptance because it lacked the explicit
bound-checkout import preflight. Neither contributes scientific evidence. The
authoritative array is 13309618. Each task first prints and verifies
`recovar.__file__` under
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_k4_exact_input_invariance_evidence_20260901`
and `jax.__file__` under
`/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_em_evidence_integration_20260901/.pixi/envs/default`.
CPU nodes may print a JAX CUDA-plugin initialization warning before falling
back to CPU; the audit itself is NumPy/SciPy and this warning is harmless when
the import preflight succeeds, a complete report is freshly written, and the
task exit code agrees with its formal report status (0 for pass, 2 for a
preserved scientific-gate failure).

From the source checkout, rerun the fail-closed audit with:

```bash
pixi run python scripts/audit_em_kclass_exact_input_invariance.py \
  --suite-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_exact_batch_invariance_91e8a30f4_20260901 \
  --seed 41001 --seed 41002 --seed 41003 \
  --comparison 25:26:image_batch_size:50:17 \
  --comparison 25:27:rotation_block_size:8192:257 \
  --expected-source-commit 91e8a30f4ebc9a88f834b1b9220dcfc3b34c31b7 \
  --output /absolute/new/path/k4_exact_input_invariance.json
```

The checked campaign scorecard under `campaigns/` binds this aggregate audit
to the per-trajectory cross-engine FSC reports and external artifacts.
