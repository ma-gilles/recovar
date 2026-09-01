# EM benchmark evidence registry

This directory is the compact, machine-validated ledger for completed RECOVAR
EM quality and performance runs. It complements the historical investigation
notes in `docs/math/em_parity_program.md`; it is not a second narrative status
board. A record may be added only after its run has completed and its evidence
has been sealed. Planned runs belong in `k4_validation_matrix.md`, not in
`entries/`.

## Status is two-dimensional

`formal_status` reports the unmodified frozen gate. It must never be changed to
make a completed run pass. The current K-class gate requires every numbered
iteration and matched class to have direct unmasked RECOVAR-to-RELION FSC-AUC
at least 0.995, exact controller topology, final RECOVAR-minus-RELION GT
FSC-AUC delta at least -0.002, and final class-assignment agreement at least
0.99.

`science_status` answers the separate question of whether both engines reached
the same scientific result. A record classified `SCIENCE_EQUIVALENT` must say
`trajectory_exact=false` and retain any formal failure. A record may be called
`TRAJECTORY_EXACT` only when the formal and scientific gates both pass.

This distinction matters for the two historical 100k-particle K=4 records.
Both reproduce the RELION scientific outcome, but both first miss the frozen
0.995 trajectory threshold at iteration 10. Neither is trajectory-exact.

The four 10k-particle/128-box pilot records broaden that evidence to white and
radial noise, uniform and nonuniform poses, linear class weights, 20% outliers,
and a second molecular family. The two Ribosembly controls pass every frozen
map-trajectory cell. The outlier case passes every map cell but misses the
final 0.99 assignment gate. The IgG pilot is a deliberately retained negative
result: two classes contain less than 1% of the particles in both engines and
their direct FSC trajectories separate at iteration 4. The optional
`quality.class_collapse` diagnostic records this condition mechanically from
the final hard populations; classes are flagged when their fraction is
strictly below the recorded threshold.

### Sealed 10k/128 K=4 pilot results

All four cases were produced from RECOVAR `0050dc54f`, ran RELION and RECOVAR
serially on the same physical H100, and stopped at the nonconverged
five-iteration cap. `Map cells` counts numbered iteration/class cells meeting
direct FSC-AUC 0.995. HBM values are MiB lower bounds from the combined
60-second monitor, filtered by each engine's sealed walltime window.

| Fixture | Job | Frozen / science | Map cells | Final agreement | Collapse (<1%) | RECOVAR wall / HBM | RELION wall / HBM |
| --- | ---: | --- | ---: | ---: | --- | ---: | ---: |
| Ribosembly, white noise 1, uniform | 13296060 | PASS / PASS | 20/20 | 99.73% | none | 1199 s / 17087 | 122 s / 79563 |
| Ribosembly, radial noise 3, nonuniform, linear weights | 13296061 | PASS / PASS | 20/20 | 99.17% | none | 1621 s / 33495 | 143 s / 79559 |
| IgG-1D, white noise 1, uniform | 13296062 | FAIL / PASS | 13/20 | 98.61% | classes 2, 3 in both | 1343 s / 17087 | 117 s / 79561 |
| Ribosembly, radial noise 3, nonuniform, 20% outliers | 13296063 | FAIL / PASS | 20/20 | 98.79% | none | 1674 s / 33495 | 143 s / 79559 |

The record JSON contains the full commands, source/input/output hashes,
controller rows, per-class metrics and populations, accounting, and known
limitations. The sealed run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_robustness_pilot_0050dc54f_20260901`.

## Required evidence

Schema v1 requires:

- exact RECOVAR commit and tree from a clean checkout, and the RELION commit,
  source-diff hash, executable path, and executable hash;
- the exact RECOVAR and RELION commands, selected environment variables, and
  variables confirmed unset;
- absolute input paths, byte sizes, and SHA-256 digests;
- Slurm job IDs, status/exit code, node, GPU model/UUID, identical requested
  and allocated TRES, logs, elapsed time, and MaxRSS;
- hashed output, command, launcher, audit, shellwise-FSC, and performance
  artifacts;
- Hungarian permutation-aware class matching, per-class cross-engine FSC-AUC,
  per-engine GT FSC-AUC, class agreement, and hard/posterior populations;
- an explicit hard-population class-collapse diagnostic when it was evaluated;
- per-engine masked and unmasked half-map FSC-AUC and masked 0.143 resolution;
  historical missing values must be explicit `null` values with a reason;
- RECOVAR and RELION wall time, peak HBM, and MaxRSS, with missing measurements
  explicitly explained; and
- the frozen formal result, independent science result, comparator, concise
  interpretation, and known limitations.

Performance ratios are formal only when the engines use the same GPU model
under a matched workload. Cross-model ratios may be stored only as diagnostic
values with `hardware_comparable=false` and `formal_speedup=null`.

## Validate

The normal check validates JSON Schema and cross-record invariants without
rehashing the 26 GB particle stack:

```bash
pixi run python scripts/validate_em_benchmark_registry.py
pixi run pytest tests/unit/test_validate_em_benchmark_registry.py
```

On Della, use the expensive checksum audit only when sealing or re-auditing a
record. Shared paths are hashed once even when multiple records use them:

```bash
pixi run python scripts/validate_em_benchmark_registry.py --verify-files
```

## Sealing a new record

1. Create an isolated run root with `SAFE_TO_DELETE`; record the clean source
   commit/tree before execution.
2. Seal all inputs and the exact launcher before submission. For generated
   data, retain the generator command, source structures, configuration, RNG
   seed, and output hashes.
3. Request one explicit GPU unless the workload truly needs more. Immediately
   compare Slurm ReqTRES and AllocTRES; cancel a mismatched allocation.
4. Capture the exact command, set/unset environment, import provenance, GPU
   UUID/model, monitor CSV, wall-clock markers, logs, and Slurm accounting.
5. Evaluate K-class maps with Hungarian matching. Preserve the full pairwise
   FSC-AUC matrix and shellwise curves, per-engine GT results, class
   populations, particle agreement, and controller trajectory.
6. For real data, generate one common hashed mask and evaluate both engines'
   half maps with the same masked and unmasked FSC implementation. Never use a
   mask made from only one engine's final map without recording that bias.
7. Populate a record from the sealed artifacts, run the validator and focused
   unit test, and commit only compact JSON/NPZ summaries—not maps or stacks.

The proposed expansion from the current single-fixture K=4 evidence is in
`k4_validation_matrix.md`. The audited inventory and unresolved provenance
gaps are in `audit_inventory_20260901.md`. The runnable 10k real-data K=4
InitialModel diagnostic and its explicit half-map limitations are documented
in `real_kclass_initialmodel_pairs.md`.
