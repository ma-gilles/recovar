# EM benchmark evidence registry

This directory is the compact, machine-validated ledger for completed RECOVAR
EM quality and performance runs. It complements the historical investigation
notes in `docs/math/em_parity_program.md`; it is not a second narrative status
board. A record may be added only after its run has completed and its evidence
has been sealed. Planned runs belong in `k4_validation_matrix.md`, not in
`entries/`.

Single-run evidence lives in `entries/` and validates against
`schema_v1.json`. Multi-case campaign scorecards live in `campaigns/` and
validate against `campaign_schema_v1.json`. Campaign scorecards retain one
row per case, the per-class signed quality result, hard/posterior occupancies,
wall time and HBM, exact launch/config/input hashes, Slurm accounting, and a
classification for successful, boundary, and unresolved outcomes. They point
to sealed external products instead of checking particle stacks or maps into
Git.

Campaign failure labels are deliberately causal. A RELION class-collapse
negative is distinct from a RECOVAR implementation failure, and both are
distinct from a completed trajectory whose endpoint remains scientifically
unresolved. Runs stopped before comparable populations exist use
`occupancy.status=NOT_EVALUATED`; class-collapse labels require measured
populations that mechanically support `NEAR_COLLAPSE` or `ZERO_CLASS`.

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

Campaign schema v1 additionally requires every case to retain its exact
configuration and launcher, primary generated-data and reference hashes,
per-class signed RECOVAR-minus-RELION GT FSC-AUC deltas, class populations,
numbered-trajectory result, matched-H100 wall/HBM measurements, and Slurm
ReqTRES/AllocTRES. Intended execution-invariance groups are admissible only
when their particle hashes match; a shared seed is not sufficient evidence.

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

Failed or harness-limited real-data InitialModel pairs are kept outside the
accepted schema-v1 registry. Validate their dedicated fail-closed ledger with:

```bash
pixi run python scripts/validate_em_real_kclass_diagnostics.py
pixi run pytest tests/unit/initial_model/test_validate_em_real_kclass_diagnostics.py
```

The validator requires full source/input/artifact hashes, exact matching Slurm
ReqTRES/AllocTRES, completed native engines, a retained failing scientific
audit, no half-map claim, and a null formal performance ratio. The checked
records and their exact rerun commands are in
`diagnostics/real-kclass-initialmodel-20260901.json`.

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

## Repeating the K-class robustness panel

`scripts/run_em_kclass_robustness_matrix_slurm.py` contains the frozen
synthetic case definitions. A three-seed launch expands each selected case over
seeds 41001, 41002, and 41003 while retaining the base case name and seed as
metadata. For example, first inspect the generated Slurm scripts without
submitting them:

```bash
export RELION_SRC_DIR=/absolute/path/to/relion/src
export EM_KCLASS_MATRIX_RELION_REFINE_MPI=/absolute/path/to/dispatch-instrumented/relion_refine_mpi
export EM_KCLASS_MATRIX_PIXI_PY="$(pixi run which python)"
pixi run python scripts/run_em_kclass_robustness_matrix_slurm.py \
  --dry-run \
  --scratch-dir /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_three_seed_preview \
  --three-seed-suite \
  --case 30 --case 31 --case 32 --case 33 --case 34
```

Remove `--dry-run` only after reviewing the source and executable provenance,
requested resources, case table, and generated scripts. The launcher refuses a
stock RELION executable because strict trajectory parity requires the sealed
dynamic-dispatch capture build. It also refuses exclusive allocations and any
pre-existing case/data/oracle root: reruns use a new scratch root rather than
mixing or resealing partial evidence.

The run root contains:

- `case_table.tsv`, the pipe-delimited immutable case/seed index;
- `submission.env`, the resolved launcher environment and job IDs;
- `em_kclass_robustness_summary.json`, the ordinary per-trajectory matrix
  summary;
- `em_kclass_multiseed_summary.json` and
  `em_kclass_multiseed_summary.md`, the validated three-seed aggregate; and
- each case's class-population audit, commands, provenance, FSC metrics, and
  performance artifacts under its case directory.

The multi-seed aggregate is intentionally not a schema-v1 registry record and
does not make a formal gate claim. It proves that exactly three frozen seeds
were indexed without scientific-axis drift, preserves every per-seed failure
(including class collapse), and provides cross-seed quality/runtime/memory
reductions. Seal each accepted trajectory separately in `entries/`; then use
the aggregate as a compact suite-level index. Each aggregate row is bound back
to the runtime-written `case_config.json` (including source PDB directory,
seed, symmetry, case name/index, and Slurm job), and that config is hashed in
the aggregate so a swapped or mislabeled case root fails validation.

Cases 25--27 are stricter than ordinary repeated simulations. Per seed, case
25 generates and hashes one shared dataset; cases 26 and 27 depend on that job
and verify the same manifest. Case 25 also runs and seals the one RELION model,
initialization, perturbation oracle, and dynamic dispatch schedule used by all
three RECOVAR runs. Cases 26 and 27 do not rerun RELION before changing only
RECOVAR's batching boundary. Selecting either consumer without its producer,
or changing any generator field within the group, fails closed. Cases 31--34
generate symmetric GT volumes and pass one identical canonical label (C4, D4,
O, or I1) to the generator, RELION, and RECOVAR.

The completed 14-case synthetic K=4 campaign is summarized in
`k4_expanded14_20260901.md` and sealed machine-readably in
`campaigns/k4-expanded14-3466e7a32-h100.json`.

The post-campaign shared-build orchestration check is documented in
`k4_shared_cuda_setup_qualification_20260901.md`. It records the rejected
CPU-only setup attempt, the corrected one-H100 setup/case/summary dependency
chain, sealed binary and launcher hashes, exact Slurm accounting, and the
one-iteration case-21 smoke result. It is infrastructure evidence, not a
converged K=4 trajectory record.

## Sealed C4/D4 three-seed symmetry results

The completed three-seed C4 and D4 campaigns are documented in
`k4_symmetry_c4_d4_multiseed_20260901.md` and sealed in
`campaigns/k4-c4-three-seed-c75cbfffc-h100.json` and
`campaigns/k4-d4-three-seed-c75cbfffc-h100.json`. All six 5,000-particle,
box-128 trajectories pass all 20 numbered iteration/class cells, final
assignment agreement is 99.18--100%, no class is below 1%, and all 24 matched
per-class GT FSC=0.143 resolutions agree exactly between RECOVAR and RELION.
The records include three seeds per symmetry, generated-input hashes,
per-class signed GT FSC-AUC, hard populations, matched-H100 wall/HBM, Slurm
accounting, source/executable provenance, independent trajectory-audit jobs,
and copy-safe rerun instructions.

## Sealed O/I1 three-seed symmetry results

The completed fixed-state O and I1 campaigns are documented in
`k4_symmetry_o_i1_multiseed_20260901.md` and sealed in
`campaigns/k4-o-three-seed-22efd8065-h100.json` and
`campaigns/k4-i1-three-seed-22efd8065-h100.json`. All six 5,000-particle,
box-128 trajectories pass all 20 numbered iteration/class cells, minimum
assignment agreement is 99.76%, no class is below 1%, and all 24 matched
per-class GT FSC=0.143 resolutions agree exactly between RECOVAR and RELION.
The runs exercise the post-`c75cbfffc` fixed non-C1 local-search path through
iteration 5 and retain generated-input hashes, per-class signed GT FSC-AUC,
hard populations, matched-H100 wall/HBM, Slurm accounting, and independent
trajectory-audit jobs.
