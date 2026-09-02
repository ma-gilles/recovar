# Real-data K-class InitialModel pairs

This runbook launches a matched RELION/RECOVAR K-class InitialModel diagnostic
on one Slurm GPU. It is the bounded first real-data K>1 gate for the EM pull
request: every numbered iteration is compared after Hungarian class matching,
and the report preserves shellwise cross-engine FSC, FSC-AUC, hard-assignment
agreement, full class populations, class-collapse status, input hashes, source
trees, commands, environment, and separate per-engine wall/HBM/RSS.

This is not a gold-standard refinement benchmark. InitialModel emits one map
per class rather than independently refined half maps. A passing result means
that RECOVAR follows RELION's InitialModel trajectory and retains the same
class populations on the selected particles. It does not establish final
resolution, biological class validity, or seed-to-seed stability. Those claims
still require matched Class3D/refinement runs with shared initial maps, frozen
halves, a common mask, and multiple seeds.

## Frozen fixtures

The launcher accepts only the following existing 10,000-particle fixtures.
It validates the STAR and selected-index file before writing a job, and the
pair runner hashes the STAR and every referenced stack before either engine
runs.

| Dataset | Fixture | `particles.star` SHA-256 | `source_indices.npy` file SHA-256 |
| --- | --- | --- | --- |
| EMPIAR-10345 | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_real10345_10k_fixture_v1_20260823/data` | `e5d9f77ff38d0e5137412892e7cc7591ba09265fb928b649cdeab58208a540f5` | `9f812a7bfd6bb9dd071786143a501c6803f6c05541faee36c7d6e07f0aa787a3` |
| EMPIAR-10076 | `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_real10076_10k_fixture_20260712/data` | `2560afeea6839dddbb38b47d26cdf8944535a799d1e6d3e1441535c96043998f` | `b58a6d11fb292a9ed9573ac75c6a0673f4a0e8c216f4dadf11a7f0537b0e2c9d` |

The fixture manifests are retained and hashed as lineage evidence. The 10345
fixture contains a balanced frozen half-set label, but InitialModel does not
consume that split as two independent refinements.

## Frozen default contract

The default pilot is K=4, C1, eight iterations, seed 0, `tau2_fudge=4`,
HEALPix order 1, oversampling 1, offset range/step 6/2 pixels, padding factor
1, 500-image batches, the exact `relion_cuda` image Fourier backend, and
5,000-rotation blocks. It audits iterations 1--8,
the numbered artifacts emitted after each completed native InitialModel update.
Iteration 0 remains available only as an explicit checkpoint for legacy or
frozen oracles that actually retain matching iteration-0 artifacts.
Every iteration requires minimum matched cross-engine FSC-AUC 0.999,
permutation-aware assignment agreement 0.995 wherever assignments exist,
identical particle identity and assigned/unassigned sets, and no class below
1% in either engine once assignments exist. Iteration 0 is normally unassigned
in both engines and is recorded as non-evaluable rather than as four collapsed
classes. Collapse is a separate scientific failure even when the two engines
reproduce the same collapse.

Fresh mode runs RELION followed by RECOVAR as separate processes on the same
physical GPU. This is the only mode that reports a RECOVAR/RELION runtime
ratio. Frozen mode reuses a validated immutable RELION directory and runs only
RECOVAR; it reports no runtime ratio and marks legacy missing HBM/RSS/source
fields explicitly.

## EMPIAR-10345 fresh pair

Run from a clean checkout containing these scripts. This first invocation is a
dry run: it validates the fixture/source/binaries and writes a sealed sbatch
script, but submits nothing.

```bash
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
PIXI_PY=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_matrix_expanded_axes_20260901/.pixi/envs/default/bin/python
RUN_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10345_initialmodel_$(git rev-parse --short=10 HEAD)_20260901
"${PIXI_PY}" -m scripts.launch_em_real_kclass_initialmodel_slurm \
  --dataset 10345 \
  --output-root "${RUN_ROOT}" \
  --relion-refine /scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/build_clean_pinned/bin/relion_refine \
  --relion-source-dir /scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/src \
  --pixi-python "${PIXI_PY}"
```

Inspect `${RUN_ROOT}/submission_manifest.json` and
`${RUN_ROOT}/scripts/run_pair.sbatch`. Submit the already sealed script with
the exact command printed by the launcher:

```bash
sbatch --parsable "${RUN_ROOT}/scripts/run_pair.sbatch"
```

Alternatively, add `--submit` to the launcher command to validate, write, and
submit in one invocation. The generated job requests exactly one H100, one
task, eight CPUs, and 192 GB without `--exclusive`; it fails if Slurm's
requested and allocated TRES differ or if `OverSubscribe` is not `OK`.

## EMPIAR-10076 and frozen replay

Use the same command with `--dataset 10076` and a distinct output root. A
legacy frozen RELION oracle is available at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k4_real10076_10k_8bb910f3/pair/pair_report.json`:

```bash
"${PIXI_PY}" -m scripts.launch_em_real_kclass_initialmodel_slurm \
  --dataset 10076 \
  --output-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_initialmodel_frozen_$(git rev-parse --short=10 HEAD)_20260901 \
  --reference-pair-report /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_k4_real10076_10k_8bb910f3/pair/pair_report.json \
  --relion-refine /scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/build_clean_pinned/bin/relion_refine \
  --relion-source-dir /scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/src \
  --pixi-python "${PIXI_PY}"
```

The legacy run at commit `8bb910f38c16c2d10f96015c233d668115ac4745`
passed its old trajectory gate: minimum matched FSC-AUC was
0.9999987694 and minimum assignment agreement was 0.9988095. RELION took
71.65 s and RECOVAR took 705.67 s on job 12796578. Both engines also converged
to a class distribution near `[0.4104, 0.0010, 0.3983, 0.1903]`; therefore the
second class collapsed under the current 1% policy. This legacy result proves
cross-engine reproduction of that collapse, not a scientifically stable K=4
partition. It predates separate HBM/RSS capture and does not record a
cryptographic RELION binary-to-source build attestation, so it is not a
registry-ready performance result.

## Sealed rejected pairs from 2026-09-01

Three fresh, same-H100, 10,000-particle K=4 pairs completed all eight native
iterations. Two used RECOVAR commit `9681a1727`, before the exact
image-preprocessing and pre-E-step operand restorations; their old wrapper
failed on a nonexistent iteration-0 checkpoint and their sealed outputs were
audited independently. The current-code 10076 pair used commit `3942224f5`,
the exact `relion_cuda` backend, and the fixed iteration-1--8 wrapper. It wrote
its complete embedded audit before exiting nonzero on the scientific gate.
None is a benchmark-registry entry.

| Dataset / batch | Pair job; audit job | Min FSC-AUC / assignment | Final matched FSC-AUC | Final RECOVAR / RELION counts | RECOVAR wall / HBM / RSS | RELION wall / HBM / RSS |
| --- | --- | --- | --- | --- | --- | --- |
| 10076 / 50 | 13301932; 13303753 | 0.02717 / 12.88% | 0.26493, 0.23100, 0.06122, 0.12403 | 1085, 5, 6992, 1918 / 4099, 10, 3990, 1901 | 2235.4 s / 17765 MiB / 15583768 KiB | 71.0 s / 79561 MiB / 2857780 KiB |
| 10345 / 500 | 13300874; 13302406 | 0.03232 / 14.12% | 0.04951, 0.17794, 0.10876, 0.04203 | 1929, 1848, 3149, 3074 / 2669, 1007, 3264, 3060 | 1605.3 s / 66771 MiB / 17569148 KiB | 71.0 s / 79561 MiB / 2850728 KiB |
| 10076 / 500, current code | 13304163; embedded | 0.02708 / 12.88% | 0.26279, 0.23099, 0.05964, 0.12461 | 1091, 5, 6963, 1941 / 4101, 10, 3985, 1904 | 2143.5 s / 66793 MiB / 17677612 KiB | 71.1 s / 79561 MiB / 2855152 KiB |

The first two wall values are monitor-span approximations; the current-code
pair uses the native process timers written into `pair_report.json`. HBM is a
one-second sampled lower bound in every row. The current batch-500 run proves
that all eight iterations now complete without OOM at 66,793 MiB peak HBM,
independently of its failed science gate. The batch-50 run used only 17,765 MiB
but did not improve wall time. The current same-GPU raw timing ratio was
30.156, but it remains diagnostic: no formal RECOVAR/RELION ratio is admitted
until the scientific gate passes.

The complete machine-readable rejected-run ledger is
`docs/benchmarks/em/diagnostics/real-kclass-initialmodel-20260901.json`. It
retains exact source trees, resolved input sizes and SHA-256 hashes, commands,
run roots, requested/allocated Slurm resources, quality, performance
limitations, and artifact hashes. Its validator fails if a rejected run tries
to claim benchmark admission, half maps, a formal ratio, mismatched resources,
or a passing scientific audit:

```bash
pixi run python scripts/validate_em_real_kclass_diagnostics.py
pixi run pytest tests/unit/initial_model/test_validate_em_real_kclass_diagnostics.py
```

## Offset-prior treatment full pairs at `92438c285`

Two independent 10,000-particle, K=4, iteration-1--8 pairs were rerun from
clean commit `92438c285172998baf958ef46d7e3053a51052d2` after the offset-prior
treatment. Both engines completed successfully on the same physical H100 in
each job. The outer Slurm jobs are `FAILED` with exit code `1:0` by design:
the pair wrapper returns nonzero when its fixed scientific gate rejects the
trajectory. ReqTRES and AllocTRES are identical in both jobs
(`billing=15,cpu=8,gres/gpu=1,mem=192G,node=1`), and neither job was
exclusive.

| Dataset | Job / terminal elapsed | Iteration-1 raw labels | Iteration-1 matched FSC-AUC | Iteration-2 map-Hungarian assignment | Final FSC-AUC / counts (RECOVAR; RELION) | RECOVAR wall / HBM / RSS | RELION wall / HBM / RSS |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 10076 | 13313941 / 2762 s | 200/200; counts 15,150,17,18 in both | 0.15528, 0.44593, 0.03516, 0.29186 | 54.5% | 0.24542, 0.22372, 0.04228, 0.14403 / 2426,14,5914,1646; 4099,10,3987,1904 | 2247.5 s / 66789 MiB / 17722084 KiB | 71.1 s / 79547 MiB / 2855564 KiB |
| 10345 | 13314322 / 2430 s | 200/200; counts 63,9,115,13 in both | 0.10812, 0.21129, 0.06886, 0.08268 | 27.0% | 0.02999, 0.18647, 0.11436, 0.03911 / 1799,1523,3138,3540; 2669,1007,3264,3060 | 1938.5 s / 34015 MiB / 16425408 KiB | 72.1 s / 79561 MiB / 2847356 KiB |

The iteration-1 raw class labels are exactly identical in both pairs. This is
not the same quantity as assignment agreement after map-based Hungarian
matching. In 10345, the already-divergent maps select permutation
`[0,2,3,1]`; applying that permutation to identical raw labels produces the
reported 31.5% map-Hungarian assignment value. The raw-label agreement remains
100%. In 10076 the map permutation happens to be the identity, so both values
are 100% at iteration 1.

The scientific conclusion from these two runs alone is narrow: the class-label
component of iteration 1 agrees, while the first artifacts they compare after
that point—the updated maps—fail FSC. At iteration 2 the hard assignments
diverge to 54.5% and 27.0%, respectively. These runs localize the gap only to
the interval between the iteration-1 E-step inputs and post-update maps; they
do not prove that the M-step is the first divergent operation. The later
shared-200 causal replay captures that interval and finds candidate-topology,
raw-score, posterior, and support differences before reconstruction. It
therefore supersedes the earlier M-step-first interpretation.

The complete compact record, including every absolute run/report/log path,
input and artifact SHA-256, terminal `sacct` fields, source tree, FSC,
assignments, counts, wall time, sampled HBM, GNU-time RSS, and the explicit
half-map/performance limitations, is
`docs/benchmarks/em/diagnostics/real-kclass-offset-prior-fullpairs-92438c285-20260901.json`.
Validate both its claims and the retained compact files with:

```bash
pixi run python scripts/validate_em_real_kclass_offset_prior_fullpairs.py
pixi run python scripts/validate_em_real_kclass_offset_prior_fullpairs.py --verify-files
pixi run pytest tests/unit/initial_model/test_validate_em_real_kclass_offset_prior_fullpairs.py
```

To reproduce on fresh roots from the sealed source commit:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_qualified_staging_20260901
test "$(git rev-parse HEAD)" = 92438c285172998baf958ef46d7e3053a51052d2
test -z "$(git status --porcelain)"
unset PYTHONPATH PYTHONHOME CONDA_PREFIX VIRTUAL_ENV
export PYTHONNOUSERSITE=1
PIXI_PY=/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_qualified_staging_20260901/.pixi/envs/default/bin/python3.11
for DATASET in 10076 10345; do
  RUN_ROOT=/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_${DATASET}_offset_prior_fullpair_replay_92438c285_20260901
  "${PIXI_PY}" -m scripts.launch_em_real_kclass_initialmodel_slurm \
    --dataset "${DATASET}" \
    --output-root "${RUN_ROOT}" \
    --relion-refine /scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/build_clean_pinned/bin/relion_refine \
    --relion-source-dir /scratch/gpfs/GILLES/mg6942/relion_clean_f2c1a384/src \
    --pixi-python "${PIXI_PY}" \
    --submit
done
```

Use new run roots if those replay paths already exist. The exact original
pair argv, including every scientific option, is retained under `pair_command`
in each hashed `submission_manifest.json` named by the compact record. These
InitialModel pairs do not have independent half maps, and the raw timing ratios
(31.59x and 26.90x) are diagnostic only because quality failed.

### Causal follow-ups

These failures are not explained by a harmless global rotational drift. A
proper-rotation search on the 10076 iteration-1 maps (job 13302694) raised the
mean matched FSC-AUC only to 0.22804, with the weakest matched class at 0.11008,
and selected different rotations for different classes. The sealed result is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_it1_rigid_alignment_9681a1727_20260901/outputs/it001_proper_rotation_alignment.json`
(SHA-256
`93eaa693af1a7244b79574c0c33aeecb05ee0beeb503c35b86a865e9a291f1fa`).

The first exact-image/pre-E-step operand replay (job 13303818) is a formal
negative, but its map scores are not a valid matched-workload trajectory
measurement. Running RECOVAR with `--nr_iter 1` activated the final K>1
all-particle schedule, so RECOVAR reconstructed from all 10,000 particles while
the frozen RELION iteration-1 artifact reconstructed from exactly 200. The
resulting matched FSC-AUC was `[0.14393, 0.53001, 0.19640, 0.22839]`, and the
STAR coverage mismatch was 10,000 assigned versus 200 assigned and 9,800
unassigned. It proves that the one-iteration harness does not reproduce the
iteration-1 controller state; it cannot be used to infer that the operand
restoration improved or worsened a matched map trajectory.

The replacement shared-coverage diagnostic is taken from a full eight-
iteration run, whose iteration-1 metadata select 200 particles in each engine.
The diagnostic asserts exact equality of the 200 image identities before
computing map FSC or assignment agreement. Job 13305983 completed with exact
ReqTRES/AllocTRES
`billing=16,cpu=4,mem=64G,node=1`. The shared-particle assignment agreement was
0.875, but the matched class-map FSC-AUC values were only
`[0.18364, 0.43645, 0.02708, 0.29571]`, so map trajectory parity still fails
decisively. Assignment agreement and map agreement are separate gates. Its
run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_it1_shared200_3942224f5_20260901`.
Even a passing result remains diagnostic-only because it covers one checkpoint
from one dataset and seed, not a complete trajectory or gold-standard
refinement.

The parent current-code run is job 13304163. Both native engines completed all
eight iterations, and exact visited-particle identity matched at every audited
checkpoint. Nevertheless, its full trajectory retained the same decisive
failure: minimum FSC-AUC 0.02708, minimum assignment agreement 0.12885, and
class 2 below 1% in both engines from iteration 5 onward. Thus the exact image
preprocessing, restored pre-E-step operands, and corrected checkpoint contract
fix harness and operand discrepancies but do not close the real K=4 map
trajectory gap. The next causal discriminator must capture the shared
200-particle iteration-1 boundary before reconstruction, beginning with the
initial model/tau2/noise state and then coarse/fine scores and winners.

The historical selected boundary below is sealed for two deterministic 10076
particles.
The coarse audit first found exact joint winners but nonidentical significance
support: 53 tuples for source row 114 (stack 1598) and 70 for source row 132
(stack 1839). Exact-local GPU capture job 13310825 then wrote every requested
particle-by-class fine table from immutable commit `e05c63a33`. The corrected
analyzer joins translations by their captured residual `(x,y)` coordinates and
joins only rotation matrices within Frobenius distance `1e-5`; unmatched native
rotations remain native-only support. This replaces and explicitly invalidates
the earlier nearest-row analysis, which folded unrelated rotations together.

| Source row / stack | Fine class winner RELION / RECOVAR | Candidate counts RELION / RECOVAR / common by class 0--3 | Max centered common raw-score delta | Max centered rotation / translation-prior delta |
| --- | --- | --- | ---: | ---: |
| 114 / 1598 | 1 / 1 | 128/96/64; 1248/608/576; 832/480/480; 640/96/96 | 0.01771 | 0 / 6.92141 |
| 132 / 1839 | 1 / 2 | 0/0/0; 512/1344/512; 128/1504/128; 0/32/0 | 0.01451 | 0 / 2.78757 |

For row 114, both engines choose the same class and the top candidate on the
physical intersection is identical, but 672 native class-1 candidates are
outside RECOVAR support and carry most of the native class-1 probability. For
row 132, all native class-1 and class-2 candidates are in the physical
intersection, while RECOVAR adds 832 and 1376 candidates; the captured class
mass flips from RELION `[0, 0.98552, 0.01448, 0]` to RECOVAR
`[0, 0.01001, 0.97866, 0.00004]`. The common raw-score residuals are small and
the centered rotation priors are exact. The first observed nonidentity is the
coarse significance support, and the fine boundary adds large
translation-prior differences. This rules out common-candidate projector/raw
score arithmetic as the explanation for these two final fine splits, but does
not yet assign the support/prior difference to one source expression.

Both selected captures predate commit `d1f2f9f93`, which fixed the
InitialModel mixed-unit offset-prior arithmetic. In particular, the large
translation-prior differences in this table are evidence for that fixed bug,
not a current-source causal localization. The geometry join and artifact
contract remain useful, but current conclusions must use the later
shared-200 v5 report documented in
`docs/benchmarks/em/real_k4_shared200_causal_replay.md`.

The compact checked-in historical record is
`docs/benchmarks/em/diagnostics/real-kclass-selected-fine-10076-20260901.json`.
The full report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_selected_fine_local_e05c63a33_20260901/audit/selected_fine_physical_report_v1.json`
(SHA-256
`ae4ce194e1913aff0ce96751a26b2cde2ca8c56f9adb0d0e627575db462c8cae`).
Reproduce the post-hoc audit without a GPU using:

```bash
pixi run python scripts/audit_em_real_kclass_selected_fine.py \
  --selection-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_selected_fine_local_e05c63a33_20260901/audit/selection.json \
  --coarse-report-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_selected_coarse_919119f82_20260901/audit/selected_coarse_report.json \
  --relion-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_selected_coarse_919119f82_20260901/relion \
  --recovar-score-directory /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_selected_fine_local_e05c63a33_20260901/recovar_scores \
  --iteration 1 \
  --output-json /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_k4_10076_selected_fine_local_e05c63a33_20260901/audit/selected_fine_physical_report_replay.json
```

## Outputs and admission

Every disposable outer run root and pair root contains `SAFE_TO_DELETE`.
The principal outputs under `<run-root>/pair/` are:

- `pair_report.json`: complete pair provenance, commands, resources, summary,
  and artifact hashes;
- `trajectory_audit.json`: per-iteration pairwise FSC-AUC matrix, Hungarian
  permutation, matched scores, assignment agreement, populations, and pass;
- `trajectory_shellwise_fsc.npz`: every candidate/reference class FSC curve;
- `relion/` and `recovar/`: exact command, combined log, one-second GPU monitor,
  GNU-time MaxRSS record, maps, and STAR artifacts.

A completed run is not checked into the benchmark registry automatically.
After the Slurm job finishes, copy only compact JSON/NPZ evidence into a schema
record, preserve the absolute run root and hashes, and run
`scripts/validate_em_benchmark_registry.py`. A failed or harness-limited run
belongs in the dedicated rejected diagnostics ledger, not in `entries/`. Do
not admit an InitialModel-only record as evidence of final real-data
resolution; retain that limitation and schedule the matched multi-seed
half-map refinement described above.

## Code and focused tests

- Pair runner: `scripts/run_em_real_kclass_initialmodel_pair.py`
- Slurm launcher: `scripts/launch_em_real_kclass_initialmodel_slurm.py`
- Permutation/FSC/population audit: `scripts/audit_em_real_kclass_initialmodel.py`
- Focused tests: `tests/unit/initial_model/test_audit_em_real_kclass_initialmodel.py`,
  `tests/unit/initial_model/test_run_em_real_kclass_initialmodel_pair.py`, and
  `tests/unit/initial_model/test_launch_em_real_kclass_initialmodel_slurm.py`
