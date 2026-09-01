# K=4 C4/D4 three-seed parity campaign (2026-09-01)

This campaign checks that symmetric K=4 RECOVAR refinement follows the same
five-iteration trajectory as RELION Class3D. It contains three independently
generated 5,000-particle Ribosembly fixtures for each of C4 and D4, all at box
128 with white-noise level 1 and uniform pose/class sampling. The machine
records are
[`k4-c4-three-seed-c75cbfffc-h100.json`](campaigns/k4-c4-three-seed-c75cbfffc-h100.json)
and
[`k4-d4-three-seed-c75cbfffc-h100.json`](campaigns/k4-d4-three-seed-c75cbfffc-h100.json).

## Result

All six trajectories are `TRAJECTORY_EXACT`: all 120 numbered
iteration/class cells (six cases times five iterations times four classes)
pass the frozen direct-FSC and GT-FSC thresholds, all final assignment
agreements exceed 0.99, and no class contains less than 1% of the particles.
The minimum direct RECOVAR-to-RELION FSC-AUC over the retained numbered and
final comparisons is 0.997613. The worst signed
RECOVAR-minus-RELION GT FSC-AUC delta is -0.000452. Thus, neither symmetry
family exposes a multi-class map, assignment, or population divergence in
these fixtures.

The table reports the final values from the frozen trajectory audit. `Res.` is
the four alignment-aware GT FSC=0.143 resolutions in Angstrom, in matched
class order. The RECOVAR and RELION resolutions are identical for every one of
the 24 final class pairs.

| Sym. | Seed | Final GT FSC-AUC, RECOVAR / RELION | Delta | Minimum direct FSC-AUC | Final agreement | Res., both engines (A) |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| C4 | 41001 | 0.211014 / 0.211108 | -9.43e-5 | 0.997837 | 99.94% | 34.00 / 32.00 / 32.00 / 24.73 |
| C4 | 41002 | 0.216669 / 0.216632 | +3.73e-5 | 0.999051 | 99.80% | 32.00 / 32.00 / 28.63 / 23.65 |
| C4 | 41003 | 0.217293 / 0.217305 | -1.11e-5 | 0.997613 | 99.18% | 32.00 / 28.63 / 28.63 / 23.65 |
| D4 | 41001 | 0.244524 / 0.244529 | -5.28e-6 | 0.999944 | 100.00% | 25.90 / 27.20 / 24.73 / 24.73 |
| D4 | 41002 | 0.257518 / 0.257462 | +5.55e-5 | 0.998976 | 99.64% | 23.65 / 25.90 / 23.65 / 24.73 |
| D4 | 41003 | 0.255019 / 0.255007 | +1.17e-5 | 0.999863 | 99.90% | 25.90 / 25.90 / 24.73 / 22.67 |

The separately sealed rigid-alignment endpoint evaluator reports mean
RECOVAR/RELION GT FSC-AUC values of 0.207840/0.207938,
0.213808/0.213790, and 0.209188/0.209179 for the C4 seeds, and
0.233358/0.233362, 0.247678/0.247595, and 0.244136/0.244117 for the D4
seeds. These values use the endpoint evaluator's alignment and shell policy,
whereas the table above uses the unmodified frozen trajectory audit. They are
retained as distinct measurements rather than silently mixing evaluators.

## Populations and performance

Each paired run executed RELION and RECOVAR serially on the same physical H100
80GB. HBM is the engine-specific five-second `nvidia-smi` peak in MiB and is a
sampled lower bound, not an allocator-exact maximum.

| Sym. | Seed | RECOVAR / RELION hard populations | RECOVAR / RELION wall (s) | RECOVAR / RELION peak HBM (MiB) | Slurm job |
| --- | ---: | --- | ---: | ---: | ---: |
| C4 | 41001 | 1141,1174,1039,1646 / 1141,1176,1037,1646 | 866 / 64 | 18109 / 79579 | 13301414 |
| C4 | 41002 | 1173,1037,1099,1691 / 1173,1033,1102,1692 | 860 / 64 | 18109 / 79587 | 13301415 |
| C4 | 41003 | 1029,1134,1123,1714 / 1033,1124,1129,1714 | 847 / 65 | 18109 / 79583 | 13301416 |
| D4 | 41001 | 1279,1172,1221,1328 / 1279,1172,1221,1328 | 720 / 65 | 18107 / 79587 | 13301417 |
| D4 | 41002 | 1308,1289,1038,1365 / 1301,1295,1038,1366 | 708 / 65 | 18107 / 79585 | 13301418 |
| D4 | 41003 | 1257,1372,1010,1361 / 1257,1375,1007,1361 | 708 / 65 | 18107 / 79585 | 13301419 |

The run jobs requested and received exactly one H100, 24 CPUs, and 256 GB of
host memory. The machine records retain each GPU UUID, node, exit status,
ReqTRES, AllocTRES, MaxRSS, launcher hash, command log, output log, and monitor
artifact. Setup job 13301401 used 8 CPUs and 64 GB with no GPU. Independent
CPU trajectory-audit jobs 13303682--13303687 used 2 CPUs and 16 GB each; all
completed successfully and wrote a sealed report for every case.

## Reproduce or re-audit

The sealed run root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_symmetry_multiseed_c75cbfffc_20260901`.
It has a `SAFE_TO_DELETE` marker, immutable `case_table.tsv`, launchers under
`jobs/`, exact commands and provenance under every case root, and the output
and error logs at the run-root level. The source of record is the clean
RECOVAR commit `c75cbfffc930e4dac7bcc2785f9220ad8472a5a5`, tree
`598f6cad6a17f3980bdded0d25524564b597c944`. RELION is bound by source commit,
tree, dirty-diff SHA-256, executable path, executable SHA-256, and size in each
machine record.

To repeat a case without modifying the sealed evidence, first copy its
launcher to a new scratch root and update every embedded case/run path. Do not
submit the retained launcher in place: its outputs already exist and are part
of the checksum seal. The original launch shape is, for example:

```bash
sbatch /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_symmetry_multiseed_c75cbfffc_20260901/jobs/em_kclass_matrix_31_ribo_k4_5k_g128_white_noise1_c4_uniform_seed41001.sh
```

To re-run only the frozen comparison on a copied or retained case, use the
audit script from the recorded RECOVAR checkout:

```bash
cd /scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/recovar_pr158_k4_symmetry_multiseed_gate_c75cbfffc_20260901
/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_symmetry_multiseed_c75cbfffc_20260901/venv/bin/python \
  scripts/audit_k4_fsc_trajectory.py \
  --case-root /scratch/gpfs/CRYOEM/gilleslab/em_work/codex/k4_symmetry_multiseed_c75cbfffc_20260901/cases/31_ribo_k4_5k_g128_white_noise1_c4_uniform_seed41001
```

To verify the checked-in schema, cross-field invariants, external sizes, and
SHA-256 digests:

```bash
pixi run python scripts/validate_em_benchmark_registry.py --verify-files
pixi run pytest -vv tests/unit/test_validate_em_benchmark_registry.py
```

## Scope and limitations

- This is a 5,000-particle, box-128, white-noise, five-iteration synthetic
  gate. It does not replace converged 100k/256, real-data, or high-resolution
  validation.
- RECOVAR had not converged at the cap. Final-all-data reconstruction was not
  forced, and its final class maps are exact copies of the final numbered
  half-map averages.
- RELION Class3D emits full class maps rather than independent half maps, so
  paired masked/unmasked half-map FSC is not available for this campaign.
- The top-level summary dependency job 13301426 remains pending because
  unrelated O/I1/no-CTF cases failed. The C4/D4 summaries were generated
  directly only after the six case jobs completed; this limitation is retained
  in both records.
- These are evidence records for `c75cbfffc`, before the later fixed-state
  hardening integrated at `22efd8065`; no current-source claim is inferred.
