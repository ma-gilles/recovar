# EM evidence inventory, 2026-09-01

## What already exists

The repository has substantial K-class implementation coverage:

- focused unit guards in `tests/unit/test_refine_relion_mode.py`,
  `tests/unit/test_em_kclass_merge_guards.py`,
  `tests/unit/test_k_class_joint_semantics.py`,
  `tests/unit/test_run_k_class_parity.py`, and the K4 audit/analyzer tests;
- permutation/GT evaluator tests in
  `tests/unit/initial_model/test_evaluate_kclass_gt.py`;
- fast parity integration in `tests/integration/test_em_parity_fast.py`;
- a production-scale K=4 case in `tests/long_test/test_em_parity_long.py` and
  `scripts/run_em_parity_long_slurm.sh`;
- `scripts/run_em_kclass_robustness_matrix_slurm.py`, whose 36 default cases
  span K=2/4/8/16, four PDB families, white/radial and low/very-high noise,
  uniform/nonuniform/Kent/no-CTF poses, class imbalance, contrast/noise
  scaling, offsets, 20%/50% outliers, 128/256 grids, three independent seeds,
  and matched image-batch/rotation-block invariance controls; and
- detailed K=4 causal and repeatability scorecards under `docs/math/`, plus
  the long historical log in `docs/math/em_parity_program.md`.

These assets are useful, but they do not constitute one reproducible result
ledger. Most medium/scale matrix rows remain single-seed, the old completion
baseline locks mean correlation rather than the current FSC policy, and the
historical scorecards do not uniformly bind source tree, all inputs, Slurm
allocation, quality, and performance in one schema.

## Records admitted now

| Record | Source | Hardware | Frozen gate | Scientific result |
| --- | --- | --- | --- | --- |
| `k4-ribosembly-100k256-ac5177d2-a100` | ac5177d2 / tree 58476f9e | A100 80 GB | FAIL at iteration 10 class 2 | Accepted SCIENCE_EQUIVALENT reference; 0.99320 assignment agreement; exact controller topology. |
| `k4-ribosembly-100k256-1b9209cd8-h100` | 1b9209cd / tree a16a976c | H100 80 GB | FAIL at iteration 10 class 2 | SCIENCE_EQUIVALENT to accepted; 0.99282 assignment agreement; exact controller topology. |

Both records include the 100k/256 input hashes, exact commands, environment,
Slurm jobs/TRES, logs and artifact hashes, per-class cross-engine and per-engine
GT FSC-AUC, populations, wall time, peak HBM, and MaxRSS. Both explicitly
record that common-mask half-map FSC is missing. The candidate's apparent
1.4858x iteration-time improvement is diagnostic only because it compares an
H100 candidate with an A100 reference.

## Evidence not admitted

### K=1 real data

`docs/math/em_k1_realdata_science_equivalence_scorecard_v1.json` is the best
current real-data summary. It seals FSC artifacts for EMPIAR-10073, 10345, and
10097 and defines the EMPIAR-10202 set-6 I1 contract. It is not copied into the
registry yet because:

- EMPIAR-10202 is still marked `relion_complete_recovar_pending` at its pinned
  subject commit;
- the three calibration entries do not uniformly seal both producer commands,
  exact source trees, requested/allocated TRES, wall time, peak HBM, and MaxRSS;
  and
- masked half-map evidence and mask provenance are not complete for every
  calibration row.

These are evidence-completeness gaps, not claims that the reported FSC results
are invalid. Once the missing fields are sealed, each dataset should become a
separate registry record rather than one aggregate scorecard.

### K=4 synthetic data

The admitted fixture is only one molecular family, generation seed, refinement
seed, noise regime, pose distribution, class distribution, grid, and symmetry.
Its consumed files are hashed, but its original generator command and staged
PDB files were not retained. RELION half maps and a common mask were not
retained, and the candidate/reference performance comparison crosses GPU
models. The matrix in `k4_validation_matrix.md` is designed to close these
specific gaps.

### K>1 real data

No real-data K>1 run presently meets the registry contract. Until matched
multi-seed refinements report half-map quality, permutation-aware class maps,
populations, stability, and matched performance, synthetic K=4 parity must not
be generalized into a real-data K>1 claim.

## Files intentionally left unchanged

`tests/baselines/em_parity_completion_k4_100k256.json` remains a historical
completion guard and still records correlation-based metrics from job 8290126.
It was not silently rewritten with the newer FSC evidence. The new registry is
the versioned location for current scientific and performance records.
