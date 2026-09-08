# PR180 K1 replay: initial noise state

Preserving each half's numbered noise spectrum removes most particle
discrepancies in this float32 replay. The default restart policy broadcasts
half 1's spectrum to both halves; that policy produces much larger half-2
errors against the saved iteration-4 target. This is a replay-state finding,
not evidence that double precision is needed.

Both runs still fail the existing Pmax gates. They do not qualify autonomous
quality, convergence, performance, K4 or the later cleanup commits.

## Controlled comparison

Both runs use frozen integration commit
`42a3d6184c6d05a9f4f97bd00120e62d0081f1d3`, 5,000 particles at 128 pixels,
serialized iteration-3 state and iteration-4 reference outputs. The only
scientific command change is `--continuous-relion-noise-state`. Scoring and
projection use float32; x-half accumulation uses complex64/float32. Deliberate
host precision is preserved, and offline FSC computation uses complex128.

| Measurement | Restart broadcast, 13624326 | Half-specific noise, 13624629 | Direction |
| --- | ---: | ---: | --- |
| Half-1 Pmax gaps ≥ 1e-3 | 3 / 2,515 | 3 / 2,515 | Same, all Pmax values exact |
| Half-2 Pmax gaps ≥ 1e-3 | 289 / 2,485 | 3 / 2,485 | Better |
| Pmax absolute-gap p95 | 0.00164212148 | 0.000172647813 | Better, still fails ≤ 1e-4 |
| Pmax maximum absolute gap | 0.0325985207 | 0.00169576613 | Better, still fails < 1e-3 |
| Maximum angular difference, degrees | 7.50000407 | 0.0000179901 | Better |
| Merged cross FSC-AUC | 0.995594079 | 0.995600026 | Better; both pass ≥ 0.995 |
| Merged GT FSC-AUC delta versus RELION | +0.000254144 | +0.000249589 | Slightly lower; both pass ≥ −0.002 |
| Qualified speed / memory comparison | Not measured | Not measured | Not measured |

The [comparison](comparison.json) verifies the unchanged source, fixture,
precision settings and particle/half mappings. Half 1 supplies a negative
control: its Pmax array is exactly identical between runs. The iteration-3
half-specific noise spectra differ by up to 9.95% relative to half 2.
The sole angular discrepancy above 0.1 degrees was at zero-based row 4932
in half 2; it disappears with half-specific noise.

Full shellwise FSC curves and particle checks are retained in the
[broadcast audit](broadcast_audit.json) and
[half-specific audit](continuous_audit.json). Passing map gates do not cancel
failed particle gates. The merged GT FSC 0.5 crossing also differs from the
oracle; inspect the saved curves rather than substituting map correlation.

The six remaining rows are **901, 1257, 1300, 1414, 3694 and 4568** (zero-based
input ordering). Matching candidate scores, posterior normalization and input
state remain to be captured before classifying their Pmax discrepancies. A
small angular difference alone does not establish tie-aware decision parity.

## Provenance and limitations

[Run record](run_record.json) includes commands, precision overrides, fixture
hashes, source-manifest identities, native libraries, instrumentation and job
outcomes. Both executions and offline audits preserve their consumed inputs.
Both Slurm jobs exit 1 because the particle audit fails; EM execution exits 0.

- Same node: `della-h20g2`, NVIDIA H100 80 GB HBM3, driver `610.57.04`.
- Same physical GPU: `GPU-121185a9-f700-f1d8-8119-4007e7bbba56`.
- Source is clean; tracked diff SHA-256 is
  `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`.
- CUDA library SHA-256 is
  `405f027530dbdcb725eaeffde993b0e852f9bace5cb3fc31f8bc8b16ae1f87b2`.
- Both run roots contain `SAFE_TO_DELETE`; curated fixtures are preserved.

The historical oracle artifacts are hashed, but their generating RELION
commit/build and uninterrupted execution history are not independently
established. No RELION restart was executed for this comparison. The intended
half-specific state is still STAR-serialized, not a complete binary capture.
These limitations prevent strict oracle or completion claims. Native kernel
qualification from Slurm13623670 is separate evidence on the same source.

## Reproduction

Preparation, immutable manifests and exact execution/audit scripts are under:

```text
/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/pr180_integration_20260908/k1_float32_replay/
/scratch/gpfs/CRYOEM/gilleslab/mg6942/em_dev/hia_source_review_20260906/pr180_integration_20260908/k1_float32_continuous_noise/
```

The underlying command, executed in the frozen checkout's pixi environment
with the precision and native-library settings in the run record, is:

```bash
python scripts/run_multi_iter_parity.py \
  --relion_dir "$FIXTURE/relion_ref_os0" \
  --data_star "$FIXTURE/particles.star" \
  --iter 3 --max_iter 1 --skip_final_iteration \
  --gt_volume "$FIXTURE/reference_gt.mrc" \
  --output_dir "$NEW_OUTPUT/result"
# The second run adds --continuous-relion-noise-state.
```

`FIXTURE` resolves to
`/scratch/gpfs/GILLES/mg6942/em_relion_proj/data_noise1_5k_normalized`.
Submit GPU execution through Slurm and preserve scheduler visibility. Prepare
new output roots and corresponding sealed manifests before resubmitting;
the runners reject completed output roots and changed input identities.

Original logs and saved arrays remain under:

```text
/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/k1_float32_replay/
/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/pr180_integration_20260908/k1_float32_continuous_noise/
```

Each root contains `replay.log`, `audit.log`, `audit.json`, `outcome.json`,
`noise_boundary.json` and `result/`. CPU-only `inspect_boundary.py` and
`compare_replays.py` in the preparation directories audit saved results
without rerunning EM. Run them with the recorded pixi Python after unsetting
Python/conda contamination and setting `PYTHONNOUSERSITE=1`,
`CUDA_VISIBLE_DEVICES=''`, `JAX_PLATFORMS=cpu` and the recorded reference-binding
directory. Their CPU imports report an unavailable CUDA plugin on the login
node; the explicit CPU backend assertion and audits pass.
