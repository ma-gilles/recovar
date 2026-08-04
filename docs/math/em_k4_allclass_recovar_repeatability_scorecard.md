# K=4 RECOVAR all-class repeatability scorecard

This fixed-denominator panel compares two independent RECOVAR
iteration-2 four-class captures on one physical A100. It establishes
a stable RECOVAR-side boundary only; it does not establish cross-engine parity.

Exact repeatability gates: **9 / 9**.

| Checked | Gate | Result |
| --- | --- | --- |
| [x] | `arm-a-valid` | pass |
| [x] | `arm-b-valid` | pass |
| [x] | `identity-exact` | pass |
| [x] | `geometry-and-candidate-tuples-exact` | pass |
| [x] | `raw-diff2-exact` | pass |
| [x] | `priors-exact` | pass |
| [x] | `unnormalized-scores-exact` | pass |
| [x] | `joint-posterior-exact` | pass |
| [x] | `global-significant-support-exact` | pass |

The fixed target contains 247,232 active and 66,986 significant class-pose tuples; every observed identity, tuple, score, prior, posterior, and support array is byte-exact across the two arms.

Classification: `all_observed_pass2_fields_exact`.

Immutable evidence:

- `repeatability_report`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_repeat_223e7e81_20260804T0651ET/analysis/RECOVAR_ALLCLASS_REPEATABILITY_11994138.json` (SHA-256 `3e2341222a1a2e00a014995245709f8c5383eed9d611adcf23128bbc34f7f4cd`)
- `launcher`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_repeat_223e7e81_20260804T0651ET/jobs/run_repeatability.sbatch` (SHA-256 `b4b0bb886ae39df51c0cde8a99c599ddbb9615049780829dc38e10544f8e7fa9`)
- `predeclaration`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_repeat_223e7e81_20260804T0651ET/provenance/EXECUTION_PREDECLARATION.md` (SHA-256 `935b032ad097816a5b6943a1a160c8ffc00adbfd47f0013f9a3a7897004ecca2`)
- `submission`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_repeat_223e7e81_20260804T0651ET/provenance/SUBMISSION_11994138.md` (SHA-256 `4948b7220138bdfa0fb91fbd6c59ade980fa9bdbe8afaad9a92b394ecb73e79f`)
- `science_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_repeat_223e7e81_20260804T0651ET/provenance/science_outputs_11994138.sha256` (SHA-256 `34b57e868fff2a9cdaab0b8dad3a77fbe313912e02c4883dd90793d1bc6bc294`)
- `static_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_repeat_223e7e81_20260804T0651ET/provenance/static_inputs_11994138.sha256` (SHA-256 `dcea997986165acdd5826e4eadbf468c54b3e98685b7c15cdfa78fc4a74dff91`)
- `science_log`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_repeat_223e7e81_20260804T0651ET/logs/science_11994138.out` (SHA-256 `2b9d6c6251f35b9066bf9d43df05fd4e5cfd3b56c32871be389d6052e0948580`)

Code references:

- `scripts/analyze_em_k4_allclass_recovar_repeatability.py`
- `scripts/summarize_em_k4_allclass_recovar_repeatability_scorecard.py`
- `scripts/report_em_parity_progress.py`

To validate:

```bash
pixi run python scripts/summarize_em_k4_allclass_recovar_repeatability_scorecard.py --check
```
