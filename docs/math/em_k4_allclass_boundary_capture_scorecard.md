# K=4 RECOVAR all-class boundary capture scorecard

This fixed-denominator panel records completion of the RECOVAR side of
one K=4 class-pose boundary. It is non-scoring and does not establish
cross-engine parity.

Captured classes: **4 / 4**.

| Checked | Class | Rotations | Active tuples | Significant tuples | Result |
| --- | ---: | ---: | ---: | ---: | ---: |
| [x] | 1 | 2968 | 109184 | 38982 | pass |
| [x] | 2 | 2432 | 65952 | 14076 | pass |
| [x] | 3 | 2096 | 64704 | 11804 | pass |
| [x] | 4 | 392 | 7392 | 2124 | pass |

The fixed iteration-2 target is stack 53723 at current size 38; stored joint probabilities replay within 4.3368086899420177e-19 maximum absolute error.

Classification: `recovar_four_class_joint_posterior_boundary_complete`.

Immutable evidence:

- `capture_report`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_retry_223e7e81_20260804T0200ET/analysis/ALLCLASS_PASS2_CAPTURE.json` (SHA-256 `45beed43d823191ca6ad2358cd3965cde80ffc534b67a5e127b3f9028f4f3d03`)
- `launcher`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_retry_223e7e81_20260804T0200ET/jobs/run_allclass_pass2.sbatch` (SHA-256 `ee4ea47352b4bb95ac0d72b6f71f9879bdb655c26fe02837d0519d011b607cc3`)
- `predeclaration`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_retry_223e7e81_20260804T0200ET/provenance/EXECUTION_PREDECLARATION.md` (SHA-256 `ca1bb1dcabf3b54052ff8d1defce78508c45d355d6901c672a62e889bccaf7eb`)
- `postterminal_audit`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_retry_223e7e81_20260804T0200ET/provenance/POSTTERMINAL_AUDIT_11987097.md` (SHA-256 `870157ef201f4517655c2f14b7637fabae9f13187bfb95308fffaa7aa20425ac`)
- `science_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_retry_223e7e81_20260804T0200ET/provenance/science_outputs_11987097.sha256` (SHA-256 `8272520263a3e5a6edc9164e6ea45821bb411b1ca9483a8e105231435116c784`)
- `static_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_retry_223e7e81_20260804T0200ET/provenance/static_inputs_11987097.sha256` (SHA-256 `253d5598b98f029a4bcc0eb5c8e6f23f88d56129490d0cd725fd5e60b5fcd45d`)
- `wrapper_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_allclass_recovar_retry_223e7e81_20260804T0200ET/provenance/wrapper_inputs_11987097.sha256` (SHA-256 `147a0aa0bd1f83ff247a9fa36322c0b59f9eb4834cf6505d9c96e2fdf02494e3`)

Code references:

- `scripts/summarize_em_k4_allclass_boundary_capture_scorecard.py`
- `scripts/report_em_parity_progress.py`

To validate:

```bash
pixi run python scripts/summarize_em_k4_allclass_boundary_capture_scorecard.py --check
```
