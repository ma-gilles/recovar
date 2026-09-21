# K=4 contribution repeatability scorecard

This fixed-denominator diagnostic tests strict same-observer archive
repeatability. It cannot change the FSC/FSC-AUC quality scorecards.

Strict byte equality: **0 / 3**.

| Checked | Case | Archive | Result | Failed arrays |
| --- | --- | --- | ---: | ---: |
| [ ] | `pass2-archive-byte-equality` | `pass2` | fail | 5 |
| [ ] | `contribution-archive-byte-equality` | `contribution` | fail | 14 |
| [ ] | `device-signature-archive-byte-equality` | `device_signature` | fail | 2 |

Classification: `same_observer_archives_do_not_repeat_bit_for_bit`.

Immutable evidence:

- `strict_audit_report`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_contribution_strict_afterany_retry3_e87565c5_20260731T1810ET/analysis/STRICT_BYTE_REPEATABILITY.json` (SHA-256 `9c791cfe7de4bc17b391ee55c896e9451db466618a9f82ab59e2393928d54b7f`)
- `strict_audit_complete`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_contribution_strict_afterany_retry3_e87565c5_20260731T1810ET/provenance/STRICT_REPEATABILITY_AUDIT_COMPLETE.json` (SHA-256 `ad355ce2ef184297fa8e5b005f152cb0b5fa67f3b1b4e610c22987f7ea1ee9db`)
- `observed_pass2`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_contribution_stop_repeat_retry3_3gpu_d1fb8e52_20260731T1805ET/capture/pass2/pass2_orig053722_class001_cs038.npz` (SHA-256 `a654eb32963659b0c7641410bde4216f270e1f9901683b463ba198d480584afc`)
- `observed_contribution`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_contribution_stop_repeat_retry3_3gpu_d1fb8e52_20260731T1805ET/capture/contribution/bpref_contribution_rows_it002_h1_call000000_dump000000_cs038.npz` (SHA-256 `a7bbd6c00a40c5a77cd3b0129aae6a177ea7f76365de1b1a324c24e978f488c3`)
- `observed_device_signature`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_it2_contribution_stop_repeat_retry3_3gpu_d1fb8e52_20260731T1805ET/device_signature/bpref_contribution_rows_it002_h1_call000000_dump000000_cs038.device.npz` (SHA-256 `63c14a7ee4fedf8b2a62ed366e7847c9c7a5643c107fae7cc1981b3a1ba10934`)

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k4_contribution_repeatability_scorecard.py --check
```
