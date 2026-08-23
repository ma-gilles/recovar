# K=1 case-22 serialized-restart causal scorecard

This is a non-scoring, fixed-denominator diagnostic. It cannot change
the frozen K=1 or K=4 FSC/FSC-AUC quality scorecards.

Fixed causal score: **24 / 42 passing** (42 / 42 evaluated).

| Gate group | Passed | Evaluated | Denominator |
| --- | ---: | ---: | ---: |
| `score` | 14 | 28 | 28 |
| `map-parity` | 3 | 6 | 6 |
| `map-gt` | 6 | 6 | 6 |
| `overall-arm` | 1 | 2 | 2 |

| Checked | Arm | Gate | Result | Observation |
| --- | --- | --- | --- | --- |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-0035` | pass | energy_removed=0.924152845367; p95=4.74995264028e-05; max=7.58280172306e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-0252` | pass | energy_removed=0.938632681784; p95=3.68732988704e-05; max=4.61507539171e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-0348` | pass | energy_removed=0.872370622862; p95=5.25808144744e-05; max=7.01284218962e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-0591` | pass | energy_removed=0.944094173831; p95=4.23989498415e-05; max=6.25315964555e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-0683` | pass | energy_removed=0.67811502363; p95=4.37375721447e-05; max=7.07510295399e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-1100` | pass | energy_removed=0.847019321271; p95=3.43374611532e-05; max=5.68441750204e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-1522` | pass | energy_removed=0.914062039989; p95=4.32092251685e-05; max=7.15126093382e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-1640` | pass | energy_removed=0.932271088404; p95=3.87941810686e-05; max=6.31072476835e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-1767` | pass | energy_removed=0.80720530869; p95=4.62459316282e-05; max=8.3439229968e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-2124` | pass | energy_removed=0.846535823702; p95=5.14464289836e-05; max=7.92416084323e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-2322` | pass | energy_removed=0.83177483966; p95=5.31982047448e-05; max=9.85313789954e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-2330` | pass | energy_removed=0.906773179167; p95=5.63914326847e-05; max=9.31909925441e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-2846` | pass | energy_removed=0.923627893323; p95=4.64314807232e-05; max=6.80495320466e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-score-stack-2994` | pass | energy_removed=0.767540412792; p95=5.52095215937e-05; max=8.85190634961e-05; dominated=true; absolute_gate=true |
| [x] | `iteration0-restart` | `iteration0-restart-map-parity-half1` | pass | restart-minus-fresh parity FSC-AUC=1.130174331898104e-08 |
| [x] | `iteration0-restart` | `iteration0-restart-map-parity-half2` | pass | restart-minus-fresh parity FSC-AUC=1.102321722790123e-08 |
| [x] | `iteration0-restart` | `iteration0-restart-map-parity-merged` | pass | restart-minus-fresh parity FSC-AUC=1.119893777712377e-08 |
| [x] | `iteration0-restart` | `iteration0-restart-map-gt-half1` | pass | restart-minus-fresh GT FSC-AUC=3.034683770664071e-07 |
| [x] | `iteration0-restart` | `iteration0-restart-map-gt-half2` | pass | restart-minus-fresh GT FSC-AUC=3.308941251556341e-07 |
| [x] | `iteration0-restart` | `iteration0-restart-map-gt-merged` | pass | restart-minus-fresh GT FSC-AUC=3.147003405090665e-07 |
| [x] | `iteration0-restart` | `iteration0-restart-overall` | pass | score=14/14; parity=3/3; GT=3/3 |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-0035` | fail | energy_removed=-264810672.507; p95=2.85504772097; max=5.21236179441; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-0252` | fail | energy_removed=-613772038.695; p95=4.00674047597; max=6.39937830098; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-0348` | fail | energy_removed=-224196618.855; p95=2.35394654049; max=3.42740662518; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-0591` | fail | energy_removed=-181512474.34; p95=2.23515774884; max=3.42298576512; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-0683` | fail | energy_removed=-389250230.582; p95=1.35010156812; max=1.96974665938; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-1100` | fail | energy_removed=-801021583.61; p95=2.83693933095; max=3.71149243916; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-1522` | fail | energy_removed=-430485621.866; p95=3.11295029851; max=4.64386103419; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-1640` | fail | energy_removed=-305885856.637; p95=2.87696972874; max=4.33169030163; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-1767` | fail | energy_removed=-1464449889.76; p95=4.72945970648; max=6.7850876509; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-2124` | fail | energy_removed=-806924848.761; p95=3.59681943431; max=5.97628501429; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-2322` | fail | energy_removed=-510863373.647; p95=2.74920741329; max=4.26982420699; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-2330` | fail | energy_removed=-338122629.922; p95=3.54206194897; max=6.40715060215; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-2846` | fail | energy_removed=-544257745.578; p95=3.70674792835; max=5.72709476062; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-score-stack-2994` | fail | energy_removed=-555610182.09; p95=2.5398630528; max=5.00251432376; dominated=false; absolute_gate=false |
| [ ] | `iteration1-restart` | `iteration1-restart-map-parity-half1` | fail | restart-minus-fresh parity FSC-AUC=-0.02792909523630627 |
| [ ] | `iteration1-restart` | `iteration1-restart-map-parity-half2` | fail | restart-minus-fresh parity FSC-AUC=-0.03369456874674115 |
| [ ] | `iteration1-restart` | `iteration1-restart-map-parity-merged` | fail | restart-minus-fresh parity FSC-AUC=-0.02828490222501823 |
| [x] | `iteration1-restart` | `iteration1-restart-map-gt-half1` | pass | restart-minus-fresh GT FSC-AUC=0.00278460934195543 |
| [x] | `iteration1-restart` | `iteration1-restart-map-gt-half2` | pass | restart-minus-fresh GT FSC-AUC=0.003047600199027445 |
| [x] | `iteration1-restart` | `iteration1-restart-map-gt-merged` | pass | restart-minus-fresh GT FSC-AUC=0.002865093814602149 |
| [ ] | `iteration1-restart` | `iteration1-restart-overall` | fail | score=0/14; parity=0/3; GT=3/3 |

Classification: `only_iteration0_restart_closes_score_and_map_gates`.

Causal interpretation: `case22_recovery_requires_replaying_iteration1_from_serialized_it0`.

Metric policy: Fixed same-allocation 28-particle score gates, 6 signed shellwise FSC-AUC parity gates, 6 GT FSC-AUC non-degradation gates, and 2 overall restart-arm gates; no fitted tolerance, scale, sign, shell boundary, or correlation.

Immutable evidence:

- `science_complete`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_autoiter2_d676d9d8_20260731T1225ET/provenance/SCIENCE_COMPLETE.json` (SHA-256 `56fd70c2eb72e750bd36762ecb2bd62bfcc17c9487a775869e40c722e186446c`)
- `restart_pair`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_autoiter2_d676d9d8_20260731T1225ET/analysis/RESTART_PAIR.json` (SHA-256 `ef05a9a55d1d339d61f0d354ae344caca128fb436b8acb3b609f1063c54e8ed0`)
- `restart0_score`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_autoiter2_d676d9d8_20260731T1225ET/analysis/restart0_SCORE_BOUNDARY.json` (SHA-256 `5fc76eca4cb90c93e4b5412b1de8e7ca679ef27a855e075242ee84a58514c370`)
- `restart0_map`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_autoiter2_d676d9d8_20260731T1225ET/analysis/restart0_MAP_FSC.json` (SHA-256 `e206ff91e54460118f7411d2fd1680074fdb1f0856b6fe6a20528ce7d2e9d2a7`)
- `restart1_score`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_autoiter2_d676d9d8_20260731T1225ET/analysis/restart1_SCORE_BOUNDARY.json` (SHA-256 `e074f54cf830aa4495d831a12a993a7ad594e3548d4188403df0a61c2bbd5c25`)
- `restart1_map`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_autoiter2_d676d9d8_20260731T1225ET/analysis/restart1_MAP_FSC.json` (SHA-256 `4da6ca5af66f03bed4fccecd5641f2a8e93ab39720b2f36265c74223a561be66`)
- `fresh_it0_control`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_autoiter2_d676d9d8_20260731T1225ET/provenance/FRESH_IT0_CONTROL.json` (SHA-256 `bbf9eca502ae0a9a23f8b6291c3feeaf98ca48874144cf2206706639af3fb016`)
- `restart0_iteration1_control`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_autoiter2_d676d9d8_20260731T1225ET/provenance/RESTART0_ITERATION1_CONTROL.json` (SHA-256 `f816fcdd9b92e11bcda59ca38b8b62ec365c12db65a01ca6a4ba95915fc663f9`)
- `analysis_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_autoiter2_d676d9d8_20260731T1225ET/provenance/analysis_outputs_11839040.sha256` (SHA-256 `6ae2ae809b8fb2b1f2e2212c96dd72210ae2342bbc93b5aee3a3af06585cc688`)
- `arm_manifest`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_sameallocation_restart_pair_autoiter2_d676d9d8_20260731T1225ET/provenance/arm_outputs_11839040.sha256` (SHA-256 `33e3f6891e684fcd0cd8dd7a5d1763a3ce8e0c7d058c3a6ddda51691cdd03bf1`)

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k1_serialized_restart_scorecard.py --check
```
