# K=1 continuation resolution-initializer A/B scorecard

This is a non-scoring, fixed-denominator causal diagnostic. It cannot
change the frozen K=1 or K=4 FSC/FSC-AUC quality scorecards.

Fixed paired score: stock **3 / 21** → patched **3 / 21** (gain **+0**; 6 / 42 total arm checks passing).

| Gate group | Stock pass | Patched pass | Evaluated | Denominator |
| --- | ---: | ---: | ---: | ---: |
| `score` | 0 | 0 | 14 | 14 |
| `map-parity` | 0 | 0 | 3 | 3 |
| `map-gt` | 3 | 3 | 3 | 3 |
| `overall-arm` | 0 | 0 | 1 | 1 |

| Checked | Gate | Stock | Patched | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [ ] | `continuation-init-score-stack-0035` | fail | fail | unchanged-fail | stock energy=-263267129.076, p95=2.85503702768, max=5.21236333243; patched energy=-263026040.064, p95=2.78847782982, max=5.23846013176 |
| [ ] | `continuation-init-score-stack-0252` | fail | fail | unchanged-fail | stock energy=-611285434.015, p95=4.00673270491, max=6.39937815932; patched energy=-612984456.676, p95=4.04289813459, max=6.48879104078 |
| [ ] | `continuation-init-score-stack-0348` | fail | fail | unchanged-fail | stock energy=-222115311.447, p95=2.3539463705, max=3.42742160065; patched energy=-242609021.127, p95=2.44971476395, max=3.97952404816 |
| [ ] | `continuation-init-score-stack-0591` | fail | fail | unchanged-fail | stock energy=-181434736.514, p95=2.23515787027, max=3.42298588655; patched energy=-185731285.444, p95=2.20955179708, max=3.49417918699 |
| [ ] | `continuation-init-score-stack-0683` | fail | fail | unchanged-fail | stock energy=-396833770.351, p95=1.35010037413, max=1.96974785337; patched energy=-371865730.929, p95=1.40628623371, max=2.23869925137 |
| [ ] | `continuation-init-score-stack-1100` | fail | fail | unchanged-fail | stock energy=-806553323.418, p95=2.83693314648, max=3.71149998761; patched energy=-808661338.558, p95=2.87474155995, max=3.67025612836 |
| [ ] | `continuation-init-score-stack-1522` | fail | fail | unchanged-fail | stock energy=-428194968.49, p95=3.11296121037, max=4.64386232936; patched energy=-428032125.647, p95=3.11573259906, max=4.62563683911 |
| [ ] | `continuation-init-score-stack-1640` | fail | fail | unchanged-fail | stock energy=-303276507.88, p95=2.87696433352, max=4.33168959333; patched energy=-303084521.237, p95=2.87752664216, max=4.3475762593 |
| [ ] | `continuation-init-score-stack-1767` | fail | fail | unchanged-fail | stock energy=-1458257710.51, p95=4.72945886057, max=6.78507352539; patched energy=-1460894854.06, p95=4.72607654545, max=6.84082422301 |
| [ ] | `continuation-init-score-stack-2124` | fail | fail | unchanged-fail | stock energy=-796757336.995, p95=3.5968164635, max=5.97628509524; patched energy=-785054601.043, p95=3.53226615175, max=5.82031108909 |
| [ ] | `continuation-init-score-stack-2322` | fail | fail | unchanged-fail | stock energy=-509461557.307, p95=2.74920718259, max=4.26982382248; patched energy=-515449538.508, p95=2.72914185145, max=4.49612259232 |
| [ ] | `continuation-init-score-stack-2330` | fail | fail | unchanged-fail | stock energy=-341091078.991, p95=3.54206447456, max=6.40715112832; patched energy=-346033022.922, p95=3.5861261271, max=6.66905411706 |
| [ ] | `continuation-init-score-stack-2846` | fail | fail | unchanged-fail | stock energy=-533343637.155, p95=3.70674780693, max=5.7270946392; patched energy=-534231655.666, p95=3.8147917122, max=5.57727437721 |
| [ ] | `continuation-init-score-stack-2994` | fail | fail | unchanged-fail | stock energy=-553130427.308, p95=2.53986262782, max=5.00249948995; patched energy=-522562976.957, p95=2.40444876155, max=4.46027825872 |
| [ ] | `continuation-init-map-parity-half1` | fail | fail | unchanged-fail | stock delta=-0.0279290945353847; patched delta=-0.0239961787861044 |
| [ ] | `continuation-init-map-parity-half2` | fail | fail | unchanged-fail | stock delta=-0.0336945558277868; patched delta=-0.0304012339552691 |
| [ ] | `continuation-init-map-parity-merged` | fail | fail | unchanged-fail | stock delta=-0.0282848964893059; patched delta=-0.025253079718949 |
| [x] | `continuation-init-map-gt-half1` | pass | pass | retained | stock delta=0.00278463056381487; patched delta=0.0065845494509035 |
| [x] | `continuation-init-map-gt-half2` | pass | pass | retained | stock delta=0.00304758231186619; patched delta=0.00706230051543272 |
| [x] | `continuation-init-map-gt-merged` | pass | pass | retained | stock delta=0.00286509361773038; patched delta=0.00691022964163143 |
| [ ] | `continuation-init-overall` | fail | fail | unchanged-fail | stock score=0/14, parity=0/3, GT=3/3; patched score=0/14, parity=0/3, GT=3/3 |

Classification: `resolution_initializer_changes_iteration2_geometry_but_is_not_sufficient_for_score_or_map_parity`.

Causal interpretation: `continuation_divergence_contains_additional_process_resident_state_beyond_resolution_initializer`.

Metric policy: Fixed same-allocation 14-particle score gates, 3 signed shellwise FSC-AUC parity gates, 3 GT FSC-AUC non-degradation gates, and 1 overall arm gate, evaluated before and after one predeclared RELION continuation-initializer change; no fitted tolerance, scale, sign, shell boundary, or correlation.

Immutable evidence:

- `fail_closed_result`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/provenance/FAIL_CLOSED_CAUSAL_RESULT.json` (SHA-256 `9fe9e2b60a9a2d2368cf10df8a46f590f559c7e7b9b0a8db2ef570c7db7a4b9c`)
- `fresh_arm_complete`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/fresh/provenance/ARM_COMPLETE.json` (SHA-256 `4bd4c2744a892b26a8bced026cf199b99075a04cbecbbd7bc1294c2f459b7443`)
- `stock_arm_complete`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/stock_restart1/provenance/ARM_COMPLETE.json` (SHA-256 `5907fd45036ccd8ff2a6f2be9a2ce81ff0510f1ec9f04aa899b8f2566e626917`)
- `patched_arm_complete`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/patched_restart1/provenance/ARM_COMPLETE.json` (SHA-256 `43e33bfccb03a4f8e563fa7ef6bf080ec10b2c363d6e956f474ab6122db3ad97`)
- `stock_score`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/analysis/stock_restart1_SCORE_BOUNDARY.json` (SHA-256 `686b170a1e96d3426647e06a27f7cd003ec46ef1dda3dc322c1d4237c1b917f2`)
- `stock_map`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/analysis/stock_restart1_MAP_FSC.json` (SHA-256 `bb03847af2ecd7dcad4fa683d10b3bfe9a3cf30bb0b788b193017ab8dff03c50`)
- `patched_score`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/analysis/patched_restart1_SCORE_BOUNDARY.json` (SHA-256 `42bfc5c437faebcf50aaa42b8646b43e0237f7c00c498464a3add4146fa08376`)
- `patched_map`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/analysis/patched_restart1_MAP_FSC.json` (SHA-256 `dff4f487a6d6154d7929bdaf5f8447d0021e794f4ffbd39db1c94f74c1d44158`)
- `science_inputs`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/provenance/science_inputs_11840907.txt` (SHA-256 `bb130eaa93ba138bdf21881c157bcf5b4096e3c7001802da9471966400deb66c`)
- `stdout`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/logs/science_11840907.out` (SHA-256 `b1fe28e8a7fcdb5e21f43f864c7cc636bbcff56bbb6fd5032a1f3d9fb5895458`)
- `stderr`: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k1_case22_continue_resolution_init_ab_c5e1280_20260731T1341ET/logs/science_11840907.err` (SHA-256 `64002ec91f04e9c74e11b772f3a753fa8aa6ea39a46e615b98140cea1e373c24`)

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k1_continuation_initializer_scorecard.py --check
```
