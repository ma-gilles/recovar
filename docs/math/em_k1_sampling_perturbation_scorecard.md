# K=1 continuation sampling-perturbation A/B scorecard

This fixed-denominator causal diagnostic is non-scoring.

| Fixed paired panel | Stock | Treatment | Gain |
| --- | ---: | ---: | ---: |
| Geometry identity | 3/5 | 3/5 | +0 |
| Score/map gates | 3/21 | 0/21 | -3 |

| Checked | Geometry case | Stock | Treatment | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [x] | `sampling-perturb-geometry-raw-input` | pass | pass | retained | baseline 14/14 exact; treatment 14/14 exact |
| [x] | `sampling-perturb-geometry-rotation-keys` | pass | pass | retained | baseline 14/14 exact; treatment 14/14 exact |
| [x] | `sampling-perturb-geometry-local-rotation-indices` | pass | pass | retained | baseline 14/14 exact; treatment 14/14 exact |
| [ ] | `sampling-perturb-geometry-euler-matrices` | fail | fail | unchanged-fail | baseline 0/14 exact, max_abs=0.143614752218; treatment 0/14 exact, max_abs=1.06543302536e-6 |
| [ ] | `sampling-perturb-geometry-translations` | fail | fail | unchanged-fail | baseline 0/14 exact, max_abs=0.0241493657231; treatment 0/14 exact, max_abs=1.86264514923e-7 |

| Checked | Score/map case | Stock | Treatment | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [ ] | `sampling-perturb-score-stack-0035` | fail | fail | unchanged-fail | stock p95=2.85503523062, max=5.21235444833; treatment p95=0.000166968357024, max=0.000245837222991 |
| [ ] | `sampling-perturb-score-stack-0252` | fail | fail | unchanged-fail | stock p95=4.00673852511, max=6.399377876; treatment p95=0.000263122032425, max=0.0004509892957 |
| [ ] | `sampling-perturb-score-stack-0348` | fail | fail | unchanged-fail | stock p95=2.35394651621, max=3.42740658471; treatment p95=0.000153881959648, max=0.000192659445077 |
| [ ] | `sampling-perturb-score-stack-0591` | fail | fail | unchanged-fail | stock p95=2.2351598535, max=3.42298786979; treatment p95=0.000230897231813, max=0.000357545181032 |
| [ ] | `sampling-perturb-score-stack-0683` | fail | fail | unchanged-fail | stock p95=1.3500960191, max=1.96974000137; treatment p95=0.000122211750994, max=0.000191209012684 |
| [ ] | `sampling-perturb-score-stack-1100` | fail | fail | unchanged-fail | stock p95=2.83694059374, max=3.71149217608; treatment p95=0.000188422392785, max=0.000283455658973 |
| [ ] | `sampling-perturb-score-stack-1522` | fail | fail | unchanged-fail | stock p95=3.11295068302, max=4.64386064968; treatment p95=0.000111341413196, max=0.000262339804465 |
| [ ] | `sampling-perturb-score-stack-1640` | fail | fail | unchanged-fail | stock p95=2.87697724885, max=4.33168888503; treatment p95=0.000132587101484, max=0.000239488933062 |
| [ ] | `sampling-perturb-score-stack-1767` | fail | fail | unchanged-fail | stock p95=4.72945935435, max=6.78508706403; treatment p95=0.000155949339631, max=0.000266273394828 |
| [ ] | `sampling-perturb-score-stack-2124` | fail | fail | unchanged-fail | stock p95=3.59681763725, max=5.97628626899; treatment p95=0.000197799768631, max=0.000274035279261 |
| [ ] | `sampling-perturb-score-stack-2322` | fail | fail | unchanged-fail | stock p95=2.74920147977, max=4.2698397491; treatment p95=0.0002012327432, max=0.000334651780065 |
| [ ] | `sampling-perturb-score-stack-2330` | fail | fail | unchanged-fail | stock p95=3.5420647174, max=6.40715088547; treatment p95=0.000215134696549, max=0.000410103671442 |
| [ ] | `sampling-perturb-score-stack-2846` | fail | fail | unchanged-fail | stock p95=3.70674764503, max=5.72710973609; treatment p95=0.000212223839696, max=0.000280825781886 |
| [ ] | `sampling-perturb-score-stack-2994` | fail | fail | unchanged-fail | stock p95=2.53986274924, max=5.00249936853; treatment p95=0.000158680022878, max=0.000386457544437 |
| [ ] | `sampling-perturb-map-parity-half1` | fail | fail | unchanged-fail | stock delta=-0.0279291111748617; treatment delta=-5.15543992207768e-6 |
| [ ] | `sampling-perturb-map-parity-half2` | fail | fail | unchanged-fail | stock delta=-0.0336945720714545; treatment delta=-6.09520865690882e-6 |
| [ ] | `sampling-perturb-map-parity-merged` | fail | fail | unchanged-fail | stock delta=-0.0282849120795929; treatment delta=-3.08654903125039e-6 |
| [ ] | `sampling-perturb-map-gt-half1` | pass | fail | regressed | stock delta=0.0027845675225863; treatment delta=-3.81127383475566e-5 |
| [ ] | `sampling-perturb-map-gt-half2` | pass | fail | regressed | stock delta=0.00304757450664922; treatment delta=-5.15433279671407e-6 |
| [ ] | `sampling-perturb-map-gt-merged` | pass | fail | regressed | stock delta=0.00286505794539424; treatment delta=-2.17440365055666e-5 |
| [ ] | `sampling-perturb-overall` | fail | fail | unchanged-fail | stock and treatment both fail signed score/map acceptance |

Classification: `sampling_perturbation_restore_collapses_geometry_score_and_map_gaps_but_star_precision_prevents_identity_and_signed_gate_closure`.
