# K=1 continuation sampling-roundtrip A/B scorecard

This fixed-denominator causal diagnostic is non-scoring.

| Fixed paired panel | Stock | Treatment | Gain |
| --- | ---: | ---: | ---: |
| Geometry identity | 3/5 | 5/5 | +2 |
| Score/map gates | 3/21 | 17/21 | +14 |

| Checked | Geometry case | Stock | Treatment | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [x] | `sampling-roundtrip-geometry-raw-input` | pass | pass | retained | baseline 14/14 exact; treatment 14/14 exact |
| [x] | `sampling-roundtrip-geometry-rotation-keys` | pass | pass | retained | baseline 14/14 exact; treatment 14/14 exact |
| [x] | `sampling-roundtrip-geometry-local-rotation-indices` | pass | pass | retained | baseline 14/14 exact; treatment 14/14 exact |
| [x] | `sampling-roundtrip-geometry-euler-matrices` | fail | pass | improved | baseline 0/14 exact, max_abs=0.143614752218; treatment 14/14 exact, max_abs=0 |
| [x] | `sampling-roundtrip-geometry-translations` | fail | pass | improved | baseline 0/14 exact, max_abs=0.0241493657231; treatment 14/14 exact, max_abs=0 |

| Checked | Score/map case | Stock | Treatment | Transition | Observation |
| --- | --- | --- | --- | --- | --- |
| [x] | `sampling-roundtrip-score-stack-0035` | fail | pass | improved | stock p95=2.85503523062, max=5.21235444833; treatment p95=4.3209762714e-5, max=7.72082520939e-5 |
| [x] | `sampling-roundtrip-score-stack-0252` | fail | pass | improved | stock p95=4.00673852511, max=6.399377876; treatment p95=4.21701122775e-5, max=7.15432812228e-5 |
| [x] | `sampling-roundtrip-score-stack-0348` | fail | pass | improved | stock p95=2.35394651621, max=3.42740658471; treatment p95=5.25854943021e-5, max=8.01644527542e-5 |
| [x] | `sampling-roundtrip-score-stack-0591` | fail | pass | improved | stock p95=2.2351598535, max=3.42298786979; treatment p95=3.98907484225e-5, max=5.8170813162e-5 |
| [x] | `sampling-roundtrip-score-stack-0683` | fail | pass | improved | stock p95=1.3500960191, max=1.96974000137; treatment p95=3.9450338096e-5, max=5.39488438278e-5 |
| [x] | `sampling-roundtrip-score-stack-1100` | fail | pass | improved | stock p95=2.83694059374, max=3.71149217608; treatment p95=3.4278014607e-5, max=6.38057445599e-5 |
| [x] | `sampling-roundtrip-score-stack-1522` | fail | pass | improved | stock p95=3.11295068302, max=4.64386064968; treatment p95=4.32590590833e-5, max=8.2359706056e-5 |
| [x] | `sampling-roundtrip-score-stack-1640` | fail | pass | improved | stock p95=2.87697724885, max=4.33168888503; treatment p95=4.11226515439e-5, max=7.17832808164e-5 |
| [x] | `sampling-roundtrip-score-stack-1767` | fail | pass | improved | stock p95=4.72945935435, max=6.78508706403; treatment p95=4.95861948821e-5, max=8.66461812166e-5 |
| [x] | `sampling-roundtrip-score-stack-2124` | fail | pass | improved | stock p95=3.59681763725, max=5.97628626899; treatment p95=5.39813813305e-5, max=7.3796241935e-5 |
| [x] | `sampling-roundtrip-score-stack-2322` | fail | pass | improved | stock p95=2.74920147977, max=4.2698397491; treatment p95=5.30231536516e-5, max=8.67862600273e-5 |
| [x] | `sampling-roundtrip-score-stack-2330` | fail | pass | improved | stock p95=3.5420647174, max=6.40715088547; treatment p95=5.79552561987e-5, max=0.000107316502834 |
| [x] | `sampling-roundtrip-score-stack-2846` | fail | pass | improved | stock p95=3.70674764503, max=5.72710973609; treatment p95=4.41818717945e-5, max=6.58721443187e-5 |
| [x] | `sampling-roundtrip-score-stack-2994` | fail | pass | improved | stock p95=2.53986274924, max=5.00249936853; treatment p95=4.96202817942e-5, max=0.00010001121845 |
| [x] | `sampling-roundtrip-map-parity-half1` | fail | pass | improved | stock delta=-0.0279291004086; treatment delta=+2.46680715965e-7 |
| [x] | `sampling-roundtrip-map-parity-half2` | fail | pass | improved | stock delta=-0.0336945697872; treatment delta=+6.20228254311e-8 |
| [x] | `sampling-roundtrip-map-parity-merged` | fail | pass | improved | stock delta=-0.0282849038637; treatment delta=+1.44716856831e-7 |
| [ ] | `sampling-roundtrip-map-gt-half1` | pass | fail | regressed | stock delta=0.0027846137881; treatment delta=-2.08306407599e-5 |
| [ ] | `sampling-roundtrip-map-gt-half2` | pass | fail | regressed | stock delta=0.00304762675882; treatment delta=-1.67426148376e-5 |
| [ ] | `sampling-roundtrip-map-gt-merged` | pass | fail | regressed | stock delta=0.00286510844287; treatment delta=-1.89465661311e-5 |
| [ ] | `sampling-roundtrip-overall` | fail | fail | unchanged-fail | stock fails score/parity; treatment passes score/parity but fails all three GT non-degradation gates |

Classification: `sampling_perturbation_binary64_roundtrip_closes_geometry_and_score_parity_but_gt_gate_remains`.
