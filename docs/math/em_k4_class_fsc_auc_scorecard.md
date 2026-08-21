# K=4 direct per-class FSC-AUC scorecard

Fixed class score: **41 / 60 passing** (60 / 60 evaluated; 9 / 15 iterations pass all classes).

Each cell is checked when its shellwise cross-engine FSC-AUC is at least `0.995`.

| Iteration | Class 1 | Class 2 | Class 3 | Class 4 | Passed |
|---:|---:|---:|---:|---:|---:|
| 1 | [x] 0.999999922 | [x] 0.999995490 | [x] 0.999999919 | [x] 0.999999898 | 4 / 4 |
| 2 | [x] 0.999998367 | [x] 0.999992369 | [x] 0.999997743 | [x] 0.999997337 | 4 / 4 |
| 3 | [x] 0.999914408 | [x] 0.999764043 | [x] 0.999661549 | [x] 0.999262938 | 4 / 4 |
| 4 | [x] 0.999191107 | [x] 0.998540927 | [x] 0.998982955 | [x] 0.999235423 | 4 / 4 |
| 5 | [x] 0.998392146 | [x] 0.997450289 | [x] 0.998108053 | [x] 0.998935992 | 4 / 4 |
| 6 | [x] 0.997276789 | [x] 0.996793399 | [x] 0.997505777 | [x] 0.998169571 | 4 / 4 |
| 7 | [x] 0.997614527 | [x] 0.996669897 | [x] 0.997395437 | [x] 0.997926600 | 4 / 4 |
| 8 | [x] 0.997331271 | [x] 0.996319247 | [x] 0.996562700 | [x] 0.997170895 | 4 / 4 |
| 9 | [x] 0.996282315 | [x] 0.995199528 | [x] 0.995699838 | [x] 0.996330822 | 4 / 4 |
| 10 | [x] 0.996185138 | [ ] 0.994889094 | [x] 0.995291875 | [x] 0.996347083 | 3 / 4 |
| 11 | [ ] 0.994825239 | [ ] 0.993483168 | [ ] 0.993361694 | [ ] 0.994653232 | 0 / 4 |
| 12 | [x] 0.995769599 | [ ] 0.994021127 | [ ] 0.993671791 | [x] 0.995012485 | 2 / 4 |
| 13 | [ ] 0.994879090 | [ ] 0.993316787 | [ ] 0.992467064 | [ ] 0.994451397 | 0 / 4 |
| 14 | [ ] 0.993536267 | [ ] 0.991901146 | [ ] 0.991581276 | [ ] 0.994310660 | 0 / 4 |
| 15 | [ ] 0.993600513 | [ ] 0.991281889 | [ ] 0.990091128 | [ ] 0.993828082 | 0 / 4 |

Remaining failed cells: `k4-it10-class2`, `k4-it11-class1`, `k4-it11-class2`, `k4-it11-class3`, `k4-it11-class4`, `k4-it12-class2`, `k4-it12-class3`, `k4-it13-class1`, `k4-it13-class2`, `k4-it13-class3`, `k4-it13-class4`, `k4-it14-class1`, `k4-it14-class2`, `k4-it14-class3`, `k4-it14-class4`, `k4-it15-class1`, `k4-it15-class2`, `k4-it15-class3`, `k4-it15-class4`.

Metric policy: Shellwise cross-engine FSC-AUC at the fixed 0.995 gate; correlation is not used.

Source snapshot: `docs/math/em_k4_backend_trajectory_snapshot_v2.json` (SHA-256 `bc10d0555488b22f0bc8d54afe5afc5288064ddb4708bd1c75f3b55dd4c0060a`).
Source trajectory: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/em_k4_full15_host_relioncuda_samegpu_4181d340_20260725T051500ET/analysis/relion_cuda/k4_fsc_trajectory.json` (SHA-256 `5e030ab63c779b8e3050c8fc63ad4efabcc3e353d3b77ce047da8c20e63076fd`).

To validate and regenerate:

```bash
pixi run python scripts/summarize_em_k4_class_fsc_auc_scorecard.py --check
```

On Della, replay all 60 values against the sealed trajectory:

```bash
pixi run python scripts/summarize_em_k4_class_fsc_auc_scorecard.py --check --verify-source-trajectory
```
