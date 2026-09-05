# Direct RELION x-half it48 H100 ABBA (job 13483360)

## Decision

**NOT PROMOTED; default remains off.**

The direct route removes InitialModel's full-cube expansion followed by an
x-half slice, and its deterministic conversion is exact in focused tests. This
four-process run does not separate its timing effect from run-to-run variance,
and the fresh-process map/BPref reductions are slightly less stable than the
controls. It is therefore retained as an implementation rung, not composed into
the production candidate.

## Sealed run

- Slurm job: `13483360` (`COMPLETED`, exit `0:0`, `00:02:54`)
- Node: `della-h19g2`; one H100; all four arms use the same allocation/GPU UUID
- Order: control 1, direct 1, direct 2, control 2
- Immutable head: `ad55bb95e8f65f7ac0de25106c483494dfc9adb2`
- Source tree: `25c8aa769145371d51b2c28ebb2c39ab434ce4c3`
- Output root: `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_direct_xhalf_abba_ad55bb95e_20260905T1618Z`
- Qualification JSON SHA256: `46a1fb2f1825a98e7ddb7a694bbe5fd77501c20226cb9f1c4af984f2f076cde0`

Post-job verification rechecked the static inputs, source files, runtime
artifacts, and every arm's closed artifact manifest. `analysis.stdout` is
byte-identical to `qualification.json`; there is no failure record, and the
immutable checkout remains clean.

## Timing

| Arm | Treatment | warm profile-free wall (s) | cold local-final-accumulator (s) |
| --- | --- | ---: | ---: |
| 1 | control | `1.317886905` | `0.327784538` |
| 2 | direct | `1.269149601` | `0.401021481` |
| 3 | direct | `1.260653425` | `0.404258013` |
| 4 | control | `1.521535531` | `0.452958822` |
| median | control / direct | `1.419711218` / `1.264901513` | `0.390371680` / `0.402639747` |

The raw wall medians differ by `10.90%`, and both paired comparisons favor the
direct route. This is not causal evidence: the `0.203649 s` within-mode repeat
spread exceeds the `0.154810 s` median saving. The purported mechanism is also
not confirmed; the cold profiled finalization median is `3.14%` slower, not the
preregistered `20%` faster.

The clean control-1 value (`1.317887 s`) and both direct values (`1.269150` and
`1.260653 s`) suggest a likely steady effect of only a few percent, while
control-2 is the noisy arm. That observation is not substituted for the failed
gate and does not justify another standalone GPU run.

## Science

All 15 tracked discrete metadata fields and all particle STAR payloads are
exact across every cold, warm, and unmeasured science phase. All maps are finite
and nonzero. The repeat oracle nevertheless fails:

- worst map cross-arm nL2: `3.0793767e-9`
- corresponding cold repeat envelope: `2.9855960e-9` (`1.0314x` over)
- science cross-arm nL2: `2.5818794e-9` versus `2.3875520e-9` (`1.0814x` over)
- warm cross pairs are within the repeat envelope, but the direct warm repeat
  is less stable than the control warm repeat (`2.5610877e-9` versus
  `2.0315112e-9`)

The canonical BPref dumps cover both halfsets and unscaled/scaled data and
weight. Their reductions remain at expected atomic scale (`4.7e-8` to
`9.7e-8` nL2), but several direct repeats are modestly less stable and the
largest cross pair exceeds its observed repeat envelope by about `2.6%`.
Nothing indicates a hard-state or layout error; equally, the strict result may
not be relabeled as parity.

## Next action

Do not spend another standalone H100 gate on this sub-5%-likely boundary. Keep
the implementation default-off and prioritize the packed-tail JIT, whose
completed CUDA smoke removes several seconds of synchronized cold-stage work
and has a stronger plausible whole-wall effect. The packed-tail crossed gate
will reuse this harness's immutable checkout, private-cache, exact route,
closed-universe, CPU-analysis, map, STAR, and accumulator checks.
