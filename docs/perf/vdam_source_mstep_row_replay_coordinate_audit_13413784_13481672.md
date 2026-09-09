# VDAM source M-step row-replay coordinate audit

## Decision

The target-particle replay does **not** show a production VDAM M-step sign
error.  It exposes a coordinate-layout bug in the diagnostic pre-scatter row
replay.  Keep the production image, CTF, projector, and BPref frame signs
unchanged.

Job `13413784` completed the requested science capture, but its outer wrapper
exited after a dtype-blind assertion compared the local-score diagnostic's
float64 posterior directly with the production float32 posterior.  Focused GPU
reanalysis job `13481672` completed `0:0` in 18 seconds on `della-h19g2`.  It
did not rerun RECOVAR or RELION science.

Evidence root:

```
/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_target1847_source_replay_8d757d592_20260904T021212Z
```

Corrected analysis:

```
reanalysis_20260905/analysis/boundary_contract.json
reanalysis_20260905/analysis/summary.json
reanalysis_20260905/analysis/candidate_vs_native_high_repeat-{01,03,04}.json
```

## Posterior result

The diagnostic float64 posterior casts bitwise to the production float32
posterior.  The contribution capture, fused production capture, and prior run
all contain the same float32 posterior and score payload.  The three sealed
native-high repeats also give the same target-particle posterior.  Replacing
the candidate posterior with the native posterior changes neither the
candidate gradient rows nor the scattered BPref arrays.

The target therefore does not support a posterior-selection explanation for
the residual trajectory drift.

## Frame algebra

For RELION image, CTF, projection, and inverse-noise operands
`I_R, C_R, P_R, W_R`, the captured RECOVAR operands obey exactly

```
I_C = N^2 I_R
C_C = -C_R
P_C = -N^2 P_R
W_C = N^-4 W_R
```

Consequently the source reducer returns `D_C = -N^-2 D_R` and
`Q_C = N^-4 Q_R`.  The existing output bridge scales by `(-N^2, N^4)` and
restores RELION's BPref frame.  Negating the RECOVAR image would break this
identity.

The captured operand checks are exact for the image, CTF, inverse-noise scale,
and source-versus-score reference.  The independently projected native
reference agrees after frame conversion at relative L2 `1.812411e-4` and
cosine `0.999999984`.

## Diagnostic defect

The compact source operands are stored in centered-row order.  The capture-only
call to `relion_vdam_mstep_sums_f32` is passed `mstep_recon_window_indices`,
which were converted for the x-half adjoint to FFTW row order.  That row
reducer, however, decodes its `pixel_indices` as centered rows.

For the first captured target pixel, FFTW row `110` represents `ky=-18`, but
the diagnostic reducer interprets it as centered `ky=+46`.  The `+64` phase
offset makes the two retained translations nearly oppose their correct image
terms.  It explains the apparent global sign flip without any image-sign bug.

| Replay | Relative L2 vs coherent score row | Cosine |
| --- | ---: | ---: |
| Wrong coordinates, original image | 1.822394 | -0.9785815 |
| Wrong coordinates, negated image | 0.2527674 | +0.9785767 |
| Correct centered coordinates, original image | 1.9136e-7 | approximately 1 |
| Correct centered coordinates, negated image | 2.000214 | -0.9999973 |

The wrong-coordinate replay reproduces the captured `active_summed` row at
relative L2 `1.68e-7`, so the earlier exact "source production replay" result
was self-consistency of the diagnostic mistake.  The denominator is
coordinate-independent and cannot validate the translation phase.

## Production scope

The production fused x-half path is unaffected.  Its CUDA kernel deliberately
decodes FFTW row indices.  The ordinary two-stage row reducer receives centered
indices, and the outer source-faithful production scatter uses the fused
x-half implementation.  The erroneous `summed` row is materialized only for
diagnostic contribution capture.

Existing production accumulator evidence is incompatible with a global sign
bug: candidate/native raw data relative L2 is `9.879e-7` and `6.909e-6` for
the two halves, raw weight relative L2 is `6.800e-7` and `6.210e-7`, and the
reconstructed reference relative L2 is `1.798e-6`.

## Required repair

1. Pass centered reconstruction indices to the capture-only row reducer.
2. Store those indices explicitly as `source_vdam_pixel_indices`; retain the
   existing FFTW `window_indices` for x-half scatter.
3. Make the analyzer consume the explicit centered field, with a documented
   legacy conversion for old captures if they remain supported.
4. Extend the focused CUDA row-reducer test with both positive and negative
   `ky`, fractional `ty`, and at least two retained translations.

This repair improves diagnostic trust only.  It does not change production
science or the frozen v3 release score.
