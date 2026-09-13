# VDAM final-support deferral — H100 job 13364363

## Decision

Retain the default-off implementation for repair, but reject this revision at
the quality gate. Projecting and accumulating only the final reconstruction
support produces a material warm EM speedup and leaves all discrete science
state exact. Packing away dense zero rows, however, changes the float32
reduction tree used for the noise sufficient statistics, and cancellation
amplifies that reorder beyond the direct-repeat envelope.

## Qualification

| Field | Value |
|---|---|
| Source | `fea9ae567e732d013860737e2cf5f2022c548818` |
| Slurm | `13364363` (`COMPLETED`, exit `0:0`, elapsed `00:05:52`) |
| Hardware | `della-h19g1`, NVIDIA H100 80GB HBM3, `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518` |
| Boundary | One exact in-memory GF46 iteration-34 state, iteration `34 -> 35` |
| Panel | direct / packed deferred / packed deferred / direct |
| Peak process RSS | 6,408.50 MiB |

The candidate requires the flat-row scorer, packed projection, exact/source-
faithful RELION x-half BPref, and
`RECOVAR_INITIAL_MODEL_DEFER_PACKED_VDAM=1`. All candidate switches remain off
by default.

## Science result

- All seven tracked E-step metadata fields, all nine particle-state fields,
  all 22 sampling-state fields, and the complete support-audit digest are
  exactly equal between the warm direct and candidate arms.
- Reconstruction accumulator normalized-L2 differences are `4.45e-8` to
  `8.04e-8`, comparable to direct/direct repeat noise (`4.10e-8` to
  `7.68e-8`).
- Final `Iref`, `Igrad1`, `Igrad2`, `data_vs_prior_class`, and
  `sigma2_class` remain at atomic-repeat scale.
- Final `sigma2_noise` does not: candidate/direct normalized L2 is
  `1.89017e-5`, versus `7.63936e-9` for direct/direct repeat.

The largest raw discrepancy is in `wsum_sigma2_noise` shell 1: the candidate
is `-6,606,272`, direct is `-7,186,080`, a delta of `579,808`; the largest
direct-repeat shell delta is `384`. Image power itself remains at repeat scale.
The exact dense CTF denominator is shared, and discrete posteriors are exact,
so the isolated cause is the row reduction: the direct helper reduces the
62,208-row zero-padded axis, whereas the candidate reduces 489 nonzero rows.
Large A2 and XA terms cancel in the low shells and expose the different
float32 tree.

## Runtime result

| Metric | Warm direct | Warm packed deferred | Change |
|---|---:|---:|---:|
| Whole iteration | 2.309482 s | 2.377314 s | +2.94% |
| Shared local EM | 1.221089 s | 1.003483 s | **-17.82%** |
| Accounted shared EM | 1.187634 s | 0.968822 s | **-18.42%** |
| Local big JIT | 1.088108 s | 0.807055 s | **-25.83%** |
| Deferred local noise | 0.000000 s | 0.054636 s | +0.054636 s |
| Final accumulator | 0.015198 s | 0.021193 s | +39.44% |

The big JIT projects 1,867 pixels for the candidate instead of 1,868, while
flat score rows fall from 62,208 to 38,016 and final reconstruction support is
489 rows. The single whole-iteration sample includes about 0.286 s of
additional time outside the profiled EM region, so it is not yet a reliable
end-to-end timing verdict. Correctness repair precedes repeated warmed timing.

Cold walls were direct `12.885531 s` and packed deferred `15.777305 s`; the
candidate cold arm includes an 11.08 s deferred-noise compilation and is not a
promotion metric.

## Repair

Keep compact projection and the shared exact denominator/posterior, form the
same A2 and complex XA row terms on final support, scatter those terms into
their original dense row positions with the shared EM flat-row mapping, and
invoke the existing noise finalization once over the original zero-padded
axis. This restores the direct reduction topology without recomputing dense
projections. The repaired candidate must rerun the same ABBA gate before any
trajectory timing claim.

## Provenance

- Report JSON SHA-256: `55bb5760935c5ea1314ad9591558f8ba281f71f88141201e8b58c18383459afe`
- Qualified CUDA checksum-file SHA-256:
  `336c6d3c9debe27a2e6535367437d6e4faf166265fb47630f3466fe16f1ce336`
- Artifact manifest SHA-256:
  `6a139ae1afa801a3ec6469a2d7c7e92840e475296f4ad0a439548ccc14dac26a`
- Disposable artifact root:
  `/scratch/gpfs/GILLES/mg6942/vdam_runs/vdam_packed_deferred_same_state_it34_fea9ae567_20260903T001431Z`
