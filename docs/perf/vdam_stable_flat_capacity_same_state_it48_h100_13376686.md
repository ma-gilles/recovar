# VDAM stable flat-row capacity gate — H100 job 13376686

## Decision

Advance the default-off stable flat-row capacity lane to a fresh-cache
trajectory gate, but do not promote it to the production default yet.  The
iteration-47-to-48 ABBA panel preserves every decision-bearing output exactly,
keeps continuous differences inside the ordinary CUDA repeat envelope, and is
warm-runtime neutral.  It also changes the packed fine-score ABI from the
data-dependent row count to the mature fixed `B * R` capacity, which is the
mechanism needed to collapse the trajectory's many `Q`-dependent JAX compile
families.

This is a one-transition diagnostic.  Its cache history includes the direct
checkpoint and all preceding arms, so the large first-occurrence difference
is evidence that the fixed ABI can reuse compiled shapes, not an accepted
trajectory speedup.  A clean four-arm trajectory with fresh processes and
fresh caches remains the promotion gate.

## Qualification

| Field | Value |
|---|---|
| Source | `2957ae6a4e4d81ca9c5524871f4e9609c2d9d39d` |
| Slurm | `13376686` (`COMPLETED`, exit `0:0`) |
| Hardware | `della-h19g3`, NVIDIA H100 80GB HBM3, `GPU-1fdb3b99-e7ff-fe6d-4f59-9d2cc85fa319` |
| Boundary | One exact in-memory GF46 state, iteration `47 -> 48` |
| Panel | stable-flat off / on / on / off |
| Diagnostics | support IDs enabled; split-noise diagnostics disabled |
| Peak monitored GPU memory | 17,593 MiB |

Every arm pins the already qualified hybrid coarse path, flat local rows,
packed local projection, deferred source-faithful VDAM statistics, and the
invalid-row CUDA early exit.  Only
`RECOVAR_INITIAL_MODEL_STABLE_FLAT_ROW_CAPACITY` changes.  The feature remains
off by default and fails closed unless flat rows are active.

## Execution proof

The harness validates the request/effective option at the adapter boundary and
inside every executed local-engine profile.  Across each of the four chunks:

- stable-off uses 4,752 packed score rows from 13,824 available `B * R` rows;
- stable-on uses exactly 13,824 rows, preserving the ordinary packed prefix and
  filling the tail with invalid row IDs;
- the CUDA scorer converts those invalid rows directly to `+inf`, before fine
  pixel work;
- stable-on has zero strict row reductions, while stable-off has four.

The physical score capacity is therefore constant in `Q` for a fixed
`(B, R, F, T)` ABI.  The companion CPU gate proves that the logical prefix is
byte-identical to ordinary packing and that the whole tail is canonical zero
padding.

## Science result

- Pose IDs, translations, classes, posterior maxima, significant counts,
  selected particles, particle state, sampling state, and the full support
  audit are exactly equal for every one of the six pair comparisons.
- The off/off accumulator repeat distance is at most `9.513e-8`; the on/on
  repeat distance is at most `9.145e-8`; the largest cross-mode distance is
  `9.822e-8`.
- The off/off final-state repeat distance is at most `5.418e-8`; the on/on
  repeat distance is at most `1.622e-8`; the largest cross-mode distance is
  `5.258e-8`.

The candidate is therefore mathematically equivalent at this boundary.  The
non-bitwise accumulator and map differences are no larger than within-backend
CUDA atomic/reduction variation and do not alter support or any discrete
state.

## Runtime result

| Metric | First off | First on | Change | Warm off | Warm on | Change |
|---|---:|---:|---:|---:|---:|---:|
| Whole iteration | 18.933491 s | 4.679533 s | -75.28% | 2.418518 s | 2.438339 s | +0.82% |
| Expectation | 18.703691 s | 4.448972 s | -76.21% | 2.196613 s | 2.205614 s | +0.41% |
| Shared local EM | 12.225674 s | 3.227074 s | -73.60% | 0.981885 s | 0.990735 s | +0.90% |
| Local big JIT | 3.746406 s | 2.429633 s | -35.15% | 0.718136 s | 0.723593 s | +0.76% |
| Local noise | 5.676483 s | 0.632016 s | -88.87% | 0.106393 s | 0.107464 s | +1.01% |
| Local packing | 1.071809 s | 0.009426 s | -99.12% | 0.009214 s | 0.009187 s | -0.30% |

The warm result is deliberately read as neutral: evaluating 2.91x as many
physical rows costs less than 1% end to end because padded rows exit in CUDA.
The first-occurrence panel is strongly favorable, but its cache history makes
it diagnostic only.  The decisive experiment is the fresh-process trajectory,
where the candidate should trade the observed `Q` churn for a small fixed set
of `B * R` compile families.

## Provenance

- Report JSON SHA-256:
  `1eb3d9a92aa9886c7a08cdca91ae9f06899b4a2ab845f8cbfaf2d5a6bf5a8b42`
- Qualified CUDA checksum-file SHA-256:
  `8f2f0dae72080f9a0c38d4addf64f5afe59c01ed584fa5fe683d952633691d31`
- Qualified CUDA binary SHA-256:
  `b4d5a24d679123faf29d438accb2a64fea6da60840be07c1e920928f72dec21d`
- Static-input checksum-file SHA-256:
  `199359188d12a1e72a2b3ae03bf1aceb8b403ccef13268f6e3ca7f42bb40dbdc`
- Artifact manifest SHA-256:
  `dce756b511757157fe5e8b8c0d1c7b8fdcc9f1b715cdb454a95ea616f1a151a6`
- Disposable artifact root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_stable_flat_capacity_it47_2957ae6a4_20260903T055737Z`
