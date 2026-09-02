# VDAM packed projection at the production seam — H100 job 13363465

## Decision

Retain packed projection as a default-off component and stack final-support
noise/M-step deferral on top of it. Projecting the shared EM flat-row plan
directly reverses the scorer-only regression and improves the warm production
iteration, but the 2.35% whole-iteration gain is not large enough to promote by
itself.

## Qualification

| Field | Value |
|---|---|
| Source | `cb5855d8367b452d48b6765f7e9265ed8d076ca4` |
| Slurm | `13363465` (`COMPLETED`, exit `0:0`, elapsed `00:05:39`) |
| Hardware | `della-h19g1`, NVIDIA H100 80GB HBM3, `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518` |
| Boundary | One exact in-memory GF46 iteration-34 state, iteration `34 -> 35` |
| Panel | direct / packed projection / packed projection / direct |
| Peak process RSS | 6,351,016 KiB |

The candidate requires both
`RECOVAR_INITIAL_MODEL_FLAT_LOCAL_ROWS=1` and
`RECOVAR_INITIAL_MODEL_PACKED_LOCAL_PROJECTION=1`. Both remain off by
default.

## Science result

- Selected particles, class and pose assignments, best rotation IDs,
  translations, Pmax values, and significant counts are exactly equal in all
  four direct/candidate comparisons.
- Every support audit and aggregate support digest is exact.
- Accumulator normalized-L2 differences remain at ordinary CUDA atomic-repeat
  scale (`4.4e-8` to `7.9e-8`). The largest cross-backend accumulator absolute
  delta is `0.00390625`, or `1.94x` the largest repeat delta.
- One derived `data_vs_prior_class` absolute tail is `9.42e-6`, `2.37x` the
  largest direct/candidate repeat delta (`3.97e-6`), while its normalized-L2
  delta is `2.37e-8`. This is tiny and does not change a discrete decision, but
  keeps this component default-off until the combined candidate is gated.

## Runtime result

| Metric | Warm direct | Warm packed projection | Change |
|---|---:|---:|---:|
| Whole iteration | 2.091715 s | 2.042595 s | -2.35% |
| Pass 1 | 0.494867 s | 0.488784 s | -1.23% |
| Pass 2 | 1.183086 s | 1.142239 s | -3.45% |
| Shared local EM | 1.170352 s | 1.130755 s | -3.38% |
| Accounted shared EM | 1.127301 s | 1.097088 s | -2.68% |
| Local big JIT | 1.023525 s | 0.998770 s | -2.42% |

The projection row count falls from 62,208 to 38,016 (`-38.89%`), while only
489 rows have nonzero reconstruction posterior. The remaining gap is therefore
the dense noise/reference topology after scoring. The next candidate keeps the
exact dense denominator reduction but projects noise and executes the existing
source-faithful BPref path only over the final packed reconstruction support.

Cold walls were direct `12.514089 s` and packed projection `5.679110 s`; they
are reported for completeness but are not used for promotion.

## Provenance

- Report JSON SHA-256: `825791a60b2e35168b807020b6bc7bc32bb149b5c82fe9c7b3be45ecb7de855c`
- Qualified CUDA checksum-file SHA-256:
  `2abeba3e96bfe56f1970345103c7f8f667b44b934c0f96553c89305712d8a968`
- Artifact manifest SHA-256:
  `afba4acea77e0b21f489a803832cd6d6beea7c9ac58225e5ce5d2851a39bde45`
- Disposable artifact root:
  `/scratch/gpfs/GILLES/mg6942/vdam_runs/vdam_packed_projection_same_state_it34_cb5855d83_20260902T234120Z`
