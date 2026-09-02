# VDAM GF46 certified-state H100 qualification: job 13340384

Status: **qualified at exact production chunk geometry**. This is a scorer
component and memory qualification, not an end-to-end runtime or trajectory
parity result.

## Result

| Boundary | Result |
|---|---:|
| Geometry | `B=500`, `T=29`, `F=5100`, `Rblock=4608`, `Rtotal=36864` |
| Warm synchronized chunk time | median `0.053900466 s`; range `0.053878273--0.053984971 s` |
| Projected eight-chunk time | `0.431203728 s` per physical 500-image batch |
| Compile / lower | `2.685025914 s` / `0.195067264 s` |
| XLA dynamic arguments | `1,428,827,344 B` |
| XLA output | `37,015,512 B` |
| XLA temporary | `2,873,091,112 B` |
| XLA peak | `4,338,930,856 B` |
| Observed device peak in use | `4,422,822,144 B` |

The five timed outputs were byte-identical. Every active image recorded exactly
`133632 = 4608 * 29` candidates, every rotation in the chunk was visited once,
unvisited rotations remained zero, and the invalid-candidate count was zero.

StableHLO and pre-optimization HLO each contain exactly two dot operations.
The optimized module contains exactly two `__cublas$gemm` calls, both with
`HIGHEST,HIGHEST` operand precision and algorithm `ALG_UNSET`. The compiled and
runtime optimized HLO texts are byte-identical.

CUDA PJRT/JAX 0.9.0.1 exposes `serialized_buffer_assignment_proto` as an empty
optional byte string on this backend. The harness records that explicitly;
all scalar compiled-memory fields, optimized HLO, and runtime modules are
present. No geometry or precision contract was weakened.

## Interpretation

The certificate update itself is not the expected runtime bottleneck. Eight
qualified chunks cost about 0.431 seconds for one 500-image batch, before exact
selected-block rescore and the unchanged RELION posterior. The next meaningful
gate is the complete hybrid call using the real projection cache and observed
candidate selection, not another synthetic certificate microbenchmark.

## Provenance

- Slurm job `13340384`, completed `0:0` in 11 seconds on `della-h19g1`
- GPU `NVIDIA H100 80GB HBM3`, UUID
  `GPU-2ee3da91-970a-6714-84df-530aefe04a08`
- Python/JAX/jaxlib `3.11.13` / `0.9.0.1` / `0.9.0.1`
- immutable source commit `2653ee00217c031f2c093f50a4686d55bc1eb0f0`
- source tree `aee581083a7db7eff798feb40dddb3964fb695cb`
- source-manifest SHA-256
  `dc59fa46be60b8b341458d8261a20a5e9f419ed3b6197d1979219d4da92bf837`
- qualification JSON SHA-256
  `7f6efc8ad7eab6378387d184e0733a3243d5403fe24b6fea4e5e9ceb6e68703e`
- artifact manifest SHA-256
  `7745f81638e57a775d700483294acfeb996c6b330745bf340e17acc57f8e029a`
- artifact root, containing `SAFE_TO_DELETE`:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_certificate_state_h100_13340384_2653ee002`

Fail-fast setup attempts `13333604`, `13333680`, `13340374`, `13340383`, and
`13340385` contain no scientific result. Job `13333741` did lower and compile
the same exact graph but stopped on the optional empty PJRT serialization
before execution; it is superseded by job `13340384`.
