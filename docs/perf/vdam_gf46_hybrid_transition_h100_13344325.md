# GF46 integrated-hybrid one-transition seal — job 13344325

> **Overall: PASS for the bounded iteration 180 to 181 checkpoint.** All six
> H100 arms completed, the artifacts are sealed, and the feature remains
> default-off pending the representative full-trajectory/no-growth gate.

## At a glance

| Gate | Result | Status |
|---|---:|---:|
| Persisted cutoff counts | exact across direct/hybrid cold/warm audits | PASS |
| Ordered support | one row varies only by a nested inclusive tie; common core covers cutoff | PASS |
| Audit discrete/STAR identity | exact | PASS |
| Clean discrete/STAR identity | exact | PASS |
| Map pooled repeat envelope | all 15 hybrid-repeat and 36 crossed pairs bounded | PASS |
| Model-state pooled repeat envelope | all 15 hybrid-repeat and 36 crossed pairs bounded | PASS |
| Median warm wall | `21.223620 -> 4.746491 s` (**4.471x**) | PASS |
| Median expectation | `20.631624 -> 4.155026 s` (**4.965x**) | PASS |
| Median pass 1 | `18.783001 -> 2.288276 s` (**8.208x**) | PASS |
| Median pass 2 | `1.115845 -> 1.122960 s` (`1.006x`) | INFO |
| Median peak RSS | `3.315 -> 3.515 GiB` (`+6.04%`) | INFO |
| Hybrid exact rescore/fallback | 6,000 selected images / 0 fallback images | PASS |

The exact-ID audit re-hashed every stored row and aggregate. The four audit
supports are pairwise nested, their common intersection covers every exact
RELION cutoff rank, and only image row 412 varies by one boundary-tie member.
The same row also varies between the two direct executions, so this is an
inclusive-threshold tie rather than a hybrid-only support substitution.

Normalized map metrics use the predeclared mature true-200 floor of
`4 * eps(float32) = 4.76837158203125e-7`; max-absolute and signed-bias gates
remain empirical and control-derived. Every map and model pair passed.

## Provenance

- Slurm: `13344325`, state `COMPLETED 0:0`, elapsed `00:06:32` on
  `della-h19g1`.
- GPU: `GPU-75c2d200-95d1-ef57-fb52-1698386c756c` (NVIDIA H100 80GB HBM3).
- Source/tree: `85c4b13bf5f7b975ea4504750fe8495556e6030c` /
  `1aab0e98a30ecda50bbfe3d30a67cb8583237611`.
- Source/input manifests: `36593fee9c8837463c621a89b196de2e9353e218410cab415968c26de0d890b5` /
  `de224471a690d1faaae4067217dbcc90b632269d62b0b3372b20aafa69157d91`.
- Report JSON/Markdown SHA-256: `10649aa2916269335f7615d399c0a3388bf722a49a0223b6f2acc7a863535351` /
  `9eccd41280204063b46b1b893ba29e600a6a41f609f065c5a4b017247b8f563a`.
- Sealed root:
  `/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_gf46_hybrid_transition_13344325_85c4b13bf`.

This PASS qualifies only the real GF46 one-transition direct-versus-hybrid
boundary. Long-trajectory no-growth, cross-dataset parity, RELION wall-time
parity, default enablement, and the frozen v3 scores remain separate gates.
