# VDAM packed fine-row scorer primitive — H100 job 13361586

## Decision

The packed flat-row CUDA primitive is bitwise exact and materially faster in
isolation. Retain it as the shared EM/InitialModel primitive, but require a
live production A/B before promotion because this benchmark excludes dense
projection, M-step, and noise work.

## Qualification

| Field | Value |
|---|---|
| Source | `bc800e4fa2e5a607a6a4ecd480a174e0ca5ca628` |
| Slurm | `13361586` (`COMPLETED`, exit `0:0`) |
| Hardware | `della-h21g4`, NVIDIA H100 80GB HBM3, `GPU-099c0d77-bb85-f2e9-f628-148b733c9176` |
| Gate | `recovar.vdam_flat_row_score_slurm_gate.v1` |
| Exactness | Active raw diff2, dense scores, and all six posterior outputs bitwise exact |
| Static tail | Poisoned padded rows are a no-op |
| Call topology | Same outer call count in every arm |

The flat and rectangular CUDA entry points instantiate the same templated
RELION arithmetic body. Only the row addressing changes; posterior conversion
continues through the mature shared EM implementation.

## H100 timing

| GF46 iteration | Static row reduction | Raw diff2 change | Score + posterior change |
|---:|---:|---:|---:|
| 20 | 37.50% | -38.80% | -32.90% |
| 40 | 49.76% | -50.58% | -35.79% |
| 60 | 79.17% | -79.03% | -54.69% |
| 80 | 57.00% | -57.16% | -32.77% |

## Provenance

- Run JSON SHA-256: `69866a5f13f64c3c4cb13dfc12647d635883188296abf23404d1ba5f65667ece`
- Exactness JSON SHA-256: `bf29f6684494b05ce6426db44011fbac2c40fc322fa4a878ec690c2ae7b0085a`
- Timing JSON SHA-256: `05a97becbcf6e0dee2946a13361af24c31a0af73584f517bb362bbb6d768a965`
- CUDA binary SHA-256: `dfa0c9a4b8dded0efe7336544656c7a256a9cb8528bf4cee944cc8f6bc3ea054`
- JUnit SHA-256: `1123e8831a8e2e83b5c9e3966c56113fec2cbaadb9230632ab9c6f14bcd423c6`
- Source manifest SHA-256: `1374b1769e931979eefa398ad664c7c4d3869ede24ed46175b0e4cf2562d9771`
- Disposable artifact root:
  `/scratch/gpfs/GILLES/mg6942/vdam_flat_row_port_h100_bc800e4fa_20260903T0040Z`

