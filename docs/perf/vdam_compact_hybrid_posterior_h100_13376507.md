# VDAM compact certified-hybrid posterior gate — H100 job 13376507

## Decision

Pass the focused CUDA correctness boundary and advance the default-off compact
posterior to a frozen GF46 same-state production gate.  The compact
fixed-capacity scan retained the dense positive-only oracle's global support,
significant count, and cutoff count on every test row.  The compact
positive-only oracle also retained the dense positive-only oracle's global
support and bitwise count/cutoff/threshold fields.

This is not a runtime or trajectory qualification.  The mode remains
default-off behind `RECOVAR_COARSE_GAUSSIAN_GEMM_COMPACT_POSTERIOR=1` until a
production checkpoint gate establishes science parity and material speed.

## Qualification

| Field | Value |
|---|---|
| Source commit | `c99253254bfb79415370373b7ec27c99e14a198c` |
| Source tree | `560e3fe9b1273cbde6c444ee480568971a3394f0` |
| Slurm | `13376507` (`COMPLETED`, exit `0:0`, elapsed `00:01:11`) |
| Hardware | `della-h19g1`, NVIDIA H100 80GB HBM3, `GPU-0d7b80c7-fef8-e346-6332-de36ae1af518` |
| Peak batch RSS | 1,715,988 KiB |
| Test | `test_compact_hybrid_gpu_positive_oracle_and_fixed_capacity_support` |
| Result | `1 passed in 12.32s` |
| Geometry | `B=3`, `R=64`, `T=3`, compact capacity `Q=3`; active source-16 counts `[2,2,1]` |
| Posterior | adaptive fraction `0.999`, maximum significants `7`, tie ULPs `0` |

The job rebuilt the custom CUDA extension for `sm_90` from the pinned clean
tree, required custom CUDA in pytest, saw exactly one H100, and checked all five
required RELION-parity ancestors before execution.

## Contracts exercised

The gate compared three live-CUDA paths using the same exact selected scores:

1. dense global scores with the positive-only CUB oracle;
2. compact ordered scores with the positive-only CUB oracle;
3. compact ordered scores with the intended fixed-capacity runtime scan.

For path 2 versus path 1, mapped global support IDs and the `n_significant`,
pre-tie cutoff-count, and threshold fields are bitwise equal.  For path 3
versus path 1, mapped global support IDs, `n_significant`, and cutoff count are
exactly equal.  The gate deliberately does not claim bitwise continuous-weight
identity for the fixed-capacity path: removing exact-zero entries changes the
float32 scan topology.  Production qualification must classify any resulting
continuous deltas against within-arm repeat noise while requiring exact
support, pose, and state decisions.

The compact index order is selected source-16 block, source rotation, then
translation.  Host validation requires the active source-block prefix to be
strictly increasing, so compact support maps monotonically back to canonical
global `rotation*T+translation` IDs and preserves the first-global-pose tie
rule.  Invalid selected output still falls back for the whole batch to the
existing full rectangular direct scorer.

## Immutable provenance

The worktree status artifact is empty (SHA-256
`e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855`).
The Slurm script asserted the same empty tracked/untracked status again after
pytest and before writing `COMPLETED`.

| Artifact or source | SHA-256 |
|---|---|
| CUDA shared library | `0ee28a925c30ebf9d530ab59c210e8e90514d8275e6f4886a50ffbe33f4ed0e2` |
| `recovar/cuda/cuda_backproject.cu` | `d887bf810aea8ed3840c9f13613235803e4112d8c9fb183a66b1f3ffcfdef97f` |
| `recovar/cuda/Makefile` | `b7965d1472ea43679f76a89df42e0765036004cdc579adb028afa76007c3b4e8` |
| `recovar/cuda_backproject.py` | `c5d18bfd76f79e8e077f345e507ae26fe030e1bcce846aba57321688dcc0ad4e` |
| `coarse_gemm_hybrid.py` | `51c203755ad3cc2ac17efe4e49331dd9a61810a4694a1d8bee4bbd5c45fa0833` |
| `significance.py` | `763eb0c2f1710358a36f3acc176f8d980a13c86f29601702f9ffcdb8927196e0` |
| `oversampling.py` | `daec51f6dfb4715db4316d9ace81a528e4c9e1705a7af04cba5645479c0f3fb5` |
| focused GPU test source | `105bbdce21f008cdc5c500b6478cc3d96cb1e9de58b7bd00b01d75cb4f9d823d` |
| pytest stdout | `7e10dd0cb2dc873ecd3bc973ecd44592466986f603acf0d8cffa8c64cf0ff565` |
| pytest stderr (empty) | `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| CUDA checksum file | `e8386e5ad25f0fa035f29e5e25a145db8a1bc82d98b9e012de6751c21ed4a42e` |
| CUDA build stdout | `b698f1e40723bc6c6e4e8a166f812de7fbbaed1e4cff99f6f8396cf37edc5ecb` |
| CUDA build stderr | `76dc5c6cd3b88768ec4633636ae2d1e39306805b5904821d61427fdd9608b133` |
| submitted sbatch script | `f79ef9c56c62aeed0170a1637d601f51ddad3589f7c741ed5a7ef156ed2fa170` |

Disposable evidence root:
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/vdam_compact_posterior_gpu_c99253254_20260903T055025Z`.
