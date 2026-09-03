# EMPIAR-10202 set-6 iteration-11 direct FSC

This is an interim high-resolution diagnostic, not the final PR #158 scoring
result.  It compares the matched numbered iteration-11 regularized half maps
from RECOVAR and RELION on all 30,515 deposited set-6 particles with I1
symmetry.  Both refinements use the same particles, deposited half assignment,
initial reference, poses, CTFs, and refinement parameters.

The important result is that no fitted alignment is needed.  RECOVAR's
canonical export was compared directly with RELION in their shared on-disk
frame; the export relation to the native RECOVAR maps was verified exactly
over every voxel.  No rotation, translation, reflection, sign, or scale was
fitted.

| Metric | RECOVAR | RELION / cross-engine | Gate |
| --- | ---: | ---: | --- |
| Three-shell-sustained half-map FSC 0.143 | shell 250 / 2.521600 A | shell 250 / 2.521600 A | pass |
| Half-map FSC band AUC | 0.8732780 | 0.8735295 | delta 0.0002515; pass |
| Half-map curve RMSE, shells 1--249 | -- | 0.0046341 | pass |
| Direct merged-map FSC-AUC, shells 1--249 | -- | 0.9912248 | pass |
| Direct half-1 FSC-AUC | -- | 0.9872068 | pass |
| Direct half-2 FSC-AUC | -- | 0.9869154 | pass |

All primary gates in
`docs/math/em_k1_realdata_science_equivalence_scorecard_v1.json` pass.  The
legacy full-spectrum non-DC AUC of 0.97236 does not meet the separate 0.995
strict numerical-parity diagnostic, but that diagnostic is explicitly not an
acceptance metric.  The resolved-band FSC and independent-half metrics show
that the two engines have produced the same-resolution, directly registered
reconstruction at this checkpoint.  A proper-SO(3) rescue was therefore not
run.

## Independent Fourier implementation check

Job `13373204` evaluated the four 800-cubed maps with exact Hermitian
half-spectrum shell reductions.  Job `13373359` independently repeated the
calculation with full complex FFTs.  The maximum curve difference inside the
accepted band was `2.65e-9`; the largest difference over any retained shell
was `2.81e-7`; and the largest primary-metric difference was `6.87e-11`.
Crossing shells, the selected band, and every pass/fail decision were
identical.

Both jobs requested and received exactly `cpu=8,mem=64G,node=1,billing=16`,
with no GPU and `OverSubscribe=OK`.  The full-complex job peaked at
55,189,160 KiB batch RSS, while the half-spectrum job peaked at 27,676,436
KiB.  The production diagnostic now uses float32 real input, complex64 rFFT
storage, exact Hermitian plane weights, and releases each transform pair before
the next FSC.  Even- and odd-sized equivalence to a full FFT is unit tested.

## Evidence and reproduction

The complete immutable, disposable audit root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/10202_it11_direct_fsc_20260903T040436Z`
and carries `SAFE_TO_DELETE`.  The authoritative full-FFT report is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/10202_it11_direct_fsc_20260903T040436Z/outputs/raw/raw_direct_metrics.json`
(SHA-256
`a412be7ebcb713a01a577548c92115a3c7bb192fbff23106cf4cf8b479c642ca`),
and its curves are in the adjacent `raw_direct_curves.npz` (SHA-256
`0d6fb5112fd5c2fff887753003ac6041640b57b672684870229d70cd430de7fc`).
The full-versus-half-spectrum comparison is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/10202_it11_direct_fsc_20260903T040436Z/outputs/raw_fft_crosscheck_v2.json`
(SHA-256
`0215314416f1946a3f096867aeccc7fd7dd038b0e333456df30a689f6e4c6c46`).
Exact commands, environment, allocation records, input hashes, the three
harness-only attempts, and launcher hashes are retained in
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/10202_it11_direct_fsc_20260903T040436Z/provenance/AUDIT.md`.

The compact checked-in record is
`docs/benchmarks/em/diagnostics/k1-empiar10202-it011-direct-fsc-20260903.json`.
The final claim remains pending natural completion of the RECOVAR refinement
and comparison with RELION's converged final maps.
