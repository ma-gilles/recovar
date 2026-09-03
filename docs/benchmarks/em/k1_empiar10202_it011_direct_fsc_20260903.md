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

## Corrected-run replication and masked support

The memory-corrected full-particle trajectory independently replicated this
matched checkpoint at source commit
`b31f7bb3a88b96885e568fa4a12d5ec265ab4aab`. Its direct shared-frame audit
again requires no fitted operation: both half-map FSC curves cross at shell
250 (`2.5215997696 A`), with merged/half-1/half-2 cross-engine FSC-AUC
`0.9911398/0.9868918/0.9869988`. The sealed direct-replication root is
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_it11_corrected_direct_fsc_20260903T0501`;
its primary JSON and validated-manifest SHA-256 values are
`a89bf4b106e2f6943e3ea001e022afd910f4fe8d6d6d371bbace74eb3b1936f2`
and
`ae7b2157bc951fb31dc472b396828f933638bf0d62edddc75174052a94c0219d`.

A separate RELION-postprocess replication applies the sealed publication
mask, `--randomize_at_fsc 0.8`, and seed 42 to those corrected maps. Both
engines cross the corrected-masked curve at shell 248 (`2.5419352516 A`);
resolved shells 1--247 have normalized-AUC delta `0.0009232` and RMSE
`0.0119231`. Masked FSC remains supporting only and cannot rescue or replace
the unmasked comparison. Exact command, resources, source hashes, and output
seals are under
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_it11_corrected_masked_fsc_20260903T0518`;
the result JSON and validated manifest have SHA-256
`6799bfefff1ec370828ec292be0dd69d799df83dda7eda3ed10e0f80566bd7a2`
and
`a748c022552f5b3e9bd34db2dd14a91dc37143097a074774024440006c78d72d`.

## Later corrected iteration-14 checkpoint

The advancing corrected trajectory provides a later, independent matched
checkpoint. At numbered iteration 14, both RECOVAR and RELION cross unmasked
half-map FSC 0.143 at shell 262 (`2.4061066504 A`). The joint shells 1--261
give direct cross-engine FSC-AUC `0.9875827` merged and
`0.9816448/0.9830144` for halves 1/2, with half-FSC RMSE `0.0059807`,
half-band AUC difference `0.0007075`, and resolution ratio `1.0`. No fitted
operation was used and all frozen primary gates pass. This remains a
diagnostic-only intermediate checkpoint until the trajectory reaches natural
completion.

CPU job `13380640` completed with exact requested/allocated
`cpu=8,mem=80G,node=1,billing=20` and retained its complete, verified audit at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_it14_corrected_direct_fsc_20260903T0529`.
The metrics JSON and manifest SHA-256 values are
`eaae20795685282f13814a9ee9ccbeb41ec9c73a7eb52a0b57201f921c90881b`
and
`5b141e0e92265881cd36744d9274c128c341ea9b02f29b4c0465843b70cf6fe9`.

The exact same publication mask and phase-randomization policy give an
iteration-14 corrected-masked crossing of shell 258 (`2.4434106295 A`) for
both engines. Resolved shells 1--257 have masked AUC delta `0.0018079` and
RMSE `0.0071828`. This is supporting-only evidence; the raw comparison passes
independently. Job `13380912` and the complete sealed record are at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/live_run_monitor/empiar10202_set6_it14_corrected_masked_fsc_20260903T0546`;
its result JSON and verified manifest have SHA-256
`04bdfb5fe7eee859dedb8bb87f38fecd12277a5d339c0d06ad7596c4f5b5fc3e`
and
`9962a2000bf8f86e63c01131e5174aae7ef5a061f2f9c7e9a967d4afcaeddf89`.

## Later corrected iteration-16 checkpoint

At numbered iteration 16, both RECOVAR and RELION cross unmasked half-map FSC
0.143 at shell 281 (`2.2434161651 A`). Joint shells 1--280 have direct
cross-engine FSC-AUC `0.9858187` merged and `0.9796343/0.9803080` for halves
1/2. Half-FSC RMSE is `0.0051946`, half-band AUC differs by `0.0005905`, and
the resolution ratio is `1.0`. All frozen primary gates pass with no fitted
operation. Natural convergence remains the final admission boundary.

CPU job `13381233` completed with exact requested/allocated
`cpu=8,mem=80G,node=1,billing=20`. Its complete verified record is at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/root/empiar10202_set6_it16_corrected_direct_fsc_20260903T100833Z`;
the metrics JSON and manifest SHA-256 values are
`c78932d1de8c6d0295820570c3e571365841e80178cabb9bbdd84e0f3106343c`
and
`bdb19f45bbf1ce1cea3382855184893fb509ee196315e23fa34e73c9e1f8229e`.

The iteration-16 masked-support replication uses the same sealed publication
mask and phase-randomization policy as the earlier checkpoints. RECOVAR and
RELION first remain below corrected-masked FSC 0.143 at shells 280
(`2.251428 A`) and 277 (`2.275812 A`), respectively. Over their common band,
shells 1--276, masked normalized AUC differs by `0.0008457` and curve RMSE is
`0.0053812`. The raw curves extracted from the same postprocess STAR files
both reproduce the shell-281 crossing. This is supporting-only evidence and
does not replace or modify the direct unmasked pass.

CPU job `13381344` completed with exact requested/allocated
`cpu=4,mem=64G,node=1,billing=16`. Its verified record is at
`/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/root/empiar10202_set6_it16_corrected_masked_fsc_20260903T101521Z`;
the comparison JSON and manifest SHA-256 values are
`f72404609a235f0cb2a09df9d0bae3dc4e23d87d907944892e7ea9e9ec8abaa7`
and
`3e69ade91053e305665cead63abcb6ac28d6af41a6154f056d560c036357cf3c`.

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
