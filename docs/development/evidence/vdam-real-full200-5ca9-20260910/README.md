# Frozen5ca9 real10076 full200 review

Job13664965, four natural200 arms on one H100, source
`5ca9c8fff30be6fbdf77471cff793388bffffe00`, 10k particles/256 pixels, K1.
This review does **not** qualify the moving cleanup tip or full production precision.

All four final cross-engine pairs fail the unchanged FSC-AUC >=0.999 gate.
Native repetition also fails. Two native runs provide one final pairwise value,
not an acceptance interval; no repeat-band or numerical-noise waiver is admitted.

| Pair | Raw-map AUC at100 | Raw-map AUC at200 |
| --- | ---: | ---: |
| native1/native2 | 0.8540848797385455 | 0.9751069897612615 |
| candidate1/candidate2 | 0.9279151369867139 | 0.9701386871327736 |
| candidate1/native1 | 0.9264593371344289 | 0.9732199296674773 |
| candidate2/native2 | 0.8533315870648044 | 0.9709502822021647 |
| candidate1/native2 | 0.8773160074630483 | 0.9683536384150864 |
| candidate2/native1 | 0.9425732463262596 | 0.9748330281709598 |

The integrator recomputed these12 curves from8 manifest-hashed MRCs using the
canonical FSC functions, verified identical to the frozen source. Every AUC
matches the producer exactly. All804 map paths exist and have manifest entries;
this is not rehashing/recomputing all804 maps. The producer samples19 checkpoints,
not all201; sampled failure locations are preserved in [result.json](result.json).
All2,317 declared candidate source files and18 manifest pins check before/after.
Full transitive native build/input closure is outside this review.

[reviewed_curves.npz](reviewed_curves.npz) stores the12 recomputed curves and8
raw/aligned saved reference curves. The reference is not registered as GT in the
fixture manifest. Independently fitted reference alignments and shell proxies do
not establish absolute accuracy. No reference fit was repeated here.

Natural process timing: native1/2 =608.101/662.085s, candidate1/2 =880.109/845.286s.
The four descriptive cross ratios span1.276704–1.447307. Candidate CLI-only timing
excludes import/library checks; aggregate arm timing includes additional work.
These scopes must not be mixed. One four-arm ordering is not broad performance
qualification, and quality remains unaccepted.

Reproduction and full pins:
[review script](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_full200_admission_20260910/review_v2.py),
[command/CPU receipt](/scratch/gpfs/CRYOEM/gilleslab/em_work/codex/real_full200_admission_20260910/verification_v2/receipt.json).
The first reviewer attempt failed on differing manifest path formats and is
preserved. No production source, gates, references or native binaries changed;
no GPU/Slurm workload was launched. Strict state/score-margin classification
still needs evidence beyond these map comparisons.
