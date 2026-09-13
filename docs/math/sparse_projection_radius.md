# Sparse projection radius propagation

The generic sparse projection path must forward the requested `max_r` to the
same projector used by coarse scoring. The radius is not merely a final pixel
selection: the CUDA texture implementation also derives its compact texture
extent and origin from the radius. Gathering the same image pixels afterward
does not make a projection computed with a different radius equivalent.

Implementation: [`_compute_sparse_pass2_projections_block`](../../recovar/em/sparse_pass2/sparse_pass2_projection_blocks.py).
The helper also reads `max_r` to infer a supplied RELION projector's output crop,
but must not consume it before the generic call. An explicit `None` and an
omitted argument retain their distinct underlying-projector meanings. Explicit
RELION output size remains independent of the model radius.

For fixed image operand \(I\), weights \(w\), and projections \(P_f,P_c\), the
projection-only change in Gaussian log score is

\[
\Delta s=\Re\sum_p w_p\overline{I_p}(P_{f,p}-P_{c,p})
-\tfrac12\sum_p w_p(|P_{f,p}|^2-|P_{c,p}|^2).
\]

This identity permits an offline diagnostic without changing production precision.
The September 13 four-particle cold-start discriminator found identical padded
volume and rotation inputs but differing coarse/pass-2 projections. For particle
1433, projection differences predicted score RMS0.101721 versus observed0.101761;
the remaining RMS was6.97e-5. A matched missing-radius intervention reduced its
Pmax from0.103265554 to0.095389739 (native STAR0.095284). This is a scoped diagnostic,
not strict-state, full-trajectory or speed acceptance. Existing mixed-precision
stages were not changed or qualified as all-float32.

Exact forwarding regressions, including whole/chunked/windowed generic paths and
supplied-RELION crop behavior, are in
[`test_sparse_projection_radius.py`](../../tests/unit/test_sparse_projection_radius.py).
Current qualification and pinned artifacts are linked from
[`em_status.md`](../development/em_status.md).
