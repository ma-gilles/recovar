# Kernel correction, prior, and masking

The typical pipeline solves
$$
\hat u = \arg\min_u F(u),
$$
then applies the post-processing map $\hat v = K^{-1}\hat u$, followed by masking. The mathematically correct kernel-corrected formulation is instead
$$
\hat v = \arg\min_v F(Kv).
$$
Since $K$ is invertible, assuming the minimum is unique, this is exactly equivalent in the unregularized case:
$$
\hat v = K^{-1}\arg\min_u F(u) = K^{-1}\hat u.
$$

We use this to formalize the full post-processing into one optimization problem.

## Notation

We work with the deblurred volume `v` in real space.

- `A` and `b` are the least-squares Hessian and right-hand side in the Fourier domain.
- `\Lambda` is the prior precision in the Fourier domain.
- `K` is the gridding/discretization operator in real space.
- `M` is the soft-mask weight in real space.
- `P` is the hard-mask projector in real space.
- `E` injects the reduced unknowns on the support `\Omega` into the full real-space grid.
- `k_{\mathrm{eff}}^2 = N^{-1}\sum_x K(x)^2`.
- `\bar m^2 = N^{-1}\sum_x M(x)^2`.

## Naive post-processing

$$
\tilde u
=
\arg\min_u
\frac12 \, u^{*} A \, u
- \operatorname{Re}\!\left(b^{*} u\right)
+ \frac12 \, u^{*} \Lambda \, u.
$$

Then apply gridding correction and masking afterward:

$$
\hat v_{\mathrm{naive}} = S K^{-1} \tilde u,
$$

where `S=P` for a hard mask and `S=I-M` for a soft mask.

No preconditioner is needed: this is a diagonal solve in the Fourier domain.

## Hard mask

Write

$$
v = E\alpha,
$$

with `\alpha` containing the values on `\Omega`.

$$
\hat \alpha
=
\arg\min_\alpha
\frac12 \, {K E\alpha}^{\,*} A \, {K E\alpha}
- \operatorname{Re}\!\left(b^{*} {K E\alpha}\right)
+ \frac12 \, {E\alpha}^{\,*} \Lambda \, {E\alpha},
$$

$$
\hat v_{\mathrm{hard}} = E\hat \alpha.
$$

Solver: PCG on the reduced variable `\alpha`.

Preconditioner:
$$
P_{\mathrm{hard}}^{-1}(\xi)
=
\bigl(k_{\mathrm{eff}}^2 A(\xi) + \Lambda(\xi)\bigr)^{-1},
$$
applied on the full grid and then restricted back to the support.

## Smoothed mask

$$
\hat v_{\mathrm{soft}}
=
\arg\min_v
\frac12 \, {Kv}^{\,*} A \, {Kv}
- \operatorname{Re}\!\left(b^{*} {Kv}\right)
+ \frac12 \, v^{*} \Lambda \, v
+ \frac{\mu}{2}\,\|Mv\|_2^2.
$$

Solver: PCG in the full variable `v`.

Preconditioner:
$$
P_{\mathrm{soft}}^{-1}(\xi)
=
\bigl(k_{\mathrm{eff}}^2 A(\xi) + \Lambda(\xi) + \mu \, \bar m^2\bigr)^{-1}.
$$

# PPCA formulations with kernel correction, prior, and masking

We work with the deblurred real-space loading field `V`.

- `A(\xi)` is the Fourier-domain `q \times q` positive semidefinite block. In the code this is `LHS(\xi)`.
- `d(\xi)` is the Fourier-domain linear term. In the code this is `RHS(\xi)`.
- `\Lambda(\xi)` is the Fourier-domain prior precision.
- `K` is the gridding/discretization operator in real space.
- `M` is the soft-mask weight in real space.
- `E` injects the reduced unknowns on the support `\Omega` into the full real-space grid.
- `Z` denotes the reduced hard-mask unknown.
- `k_{\mathrm{eff}}^2 = N^{-1}\sum_x K(x)^2`.
- `\bar m^2 = N^{-1}\sum_x M(x)^2`.

The common PPCA quadratic is

$$
J_0(V)
=
\frac12 \sum_{\xi} (KV)_F(\xi)^* A(\xi) (KV)_F(\xi)
- \operatorname{Re}\sum_{\xi} d(\xi)^* (KV)_F(\xi)
+ \frac12 \sum_{\xi} V_F(\xi)^* \Lambda(\xi) V_F(\xi).
$$

## Naive post-processing

First solve the unmasked problem in the gridded variable:

$$
\widetilde U
=
\arg\min_U
\left[
\frac12 \sum_{\xi} U(\xi)^* A(\xi) U(\xi)
- \operatorname{Re}\sum_{\xi} d(\xi)^* U(\xi)
+ \frac12 \sum_{\xi} U(\xi)^* \Lambda(\xi) U(\xi)
\right].
$$

Then apply gridding correction and masking afterward:

$$
\widehat V_{\mathrm{naive}} = S K^{-1} \widetilde U,
$$

where `S` is the chosen post-processing mask operator.

No preconditioner is needed: this is independent at each Fourier voxel,
$$
\widetilde U(\xi) = \bigl(A(\xi)+\Lambda(\xi)\bigr)^{-1} d(\xi).
$$

## Hard mask

Write

$$
V = EZ,
$$

where `Z` contains the values of `V` on the support `\Omega`.

Then solve

$$
\widehat Z
=
\arg\min_Z J_0(EZ),
$$

and set

$$
\widehat V_{\mathrm{hard}} = E\widehat Z.
$$

Equivalently,

$$
\widehat Z
=
\arg\min_Z
\left[
\frac12 \sum_{\xi} (KEZ)_F(\xi)^* A(\xi) (KEZ)_F(\xi)
- \operatorname{Re}\sum_{\xi} d(\xi)^* (KEZ)_F(\xi)
+ \frac12 \sum_{\xi} (EZ)_F(\xi)^* \Lambda(\xi) (EZ)_F(\xi)
\right].
$$

Solver: PCG on the reduced variable `Z`.

Preconditioner:
$$
P_{\mathrm{hard}}^{-1}(\xi)
=
\bigl(k_{\mathrm{eff}}^2 A(\xi) + \Lambda(\xi)\bigr)^{-1},
$$
applied blockwise in the Fourier domain and then pulled back to the support.

## Soft mask

Solve

$$
\widehat V_{\mathrm{soft}}
=
\arg\min_V
\left[
J_0(V)+\frac{\mu}{2}\|MV\|_F^2
\right].
$$

Equivalently,

$$
\widehat V_{\mathrm{soft}}
=
\arg\min_V
\left[
\frac12 \sum_{\xi} (KV)_F(\xi)^* A(\xi) (KV)_F(\xi)
- \operatorname{Re}\sum_{\xi} d(\xi)^* (KV)_F(\xi)
+ \frac12 \sum_{\xi} V_F(\xi)^* \Lambda(\xi) V_F(\xi)
+ \frac{\mu}{2}\|MV\|_F^2
\right].
$$

Solver: PCG in the full variable `V`.

Preconditioner:
$$
P_{\mathrm{soft}}^{-1}(\xi)
=
\bigl(k_{\mathrm{eff}}^2 A(\xi) + \Lambda(\xi) + \mu \, \bar m^2 I_q\bigr)^{-1}.
$$

## Code references

| Formulation | Implementation |
|-------------|---------------|
| Naive (homogeneous) | `recovar/reconstruction/homogeneous.py:get_mean_conformation_relion` |
| Naive (PPCA) | `recovar/ppca/ppca.py:EM_step_half` (standard per-voxel path, line ~1148) |
| Hard mask PCG | `recovar/reconstruction/pcg_mean.py:pcg_mstep` |
| Hard mask with K | `bench_mstep.py:solve_hard` |
| Soft mask with K | `bench_mstep.py:solve_soft` |
| Gridding kernel K | `bench_mstep.py:compute_G` |
| Alpha weight M | `bench_mstep.py:build_alpha` |
