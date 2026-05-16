# PPCA Spectrum Shrinkage: Problem Statement, Mathematical Framing, and Open Questions

This note is intended as context for a future agent or human trying to
understand a specific problem in RECOVAR:

- direct PPCA often appears to recover a better heterogeneity basis than the
  older covariance-PCA pipeline,
- but PPCA can still produce a worse spectrum and worse latent embeddings,
- and we want to understand why that happens mathematically, not just patch it
  empirically.

The key goal is to explain the following empirical pattern:

1. `W` from PPCA is often good.
2. The implied eigenvalues or latent scale from PPCA are often too small.
3. The resulting embeddings can therefore be worse than the basis quality would
   suggest.
4. A postprocessing step that keeps the PPCA span but recalibrates the spectrum
   often improves embeddings without changing the span much.

That suggests a "good subspace, bad calibration" problem.

This note is deliberately phrased as an open problem statement, not a final
theory.

## The Model

We observe cryo-EM images

$$
y_i = A_i x_i + \varepsilon_i,
$$

where:

- $y_i \in \mathbb{R}^{m_i}$ is image $i$,
- $A_i$ is the forward operator for image $i$:
  projection, CTF, masking, discretization, whitening, and any other linear
  observation effects used by the implementation,
- $x_i \in \mathbb{R}^d$ is the unknown 3D Fourier or volume-space signal for
  image $i$,
- $\varepsilon_i \sim \mathcal{N}(0, \Sigma_{\varepsilon,i})$ is noise.

We write the heterogeneity model as

$$
x_i = \mu + U \alpha_i,
$$

or equivalently in PPCA form

$$
x_i = \mu + W z_i, \qquad z_i \sim \mathcal{N}(0, I_q).
$$

If we factor

$$
W = U S^{1/2} R,
$$

with $U^\top U = I_q$, $S = \operatorname{diag}(s_1,\dots,s_q)$, and $R$ an
orthogonal rotation, then:

- the **subspace** is $\operatorname{span}(U)=\operatorname{span}(W)$,
- the **spectrum** is $S$,
- the **embedding scale** lives partly in $S$ and partly in the posterior on
  $z_i$.

That separation is the whole point of this note:

- subspace recovery,
- eigenvalue calibration,
- embedding quality

are related, but they are not the same problem.

## Two Competing Methods

### 1. Covariance-PCA

The older RECOVAR pipeline estimates a covariance operator

$$
C \approx \mathbb{E}\big[(x_i-\mu)(x_i-\mu)^\top\big],
$$

then computes its leading eigendecomposition

$$
C \approx U S U^\top.
$$

Embeddings are then computed downstream using this $U,S$ pair, for example via
a marginalized or regularized posterior solve.

Conceptually:

- estimate covariance first,
- then diagonalize,
- then embed.

This method can produce a decent spectrum even when the leading subspace is not
optimal.

### 2. Direct PPCA

PPCA instead optimizes a low-rank factor directly:

$$
x_i = \mu + W z_i,\qquad z_i \sim \mathcal{N}(0, I_q).
$$

Under Gaussian noise, the per-image covariance is

$$
\operatorname{Cov}(y_i \mid W)
=
A_i W W^\top A_i^\top + \Sigma_{\varepsilon,i}.
$$

The negative log-likelihood is therefore, up to constants,

$$
\mathcal{L}(W)
=
\frac12 \sum_i
\left[
\log \left|A_i W W^\top A_i^\top + \Sigma_{\varepsilon,i}\right|
+
(y_i-A_i\mu)^\top
\left(A_i W W^\top A_i^\top + \Sigma_{\varepsilon,i}\right)^{-1}
(y_i-A_i\mu)
\right].
$$

RECOVAR does not optimize this bare likelihood. In practice there is additional
regularization and prior structure in the PPCA path. That matters, because the
main empirical issue seems to be not "PPCA cannot find the right subspace", but
"PPCA seems to shrink the spectrum too much."

The posterior mean of the latent variable has the generic form

$$
\mathbb{E}[z_i \mid y_i]
=
M_i^{-1}
W^\top A_i^\top \Sigma_{\varepsilon,i}^{-1}(y_i-A_i\mu),
$$

where

$$
M_i
=
I_q + W^\top A_i^\top \Sigma_{\varepsilon,i}^{-1} A_i W.
$$

This formula makes the problem clear:

- if $W$ is shrunk too much in norm,
- then the implied covariance $W W^\top$ is too small,
- and the posterior means of $z_i$ are also pulled inward.

So a basis can look directionally correct while the latent coordinates are
still over-regularized.

## The Empirical Problem We Are Trying To Explain

Across the synthetic RECOVAR comparisons, the recurring pattern is:

- PPCA often improves per-PC quality / `RelVar`,
- but the PPCA eigenvalues are often too small relative to GT,
- and the embedding error is then worse than the basis quality alone would
  predict.

This is the main phenomenon to explain.

A convenient way to say it is:

> PPCA often learns a good `W`, but a badly calibrated spectrum and therefore a
> badly calibrated embedding.

This is not merely a visualization issue. It shows up quantitatively in:

- the saved eigenvalues,
- the gap between `s` and empirical `var(z)`,
- embedding squared error,
- sometimes contrast recovery as well.

## Initial Fix That Helps, But Is Probably Not The Final Answer

We tried the following postprocessing:

1. run PPCA and get $W$,
2. orthonormalize $W \mapsto U$,
3. run projected covariance restricted to $\operatorname{span}(U)$,
4. recover a refined basis/spectrum pair $(U_{\mathrm{ref}}, S_{\mathrm{ref}})$,
5. recompute embeddings using that refined $(U,S)$.

Call this **PPCA+ProjCov**.

Empirically this often:

- changes `RelVar` very little,
- leaves the span essentially as good as PPCA,
- improves embedding error materially.

That is strong evidence that the PPCA issue is often not the span itself. The
issue is more likely:

- spectrum calibration,
- posterior shrinkage,
- or a mismatch between the variable being regularized and the variable whose
  spectrum we want to interpret.

So the current fix can be summarized as:

> keep the good PPCA span, recalibrate the spectrum in that span.

Useful, but still ad hoc.

## Why The Variable Being Regularized Might Matter

One useful idea from `main_look_t_this_one.tex` is the distinction between
regularizing a deconvolved object and regularizing a gridded / blurred object.

Suppose the physically meaningful object is $V$, but the reconstruction is
performed in a gridded variable

$$
U = K V,
$$

where $K$ is an invertible gridding or interpolation kernel.

Then these are **not** the same problem:

$$
\min_V \|A K V - b\|_2^2 + \lambda \|V\|_2^2
$$

and

$$
\min_V \|A K V - b\|_2^2 + \lambda \|K V\|_2^2.
$$

The second one is equivalent to solving in the gridded variable and
deconvolving afterward; the first one is the natural Tikhonov penalty on the
deconvolved object itself. They lead to different shrinkage.

This is relevant here because the PPCA regularization may be acting in a
variable that is not the same as the one in which we want to interpret:

- eigenvalues,
- basis magnitude,
- posterior latent scale.

So one plausible mathematical explanation is:

> PPCA is regularizing the right span in the wrong coordinates.

That would naturally produce:

- good directions,
- biased low singular values,
- over-shrunk embeddings.

This should be treated as a hypothesis, not a conclusion.

## A Cleaner Mathematical Way To Phrase The Problem

Let the true heterogeneity covariance be

$$
C_\star = U_\star S_\star U_\star^\top.
$$

Suppose PPCA returns

$$
W_{\mathrm{ppca}} \approx U_\star \widetilde{S}^{1/2} R
$$

with a good column span but with

$$
\widetilde{S} \ll S_\star
$$

in magnitude.

Then two things can simultaneously be true:

1. the principal directions are good, because $\operatorname{span}(W)$ is close
   to $\operatorname{span}(U_\star)$,
2. the posterior means are over-shrunk, because the effective covariance in the
   posterior solve is too small.

That immediately explains how one can get:

- good `RelVar`,
- poor eigenvalues,
- poor embeddings.

Projected covariance in the PPCA span then acts like a restricted
re-estimation of $S$ while keeping $U$ approximately fixed.

That explains why PPCA+ProjCov can improve embedding quality while barely
changing subspace quality.

## What We Want To Understand

The mathematical questions are:

1. Why can PPCA recover a better $W$ or better subspace than covariance-PCA,
   yet produce a worse spectrum?
2. Where exactly does the shrinkage enter?
   - in the PPCA likelihood itself,
   - in the explicit regularizer,
   - in the prior estimation,
   - in whitening / deconvolution conventions,
   - in the posterior solve for $z_i$,
   - or in some combination of all of these?
3. Can the phenomenon be described as:
   - good subspace recovery,
   - bad spectrum calibration,
   - and therefore bad embedding calibration?
4. Can we derive a simple toy model showing:
   - correct span,
   - underestimated eigenvalues,
   - degraded embeddings,
   - improvement from recalibrating only $S$ in a fixed span?

## What We Actually Want From A Better Algorithm

The goal is not just to explain the current fix. The real goal is to design a
better method.

The desired method should:

- retain the good PPCA basis discovery,
- avoid the apparent spectrum shrinkage,
- produce well-calibrated embeddings directly,
- and ideally not require a separate covariance recalibration step.

So the open-ended algorithmic question is:

> Can we design a principled method that preserves the improved PPCA span while
> estimating the spectrum and embeddings in a better-calibrated way?

Possible directions to think about, without assuming any of them is correct:

- decouple subspace estimation from spectrum estimation,
- alternate between span updates and spectrum recalibration,
- regularize in a variable that matches the physically meaningful object,
- explicitly separate basis regularization from posterior embedding
  regularization,
- estimate $U$ with PPCA but estimate $S$ with a moment-matching or restricted
  covariance criterion,
- learn a posterior temperature / scale correction for embeddings,
- formulate a model where $U$ and $S$ have different priors or penalties.

Again, these are directions to analyze, not settled recommendations.

## What Not To Assume

- Do not assume covariance-PCA is "the correct answer."
- Do not assume direct PPCA is mathematically wrong just because its saved
  spectrum looks too small.
- Do not assume PPCA+ProjCov is the final algorithm.
- Do not assume the issue is only numerical.

The most plausible current reading is:

- PPCA often improves the subspace,
- but regularization / scaling / posterior calibration appears to distort the
  spectrum,
- and that hurts embeddings.

## Relevant Empirical Outputs In This Repo

The following figures summarize the existing synthetic comparisons:

### Per-PC quality

![Synthetic per-PC quality](./ppca_shrinkage_pc_quality.png)

### Raw eigenvalues

![Synthetic eigenvalues vs GT](./ppca_shrinkage_eigenvalues.png)

### Estimated-to-GT eigenvalue ratio

![Synthetic eigenvalue ratio to GT](./ppca_shrinkage_eigenvalue_ratio.png)

### Embedding and contrast metrics

![Synthetic embedding error and contrast correlation](./ppca_shrinkage_embedding_error.png)

These are evidence for the problem statement above; they are not themselves the
mathematical explanation.

## External References That Are Likely Useful

- Tipping and Bishop, *Probabilistic Principal Component Analysis*:
  <https://www.microsoft.com/en-us/research/publication/probabilistic-principal-component-analysis/>
- Katsevich, Katsevich, and Singer, *Covariance Matrix Estimation for the
  Cryo-EM Heterogeneity Problem* / *Structural Variability from Noisy
  Tomographic Projections*:
  <https://oar.princeton.edu/bitstream/88435/pr11t1k/1/1710.09791.pdf>
- Gavish and Donoho, *Optimal Shrinkage of Eigenvalues in the Spiked Covariance
  Model*:
  <https://arxiv.org/abs/1311.0851>
- SOLVAR, as a more recent covariance-based cryo-EM comparison point:
  <https://arxiv.org/abs/2602.17603>

The main use of these references here is:

- PPCA theory,
- covariance-spectrum shrinkage theory,
- cryo-EM covariance estimation structure.

## Code References

- direct PPCA and projected-covariance pipeline branch:
  - `recovar/commands/pipeline.py`
- synthetic compare harness:
  - `recovar/ppca/compare_covariance_vs_ppca_pipeline.py`
- synthetic summary writer:
  - `recovar/ppca/summarize_pipeline_compare_sweep.py`
- PPCA implementation:
  - `recovar/ppca/ppca.py`
- variance-prior note:
  - `docs/math/ppca_variance_prior_notes.md`
