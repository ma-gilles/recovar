# recovar.heterogeneity

Heterogeneity analysis pipeline: covariance estimation, PCA, latent
embedding, and volume reconstruction.

## covariance_estimation

Regularized covariance matrix estimation from half-set cryo-EM data.

::: recovar.heterogeneity.covariance_estimation
    options:
      members_order: source

## principal_components

Principal component analysis of the estimated covariance operator.

::: recovar.heterogeneity.principal_components
    options:
      members_order: source

## embedding

Per-image latent coordinate estimation via linear projection.

::: recovar.heterogeneity.embedding
    options:
      members_order: source

## heterogeneity_volume

Kernel-regression volume reconstruction from latent embeddings.

::: recovar.heterogeneity.heterogeneity_volume
    options:
      members_order: source

## latent_density

Kernel density estimation and stable-state detection in latent space.

::: recovar.heterogeneity.latent_density
    options:
      members_order: source

## trajectory

Minimum-energy path finding between states in latent space.

::: recovar.heterogeneity.trajectory
    options:
      members_order: source

## adaptive_kernel_discretization

Adaptive kernel methods for heterogeneity volume estimation.

::: recovar.heterogeneity.adaptive_kernel_discretization
    options:
      members_order: source

## locres

Local resolution estimation and local filtering of reconstructed volumes.

::: recovar.heterogeneity.locres
    options:
      members_order: source
