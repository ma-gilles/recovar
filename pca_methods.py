import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax
from pca_experiments import *

def almost_randomized_SVD(P, y, Q, noise_variance, prior_variance):

    # Form all the matrices we need
    PQ = jax.lax.batch_matmul(P, Q[None])
    QPPQ = jax.lax.batch_matmul(jnp.conj(PQ).tranpose(0,2,1),jnp.conj(PQ).tranpose(0,2,1))
    H = QPPQ + jnp.diag(1/prior_variance)
    Hinv = jax.numpy.linalg.pinv(H) # oof
    # M = batch_solve_hermitian(H, QPPQ)
    M = batch_matmat(Hinv, M)
    new_noise_variance = Hinv * noise_variance # May want this to be white for things to be less annoying
    z = batch_matvec(M, y)

    # option 1: just svd Z
    U, S, V =jnp.linalg.svd(z)
    pred_eigs1 = S**2 / y.shape[0]

    # covariance estimation again
    z_covar = covar_estimate(z, M, new_noise_variance)
    # U, S, V =jnp.linalg.svd(z)
    # pred_eigs1 = S**2 / y.shape[0]

    return

## SVD based PCA

def pca_by_svd(z, **kwargs):
    u,s,v = jnp.linalg.svd(z.T, full_matrices = False)
    return u, s**2

def pca_by_whitened_svd(z, noise_variances):
    # n = z.shape[0]
    z= z.T
    z /= jnp.sqrt(noise_variances)[...,None]
    u,s,_ = jnp.linalg.svd(z, full_matrices = False)
    # covariances = z @ z.T / n  - jnp.diag(noise_variances)
    # s, u = jnp.linalg.eigh(covariances)
    u*= jnp.sqrt(noise_variances)
    return u, s**2


## Covariance-based PCAs

def vec(X):
    return X.T.reshape(-1)

## Inverse of vec function.
def unvec(x):
    n = np.sqrt(x.size).astype(int)
    return x.reshape(n,n).T


def covar_estimate(y, P, noise_covar):
    PTP = jax.lax.batch_matmul(np.conj(P).transpose(0,2,1),P)
    lhs = jnp.sum(batch_kron(PTP,PTP), axis=0)
    yy_m_Sigma = batch_outer(y,y) - noise_covar
    P_yy_m_Sigma = jax.lax.batch_matmul(np.conj(P).transpose(0,2,1),yy_m_Sigma)
    P_yy_m_Sigma = jax.lax.batch_matmul(P_yy_m_Sigma,P)
    P_yy_m_Sigma = jnp.sum(P_yy_m_Sigma, axis=0)

    rhs = vec(P_yy_m_Sigma)
    covar = jnp.linalg.solve(lhs, rhs)
    covar = unvec(covar)
    return unvec(covar)


def covar_estimate_diag(y, P, noise_covar):
    PTP = P**2
    lhs = jnp.sum(batch_outer(PTP, PTP), axis=0)
    Py = P * y
    rhs = jnp.sum(batch_outer(Py, Py) - noise_covar, axis=0)
    return rhs / lhs

matmat_broadcast_first = jax.vmap(lambda A, x : A@x, in_axes = (None, 0))

def solve_regularized_least_squares(y, P, noise_covariance_inverse, prior_variance ):
    # Form P^T \Lambda^{inv} 
    PTP = jax.lax.batch_matmul(np.conj(P).transpose(0,2,1),matmat_broadcast_first(noise_covariance_inverse, P))
    P_y = batch_matvec(np.conj(P).transpose(0,2,1),y)
    A_mat = PTP + jnp.diag(1/prior_variance)
    x_hat = batch_solve_hermitian(A_mat, P_y)
    return x_hat

matmat_broadcast_second = jax.vmap(lambda A, x : A@x, in_axes = (0, None))

def almost_randomized_SVD_by_SVD(P, y, Q, noise_covariance, prior_variance, P_diag = True):
    noise_covariance_inverse = jnp.linalg.pinv(noise_covariance) # sloppy but oh well...
    if P_diag:
        PQ = scale_columns(P, Q)
    else:
        PQ = matmat_broadcast_second(P, Q)

    x_hat = solve_regularized_least_squares(y, PQ, noise_covariance_inverse, prior_variance)

    # option 1: just svd Z
    U, S = pca_by_svd(x_hat)
    # U, S, V =jnp.linalg.svd(x_hat.T)
    # n =  y.shape[0]
    U_hat = Q@ U 
    return U_hat,  S


scale_columns = jax.vmap(lambda d, x : d[...,None]*x, in_axes = (0, None))

def almost_randomized_SVD_by_covar(P, y, Q, noise_covariance, P_diag = True):
    if P_diag:
        PQ = scale_columns(P, Q)
    else:
        PQ = matmat_broadcast_second(P, Q)

    u, s  = P_pca_by_covar(y, PQ, noise_covariance) 
    U2 = Q@ u

    return U2, s




def P_pca_by_covar(y, P, noise_covar):
    covariance = covar_estimate(y, P, noise_covar)
    s, u = jnp.linalg.eigh(covariance)
    return np.fliplr(u), np.flip(s)

def P_pca_by_covar_diag(y, P, noise_covar):
    covariance = covar_estimate_diag(y, P, noise_covar)
    s, u = jnp.linalg.eigh(covariance)
    return np.fliplr(u), np.flip(s)


def pca_by_covar(z, noise_covariances):
    n = z.shape[0]
    z= z.T
    covariances = z @ z.T / n  - noise_covariances
    s, u = jnp.linalg.eigh(covariances)
    # import pdb; pdb.set_trace()
    return np.fliplr(u), np.flip(s)

def pca_by_whitened_covar(z, noise_variances):
    z_whitened = z / jnp.sqrt(noise_variances)
    u, s = pca_by_covar(z_whitened, jnp.eye(z.shape[1]))
    u = u * jnp.sqrt(noise_variances)
    return u, s


def pca_by_whitened_covar(z, noise_variances):
    z_whitened = z / jnp.sqrt(noise_variances)
    u, s = pca_by_covar(z_whitened, jnp.eye(z.shape[1]))
    u = u * jnp.sqrt(noise_variances)
    return u, s



