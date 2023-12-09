import numpy as np
import jax.numpy as jnp
import matplotlib.pyplot as plt
import jax

def captured_variance(test_v, U, s):
    test_v, _ = jnp.linalg.qr(test_v)
    x = (jnp.conj(test_v.T) @ U) * np.sqrt(s)
    norms = np.linalg.norm(x, axis=-1)**2
    return np.cumsum(norms)

def relative_variance_from_captured_variance(variance, s):
    all_variance = np.sum(s)
    return (variance) / all_variance

def relative_variance(test_v, U, s):
    variance = captured_variance(test_v, U, s)
    return relative_variance_from_captured_variance(variance, s)

def normalized_variance(test_v, U, s):
    variance = captured_variance(test_v, U, s)
    return normalized_variance_from_captured_variance(variance, s)

def normalized_variance_from_captured_variance(variance, s):
    all_variance_up_to = np.cumsum(np.asarray(s))
    if variance.size > all_variance_up_to.size:
        all_variance_up_to_padded = np.ones(variance.size) * all_variance_up_to[-1]
        all_variance_up_to_padded[:all_variance_up_to.size] = all_variance_up_to
    else:
        all_variance_up_to_padded = all_variance_up_to
    return (variance) / all_variance_up_to_padded[:variance.size]


def get_all_variance_scores(test_v, U, s):
    variance = captured_variance(test_v, U, s)
    rel_variance = relative_variance_from_captured_variance(variance, s)
    normalized_variance = normalized_variance_from_captured_variance(variance, s)
    return variance, rel_variance, normalized_variance



# batch_solve_hermitian = jax.vmap(lambda a, b : jax.scipy.linalg.solve( a ,b, assume_a='pos'), in_axes = (0,0))
batch_solve_hermitian = jax.vmap(lambda a, b : jax.scipy.linalg.solve( a ,b), in_axes = (0,0))

batch_matmat = jax.lax.batch_matmul
def batch_matvec(A,b):
    return batch_matmat(A, b[...,None])[...,0]

broadcast_A_matvec = jax.vmap(lambda A, x : (A@x[...,None])[...,0], in_axes = (None, 0))

batch_outer = jax.vmap(jnp.outer, in_axes=(0,0))
batch_kron = jax.vmap(jnp.kron, in_axes = (0,0))


def generate_PCA_experiment(m,n,b,y_dim, snr = 1,P_option = "identity", noise_model = "gray"):

    U = np.random.randn(n,b)
    U,_ = np.linalg.qr(U)
    eigs = (0.75)**np.arange(b) #np.flip(np.arange(0,b))#np.ones(np.arange(10))
    # eigs_init = eigs
    
    Z = np.sqrt(eigs)[...,None] * np.random.randn(b, m)
    x = (U @ Z).T
    use_sample_as_gt = True
    if use_sample_as_gt:
        U,s_t,_ = jnp.linalg.svd(x.T)
        eigs = (s_t**2)/ x.shape[0]

    if P_option == "rand":
        P = 1 + np.random.rand(m, y_dim, n)   
    elif P_option == "subsampled_diag" :
        key = jax.random.PRNGKey(0)
        perms = jax.random.permutation(key, np.repeat(np.arange(y_dim)[None], repeats = m, axis=0), axis=1, independent=True)
        del_entries = np.random.randint(0, 2, size = perms.shape).astype(bool)
        perms = perms.at[del_entries].set(0)
        batch_diag = jax.vmap(jnp.diag)
        P = batch_diag(perms).astype(x.dtype)
    elif P_option == 'identity_scaled':
        P = 2*np.repeat(-jnp.eye(y_dim)[None], m, axis =0)
    elif P_option == 'identity':
        P = 1*np.repeat(jnp.eye(y_dim)[None], m, axis =0)
        
    if noise_model == "white":
        # snr = eigs[0] *1
        noise_var = eigs[0] / snr
        noise_variances = jnp.ones(y_dim) * noise_var
        noise_covariance = jnp.eye(y_dim) * noise_var
        noise_covariance_root = jnp.diag(np.sqrt(noise_variances))
    elif noise_model == "gray":
        # snr = eigs[0] *1
        noise_var0 = eigs[0] / snr
        offset_from_0 = 0.1
        noise_variances = (np.random.rand(y_dim)*(2- 2*offset_from_0) + offset_from_0)*noise_var0 #noise_var0 * (0.75)**np.arange(y_dim)
        noise_covariance = jnp.diag(noise_variances)
        noise_covariance_root = jnp.diag(np.sqrt(noise_variances))

    noise = broadcast_A_matvec(noise_covariance_root, np.random.randn(m, y_dim ))
    y = batch_matvec(P, x)  + noise
    return U, eigs, y, P, noise_covariance, noise_covariance_root