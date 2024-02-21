import jax
import jax.numpy as jnp
import numpy as np
import pickle
from recovar import core, utils, simulator, linalg, mask, constants
from recovar.fourier_transform_utils import fourier_transform_utils
ftu = fourier_transform_utils(jnp)
ftu_np = fourier_transform_utils(np)

# Maybe should take out these dependencies?
from cryodrgn import mrc

def qr_on_cpu(Q):
    Q = jax.device_put(Q, device=jax.devices("cpu")[0])
    Q,R = jnp.linalg.qr(Q)
    Q = np.array(Q) # I don't know why but not doing this causes massive slowdowns sometimes?
    R = np.array(R)
    return Q, R

def captured_variance(test_v, U, s):
    # test_v, _ = qr_on_cpu(test_v)
    
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


