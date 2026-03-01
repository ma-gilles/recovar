"""EM state containers: EMState, SGDState, HeterogeneousEMState."""

import logging
import numpy as np
import jax
from recovar import utils, jax_config
from recovar.reconstruction import relion_functions
from recovar.core import mask as mask_fn
from recovar.heterogeneity import principal_components
from .e_step import E_with_precompute
from .m_step import M_with_precompute
from .heterogeneity import compute_H_B, compute_projected_covariance_rhs_lhs, solve_covariance
logger = logging.getLogger(__name__)

def get_default_sgd_options():
    options = {}
    options['minibatch_size'] = 30
    options['steps_size'] = 'hess'
    options['mu'] = 0.9
    return options



## Probably should implement these so we don't have to pass around so many arguments
class EMState():
    mean = None
    mean_variance = None
    noise_variance = None
    name = "EM"
    Ft_CTF = 0
    Ft_y = 0
    def __init__(self, mean, mean_variance, noise_variance):
        self.mean = mean
        self.mean_variance = mean_variance
        self.noise_variance = noise_variance

    def E_step(self, experiment_dataset, rotations, translations, disc_type, big_image_batch):
        probabilities = E_with_precompute(experiment_dataset, self.mean, rotations, translations, self.noise_variance, disc_type, big_image_batch)
        return probabilities

    def M_step(self, experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch):
        Ft_y_this, Ft_CTF_this = M_with_precompute(experiment_dataset, probabilities, rotations, translations, self.noise_variance, disc_type, big_image_batch)
        self.Ft_y += Ft_y_this
        self.Ft_CTF += Ft_CTF_this

    def finish_up_M_step(self, experiment_dataset, disc_type):
        self.mean = relion_functions.post_process_from_filter(experiment_dataset, self.Ft_CTF, self.Ft_y, tau = self.mean_variance, disc_type = disc_type).reshape(-1)



class SGDState():
    mean = None
    mean_variance = None
    noise_variance = None
    update = 0
    name = "SGD"
    sgd_projection = lambda x : x
    sgd_batchsize = 100

    def __init__(self, mean, mean_variance, noise_variance):
        self.mean = mean
        self.mean_variance = mean_variance
        self.noise_variance = noise_variance

    def E_step(self, experiment_dataset, rotations, translations, disc_type, big_image_batch):
        probabilities = E_with_precompute(experiment_dataset, self.mean, rotations, translations, self.noise_variance, disc_type, big_image_batch)
        return probabilities

    def M_step(self, experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch, iter, volume_mask = None):

        Ft_y_this, Ft_CTF_this = M_with_precompute(experiment_dataset, probabilities, rotations, translations, self.noise_variance, disc_type, big_image_batch)
        n_images_batch = len(big_image_batch)

        mean = self.mean
        mu = 0.9
        grad = 2 * ((Ft_CTF_this) * mean - Ft_y_this) *  experiment_dataset.n_images / n_images_batch + 2/ self.mean_variance * mean

        step = 1 / np.max(np.abs(Ft_CTF_this))
        self.update = mu * self.update + (1 - mu) * step * grad 
        if np.isnan(self.update).any() or np.isinf(self.update).any():
            logger.error("|update|: %s", np.linalg.norm(self.update))

        if iter % 10 == 0:
            logger.debug('|dx| / |x|: %s', np.linalg.norm(self.update) / np.linalg.norm(mean))
            logger.debug('|prior|/ grad: %s', np.linalg.norm( 2/ self.mean_variance * mean) / np.linalg.norm(grad))
            logger.debug('|x|: %s', np.linalg.norm( mean))
            logger.debug('|dx|: %s', np.linalg.norm( self.update))
        mean -= self.update * 0.1
        mean = self.sgd_projection(mean)

        std_multiplier = 10
        mean = np.clip(mean.real, -std_multiplier * np.sqrt(self.mean_variance), std_multiplier * np.sqrt(self.mean_variance)) + 1j * np.clip(mean.imag, -std_multiplier * np.sqrt(self.mean_variance), std_multiplier * np.sqrt(self.mean_variance))

        if np.isnan(mean).any() or np.isinf(mean).any() or np.isnan(np.linalg.norm(mean)) or np.isinf(np.linalg.norm(mean)):
            logger.error('|dx| / |x|: %s', np.linalg.norm(self.update) / np.linalg.norm(mean))
            logger.error('|prior|/ grad: %s', np.linalg.norm( 2/ self.mean_variance * mean) / np.linalg.norm(grad))
            logger.error('|x|: %s', np.linalg.norm( mean))
            logger.error('|dx|: %s', np.linalg.norm( self.update))

            raise ValueError(
                f"NaN/Inf detected in mean estimate. "
                f"|update|={np.linalg.norm(self.update)}, |x|={np.linalg.norm(mean)}"
            )

        self.mean = mean

    def finish_up_M_step(self, experiment_dataset, disc_type):
        pass


class HeterogeneousEMState():

    # Mean stuff
    mean = None
    mean_prior = None
    Ft_y = 0
    Ft_CTF = 0 

    # Covariance stuff
    H = 0
    B = 0
    cov_cols = None
    covariance_prior = None
    covariance_options = None
    picked_frequency_indices = None

    # Projected covariance stuff
    projected_cov_lhs = 0
    projected_cov_rhs = 0
    subspace = None

    # PCA stuff
    u = None
    s = None

    # Other stuff
    volume_mask = None
    noise_variance = None
    name = "HeterogeneousEM"


    def __init__(self, mean, mean_variance, noise_variance):
        self.grid_size = utils.guess_grid_size_from_vol_size(mean.size)
        self.mean = mean
        self.mean_variance = mean_variance
        self.noise_variance = noise_variance
        grid_size = utils.guess_grid_size_from_vol_size(mean.size)
        self.volume_mask = mask_fn.raised_cosine_mask(3 * [grid_size], grid_size//2 -3, grid_size//2, -1)

    def E_step(self, experiment_dataset, rotations, translations, disc_type, big_image_batch):
        probabilities = E_with_precompute(experiment_dataset, self.mean, rotations, translations, self.noise_variance, disc_type, big_image_batch, u = self.u, s = self.s)
        return probabilities

    def M_step(self, experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch):

        ## Accumulate Ft_y and Ft_CTF
        Ft_y_this, Ft_CTF_this = M_with_precompute(experiment_dataset, probabilities, rotations, translations, self.noise_variance, disc_type, big_image_batch)
        self.Ft_y += Ft_y_this
        self.Ft_CTF += Ft_CTF_this

        ## Accumulate H, B, and covs
        H_this,B_this = compute_H_B(experiment_dataset, self.mean, probabilities, rotations, translations, self.noise_variance, None, self.picked_frequency_indices, big_image_batch, self.covariance_options['disc_type'])
        H_this = np.array(H_this)
        B_this = np.array(B_this)
        self.H += H_this
        self.B += B_this

        if self.subspace is not None:
            projected_cov_lhs_this, projected_cov_rhs_this = compute_projected_covariance_rhs_lhs(experiment_dataset, self.mean, self.subspace, rotations, translations, probabilities, None, self.noise_variance, disc_type_mean = self.covariance_options['disc_type'], disc_type_u = self.covariance_options['disc_type_u'], image_indices = big_image_batch)
            self.projected_cov_lhs += projected_cov_lhs_this
            self.projected_cov_rhs += projected_cov_rhs_this

    def finish_up_M_step(self, experiment_dataset, disc_type):
        self.mean = relion_functions.post_process_from_filter(experiment_dataset, self.Ft_CTF, self.Ft_y, tau = self.mean_variance, disc_type = disc_type).reshape(-1)

        if self.subspace is not None:
            projected_covar = solve_covariance(self.projected_cov_lhs, self.projected_cov_rhs)
            s, u_small = np.linalg.eigh(projected_covar)
            u_small =  np.fliplr(u_small)
            s = np.flip(s)
            self.u = (self.subspace @ u_small).T
            self.s = np.where(s >0 , s, np.ones_like(s)*jax_config.EPSILON)

        post_process_vmap = jax.vmap(relion_functions.post_process_from_filter_v2, in_axes = (0, 0, None, None, 0, None,None, None, None, None, None))
        
        self.cov_cols = post_process_vmap(self.H, self.B, experiment_dataset.volume_shape, 1, self.covariance_prior, self.covariance_options['left_kernel'], False, self.covariance_options['grid_correct'],  "square",  1, self.volume_mask ).reshape(self.H.shape[0], -1).T

        memory_to_use = utils.get_gpu_memory_total()
        self.subspace, _ , _ = principal_components.randomized_real_svd_of_columns(self.cov_cols, self.picked_frequency_indices, None, experiment_dataset.volume_shape, 50, test_size=self.covariance_options['randomized_sketch_size'], gpu_memory_to_use=memory_to_use)
        self.subspace = self.subspace[:,:self.covariance_options['n_pcs_to_compute']]
