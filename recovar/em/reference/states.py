"""Reference state containers for homogeneous and heterogeneous EM."""

import jax
import numpy as np

from recovar import jax_config, utils
from recovar.core import mask as mask_fn
from recovar.heterogeneity import principal_components
from recovar.reconstruction import relion_functions

from .e_step import compute_pose_probabilities
from .heterogeneity import compute_H_B, compute_projected_covariance_rhs_lhs, solve_covariance
from .m_step import accumulate_mean_statistics


class EMState:
    name = "EM"

    def __init__(self, mean, mean_variance, noise_variance):
        self.mean = mean
        self.mean_variance = mean_variance
        self.noise_variance = noise_variance
        self.Ft_y = 0
        self.Ft_CTF = 0

    def E_step(self, experiment_dataset, rotations, translations, disc_type, big_image_batch):
        probabilities = compute_pose_probabilities(
            experiment_dataset, self.mean, rotations, translations, self.noise_variance, disc_type, big_image_batch
        )
        return probabilities

    def M_step(self, experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch):
        Ft_y_this, Ft_CTF_this = accumulate_mean_statistics(
            experiment_dataset, probabilities, rotations, translations, self.noise_variance, disc_type, big_image_batch
        )
        self.Ft_y += Ft_y_this
        self.Ft_CTF += Ft_CTF_this

    def finish_up_M_step(self, experiment_dataset, disc_type):
        self.mean = relion_functions.post_process_from_filter(
            experiment_dataset, self.Ft_CTF, self.Ft_y, tau=self.mean_variance, disc_type=disc_type
        ).reshape(-1)


class HeterogeneousEMState:
    name = "HeterogeneousEM"

    def __init__(self, mean, mean_variance, noise_variance):
        self.grid_size = utils.guess_grid_size_from_vol_size(mean.size)
        self.mean = mean
        self.mean_variance = mean_variance
        self.noise_variance = noise_variance
        self.Ft_y = 0
        self.Ft_CTF = 0
        self.H = 0
        self.B = 0
        self.projected_cov_lhs = 0
        self.projected_cov_rhs = 0
        self.cov_cols = None
        self.covariance_prior = None
        self.covariance_options = None
        self.picked_frequency_indices = None
        self.subspace = None
        self.u = None
        self.s = None
        self.volume_mask = mask_fn.raised_cosine_mask(
            3 * [self.grid_size], self.grid_size // 2 - 3, self.grid_size // 2, -1
        )

    def E_step(self, experiment_dataset, rotations, translations, disc_type, big_image_batch):
        probabilities = compute_pose_probabilities(
            experiment_dataset,
            self.mean,
            rotations,
            translations,
            self.noise_variance,
            disc_type,
            big_image_batch,
            u=self.u,
            s=self.s,
        )
        return probabilities

    def M_step(self, experiment_dataset, probabilities, rotations, translations, disc_type, big_image_batch):

        ## Accumulate Ft_y and Ft_CTF
        Ft_y_this, Ft_CTF_this = accumulate_mean_statistics(
            experiment_dataset, probabilities, rotations, translations, self.noise_variance, disc_type, big_image_batch
        )
        self.Ft_y += Ft_y_this
        self.Ft_CTF += Ft_CTF_this

        ## Accumulate H, B, and covs
        H_this, B_this = compute_H_B(
            experiment_dataset,
            self.mean,
            probabilities,
            rotations,
            translations,
            self.noise_variance,
            self.picked_frequency_indices,
            big_image_batch,
            self.covariance_options["disc_type"],
        )
        H_this = np.array(H_this)
        B_this = np.array(B_this)
        self.H += H_this
        self.B += B_this

        if self.subspace is not None:
            projected_cov_lhs_this, projected_cov_rhs_this = compute_projected_covariance_rhs_lhs(
                experiment_dataset,
                self.mean,
                self.subspace,
                rotations,
                translations,
                probabilities,
                self.noise_variance,
                disc_type_mean=self.covariance_options["disc_type"],
                disc_type_u=self.covariance_options["disc_type_u"],
                image_indices=big_image_batch,
            )
            self.projected_cov_lhs += projected_cov_lhs_this
            self.projected_cov_rhs += projected_cov_rhs_this

    def finish_up_M_step(self, experiment_dataset, disc_type):
        self.mean = relion_functions.post_process_from_filter(
            experiment_dataset, self.Ft_CTF, self.Ft_y, tau=self.mean_variance, disc_type=disc_type
        ).reshape(-1)

        if self.subspace is not None:
            projected_covar = solve_covariance(self.projected_cov_lhs, self.projected_cov_rhs)
            s, u_small = np.linalg.eigh(projected_covar)
            u_small = np.fliplr(u_small)
            s = np.flip(s)
            self.u = (self.subspace @ u_small).T
            self.s = np.where(s > 0, s, np.ones_like(s) * jax_config.EPSILON)

        post_process_vmap = jax.vmap(
            relion_functions.post_process_from_filter_v2,
            in_axes=(0, 0, None, None, 0, None, None, None, None, None, None),
        )

        self.cov_cols = (
            post_process_vmap(
                self.H,
                self.B,
                experiment_dataset.volume_shape,
                1,
                self.covariance_prior,
                self.covariance_options["left_kernel"],
                False,
                self.covariance_options["grid_correct"],
                "square",
                1,
                self.volume_mask,
            )
            .reshape(self.H.shape[0], -1)
            .T
        )

        memory_to_use = utils.get_gpu_memory_total()
        self.subspace, _, _ = principal_components.randomized_real_svd_of_columns(
            self.cov_cols,
            self.picked_frequency_indices,
            None,
            experiment_dataset.volume_shape,
            50,
            test_size=self.covariance_options["randomized_sketch_size"],
            gpu_memory_to_use=memory_to_use,
        )
        self.subspace = self.subspace[:, : self.covariance_options["n_pcs_to_compute"]]
