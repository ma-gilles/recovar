"""Synthetic heterogeneous dataset generation for testing."""

import logging

import jax
import jax.numpy as jnp
import numpy as np
from recovar import core, utils, jax_config

logger = logging.getLogger(__name__)
from recovar.simulation import simulator
from recovar.core import linalg, mask


def load_heterogeneous_reconstruction(simulation_info_file, volumes_path_root=None, load_volumes=True):
    if isinstance(simulation_info_file, dict):
        simulation_info = simulation_info_file
    else:
        simulation_info = utils.pickle_load(simulation_info_file)

    volumes_path_root = simulation_info["volumes_path_root"] if volumes_path_root is None else volumes_path_root

    if load_volumes:
        if "scale_vol" in simulation_info:
            volumes = simulator.load_volumes_from_folder(
                volumes_path_root,
                simulation_info["grid_size"],
                simulation_info["trailing_zero_format_in_vol_name"],
                normalize=False,
            )
            volumes = volumes * simulation_info["scale_vol"]
        else:
            volumes = simulator.load_volumes_from_folder(
                volumes_path_root,
                simulation_info["grid_size"],
                simulation_info["trailing_zero_format_in_vol_name"],
                normalize=True,
            )

    else:
        volumes = None

    return HeterogeneousVolumeDistribution(
        volumes, simulation_info["image_assignment"], simulation_info["per_image_contrast"]
    )


class HeterogeneousVolumeDistribution:
    def __init__(self, volumes, image_assignments, contrasts, valid_indices=None, vol_batch_size=None):

        self.volume_shape = utils.guess_vol_shape_from_vol_size(volumes.shape[-1])
        self.vol_batch_size = (
            utils.get_vol_batch_size(self.volume_shape[0], utils.get_gpu_memory_total())
            if vol_batch_size is None
            else vol_batch_size
        )
        self.volumes = volumes
        valid_indices = mask.get_radial_mask(self.volume_shape, radius=None) if valid_indices is None else valid_indices
        self.valid_indices = np.array(valid_indices.reshape(-1))
        if self.volumes is not None:
            self.volumes *= self.valid_indices[None, :]
        self.image_assignments = image_assignments
        self.contrasts = contrasts

        # Get Image assignment
        self.probs_of_state = None
        self.percent_outliers = None
        self.compute_probs_of_state()
        self.u = None
        self.s = None

        self.mean = None
        self.covariance_cols = None

    def get_u(self):
        if self.u is None:
            self.compute_u_s()
        return self.u

    def get_s(self):
        if self.s is None:
            self.compute_u_s()
        return self.s

    def compute_u_s(self, contrasted=False):
        self.u, self.s = self.get_covariance_eigendecomposition(contrasted=contrasted)

    def get_probs_of_state(self):
        self.compute_probs_of_state()
        return self.probs_of_state

    def get_mean(self):
        if self.mean is None:
            self.compute_mean_conformation()
        return self.mean

    def compute_probs_of_state(self):
        n_gt_molecules = self.volumes.shape[0]

        probs_of_molecule = np.zeros(n_gt_molecules, dtype=np.float32).real
        for k in range(n_gt_molecules):
            probs_of_molecule[k] = np.sum(self.image_assignments == k)

        probs_of_molecule /= np.sum(probs_of_molecule)
        self.probs_of_state = probs_of_molecule
        self.percent_outliers = np.sum(self.image_assignments == -1) / self.image_assignments.size

    def compute_mean_conformation(self):
        self.mean = np.array(np.sum(self.volumes * self.get_probs_of_state()[:, None], axis=0))

    def get_covariance_columns(self, picked_frequencies, contrasted=False):
        vols = self.get_covariance_square_root(contrasted)
        return vols @ np.conj(vols[picked_frequencies, :]).T

    def get_covariance_square_root(self, contrasted):
        contrast_variance = np.var(self.contrasts[self.image_assignments != -1]) if contrasted else 0

        vols = (
            np.sqrt(1 + contrast_variance)
            * (self.volumes - self.get_mean()[None, :])
            * np.sqrt(self.get_probs_of_state()[:, None])
        )

        if contrast_variance > 0:
            vols = np.concatenate([np.sqrt(contrast_variance) * self.get_mean()[None], vols])
        return vols.T

    def get_vol_svd(self, contrasted=False, real_space=False, random_svd_pcs=None):

        vols = self.get_covariance_square_root(contrasted)

        if real_space:
            vols = linalg.batch_idft3(vols, self.volume_shape, self.vol_batch_size).real

        if random_svd_pcs is None:
            u, s, v = np.linalg.svd(vols, full_matrices=False)
        else:
            u, s, v = linalg.randomized_svd(vols, random_svd_pcs)

        return u, s, v

    def get_covariance_eigendecomposition(self, contrasted=False):
        u, s, _ = self.get_vol_svd(contrasted)
        zero_freq = core.frequencies_to_vec_indices(jnp.array([[0, 0, 0]]), self.volume_shape)
        ip = np.where(np.abs(u[zero_freq, :]) > jax_config.ROOT_EPSILON, u[zero_freq, :] / np.abs(u[zero_freq, :]), 1)
        u /= ip
        return u, s**2

    def get_fourier_variances(self, contrasted=False):
        vols = self.get_covariance_square_root(contrasted)
        return np.linalg.norm(vols, axis=-1) ** 2

    def get_spatial_variances(self, contrasted=False):
        vols = self.get_covariance_square_root(contrasted)
        vols = linalg.batch_idft3(vols, self.volume_shape, self.vol_batch_size)
        return np.linalg.norm(vols, axis=-1) ** 2


def get_col_covariance_for_one_X_one_index(X, Xmean, vec_index):
    return (X - Xmean) * jnp.conj(X[vec_index] - Xmean[vec_index])


get_col_covariance_for_one_X_many_index = jax.vmap(get_col_covariance_for_one_X_one_index, in_axes=(None, None, 0))
get_col_covariance_for_many_X_one_index = jax.vmap(get_col_covariance_for_one_X_one_index, in_axes=(0, None, None))


def get_col_covariance(Xs, X_mean, vec_indices, prob_of_X):
    cov = np.zeros_like(Xs, shape=[X_mean.size, vec_indices.size])
    Xs_j = jnp.array(Xs)
    for v_idx, v in enumerate(vec_indices):
        cov[:, v_idx] = jnp.sum(prob_of_X[..., None] * get_col_covariance_for_many_X_one_index(Xs_j, X_mean, v), axis=0)

    return cov
