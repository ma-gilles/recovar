#!/usr/bin/env python3
"""
Debug script: Compare old vs new embedding function outputs.

Run on a GPU node to identify where numerical divergence occurs.
This creates a tiny synthetic dataset and compares the old
compute_single_batch_coords_split with the new compute_batch_coords.
"""
import sys
import numpy as np
import jax
import jax.numpy as jnp

# Use the local repo
sys.path.insert(0, '/scratch/gpfs/GILLES/mg6942/heterogeneity_dev-1')

from recovar import core
from recovar.reconstruction import noise
from recovar.heterogeneity import covariance_core, embedding
from recovar.core.configs import ForwardModelConfig, BatchData, ModelState, EmbeddingOpts
import recovar.core.forward as core_forward

np.random.seed(42)

# Tiny synthetic data matching test params: grid_size=32, batch of 10 images
grid_size = 32
n_images = 10
basis_size = 4
image_shape = (grid_size, grid_size)
volume_shape = (grid_size, grid_size, grid_size)
image_size = grid_size ** 2
volume_size = grid_size ** 3

print(f"grid_size={grid_size}, n_images={n_images}, basis_size={basis_size}")
print(f"JAX devices: {jax.devices()}")

# Create synthetic data
batch = np.random.randn(n_images, image_size).astype(np.complex64)
mean_estimate = np.random.randn(volume_size).astype(np.complex64)
volume_mask = np.ones(volume_size, dtype=np.float32)
basis = np.random.randn(basis_size, volume_size).astype(np.complex64)
eigenvalues = np.abs(np.random.randn(basis_size)).astype(np.float32) + 0.01
CTF_params = np.random.randn(n_images, 9).astype(np.float32)
# Set realistic CTF params
CTF_params[:, 0] = 10000  # dfu
CTF_params[:, 1] = 10000  # dfv
CTF_params[:, 2] = 0      # dfang
CTF_params[:, 3] = 300    # voltage
CTF_params[:, 4] = 2.7    # cs
CTF_params[:, 5] = 0.1    # w
CTF_params[:, 6] = 0      # phase_shift
CTF_params[:, 7] = 0      # bfactor
CTF_params[:, 8] = 1      # scale
rotation_matrices = np.eye(3, dtype=np.float32)[None].repeat(n_images, axis=0)
translations = np.zeros((n_images, 2), dtype=np.float32)
image_mask = np.ones(image_size, dtype=np.float32)
noise_variance = np.ones((n_images, image_size), dtype=np.float32)
contrast_grid = np.linspace(0.02, 2, 50).astype(np.float32)
voxel_size = 1.0
padding = 0
disc_type = 'linear_interp'
premultiplied_ctf = False
CTF_fun = core.evaluate_ctf_wrapper

# Identity process function
def process_fn(x):
    return x

print("\n=== Testing P1 functions ===")

# Old P1
old_result = embedding.compute_single_batch_coords_p1(
    batch, mean_estimate, volume_mask, basis, eigenvalues,
    CTF_params, rotation_matrices, translations, image_mask,
    4.0,  # volume_mask_threshold
    image_shape, volume_shape, grid_size, voxel_size, padding,
    disc_type, noise_variance, process_fn, CTF_fun, premultiplied_ctf
)
print(f"Old P1 AU_t_images shape: {old_result[0].shape}, dtype: {old_result[0].dtype}")
print(f"Old P1 AU_t_images[:2,:2]:\n{old_result[0][:2,:2]}")

# New P1
config = ForwardModelConfig(
    image_shape=image_shape,
    volume_shape=volume_shape,
    grid_size=grid_size,
    voxel_size=voxel_size,
    padding=padding,
    disc_type=disc_type,
    CTF_fun=CTF_fun,
    premultiplied_ctf=premultiplied_ctf,
    volume_mask_threshold=4.0,
    process_fn=process_fn,
)
batch_data = BatchData(
    images=batch,
    ctf_params=CTF_params,
    rotation_matrices=rotation_matrices,
    translations=translations,
    noise_variance=noise_variance,
)
model = ModelState(
    mean_estimate=jnp.asarray(mean_estimate),
    volume_mask=jnp.asarray(volume_mask),
    basis=jnp.asarray(basis),
    eigenvalues=jnp.asarray(eigenvalues),
)

new_result = embedding._compute_batch_coords_p1(config, batch_data, model)
print(f"\nNew P1 AU_t_images shape: {new_result[0].shape}, dtype: {new_result[0].dtype}")
print(f"New P1 AU_t_images[:2,:2]:\n{new_result[0][:2,:2]}")

# Compare
names = ['AU_t_images', 'AU_t_Amean', 'AU_t_AU', 'image_norms_sq', 'image_T_A_mean', 'A_mean_norm_sq']
all_match = True
for i, name in enumerate(names):
    old_val = np.array(old_result[i])
    new_val = np.array(new_result[i])
    if old_val.shape != new_val.shape:
        print(f"  {name}: SHAPE MISMATCH old={old_val.shape} new={new_val.shape}")
        all_match = False
        continue
    max_diff = np.max(np.abs(old_val - new_val))
    rel_diff = max_diff / (np.max(np.abs(old_val)) + 1e-10)
    match = "OK" if rel_diff < 1e-5 else "MISMATCH"
    if match == "MISMATCH":
        all_match = False
    print(f"  {name}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e} [{match}]")

print(f"\nP1 functions match: {all_match}")

print("\n=== Testing full coord computation ===")

# Old full
old_xs, old_contrast, old_cov, old_bias = embedding.compute_single_batch_coords_split(
    batch, mean_estimate, volume_mask, basis, eigenvalues,
    CTF_params, rotation_matrices, translations, image_mask,
    4.0, image_shape, volume_shape, grid_size, voxel_size, padding,
    disc_type, True, noise_variance, process_fn, CTF_fun,
    contrast_grid, 1.0, np.inf, False,
    shared_label=False, contrast_shared_across_tilt_series=True,
    premultiplied_ctf=premultiplied_ctf
)
print(f"Old xs shape: {old_xs.shape}, contrast: {old_contrast[:3]}")

# New full
opts = EmbeddingOpts(
    compute_covariances=True,
    compute_bias=False,
    shared_label=False,
    contrast_shared_across_tilt_series=True,
)
new_xs, new_contrast, new_cov, new_bias = embedding.compute_batch_coords(
    config, batch_data, model, opts,
    jnp.asarray(image_mask), jnp.asarray(contrast_grid),
    1.0, np.inf,
)
print(f"New xs shape: {new_xs.shape}, contrast: {new_contrast[:3]}")

# Compare
xs_diff = np.max(np.abs(np.array(old_xs) - np.array(new_xs)))
contrast_diff = np.max(np.abs(np.array(old_contrast) - np.array(new_contrast)))
cov_diff = np.max(np.abs(np.array(old_cov) - np.array(new_cov))) if old_cov is not None and new_cov is not None else 0

print(f"\nFull function comparison:")
print(f"  xs max_diff: {xs_diff:.2e}")
print(f"  contrast max_diff: {contrast_diff:.2e}")
print(f"  cov max_diff: {cov_diff:.2e}")

if xs_diff > 1e-4 or contrast_diff > 1e-4:
    print("\n*** FUNCTIONS DIFFER! ***")
    print(f"  Old xs[:3]: {np.array(old_xs[:3])}")
    print(f"  New xs[:3]: {np.array(new_xs[:3])}")
    print(f"  Old contrast[:5]: {np.array(old_contrast[:5])}")
    print(f"  New contrast[:5]: {np.array(new_contrast[:5])}")
else:
    print("\nFunctions match within tolerance.")

print("\n=== Testing variance estimation ===")

# Old variance
from recovar.heterogeneity.covariance_estimation import (
    variance_relion_kernel_trilinear,
    variance_relion_style_triangular_kernel_batch_trilinear,
)

old_var = variance_relion_style_triangular_kernel_batch_trilinear(
    mean_estimate, batch, CTF_params, rotation_matrices, translations,
    image_shape, volume_shape, voxel_size, CTF_fun, noise_variance,
    volume_mask, image_mask, 4.0, grid_size, padding,
    soften=5, disc_type=disc_type, premultiplied_ctf=premultiplied_ctf,
)

var_batch_data = BatchData(
    images=batch,
    ctf_params=CTF_params,
    rotation_matrices=rotation_matrices,
    translations=translations,
    noise_variance=noise_variance,
)
new_var = variance_relion_kernel_trilinear(
    config, var_batch_data, mean_estimate, volume_mask, image_mask, soften=5,
)

var_names = ['Ft_y', 'Ft_ctf', 'Ft_im', 'Ft_one']
for i, name in enumerate(var_names):
    old_v = np.array(old_var[i])
    new_v = np.array(new_var[i])
    max_diff = np.max(np.abs(old_v - new_v))
    rel_diff = max_diff / (np.max(np.abs(old_v)) + 1e-10)
    match = "OK" if rel_diff < 1e-5 else "MISMATCH"
    print(f"  {name}: max_diff={max_diff:.2e}, rel_diff={rel_diff:.2e} [{match}]")

print("\nDone.")
