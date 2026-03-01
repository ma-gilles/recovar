"""Reconstruction quality metrics: FSC, subspace angles, per-voxel error."""

import logging
import os.path

import jax.numpy as jnp
import numpy as np

from recovar import utils
from recovar.core import linalg, mask
from recovar.heterogeneity import locres

logger = logging.getLogger(__name__)

def captured_variance(test_v, U, s):
    """Compute cumulative captured variance of test vectors in a subspace.

    Args:
        test_v: Test vectors, shape (n_voxels, n_test).
        U: Eigenvector matrix, shape (n_voxels, n_pcs).
        s: Eigenvalue array, shape (n_pcs,).

    Returns:
        Cumulative captured variance array, shape (n_test,).
    """
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

def find_angle_between_subspaces(v1,v2, max_rank):
    ss = np.conj(v1[:,:max_rank]).T @ v2[:,:max_rank]
    s,v,d = np.linalg.svd(ss)
    if np.any(v > 1.2):
        logger.warning('v too big!')
    v = np.where(v < 1, v, 1)
    return np.sqrt( 1 - v[-1]**2)


def subspace_angles(u ,v, max_rank = None, check_orthogonalize = False):
    """Compute principal angles between two subspaces of increasing rank.

    Args:
        u: First set of basis vectors, shape (n_voxels, n_pcs).
        v: Second set of basis vectors, shape (n_voxels, n_pcs).
        max_rank: Maximum subspace rank to evaluate.
        check_orthogonalize: If True, QR-orthogonalize u and v first.

    Returns:
        Array of sine of principal angles, shape (max_rank,).
    """
    max_rank = u.shape[-1] if max_rank is None else max_rank
    sine_angles = np.zeros(max_rank)
    if check_orthogonalize:
        u,_ = np.linalg.qr(u)
        v,_ = np.linalg.qr(v)

    for k in range(1,max_rank+1):
        if k > u.shape[-1]:
            sine_angles[k-1] = 1
        else:
            sine_angles[k-1] = find_angle_between_subspaces(u[:,:k], v[:,:k], max_rank = k )
    return sine_angles  

def local_fsc_metric(map1, map2, voxel_size, mask, fsc_threshold=1/7, locres_sampling = 25 ):
    """Compute local resolution and local AUC metrics within a mask.

    Args:
        map1: First half-map (3-D real-space array).
        map2: Second half-map (3-D real-space array).
        voxel_size: Voxel size in Angstroms.
        mask: Boolean mask selecting voxels to evaluate.
        fsc_threshold: FSC threshold for resolution (default 1/7).
        locres_sampling: Sampling factor for local resolution windows.

    Returns:
        Tuple of (median_locres, ninety_pc_locres, median_auc, ten_pc_auc).
    """
    fscs, local_resols, i_loc_res, i_loc_auc = locres.local_resolution(map1, map2, 0, voxel_size, locres_sampling = locres_sampling, locres_maskrad= None, locres_edgwidth= None, locres_minres =50, use_filter = False, use_v2 = True, fsc_threshold = fsc_threshold)
    
    good_resols = i_loc_res[mask]
    good_aucs = i_loc_auc[mask]
    
    i_loc_res = np.array(i_loc_res)
    i_loc_res[~mask] = None

    median_locres = np.median(good_resols)
    ninety_pc_locres = np.percentile(good_resols, 90)

    median_auc = np.median(good_aucs)
    ninety_pc_auc = np.percentile(good_aucs, 10)
    return median_locres, ninety_pc_locres, median_auc, ninety_pc_auc


def local_error_metric(map1, map2, voxel_size, mask, normalize_by_map1 = True, locres_sampling = 15 ):
    
    errors_gt = locres.local_error(map1, map2, voxel_size, locres_sampling = locres_sampling)

    good_errors = errors_gt[mask]

    median_error = np.median(good_errors) 
    ninety_pc_error = np.percentile(good_errors, 90)

    if normalize_by_map1:
        errors_scale = locres.local_error(map1, map2 * 0, voxel_size, locres_sampling = locres_sampling)
        med_local_norm = np.median(errors_scale[mask])
        median_error = median_error/ med_local_norm
        ninety_pc_error = ninety_pc_error/ med_local_norm

    return median_error, ninety_pc_error


def masked_l2_difference(gt_map, target_map, voxel_size, mask= None ):
    
    if mask is None:
        mask = gt_mask_fn(gt_map)

    l2_error = np.mean(np.abs((gt_map-target_map)[mask])**2)
    bias = np.mean((gt_map-target_map)[mask])
    return l2_error, bias

def gt_mask_fn(gt_map):
    from recovar.core import mask as mask_fn
    mask = mask_fn.make_mask_from_gt(gt_map, smax = 3, iter = 1 , from_ft = False)
    mask = (mask > 0.5).astype(bool)

    return mask


def compute_volume_error_metrics_from_gt(gt_map, estimate_map, voxel_size, mask , partial_mask = None , normalize_by_map1 = True ):
    
    if mask is None:
        mask = gt_mask_fn(gt_map)
    
    errors_metrics = {}    
    errors_metrics['median_locres'], errors_metrics['ninety_pc_locres'], errors_metrics['median_auc'], errors_metrics['ten_pc_auc'] =  local_fsc_metric(estimate_map, gt_map, voxel_size, mask, fsc_threshold=1/2 )
    errors_metrics['median_error'], errors_metrics['ninety_pc_error'] =  local_error_metric(gt_map , estimate_map, voxel_size, mask, normalize_by_map1 = normalize_by_map1)
    errors_metrics['mask'] = mask

    if partial_mask is not None:
        errors_metrics['partial_median_locres'], errors_metrics['partial_ninety_pc_locres'], errors_metrics['partial_median_auc'], errors_metrics['partial_ten_pc_auc'] =  local_fsc_metric(estimate_map, gt_map, voxel_size, partial_mask, fsc_threshold=1/2 )
        errors_metrics['partial_median_error'], errors_metrics['partial_ninety_pc_error'] =  local_error_metric(gt_map, estimate_map, voxel_size, partial_mask,normalize_by_map1 = normalize_by_map1)
        errors_metrics['partial_mask'] = partial_mask

    return errors_metrics


def compute_volume_error_metrics_from_halfmaps(estimate1, estimate2, voxel_size, mask , partial_mask = None , normalize_by_map1 = False ):
    
    if mask is None:
        from recovar.core import mask as mask_fn
        mask = mask_fn.make_mask_from_half_maps(estimate1, estimate2)
        mask = (mask > 1e-3).astype(bool)
    
    errors_metrics = {}    
    errors_metrics['median_locres'], errors_metrics['ninety_pc_locres'], errors_metrics['median_auc'], errors_metrics['ten_pc_auc'] =  local_fsc_metric(estimate1, estimate2, voxel_size, mask, fsc_threshold=1/7 )
    errors_metrics['median_error'], errors_metrics['ninety_pc_error'] =  local_error_metric(estimate1, estimate2, voxel_size, mask, normalize_by_map1 = normalize_by_map1 )
    errors_metrics['mask'] = mask

    if partial_mask is not None:
        errors_metrics['partial_median_locres'], errors_metrics['partial_ninety_pc_locres'], errors_metrics['partial_median_auc'], errors_metrics['partial_ten_pc_auc'] =  local_fsc_metric(estimate1, estimate2, voxel_size, partial_mask, fsc_threshold=1/7 )
        errors_metrics['partial_median_error'], errors_metrics['partial_ninety_pc_error'] =  local_error_metric(estimate1, estimate2, voxel_size, partial_mask, normalize_by_map1 =normalize_by_map1)
        errors_metrics['partial_mask'] = partial_mask

    return errors_metrics


def variance_of_zs(z, gt_image_assignment):
    """
    Estimate per-label variances and overall variance of z.

    Parameters
    ----------
    z : np.ndarray
        Array of shape (n_samples, features).
    gt_image_assignment : np.ndarray
        Array of shape (n_samples,) with integer labels for each sample.

    Returns
    -------
    label_variances : np.ndarray
        Variance computed for each label (flattened across features).
    weighted_avg_variance : float
        Overall variance computed as the weighted average of per-label variances.
    overall_variance : float
        Variance computed over the entire z data.
    """
    labels = np.unique(gt_image_assignment)
    label_variances = np.zeros(len(labels))
    total_sq_diff = 0
    total_samples = z.shape[0]
    
    for idx, lab in enumerate(labels):
        sub_zs = z[gt_image_assignment == lab]
        if sub_zs.size == 0:
            continue
        
        # Compute variance of the sub-array (over all elements)
        label_variances[idx] = np.var(sub_zs)
        
        # Sum of squared differences for weighted average variance
        mean_lab = np.mean(sub_zs, axis=0)
        total_sq_diff += np.sum((sub_zs - mean_lab) ** 2)
    
    weighted_avg_variance = total_sq_diff / total_samples
    overall_variance = np.var(z)
    
    return label_variances, weighted_avg_variance / overall_variance


def fro_norm_diff_low_rank(U, s, V, d):
    """Compute the Frobenius norm of (A - B) where
    A = U * diag(s) * U^T and B = V * diag(d) * V^T,
    using only their low-rank representations.

    Parameters
    ----------
    U : jnp.ndarray
        An n x r matrix (orthonormal columns).
    s : jnp.ndarray
        A length-r vector for A's eigenvalues.
    V : jnp.ndarray
        An n x r matrix (orthonormal columns).
    d : jnp.ndarray
        A length-r vector for B's eigenvalues.

    Returns
    -------
    jnp.ndarray
        The Frobenius norm of A - B.
    """
    # Compute X = U^T V (r x r)
    X = jnp.matmul(U.T, V)

    # ||A||_F^2 = sum of s_i^2; ||B||_F^2 = sum of d_j^2
    norm_A_sq = jnp.sum(s**2)
    norm_B_sq = jnp.sum(d**2)

    # trace(A B) = sum_{i,j} s[i]*d[j]*X[i,j]^2
    trace_AB = jnp.sum((s[:, None] * d[None, :]) * (X**2))

    # ||A - B||_F^2
    diff_norm_sq = norm_A_sq + norm_B_sq - 2.0 * trace_AB
    return jnp.sqrt(diff_norm_sq)