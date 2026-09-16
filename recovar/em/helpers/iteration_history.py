"""
Per-iteration trajectory bookkeeping for ``refine_single_volume``.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def add_significant_count_artifacts(save_dict, significant_counts, half_indices, n_images):
    """Save significant-count history in both half and original image order."""
    half_order_indices = np.concatenate(
        [np.asarray(indices, dtype=np.int64) for indices in half_indices],
    )
    for iteration, counts in enumerate(significant_counts):
        if counts is None:
            continue
        counts_half_order = np.asarray(counts)
        # Keep the legacy key value/shape/dtype-compatible: it has always
        # stored the concatenated half-1, half-2 refinement-loop order.
        save_dict[f"sig_counts_iter_{iteration:03d}"] = counts_half_order
        save_dict[f"sig_counts_half_order_iter_{iteration:03d}"] = counts_half_order
        flat_counts = counts_half_order.reshape(-1)
        if flat_counts.shape[0] != half_order_indices.shape[0]:
            continue
        counts_by_image = np.full(int(n_images), -1, dtype=flat_counts.dtype)
        counts_by_image[half_order_indices] = flat_counts
        save_dict[f"sig_counts_by_image_iter_{iteration:03d}"] = counts_by_image



@dataclass(frozen=True)
class RefinementHistory:
    """Accumulates per-iteration trajectories for one RELION refinement run."""

    current_sizes: list = field(default_factory=list)
    state_swap_probe_applied_relion_iterations: list = field(default_factory=list)
    fsc_history: list = field(default_factory=list)
    fsc_for_growth_history: list = field(default_factory=list)
    pixel_resolutions: list = field(default_factory=list)
    wall_times: list = field(default_factory=list)
    significant_counts: list = field(default_factory=list)
    data_vs_prior_trajectory: list = field(default_factory=list)
    healpix_order_trajectory: list = field(default_factory=list)
    ave_Pmax_trajectory: list = field(default_factory=list)
    ave_Pmax_denominator_trajectory: list = field(default_factory=list)
    pmax_per_image_history: list = field(default_factory=list)
    noise_radial_trajectory: list = field(default_factory=list)
    noise_radial_per_half_trajectory: list = field(default_factory=list)
    tau2_radial_trajectory: list = field(default_factory=list)
    tau2_sigma2_trajectory: list = field(default_factory=list)
    tau2_avg_weight_trajectory: list = field(default_factory=list)
    tau2_shell_sum_trajectory: list = field(default_factory=list)
    tau2_shell_count_trajectory: list = field(default_factory=list)
    tau2_fsc_used_trajectory: list = field(default_factory=list)
    tau2_ssnr_trajectory: list = field(default_factory=list)
    sigma_offset_used_trajectory: list = field(default_factory=list)
    sigma_offset_used_per_half_trajectory: list = field(default_factory=list)
    sigma_offset_trajectory: list = field(default_factory=list)
    sigma_offset_per_half_trajectory: list = field(default_factory=list)
    per_class_sigma_offset_trajectory: list = field(default_factory=list)
    direction_prior_trajectory_per_half: list = field(default_factory=list)
    rotation_posterior_trajectory_per_half: list = field(default_factory=list)
    frac_changed_trajectory: list = field(default_factory=list)
    acc_rot_trajectory: list = field(default_factory=list)
    acc_trans_trajectory: list = field(default_factory=list)
    acc_rot_per_class_trajectory: list = field(default_factory=list)
    acc_trans_per_class_trajectory: list = field(default_factory=list)
    expected_accuracy_class_counts_trajectory: list = field(default_factory=list)
    expected_accuracy_status_trajectory: list = field(default_factory=list)
    smallest_change_angles_trajectory: list = field(default_factory=list)
    smallest_change_offsets_trajectory: list = field(default_factory=list)
    best_rotation_eulers_history: list = field(default_factory=list)
    best_translations_history: list = field(default_factory=list)
    class_weight_trajectory: list = field(default_factory=list)
    class_mstep_weight_trajectory: list = field(default_factory=list)
    class_full_posterior_weight_trajectory: list = field(default_factory=list)
    class_assignment_history: list = field(default_factory=list)
    local_profile_history: list = field(default_factory=list)
    global_profile_history: list = field(default_factory=list)
    relion_follower_owners_half1_trajectory: list = field(default_factory=list)
    relion_scale_follower_scales_numbered_pre_score_trajectory: list = field(default_factory=list)
    relion_scale_follower_scales_numbered_post_mstep_trajectory: list = field(default_factory=list)
    relion_follower_scale_replay_applied_iterations: list = field(default_factory=list)

    # -- scheduling / sampling grid ------------------------------------

    def record_scheduling(self, cs, healpix_order, sigma_offset_used, sigma_offset_used_per_half) -> None:
        self.current_sizes.append(cs)
        self.healpix_order_trajectory.append(healpix_order)
        self.sigma_offset_used_trajectory.append(sigma_offset_used)
        self.sigma_offset_used_per_half_trajectory.append(sigma_offset_used_per_half)

    # -- RELION follower-scale replay/dispatch bookkeeping -------------

    def record_follower_scale_pre_score(self, pre_score_scales, owners_half1) -> None:
        self.relion_scale_follower_scales_numbered_pre_score_trajectory.append(pre_score_scales)
        self.relion_follower_owners_half1_trajectory.append(owners_half1)

    # -- per-iteration E-step / M-step outputs --------------------------

    def record_direction_prior(
        self, class_direction_prior_per_half, global_direction_prior_per_half, *, k_class_enabled: bool
    ) -> None:
        """Record float64 copies of each half's learned direction prior.

        K-class runs record class 0 of each half's ``(n_classes, n_pixels)``
        prior; K=1 records the half's global prior. Missing priors stay ``None``.
        """
        source = class_direction_prior_per_half if k_class_enabled else global_direction_prior_per_half
        self.direction_prior_trajectory_per_half.append(
            [
                None
                if prior_k is None
                else np.asarray(prior_k[0] if k_class_enabled else prior_k, dtype=np.float64).copy()
                for prior_k in source
            ]
        )

    def record_rotation_posterior(self, rotation_posterior_per_half) -> None:
        """Record float64 copies of the pre-collapse orientation posterior.

        Kept separate from ``record_direction_prior`` so a direction-prior
        mismatch can be localized to posterior aggregation versus collapse.
        """
        self.rotation_posterior_trajectory_per_half.append(
            [None if value is None else np.asarray(value, dtype=np.float64).copy() for value in rotation_posterior_per_half]
        )

    def record_fsc(self, fsc, fsc_for_growth) -> None:
        self.fsc_history.append(fsc)
        self.fsc_for_growth_history.append(fsc_for_growth)

    def record_pmax(self, ave_pmax, ave_pmax_denominator, per_image_pmax) -> None:
        self.ave_Pmax_trajectory.append(ave_pmax)
        self.ave_Pmax_denominator_trajectory.append(ave_pmax_denominator)
        self.pmax_per_image_history.append(per_image_pmax)

    def record_class_weights(self, mstep_weights, posterior_weights) -> None:
        """Snapshot both weight definitions, preserving the two published M-step histories."""
        self.class_weight_trajectory.append(mstep_weights.copy())
        self.class_mstep_weight_trajectory.append(mstep_weights.copy())
        self.class_full_posterior_weight_trajectory.append(posterior_weights.copy())

    def record_pose_history(self, euler_snapshot_per_half, translation_snapshot_per_half) -> None:
        self.best_rotation_eulers_history.append(euler_snapshot_per_half)
        self.best_translations_history.append(translation_snapshot_per_half)

    def record_noise_and_tau2(self, noise_radial, noise_radial_per_half, tau2_details, *, k_class_enabled: bool) -> None:
        """Format shell diagnostics before appending a complete iteration.

        Float64 input shell arrays remain shared; stacking the halves creates
        a fresh array. Prepare every field before mutating history so malformed
        input cannot leave the trajectory lists at different lengths.
        """
        noise_radial = np.asarray(noise_radial, dtype=np.float64)
        noise_radial_per_half = np.stack(
            [np.asarray(noise_k, dtype=np.float64) for noise_k in noise_radial_per_half],
            axis=0,
        )
        tau2_details = (
            None
            if tau2_details is None
            else {
                "prior_shells": np.asarray(tau2_details["prior_shells"], dtype=np.float64),
                "sigma2_shells": np.asarray(tau2_details["sigma2_shells"], dtype=np.float64),
                "avg_weight_shells": np.asarray(tau2_details["avg_weight_shells"], dtype=np.float64),
                "shell_sum": np.asarray(tau2_details["shell_sum"], dtype=np.float64),
                "shell_count": np.asarray(tau2_details["shell_count"], dtype=np.float64),
                "fsc_shells": None if k_class_enabled else np.asarray(tau2_details["fsc_shells"], dtype=np.float64),
                "ssnr_shells": np.asarray(tau2_details["ssnr_shells"], dtype=np.float64),
            }
        )
        self.noise_radial_trajectory.append(noise_radial)
        self.noise_radial_per_half_trajectory.append(noise_radial_per_half)
        if tau2_details is None:
            self.tau2_radial_trajectory.append(None)
            self.tau2_sigma2_trajectory.append(None)
            self.tau2_avg_weight_trajectory.append(None)
            self.tau2_shell_sum_trajectory.append(None)
            self.tau2_shell_count_trajectory.append(None)
            self.tau2_fsc_used_trajectory.append(None)
            self.tau2_ssnr_trajectory.append(None)
            return
        self.tau2_radial_trajectory.append(tau2_details["prior_shells"])
        self.tau2_sigma2_trajectory.append(tau2_details["sigma2_shells"])
        self.tau2_avg_weight_trajectory.append(tau2_details["avg_weight_shells"])
        self.tau2_shell_sum_trajectory.append(tau2_details["shell_sum"])
        self.tau2_shell_count_trajectory.append(tau2_details["shell_count"])
        self.tau2_ssnr_trajectory.append(tau2_details["ssnr_shells"])
        self.tau2_fsc_used_trajectory.append(None if k_class_enabled else tau2_details["fsc_shells"])

    def record_sigma_offset_update(self, sigma_offset, sigma_offset_per_half, per_class_sigma_offset) -> None:
        self.sigma_offset_trajectory.append(sigma_offset)
        self.sigma_offset_per_half_trajectory.append(sigma_offset_per_half)
        self.per_class_sigma_offset_trajectory.append(per_class_sigma_offset)

    def record_pose_accuracy_diagnostics(
        self,
        acc_rot,
        acc_trans,
        acc_rot_per_class,
        acc_trans_per_class,
        expected_accuracy_class_counts,
        expected_accuracy_status,
        smallest_change_angles,
        smallest_change_offsets,
    ) -> None:
        self.acc_rot_trajectory.append(acc_rot)
        self.acc_trans_trajectory.append(acc_trans)
        self.acc_rot_per_class_trajectory.append(acc_rot_per_class)
        self.acc_trans_per_class_trajectory.append(acc_trans_per_class)
        self.expected_accuracy_class_counts_trajectory.append(expected_accuracy_class_counts)
        self.expected_accuracy_status_trajectory.append(expected_accuracy_status)
        self.smallest_change_angles_trajectory.append(smallest_change_angles)
        self.smallest_change_offsets_trajectory.append(smallest_change_offsets)

    def to_dict(self) -> dict:
        """Return the trajectory entries of the function's result dict.

        Reproduces the exact key strings (including two pre-existing
        aliased-duplicate keys) that all three ``return {...}`` sites in
        ``refine_single_volume`` have always used, so callers can
        merge this in with ``**history.to_dict()`` unchanged.
        """
        return {
            "fsc": self.fsc_history[-1] if self.fsc_history else None,
            "current_sizes": self.current_sizes,
            "fsc_history": self.fsc_history,
            "pixel_resolutions": self.pixel_resolutions,
            "wall_times": self.wall_times,
            "significant_counts": self.significant_counts,
            "data_vs_prior_trajectory": self.data_vs_prior_trajectory,
            "healpix_order_trajectory": self.healpix_order_trajectory,
            "ave_Pmax_trajectory": self.ave_Pmax_trajectory,
            "ave_Pmax_denominator_trajectory": self.ave_Pmax_denominator_trajectory,
            "pmax_per_image_history": self.pmax_per_image_history,
            "noise_radial_trajectory": self.noise_radial_trajectory,
            "noise_radial_per_half_trajectory": self.noise_radial_per_half_trajectory,
            "tau2_radial_trajectory": self.tau2_radial_trajectory,
            "tau2_sigma2_trajectory": self.tau2_sigma2_trajectory,
            "tau2_avg_weight_trajectory": self.tau2_avg_weight_trajectory,
            "tau2_shell_sum_trajectory": self.tau2_shell_sum_trajectory,
            "tau2_shell_count_trajectory": self.tau2_shell_count_trajectory,
            "tau2_fsc_used_trajectory": self.tau2_fsc_used_trajectory,
            "tau2_ssnr_trajectory": self.tau2_ssnr_trajectory,
            "sigma_offset_used_trajectory": self.sigma_offset_used_trajectory,
            "sigma_offset_used_per_half_trajectory": self.sigma_offset_used_per_half_trajectory,
            "sigma_offset_used_trajectory_per_half": self.sigma_offset_used_per_half_trajectory,
            "sigma_offset_trajectory": self.sigma_offset_trajectory,
            "sigma_offset_per_half_trajectory": self.sigma_offset_per_half_trajectory,
            "sigma_offset_trajectory_per_half": self.sigma_offset_per_half_trajectory,
            "per_class_sigma_offset_trajectory": self.per_class_sigma_offset_trajectory,
            "direction_prior_trajectory_per_half": self.direction_prior_trajectory_per_half,
            "rotation_posterior_trajectory_per_half": self.rotation_posterior_trajectory_per_half,
            "frac_changed_trajectory": self.frac_changed_trajectory,
            "acc_rot_trajectory": self.acc_rot_trajectory,
            "acc_trans_trajectory": self.acc_trans_trajectory,
            "acc_rot_per_class_trajectory": self.acc_rot_per_class_trajectory,
            "acc_trans_per_class_trajectory": self.acc_trans_per_class_trajectory,
            "expected_accuracy_class_counts_trajectory": self.expected_accuracy_class_counts_trajectory,
            "expected_accuracy_status_trajectory": self.expected_accuracy_status_trajectory,
            "smallest_change_angles_trajectory": self.smallest_change_angles_trajectory,
            "smallest_change_offsets_trajectory": self.smallest_change_offsets_trajectory,
            "best_rotation_eulers_history": self.best_rotation_eulers_history,
            "best_translations_history": self.best_translations_history,
            "class_weight_trajectory": self.class_weight_trajectory,
            "class_mstep_weight_trajectory": self.class_mstep_weight_trajectory,
            "class_full_posterior_weight_trajectory": self.class_full_posterior_weight_trajectory,
            "class_assignment_history": self.class_assignment_history,
            "state_swap_probe_applied_relion_iterations": list(self.state_swap_probe_applied_relion_iterations),
            "local_profile_history": self.local_profile_history,
            "global_profile_history": self.global_profile_history,
        }
