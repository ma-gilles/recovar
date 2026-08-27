#!/usr/bin/env python3
"""Compare a live K=1 reconstruction call with its saved and native boundaries."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from recovar.core import fourier_transform_utils, mask
from recovar.em.dense_single_volume.mean_helpers import (
    _apply_relion_initial_lowpass_filter,
)
from recovar.reconstruction import regularization
from recovar.utils import helpers
from scripts.analyze_k1_reconstruction_stage_boundary import (
    _load,
    _metrics,
    _stage_path,
)
from scripts.analyze_k1_recovar_reference_write_boundary import (
    _choose_sign,
    _comparison,
    _firstiter_postprocess,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recovar-root", type=Path, required=True)
    parser.add_argument("--native-stage-dir", type=Path, required=True)
    parser.add_argument("--native-relion-dir", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    args = parser.parse_args()

    intermediates = args.recovar_root / "output" / "intermediates"
    input_dir = args.recovar_root / "reconstruction_inputs"
    premask_dir = args.recovar_root / "premask"
    lifecycle_dir = args.recovar_root / "reference_lifecycle"
    fsc = jnp.asarray(np.load(intermediates / "it000_fsc.npy", allow_pickle=False))

    halves: dict[str, object] = {}
    for half in (1, 2):
        with np.load(
            input_dir / f"recovar_reconstruction_input_it001_half{half}.npz",
            allow_pickle=False,
        ) as archive:
            live_y = np.asarray(archive["Ft_y"])
            live_ctf = np.asarray(archive["Ft_ctf"])
            live_tau = np.asarray(archive["reconstruction_tau"])
            current_size = int(archive["current_size"])
            padding_factor = int(archive["padding_factor"])
            accumulator_shape = tuple(
                int(value) for value in archive["accumulator_volume_shape"]
            )
            tau2_fudge = float(archive["tau2_fudge"])

        saved_y = np.load(intermediates / f"it000_Ft_y_{half - 1}.npy", allow_pickle=False)
        saved_ctf = np.load(
            intermediates / f"it000_Ft_ctf_{half - 1}.npy", allow_pickle=False
        )
        replay_tau, _, replay_tau_details = regularization.compute_relion_tau2_from_weights(
            jnp.asarray(live_ctf),
            jnp.asarray(live_ctf),
            fsc,
            (256, 256, 256),
            tau2_fudge=tau2_fudge,
            padding_factor=padding_factor,
            r_max=current_size // 2,
            return_details=True,
            full_half_axis=0,
            accumulator_volume_shape=accumulator_shape,
        )

        with np.load(
            premask_dir / f"recovar_premask_it001_half{half}.npz",
            allow_pickle=False,
        ) as archive:
            premask_fourier = np.asarray(archive["means_premask"])
            premask_real = np.asarray(archive["means_premask_real"], dtype=np.float64)

        native_after = helpers.relion_volume_to_recovar(
            _load(
                _stage_path(args.native_stage_dir, half, "volume_after_gridding", 0),
                np.dtype("<f8"),
            )
        )
        premask_sign_aligned, selected_sign = _choose_sign(premask_real, native_after)
        solvent_radius = 200.0 / (2.0 * 2.125)
        solvent_mask = np.asarray(
            mask.raised_cosine_mask(
                (256, 256, 256),
                radius=solvent_radius,
                radius_p=solvent_radius + 5.0,
                offset=jnp.zeros(3),
            ),
            dtype=np.float64,
        )
        lowpass_fourier = _apply_relion_initial_lowpass_filter(
            jnp.asarray(premask_fourier),
            (256, 256, 256),
            2.125,
            30.0,
            filter_edgewidth=2.0,
        )
        lowpass_complex_realspace = fourier_transform_utils.get_idft3(
            lowpass_fourier.reshape((256, 256, 256))
        )
        current_path_fourier = fourier_transform_utils.get_dft3(
            lowpass_complex_realspace * jnp.asarray(solvent_mask)
        ).reshape(-1)
        real_input_path_fourier = fourier_transform_utils.get_dft3(
            jnp.real(lowpass_complex_realspace) * jnp.asarray(solvent_mask)
        ).reshape(-1)
        if selected_sign < 0:
            current_path_fourier = -current_path_fourier
            real_input_path_fourier = -real_input_path_fourier
        current_path_written = np.asarray(
            fourier_transform_utils.get_idft3(
                current_path_fourier.reshape((256, 256, 256))
            )
        ).real
        real_input_path_written = np.asarray(
            fourier_transform_utils.get_idft3(
                real_input_path_fourier.reshape((256, 256, 256))
            )
        ).real
        postprocessed = _firstiter_postprocess(
            premask_sign_aligned,
            volume_shape=(256, 256, 256),
            voxel_size=2.125,
            ini_high_angstrom=30.0,
            fourier_mask_edge=2.0,
            solvent_mask=solvent_mask,
        )
        recovar_written = np.asarray(
            helpers.load_mrc(intermediates / f"it000_half{half}_reg.mrc"),
            dtype=np.float64,
        )
        native_written = np.asarray(
            helpers.load_relion_volume(
                args.native_relion_dir / f"run_it001_half{half}_class001.mrc"
            ),
            dtype=np.float64,
        )

        lifecycle: dict[str, object] = {}
        if lifecycle_dir.is_dir():
            stage_values: dict[str, tuple[np.ndarray, np.ndarray]] = {}
            for stage in ("post_lowpass", "post_mask", "presave"):
                with np.load(
                    lifecycle_dir
                    / f"recovar_reference_it001_half{half}_{stage}.npz",
                    allow_pickle=False,
                ) as archive:
                    stage_values[stage] = (
                        np.asarray(archive["value_fourier"]),
                        np.asarray(archive["value_real"], dtype=np.float64),
                    )
            post_lowpass_fourier, post_lowpass_real = stage_values["post_lowpass"]
            post_mask_fourier, post_mask_real = stage_values["post_mask"]
            presave_fourier, presave_real = stage_values["presave"]
            captured_lowpass_complex_real = fourier_transform_utils.get_idft3(
                jnp.asarray(post_lowpass_fourier).reshape((256, 256, 256))
            )
            expected_post_mask_fourier = fourier_transform_utils.get_dft3(
                captured_lowpass_complex_real * jnp.asarray(solvent_mask)
            ).reshape(-1)
            expected_post_mask_real = np.asarray(
                fourier_transform_utils.get_idft3(
                    expected_post_mask_fourier.reshape((256, 256, 256))
                )
            ).real
            lifecycle = {
                "post_lowpass_vs_same_input_recompute": {
                    "fourier": _metrics(
                        post_lowpass_fourier, np.asarray(lowpass_fourier)
                    ),
                    "real": _metrics(
                        post_lowpass_real,
                        np.asarray(lowpass_complex_realspace).real,
                    ),
                },
                "post_mask_vs_direct_postprocess": _comparison(
                    selected_sign * post_mask_real, postprocessed
                ),
                "post_mask_vs_same_lowpass_expected_mask": {
                    "fourier": _metrics(
                        post_mask_fourier, np.asarray(expected_post_mask_fourier)
                    ),
                    "real": _metrics(post_mask_real, expected_post_mask_real),
                },
                "presave_vs_sign_aligned_post_mask": {
                    "fourier": _metrics(
                        presave_fourier, selected_sign * post_mask_fourier
                    ),
                    "real": _metrics(presave_real, selected_sign * post_mask_real),
                },
                "presave_real_vs_live_written": _comparison(
                    presave_real, recovar_written
                ),
                "presave_real_vs_native_written": _comparison(
                    presave_real, native_written
                ),
            }

        halves[str(half)] = {
            "selected_native_sign": selected_sign,
            "live_input_vs_saved_intermediate": {
                "Ft_y": _metrics(live_y, saved_y),
                "Ft_ctf": _metrics(live_ctf, saved_ctf),
            },
            "live_tau_vs_same_input_recompute": _metrics(
                live_tau, np.asarray(replay_tau, dtype=live_tau.dtype)
            ),
            "recomputed_tau_prior_shells_head": np.asarray(
                replay_tau_details["prior_shells"][:8], dtype=np.float64
            ).tolist(),
            "live_premask_vs_native_after_gridding": _comparison(
                premask_sign_aligned, native_after
            ),
            "postprocessed_live_premask_vs_live_written": _comparison(
                postprocessed, recovar_written
            ),
            "current_fourier_roundtrip_vs_live_written": _comparison(
                current_path_written, recovar_written
            ),
            "real_input_fourier_roundtrip_vs_live_written": _comparison(
                real_input_path_written, recovar_written
            ),
            "current_fourier_roundtrip_vs_direct_postprocess": _comparison(
                current_path_written, postprocessed
            ),
            "real_input_fourier_roundtrip_vs_direct_postprocess": _comparison(
                real_input_path_written, postprocessed
            ),
            "postprocessed_live_premask_vs_native_written": _comparison(
                postprocessed, native_written
            ),
            "live_written_vs_native_written": _comparison(
                recovar_written, native_written
            ),
            "reference_lifecycle": lifecycle,
        }

    report = {
        "schema": "recovar.em.k1_live_reconstruction_capture.v1",
        "metric_policy": (
            "exact input comparison plus scale-sensitive relative-L2 and signed "
            "non-DC FSC-AUC; no fitted rescaling"
        ),
        "recovar_root": str(args.recovar_root.resolve()),
        "native_stage_dir": str(args.native_stage_dir.resolve()),
        "native_relion_dir": str(args.native_relion_dir.resolve()),
        "halves": halves,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
