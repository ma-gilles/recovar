### spa quality

| Metric | Baseline | Current | Change | Status |
| --- | ---: | ---: | ---: | --- |
| contrast_abs_error_10 | 0.026186647 | 0.02622628 | +0.151% | OK |
| contrast_abs_error_10_noreg | 0.026248175 | 0.025935474 | -1.191% | OK |
| contrast_abs_error_4 | 0.025997532 | 0.026087358 | +0.346% | OK |
| contrast_abs_error_4_noreg | 0.02610193 | 0.026049558 | -0.201% | OK |
| embedding_squared_error_10 | 1.9783058 | 1.9529953 | -1.279% | OK |
| embedding_squared_error_4 | 0.379676 | 0.38310701 | +0.904% | OK |
| mean_fsc | 0.23345588 | 0.23345588 | +0.000% | OK |
| noise_correlation | 0.97989173 | 0.98024921 | +0.036% | OK |
| noise_max_relative_error | 2.1122448 | 2.1146488 | +0.114% | OK |
| noise_mean_relative_error | 0.051014002 | 0.049130641 | -3.692% | OK |
| noise_median_relative_error | 0.0059617274 | 0.0057464954 | -3.610% | OK |
| svd_relative_variance_10 | 0.46046033 | 0.47868237 | +3.957% | OK |
| svd_relative_variance_4 | 0.45175288 | 0.47531831 | +5.216% | OK |
| variance_fourier_fsc | 0.23345588 | 0.23345588 | +0.000% | OK |
| variance_fsc | 0.23345588 | 0.23345588 | +0.000% | OK |
| variance_spatial_fsc | 0.23345588 | 0.23345588 | +0.000% | OK |

### spa historical performance

| Stage/measurement | Baseline | Current | Change | Status |
| --- | ---: | ---: | ---: | --- |
| dataset_generation/wall_seconds | 136.15 | 164.44 | +20.779% | REGRESSED |
| dataset_generation/peak_cpu_memory_gb | 5.301 | 5.683 | +7.206% | OK |
| dataset_generation/peak_gpu_memory_gb | 4.916 | 4.698 | -4.434% | OK |
| pipeline/wall_seconds | 578.54 | 584.57 | +1.042% | OK |
| pipeline/peak_cpu_memory_gb | 42.681 | 35.2 | -17.528% | OK |
| pipeline/peak_gpu_memory_gb | 40.358 | 40.358 | +0.000% | OK |
| compute_state/wall_seconds | 317.34 | 178.56 | -43.732% | OK |
| compute_state/peak_cpu_memory_gb | 46.719 | 35.36 | -24.313% | OK |
| compute_state/peak_gpu_memory_gb | 0.235 | 0.999 | +325.106% | REGRESSED |
| metrics/wall_seconds | 20.8 | 24.36 | +17.115% | REGRESSED |
| metrics/peak_cpu_memory_gb | 50.666 | 39.312 | -22.410% | OK |
| metrics/peak_gpu_memory_gb | 0.076 | 1.074 | +1313.158% | REGRESSED |

Single candidate measurement against a historical hardware baseline; this is not paired performance qualification.

### cryo_et quality

| Metric | Baseline | Current | Change | Status |
| --- | ---: | ---: | ---: | --- |
| contrast_abs_error_10 | 0.024244158 | 0.024065136 | -0.738% | OK |
| contrast_abs_error_10_noreg | 0.024257016 | 0.024057518 | -0.822% | OK |
| contrast_abs_error_4 | 0.024024975 | 0.023826493 | -0.826% | OK |
| contrast_abs_error_4_noreg | 0.024021479 | 0.02381861 | -0.845% | OK |
| embedding_squared_error_10 | 1.4602945 | 1.4117036 | -3.327% | OK |
| embedding_squared_error_4 | 0.21408913 | 0.2032827 | -5.048% | OK |
| mean_fsc | 0.23345588 | 0.23345588 | +0.000% | OK |
| noise_correlation | 0.97918311 | 0.97974371 | +0.057% | OK |
| noise_max_relative_error | 2.1527984 | 2.1405902 | -0.567% | OK |
| noise_mean_relative_error | 0.051816367 | 0.049609475 | -4.259% | OK |
| noise_median_relative_error | 0.0056973845 | 0.0056769 | -0.360% | OK |
| svd_relative_variance_10 | 0.6345216 | 0.64048725 | +0.940% | OK |
| svd_relative_variance_4 | 0.63259668 | 0.63851832 | +0.936% | OK |
| variance_fourier_fsc | 0.23345588 | 0.23345588 | +0.000% | OK |
| variance_fsc | 0.23345588 | 0.23345588 | +0.000% | OK |
| variance_spatial_fsc | 0.23345588 | 0.23345588 | +0.000% | OK |

### cryo_et historical performance

| Stage/measurement | Baseline | Current | Change | Status |
| --- | ---: | ---: | ---: | --- |
| dataset_generation/wall_seconds | 128.54 | 147.7 | +14.906% | REGRESSED |
| dataset_generation/peak_cpu_memory_gb | 5.059 | 5.416 | +7.057% | OK |
| dataset_generation/peak_gpu_memory_gb | 4.547 | 2.92 | -35.782% | OK |
| pipeline/wall_seconds | 1886.69 | 925.71 | -50.935% | OK |
| pipeline/peak_cpu_memory_gb | 38.107 | 37.461 | -1.695% | OK |
| pipeline/peak_gpu_memory_gb | 40.358 | 40.358 | +0.000% | OK |
| compute_state/wall_seconds | 145.94 | 166.99 | +14.424% | REGRESSED |
| compute_state/peak_cpu_memory_gb | 38.164 | 37.461 | -1.842% | OK |
| compute_state/peak_gpu_memory_gb | 0.235 | 0.999 | +325.106% | REGRESSED |
| metrics/wall_seconds | 19.92 | 24.32 | +22.088% | REGRESSED |
| metrics/peak_cpu_memory_gb | 41.644 | 40.773 | -2.092% | OK |
| metrics/peak_gpu_memory_gb | 0.076 | 1.074 | +1313.158% | REGRESSED |

Single candidate measurement against a historical hardware baseline; this is not paired performance qualification.
