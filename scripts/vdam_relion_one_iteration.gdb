set pagination off
set confirm off
set breakpoint pending on
set $vdam_expectation_hits = 0

break MlOptimiser::expectation()
commands
  silent
  set $vdam_expectation_hits = $vdam_expectation_hits + 1
  if $vdam_expectation_hits == 1
    printf "VDAM_GDB_FIRST_EXPECTATION\n"
    if $vdam_profile_cuda != 0
      set $vdam_cuda_start_status = (int) cudaProfilerStart()
      printf "VDAM_GDB_CUDA_PROFILER_START status=%d\n", $vdam_cuda_start_status
      if $vdam_cuda_start_status != 0
        call (void) exit(86)
      end
    end
    continue
  end
  if $vdam_expectation_hits == 2
    printf "VDAM_GDB_SECOND_EXPECTATION\n"
    if $vdam_profile_cuda != 0
      set $vdam_cuda_stop_status = (int) cudaProfilerStop()
      printf "VDAM_GDB_CUDA_PROFILER_STOP status=%d\n", $vdam_cuda_stop_status
      if $vdam_cuda_stop_status != 0
        call (void) exit(87)
      end
    end
    call (void) exit(0)
  end
  continue
end

run
