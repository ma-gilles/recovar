// Passive CUDA primary-context probe for VDAM host-runtime parity experiments.
//
// Build as either a shared object (load after JAX initialises CUDA) or a small
// executable (native CUDA-runtime control).  The probe performs no kernels and
// does not change device flags after context creation.

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstdint>
#include <sstream>
#include <string>

namespace {

std::string cuda_error_json(const char* operation, cudaError_t error)
{
    std::ostringstream output;
    output << "{\"status\":\"error\",\"operation\":\"" << operation
           << "\",\"cuda_error\":" << static_cast<int>(error)
           << ",\"cuda_error_name\":\"" << cudaGetErrorName(error) << "\"}";
    return output.str();
}

std::string driver_error_json(const char* operation, CUresult error)
{
    const char* name = nullptr;
    cuGetErrorName(error, &name);
    std::ostringstream output;
    output << "{\"status\":\"error\",\"operation\":\"" << operation
           << "\",\"driver_error\":" << static_cast<int>(error)
           << ",\"driver_error_name\":\"" << (name == nullptr ? "unknown" : name)
           << "\"}";
    return output.str();
}

std::string collect_context_report()
{
    int device = -1;
    cudaError_t runtime_error = cudaGetDevice(&device);
    if (runtime_error != cudaSuccess)
        return cuda_error_json("cudaGetDevice", runtime_error);

    unsigned runtime_flags = 0;
    runtime_error = cudaGetDeviceFlags(&runtime_flags);
    if (runtime_error != cudaSuccess)
        return cuda_error_json("cudaGetDeviceFlags", runtime_error);

    int least_priority = 0;
    int greatest_priority = 0;
    runtime_error = cudaDeviceGetStreamPriorityRange(
        &least_priority, &greatest_priority);
    if (runtime_error != cudaSuccess)
        return cuda_error_json("cudaDeviceGetStreamPriorityRange", runtime_error);

    CUcontext current_context = nullptr;
    CUresult driver_error = cuCtxGetCurrent(&current_context);
    if (driver_error != CUDA_SUCCESS)
        return driver_error_json("cuCtxGetCurrent", driver_error);

    unsigned primary_flags = 0;
    int primary_active = 0;
    driver_error = cuDevicePrimaryCtxGetState(
        static_cast<CUdevice>(device), &primary_flags, &primary_active);
    if (driver_error != CUDA_SUCCESS)
        return driver_error_json("cuDevicePrimaryCtxGetState", driver_error);

    cudaStream_t stream = nullptr;
    runtime_error = cudaStreamCreate(&stream);
    if (runtime_error != cudaSuccess)
        return cuda_error_json("cudaStreamCreate", runtime_error);

    unsigned stream_flags = 0;
    int stream_priority = 0;
    runtime_error = cudaStreamGetFlags(stream, &stream_flags);
    if (runtime_error == cudaSuccess)
        runtime_error = cudaStreamGetPriority(stream, &stream_priority);
    const cudaError_t destroy_error = cudaStreamDestroy(stream);
    if (runtime_error != cudaSuccess)
        return cuda_error_json("cudaStreamGetFlags/Priority", runtime_error);
    if (destroy_error != cudaSuccess)
        return cuda_error_json("cudaStreamDestroy", destroy_error);

    std::ostringstream output;
    output << "{\"status\":\"complete\""
           << ",\"device\":" << device
           << ",\"runtime_device_flags\":" << runtime_flags
           << ",\"primary_context_flags\":" << primary_flags
           << ",\"primary_context_active\":" << primary_active
           << ",\"current_context\":\"0x" << std::hex
           << reinterpret_cast<std::uintptr_t>(current_context) << std::dec << "\""
           << ",\"stream_flags\":" << stream_flags
           << ",\"stream_priority\":" << stream_priority
           << ",\"least_stream_priority\":" << least_priority
           << ",\"greatest_stream_priority\":" << greatest_priority
           << "}";
    return output.str();
}

}  // namespace

extern "C" const char* recovar_cuda_runtime_context_probe_json()
{
    static thread_local std::string report;
    report = collect_context_report();
    return report.c_str();
}

#ifdef RECOVAR_CUDA_CONTEXT_PROBE_MAIN
int main()
{
    const char* report = recovar_cuda_runtime_context_probe_json();
    std::puts(report);
    return std::string(report).find("\"status\":\"complete\"") == std::string::npos;
}
#endif
