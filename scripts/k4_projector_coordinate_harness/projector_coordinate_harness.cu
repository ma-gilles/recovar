#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kRotations = 8;
constexpr int kImageSize = 40;
constexpr int kHalfWidth = 21;
constexpr int kPixels = kImageSize * kHalfWidth;
constexpr int kPprefX = 42;
constexpr int kPprefY = 83;
constexpr int kPprefZ = 83;
constexpr int kPprefYInit = -41;
constexpr int kPprefZInit = -41;
constexpr int kPaddingFactor = 2;
constexpr int kMaxR2Padded = 1600;
constexpr int kCoordinateFields = 9;

#define CUDA_CHECK(call)                                                                  \
    do {                                                                                  \
        const cudaError_t error_ = (call);                                                 \
        if (error_ != cudaSuccess) {                                                       \
            throw std::runtime_error(std::string(#call) + ": " + cudaGetErrorString(error_)); \
        }                                                                                 \
    } while (false)

template <typename T>
std::vector<T> read_exact(const std::filesystem::path& path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) throw std::runtime_error("cannot open " + path.string());
    const auto bytes = stream.tellg();
    if (bytes != static_cast<std::streamoff>(count * sizeof(T))) {
        throw std::runtime_error("unexpected byte count for " + path.string());
    }
    stream.seekg(0);
    std::vector<T> values(count);
    stream.read(reinterpret_cast<char*>(values.data()), bytes);
    if (!stream) throw std::runtime_error("short read from " + path.string());
    return values;
}

template <typename T>
void write_exact(const std::filesystem::path& path, const std::vector<T>& values) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) throw std::runtime_error("cannot create " + path.string());
    stream.write(reinterpret_cast<const char*>(values.data()), values.size() * sizeof(T));
    if (!stream) throw std::runtime_error("short write to " + path.string());
}

__global__ void stage_recovar_texture_source(
    const float2* __restrict__ full_volume,
    float* __restrict__ real,
    float* __restrict__ imag) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    constexpr int count = kPprefX * kPprefY * kPprefZ;
    if (index >= count) return;
    const int x = index % kPprefX;
    const int y_index = (index / kPprefX) % kPprefY;
    const int z_index = index / (kPprefX * kPprefY);
    const int y = y_index + kPprefYInit;
    const int z = z_index + kPprefZInit;
    const int i0 = kPprefZ / 2 + x;
    const int i1 = kPprefY / 2 + y;
    const int i2 = kPprefZ / 2 + z;
    const float2 value = full_volume[i0 * kPprefY * kPprefZ + i1 * kPprefZ + i2];
    real[index] = value.x;
    imag[index] = value.y;
}

struct ProjectorTextures {
    cudaArray_t real_array = nullptr;
    cudaArray_t imag_array = nullptr;
    cudaTextureObject_t real = 0;
    cudaTextureObject_t imag = 0;
};

ProjectorTextures make_textures(const float* real, const float* imag) {
    ProjectorTextures textures;
    const cudaChannelFormatDesc desc = cudaCreateChannelDesc<float>();
    const cudaExtent extent = make_cudaExtent(kPprefX, kPprefY, kPprefZ);
    CUDA_CHECK(cudaMalloc3DArray(&textures.real_array, &desc, extent));
    CUDA_CHECK(cudaMalloc3DArray(&textures.imag_array, &desc, extent));

    cudaMemcpy3DParms copy = {};
    copy.extent = extent;
    copy.kind = cudaMemcpyHostToDevice;
    copy.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(real), kPprefX * sizeof(float), kPprefY, kPprefZ);
    copy.dstArray = textures.real_array;
    CUDA_CHECK(cudaMemcpy3D(&copy));
    copy.srcPtr = make_cudaPitchedPtr(
        const_cast<float*>(imag), kPprefX * sizeof(float), kPprefY, kPprefZ);
    copy.dstArray = textures.imag_array;
    CUDA_CHECK(cudaMemcpy3D(&copy));

    cudaResourceDesc real_resource = {};
    cudaResourceDesc imag_resource = {};
    real_resource.resType = cudaResourceTypeArray;
    real_resource.res.array.array = textures.real_array;
    imag_resource.resType = cudaResourceTypeArray;
    imag_resource.res.array.array = textures.imag_array;
    cudaTextureDesc texture_desc = {};
    texture_desc.filterMode = cudaFilterModeLinear;
    texture_desc.readMode = cudaReadModeElementType;
    texture_desc.normalizedCoords = false;
    texture_desc.addressMode[0] = cudaAddressModeClamp;
    texture_desc.addressMode[1] = cudaAddressModeClamp;
    texture_desc.addressMode[2] = cudaAddressModeClamp;
    CUDA_CHECK(cudaCreateTextureObject(&textures.real, &real_resource, &texture_desc, nullptr));
    CUDA_CHECK(cudaCreateTextureObject(&textures.imag, &imag_resource, &texture_desc, nullptr));
    return textures;
}

void destroy_textures(ProjectorTextures& textures) {
    if (textures.real) CUDA_CHECK(cudaDestroyTextureObject(textures.real));
    if (textures.imag) CUDA_CHECK(cudaDestroyTextureObject(textures.imag));
    if (textures.real_array) CUDA_CHECK(cudaFreeArray(textures.real_array));
    if (textures.imag_array) CUDA_CHECK(cudaFreeArray(textures.imag_array));
    textures = {};
}

__device__ __forceinline__ void store_projection(
    int index,
    float raw_x,
    float raw_y,
    float raw_z,
    cudaTextureObject_t real_texture,
    cudaTextureObject_t imag_texture,
    float* coordinates,
    float2* projection) {
    float x = raw_x;
    float y = raw_y;
    float z = raw_z;
    float imag_sign = 1.0f;
    if (x < 0.0f) {
        x = -x;
        y = -y;
        z = -z;
        imag_sign = -1.0f;
    }
    const float texture_x = x + 0.5f;
    const float texture_y = y - static_cast<float>(kPprefYInit) + 0.5f;
    const float texture_z = z - static_cast<float>(kPprefZInit) + 0.5f;
    float* output_coordinates = coordinates + index * kCoordinateFields;
    output_coordinates[0] = raw_x;
    output_coordinates[1] = raw_y;
    output_coordinates[2] = raw_z;
    output_coordinates[3] = x;
    output_coordinates[4] = y;
    output_coordinates[5] = z;
    output_coordinates[6] = texture_x;
    output_coordinates[7] = texture_y;
    output_coordinates[8] = texture_z;
    const int radius_squared = static_cast<int>(raw_x * raw_x + raw_y * raw_y + raw_z * raw_z);
    if (radius_squared > kMaxR2Padded) {
        projection[index] = make_float2(0.0f, 0.0f);
        return;
    }
    projection[index] = make_float2(
        tex3D<float>(real_texture, texture_x, texture_y, texture_z),
        imag_sign * tex3D<float>(imag_texture, texture_x, texture_y, texture_z));
}

__device__ __forceinline__ void pixel_xy(int pixel, int& x, int& y) {
    x = pixel % kHalfWidth;
    const int row = pixel / kHalfWidth;
    y = row <= kImageSize / 2 ? row : row - kImageSize;
}

// This expression is intentionally source-ordered like RECOVAR's current
// project_texture_kernel after expanding its compact row-swapped matrix.
__global__ void project_current_source(
    cudaTextureObject_t real_texture,
    cudaTextureObject_t imag_texture,
    const float* __restrict__ eulers,
    float* __restrict__ coordinates,
    float2* __restrict__ projection) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kRotations * kPixels) return;
    const int rotation = index / kPixels;
    const int pixel = index % kPixels;
    int x, y;
    pixel_xy(pixel, x, y);
    const float* e = eulers + rotation * 9;
    const float rk0 = (static_cast<float>(y) * e[1] + static_cast<float>(x) * e[0]) * kPaddingFactor;
    const float rk1 = (static_cast<float>(y) * e[4] + static_cast<float>(x) * e[3]) * kPaddingFactor;
    const float rk2 = (static_cast<float>(y) * e[7] + static_cast<float>(x) * e[6]) * kPaddingFactor;
    store_projection(index, rk0, rk1, rk2, real_texture, imag_texture, coordinates, projection);
}

// This is the exact source order in RELION AccProjectorKernel::project3Dmodel.
__global__ void project_relion_source(
    cudaTextureObject_t real_texture,
    cudaTextureObject_t imag_texture,
    const float* __restrict__ eulers,
    float* __restrict__ coordinates,
    float2* __restrict__ projection) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kRotations * kPixels) return;
    const int rotation = index / kPixels;
    const int pixel = index % kPixels;
    int x, y;
    pixel_xy(pixel, x, y);
    const float* e = eulers + rotation * 9;
    const float xp = (e[0] * static_cast<float>(x) + e[1] * static_cast<float>(y)) * kPaddingFactor;
    const float yp = (e[3] * static_cast<float>(x) + e[4] * static_cast<float>(y)) * kPaddingFactor;
    const float zp = (e[6] * static_cast<float>(x) + e[7] * static_cast<float>(y)) * kPaddingFactor;
    store_projection(index, xp, yp, zp, real_texture, imag_texture, coordinates, projection);
}

__global__ void project_explicit_fma_current(
    cudaTextureObject_t real_texture,
    cudaTextureObject_t imag_texture,
    const float* __restrict__ eulers,
    float* __restrict__ coordinates,
    float2* __restrict__ projection) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kRotations * kPixels) return;
    const int rotation = index / kPixels;
    const int pixel = index % kPixels;
    int x_int, y_int;
    pixel_xy(pixel, x_int, y_int);
    const float x = static_cast<float>(x_int);
    const float y = static_cast<float>(y_int);
    const float* e = eulers + rotation * 9;
    const float xp = __fmul_rn(__fmaf_rn(y, e[1], __fmul_rn(x, e[0])), static_cast<float>(kPaddingFactor));
    const float yp = __fmul_rn(__fmaf_rn(y, e[4], __fmul_rn(x, e[3])), static_cast<float>(kPaddingFactor));
    const float zp = __fmul_rn(__fmaf_rn(y, e[7], __fmul_rn(x, e[6])), static_cast<float>(kPaddingFactor));
    store_projection(index, xp, yp, zp, real_texture, imag_texture, coordinates, projection);
}

__global__ void project_explicit_fma_relion(
    cudaTextureObject_t real_texture,
    cudaTextureObject_t imag_texture,
    const float* __restrict__ eulers,
    float* __restrict__ coordinates,
    float2* __restrict__ projection) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kRotations * kPixels) return;
    const int rotation = index / kPixels;
    const int pixel = index % kPixels;
    int x_int, y_int;
    pixel_xy(pixel, x_int, y_int);
    const float x = static_cast<float>(x_int);
    const float y = static_cast<float>(y_int);
    const float* e = eulers + rotation * 9;
    const float xp = __fmul_rn(__fmaf_rn(x, e[0], __fmul_rn(y, e[1])), static_cast<float>(kPaddingFactor));
    const float yp = __fmul_rn(__fmaf_rn(x, e[3], __fmul_rn(y, e[4])), static_cast<float>(kPaddingFactor));
    const float zp = __fmul_rn(__fmaf_rn(x, e[6], __fmul_rn(y, e[7])), static_cast<float>(kPaddingFactor));
    store_projection(index, xp, yp, zp, real_texture, imag_texture, coordinates, projection);
}

__global__ void project_noncontracted_relion(
    cudaTextureObject_t real_texture,
    cudaTextureObject_t imag_texture,
    const float* __restrict__ eulers,
    float* __restrict__ coordinates,
    float2* __restrict__ projection) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kRotations * kPixels) return;
    const int rotation = index / kPixels;
    const int pixel = index % kPixels;
    int x_int, y_int;
    pixel_xy(pixel, x_int, y_int);
    const float x = static_cast<float>(x_int);
    const float y = static_cast<float>(y_int);
    const float* e = eulers + rotation * 9;
    const float xp = __fmul_rn(
        __fadd_rn(__fmul_rn(e[0], x), __fmul_rn(e[1], y)), static_cast<float>(kPaddingFactor));
    const float yp = __fmul_rn(
        __fadd_rn(__fmul_rn(e[3], x), __fmul_rn(e[4], y)), static_cast<float>(kPaddingFactor));
    const float zp = __fmul_rn(
        __fadd_rn(__fmul_rn(e[6], x), __fmul_rn(e[7], y)), static_cast<float>(kPaddingFactor));
    store_projection(index, xp, yp, zp, real_texture, imag_texture, coordinates, projection);
}

__global__ void project_relion_adjacent_y_bins(
    cudaTextureObject_t real_texture,
    cudaTextureObject_t imag_texture,
    const float* __restrict__ eulers,
    float2* __restrict__ lower_projection,
    float2* __restrict__ upper_projection) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= kRotations * kPixels) return;
    const int rotation = index / kPixels;
    const int pixel = index % kPixels;
    int x_int, y_int;
    pixel_xy(pixel, x_int, y_int);
    const float x_image = static_cast<float>(x_int);
    const float y_image = static_cast<float>(y_int);
    const float* e = eulers + rotation * 9;
    float x = (e[0] * x_image + e[1] * y_image) * kPaddingFactor;
    float y = (e[3] * x_image + e[4] * y_image) * kPaddingFactor;
    float z = (e[6] * x_image + e[7] * y_image) * kPaddingFactor;
    const int radius_squared = static_cast<int>(x * x + y * y + z * z);
    float imag_sign = 1.0f;
    if (x < 0.0f) {
        x = -x;
        y = -y;
        z = -z;
        imag_sign = -1.0f;
    }
    if (radius_squared > kMaxR2Padded) {
        lower_projection[index] = make_float2(0.0f, 0.0f);
        upper_projection[index] = make_float2(0.0f, 0.0f);
        return;
    }
    const float y_scaled = y * 256.0f;
    const float y_lower = floorf(y_scaled) / 256.0f;
    const float y_upper = ceilf(y_scaled) / 256.0f;
    const float texture_x = x + 0.5f;
    const float texture_z = z - static_cast<float>(kPprefZInit) + 0.5f;
    const float texture_y_lower = y_lower - static_cast<float>(kPprefYInit) + 0.5f;
    const float texture_y_upper = y_upper - static_cast<float>(kPprefYInit) + 0.5f;
    lower_projection[index] = make_float2(
        tex3D<float>(real_texture, texture_x, texture_y_lower, texture_z),
        imag_sign * tex3D<float>(imag_texture, texture_x, texture_y_lower, texture_z));
    upper_projection[index] = make_float2(
        tex3D<float>(real_texture, texture_x, texture_y_upper, texture_z),
        imag_sign * tex3D<float>(imag_texture, texture_x, texture_y_upper, texture_z));
}

template <void (*Kernel)(cudaTextureObject_t, cudaTextureObject_t, const float*, float*, float2*)>
void run_projection(
    const std::filesystem::path& output_dir,
    const std::string& name,
    const ProjectorTextures& textures,
    const float* device_eulers) {
    constexpr int count = kRotations * kPixels;
    float* device_coordinates = nullptr;
    float2* device_projection = nullptr;
    CUDA_CHECK(cudaMalloc(&device_coordinates, count * kCoordinateFields * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&device_projection, count * sizeof(float2)));
    Kernel<<<(count + 255) / 256, 256>>>(
        textures.real, textures.imag, device_eulers, device_coordinates, device_projection);
    CUDA_CHECK(cudaGetLastError());
    std::vector<float> coordinates(count * kCoordinateFields);
    std::vector<float2> projection(count);
    CUDA_CHECK(cudaMemcpy(
        coordinates.data(), device_coordinates, coordinates.size() * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(
        projection.data(), device_projection, projection.size() * sizeof(float2), cudaMemcpyDeviceToHost));
    write_exact(output_dir / ("coordinates_" + name + ".f32"), coordinates);
    write_exact(output_dir / ("projection_" + name + ".f32x2"), projection);
    CUDA_CHECK(cudaFree(device_coordinates));
    CUDA_CHECK(cudaFree(device_projection));
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr << "usage: " << argv[0] << " INPUT_DIR OUTPUT_DIR\n";
            return 2;
        }
        const std::filesystem::path input_dir(argv[1]);
        const std::filesystem::path output_dir(argv[2]);
        if (!std::filesystem::is_directory(input_dir)) {
            throw std::runtime_error("input directory does not exist");
        }
        if (std::filesystem::exists(output_dir) && !std::filesystem::is_empty(output_dir)) {
            throw std::runtime_error("output directory is not empty");
        }
        std::filesystem::create_directories(output_dir);

        constexpr std::size_t ppref_count = kPprefX * kPprefY * kPprefZ;
        const auto ppref_real = read_exact<float>(input_dir / "ppref_real.f32", ppref_count);
        const auto ppref_imag = read_exact<float>(input_dir / "ppref_imag.f32", ppref_count);
        const auto eulers = read_exact<float>(input_dir / "eulers.f32", kRotations * 9);

        std::vector<float2> full_volume(kPprefZ * kPprefY * kPprefZ, make_float2(0.0f, 0.0f));
        for (int z = 0; z < kPprefZ; ++z) {
            for (int y = 0; y < kPprefY; ++y) {
                for (int x = 0; x < kPprefX; ++x) {
                    const int ppref_index = z * kPprefY * kPprefX + y * kPprefX + x;
                    const int full_index = (kPprefZ / 2 + x) * kPprefY * kPprefZ + y * kPprefZ + z;
                    full_volume[full_index] = make_float2(ppref_real[ppref_index], ppref_imag[ppref_index]);
                }
            }
        }

        float2* device_full_volume = nullptr;
        float* device_staged_real = nullptr;
        float* device_staged_imag = nullptr;
        CUDA_CHECK(cudaMalloc(&device_full_volume, full_volume.size() * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&device_staged_real, ppref_count * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&device_staged_imag, ppref_count * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(
            device_full_volume, full_volume.data(), full_volume.size() * sizeof(float2), cudaMemcpyHostToDevice));
        stage_recovar_texture_source<<<(ppref_count + 255) / 256, 256>>>(
            device_full_volume, device_staged_real, device_staged_imag);
        CUDA_CHECK(cudaGetLastError());

        // This readback occurs before either staged array is passed to
        // cudaMemcpy3D, so its hashes distinguish staging from texture sampling.
        std::vector<float> staged_real(ppref_count);
        std::vector<float> staged_imag(ppref_count);
        CUDA_CHECK(cudaMemcpy(
            staged_real.data(), device_staged_real, ppref_count * sizeof(float), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(
            staged_imag.data(), device_staged_imag, ppref_count * sizeof(float), cudaMemcpyDeviceToHost));
        write_exact(output_dir / "staged_relion_direct_real.f32", ppref_real);
        write_exact(output_dir / "staged_relion_direct_imag.f32", ppref_imag);
        write_exact(output_dir / "staged_recovar_real.f32", staged_real);
        write_exact(output_dir / "staged_recovar_imag.f32", staged_imag);

        // Every arithmetic variant below uses this one captured RECOVAR-staged
        // byte sequence. The direct RELION texture is used only for the oracle
        // replay, so a staging mismatch can be classified before coordinates.
        ProjectorTextures recovar_textures = make_textures(staged_real.data(), staged_imag.data());
        ProjectorTextures relion_textures = make_textures(ppref_real.data(), ppref_imag.data());
        float* device_eulers = nullptr;
        CUDA_CHECK(cudaMalloc(&device_eulers, eulers.size() * sizeof(float)));
        CUDA_CHECK(cudaMemcpy(
            device_eulers, eulers.data(), eulers.size() * sizeof(float), cudaMemcpyHostToDevice));

        run_projection<project_current_source>(output_dir, "current_source", recovar_textures, device_eulers);
        run_projection<project_relion_source>(output_dir, "relion_source", recovar_textures, device_eulers);
        run_projection<project_explicit_fma_current>(
            output_dir, "explicit_fma_current", recovar_textures, device_eulers);
        run_projection<project_explicit_fma_relion>(
            output_dir, "explicit_fma_relion", recovar_textures, device_eulers);
        run_projection<project_noncontracted_relion>(
            output_dir, "noncontracted_relion", recovar_textures, device_eulers);
        run_projection<project_relion_source>(
            output_dir, "relion_direct_source", relion_textures, device_eulers);

        constexpr int count = kRotations * kPixels;
        float2* device_lower = nullptr;
        float2* device_upper = nullptr;
        CUDA_CHECK(cudaMalloc(&device_lower, count * sizeof(float2)));
        CUDA_CHECK(cudaMalloc(&device_upper, count * sizeof(float2)));
        project_relion_adjacent_y_bins<<<(count + 255) / 256, 256>>>(
            recovar_textures.real,
            recovar_textures.imag,
            device_eulers,
            device_lower,
            device_upper);
        CUDA_CHECK(cudaGetLastError());
        std::vector<float2> lower(count), upper(count);
        CUDA_CHECK(cudaMemcpy(lower.data(), device_lower, count * sizeof(float2), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(upper.data(), device_upper, count * sizeof(float2), cudaMemcpyDeviceToHost));
        write_exact(output_dir / "projection_relion_y_bin_lower.f32x2", lower);
        write_exact(output_dir / "projection_relion_y_bin_upper.f32x2", upper);

        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaFree(device_lower));
        CUDA_CHECK(cudaFree(device_upper));
        CUDA_CHECK(cudaFree(device_eulers));
        destroy_textures(recovar_textures);
        destroy_textures(relion_textures);
        CUDA_CHECK(cudaFree(device_full_volume));
        CUDA_CHECK(cudaFree(device_staged_real));
        CUDA_CHECK(cudaFree(device_staged_imag));
        std::cout << "K4_PROJECTOR_COORDINATE_HARNESS_COMPLETE\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "K4_PROJECTOR_COORDINATE_HARNESS_ERROR: " << error.what() << "\n";
        return 1;
    }
}
