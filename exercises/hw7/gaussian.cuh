#pragma once
#include "../hw6/cuda_utils.cuh"
#include <memory>
constexpr int gaussian_samples = 22;
__host__ __device__ inline float gaussian_value(float value) {
    float x = value - (gaussian_samples / 2) * 0.01f, sum = 0;
    for (int k = 0; k < gaussian_samples; ++k) {
        sum += expf(-0.5f * x * x) / 2.506628274794649323f;
        x += 0.01f;
    }
    return sum / gaussian_samples;
}
__global__ void gaussian_pdf(const float* x, float* y, size_t n) {
    size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i < n) y[i] = gaussian_value(x[i]);
}
inline std::vector<float> gaussian_reference() {
    std::vector<float> ref(4096);
    for (int i = 0; i < 4096; ++i) ref[i] = gaussian_value(float(i) / 4096);
    return ref;
}
inline void verify_gaussian(const float* result, size_t n, int offset = 0) {
    auto ref = gaussian_reference();
    for (size_t i = 0; i < n; ++i)
        if (!std::isfinite(result[i]) || std::fabs(result[i] - ref[(i + offset) % 4096]) > 2e-6f)
            throw std::runtime_error("Gaussian CPU reference mismatch at " + std::to_string(i));
}
