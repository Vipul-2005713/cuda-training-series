#include "../hw1/cuda_helpers.cuh"
#include <math_constants.h>
constexpr int BLOCK_SIZE = 256;
constexpr int BLOCKS = 640;
// The identity for max is -infinity, NOT zero: zero fails on all-negative input.
__global__ void reduce_max(const float* input, float* out, int n) {
    __shared__ float partial[BLOCK_SIZE];
    const int tid = threadIdx.x;
    float value = -CUDART_INF_F;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x)
        value = fmaxf(value, input[i]);
    partial[tid] = value;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] = fmaxf(partial[tid], partial[tid + stride]);
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = partial[0];
}
// Usage: max_reduction [elements=8388608] [repeats=10]
int main(int argc, char** argv) {
    return checked_main([&] {
        const int n = int_arg(argc, argv, 1, 8388608);
        const int repeats = int_arg(argc, argv, 2, 10, 1, 1000);
        const size_t bytes = size_t(n) * sizeof(float);
        std::vector<float> input(n);
        DeviceBuffer<float> din(n), partial(BLOCKS), dout(1);
        for (int negative = 0; negative < 2; ++negative) {
            for (int i = 0; i < n; ++i) input[i] = negative ? -float(2 + i % 101) : float(i % 101 - 50);
            input[n - 1] = negative ? -1.0f : 123.0f; // exercise the final element/tail
            const float expected = *std::max_element(input.begin(), input.end());
            CUDA_CHECK(cudaMemcpy(din.data, input.data(), bytes, cudaMemcpyHostToDevice));
            const float ms = kernel_ms([&] {
                reduce_max<<<BLOCKS, BLOCK_SIZE>>>(din.data, partial.data, n);
                CUDA_CHECK(cudaGetLastError());
                reduce_max<<<1, BLOCK_SIZE>>>(partial.data, dout.data, BLOCKS);
            }, repeats);
            float result = 0;
            CUDA_CHECK(cudaMemcpy(&result, dout.data, sizeof(float), cudaMemcpyDeviceToHost));
            if (result != expected) throw std::runtime_error("max reduction mismatch");
            std::printf("PASS max_reduction case=%s N=%d repeats=%d two_stage_ms=%.6f input_GB_s=%.3f result=%g expected=%g\n",
                        negative ? "all_negative" : "mixed", n, repeats, ms, gb_per_second(bytes, ms), result, expected);
        }
        return 0;
    });
}

