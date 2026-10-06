#include "../hw1/cuda_helpers.cuh"
constexpr int BLOCK_SIZE = 256;
constexpr int BLOCKS = 640;
// One global atomic for every input: the destination becomes a serialization point.
__global__ void atomic_red(const float* input, float* out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) atomicAdd(out, input[i]);
}
// Registers accumulate a grid-stride subset, then shared memory combines the block.
__global__ void reduce_a(const float* input, float* out, int n) {
    __shared__ float partial[BLOCK_SIZE];
    const int tid = threadIdx.x;
    float value = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) value += input[i];
    partial[tid] = value;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) atomicAdd(out, partial[0]);
}
__device__ float warp_sum(float value) {
    for (int offset = 16; offset; offset >>= 1)
        value += __shfl_down_sync(0xffffffffu, value, offset);
    return value;
}
template<class Output>
__global__ void reduce_ws(const float* input, Output* out, int n) {
    __shared__ float warp_results[BLOCK_SIZE / 32];
    const int tid = threadIdx.x, lane = tid % 32, warp = tid / 32;
    float value = 0;
    for (int i = blockIdx.x * blockDim.x + tid; i < n; i += gridDim.x * blockDim.x) value += input[i];
    value = warp_sum(value);
    if (lane == 0) warp_results[warp] = value;
    __syncthreads();
    // A complete first warp participates, including the lanes with no partial.
    if (warp == 0) {
        value = lane < BLOCK_SIZE / 32 ? warp_results[lane] : 0;
        value = warp_sum(value);
        if (lane == 0) atomicAdd(out, static_cast<Output>(value));
    }
}
// Usage: reductions [elements=8388608] [repeats=3]
// Homework experiments: 8388608, 163840 (=640*256), 33554432.
int main(int argc, char** argv) {
    return checked_main([&] {
        const int n = int_arg(argc, argv, 1, 8388608);
        const int repeats = int_arg(argc, argv, 2, 3, 1, 1000);
        const size_t bytes = size_t(n) * sizeof(float);
        std::vector<float> input(n, 1);
        DeviceBuffer<float> din(n), dout(1);
        CUDA_CHECK(cudaMemcpy(din.data, input.data(), bytes, cudaMemcpyHostToDevice));
        const char* names[] = {"atomic_red", "reduce_a", "reduce_ws"};
        for (int method = 0; method < 3; ++method) {
            const float ms = reset_kernel_ms([&] { CUDA_CHECK(cudaMemset(dout.data, 0, sizeof(float))); }, [&] {
                if (method == 0) atomic_red<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(din.data, dout.data, n);
                else if (method == 1) reduce_a<<<BLOCKS, BLOCK_SIZE>>>(din.data, dout.data, n);
                else reduce_ws<<<BLOCKS, BLOCK_SIZE>>>(din.data, dout.data, n);
            }, repeats);
            float result = 0;
            CUDA_CHECK(cudaMemcpy(&result, dout.data, sizeof(float), cudaMemcpyDeviceToHost));
            // Adding 1 to 2^24 in fp32 rounds back to 2^24 (ties-to-even).
            const double expected = method == 0 ? std::min(n, 16777216) : double(n);
            // Above 2^24, irregular N can round intermediate atomic partial sums.
            const double tolerance = method == 0 || n <= 16777216 || n % BLOCK_SIZE == 0
                ? 0 : BLOCKS * std::numeric_limits<float>::epsilon() * n;
            if (!std::isfinite(result) || std::fabs(double(result) - expected) > tolerance)
                throw std::runtime_error(std::string(names[method]) + " reduction mismatch");
            const bool precision_demo = method == 0 && n > 16777216;
            std::printf("%s %s N=%d repeats=%d kernel_ms=%.6f input_GB_s=%.3f result=%.0f exact=%.0f absolute_error=%.0f\n",
                        precision_demo ? "PASS_EXPECTED_FP32_SATURATION" : "PASS",
                        names[method], n, repeats, ms, gb_per_second(bytes, ms), result, double(n), std::fabs(double(result) - n));
        }
        if (n > 16777216) {
            // Block sums are exactly representable for these all-ones inputs.
            // A double atomic finish also handles odd tails above fp32's exact-integer range.
            DeviceBuffer<double> double_out(1);
            const float ms = reset_kernel_ms([&] { CUDA_CHECK(cudaMemset(double_out.data, 0, sizeof(double))); }, [&] {
                reduce_ws<<<BLOCKS, BLOCK_SIZE>>>(din.data, double_out.data, n);
            }, repeats);
            double result = 0;
            CUDA_CHECK(cudaMemcpy(&result, double_out.data, sizeof(double), cudaMemcpyDeviceToHost));
            if (result != double(n)) throw std::runtime_error("double-finish reduction mismatch");
            std::printf("PASS reduce_ws_double_finish N=%d kernel_ms=%.6f input_GB_s=%.3f result=%.0f exact=%.0f\n",
                        n, ms, gb_per_second(bytes, ms), result, double(n));
        }
        return 0;
    });
}

