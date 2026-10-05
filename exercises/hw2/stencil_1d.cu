#include "../hw1/cuda_helpers.cuh"
constexpr int RADIUS = 3;
constexpr int BLOCK_SIZE = 16;
// in/out point to the first interior element; each allocation also has two halos.
__global__ void stencil_1d(const int* in, int* out, int n) {
    __shared__ int temp[BLOCK_SIZE + 2 * RADIUS];
    const int base = blockIdx.x * BLOCK_SIZE;
    // Cooperative loads include both halos. Bounds also protect the partial last block.
    for (int local = threadIdx.x; local < BLOCK_SIZE + 2 * RADIUS; local += BLOCK_SIZE) {
        const int index = base + local - RADIUS;
        temp[local] = index < n + RADIUS ? in[index] : 0;
    }
    __syncthreads();
    const int index = base + threadIdx.x;
    if (index < n) {
        int sum = 0;
        for (int offset = -RADIUS; offset <= RADIUS; ++offset)
            sum += temp[threadIdx.x + RADIUS + offset];
        out[index] = sum;
    }
}
// Usage: stencil_1d [interior_elements=1048576] [repeats=20]
int main(int argc, char** argv) {
    return checked_main([&] {
        const int n = int_arg(argc, argv, 1, 1048576);
        const int repeats = int_arg(argc, argv, 2, 20, 1, 10000);
        const size_t count = size_t(n) + 2 * RADIUS, bytes = count * sizeof(int);
        std::vector<int> input(count), output(count, 1);
        for (size_t i = 0; i < count; ++i) input[i] = int(i % 17) - 8;
        DeviceBuffer<int> din(count), dout(count);
        CUDA_CHECK(cudaMemcpy(din.data, input.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(dout.data, output.data(), bytes, cudaMemcpyHostToDevice));
        const float ms = kernel_ms([&] {
            stencil_1d<<<(n + BLOCK_SIZE - 1) / BLOCK_SIZE, BLOCK_SIZE>>>(din.data + RADIUS, dout.data + RADIUS, n);
        }, repeats);
        CUDA_CHECK(cudaMemcpy(output.data(), dout.data, bytes, cudaMemcpyDeviceToHost));
        for (size_t i = 0; i < count; ++i) {
            int expected = 1; // output halos must be untouched
            if (i >= RADIUS && i < size_t(n) + RADIUS) {
                expected = 0;
                for (int offset = -RADIUS; offset <= RADIUS; ++offset) expected += input[size_t(static_cast<long long>(i) + offset)];
            }
            if (output[i] != expected) throw std::runtime_error("stencil mismatch at " + std::to_string(i));
        }
        std::printf("PASS stencil_1d N=%d repeats=%d kernel_ms=%.6f checked=%zu (including halos)\n", n, repeats, ms, count);
        return 0;
    });
}

