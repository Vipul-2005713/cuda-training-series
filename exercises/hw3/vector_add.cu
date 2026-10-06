#include "../hw1/cuda_helpers.cuh"
__global__ void vadd(const float* a, const float* b, float* c, int n) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        c[i] = a[i] + b[i];
}
// Usage: vector_add [elements=262144] [blocks=160] [threads=1024] [repeats=5]
int main(int argc, char** argv) {
    return checked_main([&] {
        const int n = int_arg(argc, argv, 1, 262144);
        const int blocks = int_arg(argc, argv, 2, 160, 1, 65535);
        const int threads = int_arg(argc, argv, 3, 1024, 1, 1024);
        const int repeats = int_arg(argc, argv, 4, 5, 1, 10000);
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
        if (threads > prop.maxThreadsPerBlock) throw std::invalid_argument("threads exceed device limit");
        std::printf("device=%s SMs=%d max_threads_per_SM=%d\n", prop.name, prop.multiProcessorCount, prop.maxThreadsPerMultiProcessor);
        const size_t bytes = size_t(n) * sizeof(float);
        std::vector<float> a(n), b(n), c(n);
        for (int i = 0; i < n; ++i) {
            a[i] = float(i % 101 - 50) / 8;
            b[i] = float(i % 37 - 18) / 4;
        }
        DeviceBuffer<float> da(n), db(n), dc(n);
        CUDA_CHECK(cudaMemcpy(da.data, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db.data, b.data(), bytes, cudaMemcpyHostToDevice));
        const float ms = kernel_ms([&] { vadd<<<blocks, threads>>>(da.data, db.data, dc.data, n); }, repeats);
        CUDA_CHECK(cudaMemcpy(c.data(), dc.data, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i)
            if (c[i] != a[i] + b[i]) throw std::runtime_error("vector mismatch at " + std::to_string(i));
        std::printf("PASS grid_stride_vadd N=%d blocks=%d threads=%d repeats=%d kernel_ms=%.6f effective_GB_s=%.3f checked=%d\n",
                    n, blocks, threads, repeats, ms, gb_per_second(3 * bytes, ms), n);
        return 0;
    });
}

