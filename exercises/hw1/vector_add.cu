#include "cuda_helpers.cuh"
__global__ void vadd(const float* a, const float* b, float* c, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) c[i] = a[i] + b[i];
}
// Usage: vector_add [elements=1048576] [repeats=20]
int main(int argc, char** argv) {
    return checked_main([&] {
        const int n = int_arg(argc, argv, 1, 1048576);
        const int repeats = int_arg(argc, argv, 2, 20, 1, 10000);
        const size_t bytes = size_t(n) * sizeof(float);
        std::vector<float> a(n), b(n), c(n);
        for (int i = 0; i < n; ++i) {
            a[i] = float(i % 101 - 50) / 8;
            b[i] = float(i % 37 - 18) / 4;
        }
        DeviceBuffer<float> da(n), db(n), dc(n);
        CUDA_CHECK(cudaMemcpy(da.data, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db.data, b.data(), bytes, cudaMemcpyHostToDevice));
        const float ms = kernel_ms([&] { vadd<<<(n + 255) / 256, 256>>>(da.data, db.data, dc.data, n); }, repeats);
        CUDA_CHECK(cudaMemcpy(c.data(), dc.data, bytes, cudaMemcpyDeviceToHost));
        for (int i = 0; i < n; ++i)
            if (c[i] != a[i] + b[i]) throw std::runtime_error("vector mismatch at " + std::to_string(i));
        std::printf("A[0]=%g B[0]=%g C[0]=%g\n", a[0], b[0], c[0]);
        std::printf("PASS vector_add N=%d repeats=%d kernel_ms=%.6f effective_GB_s=%.3f checked=%d\n",
                    n, repeats, ms, gb_per_second(3 * bytes, ms), n);
        return 0;
    });
}

