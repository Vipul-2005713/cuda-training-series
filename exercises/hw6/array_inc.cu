#include "cuda_utils.cuh"

__global__ void inc(int* array, size_t n) {
    for (size_t i = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
         i < n; i += size_t(blockDim.x) * gridDim.x) ++array[i];
}
void prefetch(int* p, size_t bytes, bool to_host) {
#if CUDART_VERSION >= 13000
    cudaMemLocation location{};
    location.type = to_host ? cudaMemLocationTypeHost : cudaMemLocationTypeDevice;
    location.id = 0;
    CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, location, 0));
#else
    CUDA_CHECK(cudaMemPrefetchAsync(p, bytes, to_host ? cudaCpuDeviceId : 0));
#endif
}
void run(const std::string& mode, size_t n, int iterations, const cudaDeviceProp& prop) {
    bool managed = mode != "explicit", prefetched = mode == "prefetch";
    if (managed && !prop.managedMemory) {
        printf("SKIP mode=%s: managedMemory=0\n", mode.c_str()); return;
    }
    if (prefetched && !prop.concurrentManagedAccess) {
        printf("SKIP mode=prefetch: concurrentManagedAccess=0; demand migration/prefetch experiment unsupported\n"); return;
    }
    size_t bytes = n * sizeof(int);
    CudaBuffer<int> data(n, managed);
    std::vector<int> host(managed ? 0 : n, 0);
    int* cpu = managed ? data.get() : host.data();
    if (managed) std::fill(cpu, cpu + n, 0);
    EventTimer timer;
    auto wall_start = Clock::now();
    if (!managed) CUDA_CHECK(cudaMemcpy(data.get(), cpu, bytes, cudaMemcpyHostToDevice));
    if (prefetched) prefetch(data.get(), bytes, false);
    timer.start();
    for (int k = 0; k < iterations; ++k) inc<<<256, 256>>>(data.get(), n);
    CUDA_CHECK(cudaGetLastError());
    float kernel_ms = timer.stop();
    if (prefetched) prefetch(data.get(), bytes, true);
    if (managed) CUDA_CHECK(cudaDeviceSynchronize());
    else CUDA_CHECK(cudaMemcpy(cpu, data.get(), bytes, cudaMemcpyDeviceToHost));
    // CPU validation can itself migrate data on demand-paged UM systems.
    double ready_ms = elapsed_ms(wall_start);
    for (size_t i = 0; i < n; ++i) if (cpu[i] != iterations)
        throw std::runtime_error("array_inc mismatch at " + std::to_string(i));
    double validated_ms = elapsed_ms(wall_start);
    printf("PASS mode=%s n=%zu iterations=%d kernel_total_ms=%.6f kernel_per_launch_ms=%.6f ready_ms=%.6f validated_ms=%.6f\n",
        mode.c_str(), n, iterations, kernel_ms, kernel_ms / iterations, ready_ms, validated_ms);
}
int main(int argc, char** argv) try {
    Args args(argc, argv, {"n", "iterations", "mode"});
    size_t n = args.number("n", 4 * 1024 * 1024, 128ULL * 1024 * 1024);
    int iterations = int(args.number("iterations", 1, 1000000));
    std::string mode = args.str("mode", "all");
    if (mode != "all" && mode != "explicit" && mode != "managed" && mode != "prefetch")
        throw std::runtime_error("mode must be all, explicit, managed, or prefetch");
    cudaDeviceProp prop = device_info();
    printf("device=%s managedMemory=%d concurrentManagedAccess=%d\n", prop.name, prop.managedMemory, prop.concurrentManagedAccess);
    CudaBuffer<int> warm(1); CUDA_CHECK(cudaMemset(warm.get(), 0, sizeof(int)));
    inc<<<1, 1>>>(warm.get(), 1); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    for (const char* variant : {"explicit", "managed", "prefetch"})
        if (mode == "all" || mode == variant) run(variant, n, iterations, prop);
    return 0;
} catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
