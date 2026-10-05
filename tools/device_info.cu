#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { cudaError_t e = (call); if (e != cudaSuccess) { \
    std::fprintf(stderr, "%s: %s\n", #call, cudaGetErrorString(e)); return EXIT_FAILURE; } } while (0)

int main() {
    int count = 0, driver = 0, runtime = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::printf("driver_api_version=%d runtime_version=%d device_count=%d\n", driver, runtime, count);
    for (int d = 0; d < count; ++d) {
        cudaDeviceProp p{};
        CUDA_CHECK(cudaGetDeviceProperties(&p, d));
        CUDA_CHECK(cudaSetDevice(d));
        size_t free_bytes = 0, total_bytes = 0;
        CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
        std::printf("device=%d name=%s compute_capability=%d.%d SMs=%d\n", d, p.name, p.major, p.minor, p.multiProcessorCount);
        std::printf("total_memory_MiB=%.1f free_memory_MiB=%.1f max_threads_per_block=%d max_threads_per_SM=%d\n",
            total_bytes / 1048576.0, free_bytes / 1048576.0, p.maxThreadsPerBlock, p.maxThreadsPerMultiProcessor);
        std::printf("managed_memory=%d concurrent_managed_access=%d pageable_memory_access=%d cooperative_launch=%d\n",
            p.managedMemory, p.concurrentManagedAccess, p.pageableMemoryAccess, p.cooperativeLaunch);
        std::printf("concurrent_kernels=%d async_engines=%d memory_bus_width_bits=%d\n",
            p.concurrentKernels, p.asyncEngineCount, p.memoryBusWidth);
    }
    return count ? EXIT_SUCCESS : EXIT_FAILURE;
}
