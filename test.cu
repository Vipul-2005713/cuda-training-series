#include <iostream>
#include <cuda_runtime.h>

#define CUDA_CHECK(call) do { cudaError_t status = (call); if (status != cudaSuccess) { \
    std::cerr << #call << ": " << cudaGetErrorString(status) << '\n'; return 1; } } while (0)

__global__ void cudaHello() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    int deviceCount = 0;
    CUDA_CHECK(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0) {
        std::cerr << "No CUDA-compatible devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";

    // Launch 4 threads on the GPU
    cudaHello<<<1, 4>>>();
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    std::cout << "PASS CUDA smoke test\n";

    return 0;
}
