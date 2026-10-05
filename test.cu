#include <iostream>
#include <cuda_runtime.h>

__global__ void cudaHello() {
    printf("Hello from GPU thread %d!\n", threadIdx.x);
}

int main() {
    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);

    if (deviceCount == 0) {
        std::cerr << "No CUDA-compatible devices found!\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024 * 1024) << " MB\n";

    // Launch 4 threads on the GPU
    cudaHello<<<1, 4>>>();
    cudaDeviceSynchronize();

    return 0;
}