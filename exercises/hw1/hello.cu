#include "cuda_helpers.cuh"
__global__ void hello() {
    printf("Hello from block: %u, thread: %u\n", blockIdx.x, threadIdx.x);
}
int main() {
    return checked_main([] {
        hello<<<2, 2>>>();
        CUDA_CHECK(cudaGetLastError());
        // Launches are asynchronous; wait for all four printf calls to complete.
        CUDA_CHECK(cudaDeviceSynchronize());
        return 0;
    });
}

