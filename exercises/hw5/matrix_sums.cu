#include "../hw4/matrix_sums_kernels.cuh"
#include "../hw4/matrix_sums_driver.cuh"
constexpr int BLOCK_SIZE = 256;
// One block per row gives coalesced loads and many more independent workers.
__global__ void row_sums_block(const float* a, float* sums, int n) {
    __shared__ float partial[BLOCK_SIZE];
    const int row = blockIdx.x, tid = threadIdx.x;
    float value = 0;
    for (int col = tid; col < n; col += BLOCK_SIZE) value += a[size_t(row) * n + col];
    partial[tid] = value;
    __syncthreads();
    for (int stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
        if (tid < stride) partial[tid] += partial[tid + stride];
        __syncthreads();
    }
    if (tid == 0) sums[row] = partial[0];
}
// Usage: matrix_sums [matrix_side=2048] [repeats=10]
int main(int argc, char** argv) {
    return matrix_sums_main(argc, argv, {"row_sums_naive", "row_sums_block", "column_sums"}, [](const float* a, float* sums, int n, int method) {
        if (method == 0) row_sums<<<(n + 255) / 256, 256>>>(a, sums, n);
        else if (method == 1) row_sums_block<<<n, BLOCK_SIZE>>>(a, sums, n);
        else column_sums<<<(n + 255) / 256, 256>>>(a, sums, n);
    });
}

