#include "matrix_driver.cuh"
__global__ void mmul(const float* a, const float* b, float* c, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < n && col < n) {
        float sum = 0;
        for (int k = 0; k < n; ++k) sum += a[size_t(row) * n + k] * b[size_t(k) * n + col];
        c[size_t(row) * n + col] = sum;
    }
}
// Usage: matrix_mul [matrix_side=512] [repeats=10]
int main(int argc, char** argv) {
    return matrix_main(argc, argv, "matrix_mul_naive", [](const float* a, const float* b, float* c, int n) {
        mmul<<<dim3((n + 15) / 16, (n + 15) / 16), dim3(16, 16)>>>(a, b, c, n);
    });
}

