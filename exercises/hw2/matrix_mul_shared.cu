#include "../hw1/matrix_driver.cuh"
constexpr int TILE = 16;
__global__ void mmul_shared(const float* a, const float* b, float* c, int n) {
    __shared__ float as[TILE][TILE], bs[TILE][TILE];
    const int col = blockIdx.x * TILE + threadIdx.x;
    const int row = blockIdx.y * TILE + threadIdx.y;
    float sum = 0;
    for (int tile = 0; tile < (n + TILE - 1) / TILE; ++tile) {
        const int ak = tile * TILE + threadIdx.x;
        const int bk = tile * TILE + threadIdx.y;
        as[threadIdx.y][threadIdx.x] = row < n && ak < n ? a[size_t(row) * n + ak] : 0;
        bs[threadIdx.y][threadIdx.x] = bk < n && col < n ? b[size_t(bk) * n + col] : 0;
        // Every thread, including those outside the matrix, must reach both barriers.
        __syncthreads();
        for (int k = 0; k < TILE; ++k) sum += as[threadIdx.y][k] * bs[k][threadIdx.x];
        __syncthreads(); // prevent the next tile overwriting data still being consumed
    }
    if (row < n && col < n) c[size_t(row) * n + col] = sum;
}
// Usage: matrix_mul_shared [matrix_side=512] [repeats=10]
int main(int argc, char** argv) {
    return matrix_main(argc, argv, "matrix_mul_shared", [](const float* a, const float* b, float* c, int n) {
        mmul_shared<<<dim3((n + TILE - 1) / TILE, (n + TILE - 1) / TILE), dim3(TILE, TILE)>>>(a, b, c, n);
    });
}

