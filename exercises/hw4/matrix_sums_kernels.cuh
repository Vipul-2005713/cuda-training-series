#pragma once
#include "../hw1/cuda_helpers.cuh"
// Adjacent lanes visit DIFFERENT rows: their same-instruction loads are strided.
__global__ void row_sums(const float* a, float* sums, int n) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < n) {
        float sum = 0;
        for (int col = 0; col < n; ++col) sum += a[size_t(row) * n + col];
        sums[row] = sum;
    }
}
// Adjacent lanes visit adjacent columns, so their loads are coalesced.
__global__ void column_sums(const float* a, float* sums, int n) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col < n) {
        float sum = 0;
        for (int row = 0; row < n; ++row) sum += a[size_t(row) * n + col];
        sums[col] = sum;
    }
}

