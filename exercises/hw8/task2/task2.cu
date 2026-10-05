/*
 * Copyright 2014 NVIDIA Corporation
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#include "../transpose_benchmark.cuh"

__global__ void smem_cuda_transpose(int n, const double* a, double* b) {
    __shared__ double tile[32][32];
    int x = threadIdx.x, y = threadIdx.y;
    int row = blockIdx.x * 32 + x, col = blockIdx.y * 32 + y;
    if (row < n && col < n) tile[x][y] = a[INDX(row, col, n)];
    __syncthreads(); // Every thread participates, including partial edge tiles.
    int out_row = blockIdx.y * 32 + x, out_col = blockIdx.x * 32 + y;
    // The output predicate must use TRANSPOSED coordinates on boundary tiles.
    if (out_row < n && out_col < n) b[INDX(out_row, out_col, n)] = tile[y][x];
}

void launch(int n, const double* a, double* b, dim3 grid, dim3 block) {
    smem_cuda_transpose<<<grid, block>>>(n, a, b);
}
int main(int argc, char** argv) try {
    return transpose_benchmark(argc, argv, "shared_unpadded", launch);
} catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }

