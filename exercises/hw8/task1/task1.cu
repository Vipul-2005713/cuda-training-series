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

__global__ void naive_cuda_transpose(int n, const double* a, double* b) {
    int row = blockIdx.x * 32 + threadIdx.x;
    int col = blockIdx.y * 32 + threadIdx.y;
    if (row < n && col < n) b[INDX(col, row, n)] = a[INDX(row, col, n)];
}

void launch(int n, const double* a, double* b, dim3 grid, dim3 block) {
    naive_cuda_transpose<<<grid, block>>>(n, a, b);
}
int main(int argc, char** argv) try {
    return transpose_benchmark(argc, argv, "naive", launch);
} catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }

