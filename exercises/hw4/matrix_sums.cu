#include "matrix_sums_kernels.cuh"
#include "matrix_sums_driver.cuh"
// Usage: matrix_sums [matrix_side=2048] [repeats=10]
int main(int argc, char** argv) {
    return matrix_sums_main(argc, argv, {"row_sums_naive", "column_sums"}, [](const float* a, float* sums, int n, int method) {
        if (method == 0) row_sums<<<(n + 255) / 256, 256>>>(a, sums, n);
        else column_sums<<<(n + 255) / 256, 256>>>(a, sums, n);
    });
}

