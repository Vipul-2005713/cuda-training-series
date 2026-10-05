#pragma once
#include "cuda_helpers.cuh"

// Nonuniform integer inputs detect swapped indices. A[r,k]=ar[r]+ak[k],
// B[k,c]=bk[k]+bc[c]. An independently expanded dot product checks EVERY
// output in O(N^2), even at original sizes. Tiny cases also use direct CPU dots.
template<class Launch> int matrix_main(int argc, char** argv, const char* name, Launch launch) {
    return checked_main([&] {
        const int n = int_arg(argc, argv, 1, 512, 1, 8192);
        const int repeats = int_arg(argc, argv, 2, 10, 1, 10000);
        const size_t count = size_t(n) * n, bytes = count * sizeof(float);
        std::vector<float> a(count), b(count), c(count);
        double sum_ak = 0, sum_bk = 0, sum_product = 0;
        for (int k = 0; k < n; ++k) {
            const int ak = k % 5 - 2, bk = k % 11 - 5;
            sum_ak += ak; sum_bk += bk; sum_product += ak * bk;
        }
        for (int r = 0; r < n; ++r)
            for (int col = 0; col < n; ++col) {
                a[size_t(r) * n + col] = float((r % 7 - 3) + (col % 5 - 2));
                b[size_t(r) * n + col] = float((r % 11 - 5) + (col % 13 - 6));
            }
        DeviceBuffer<float> da(count), db(count), dc(count);
        CUDA_CHECK(cudaMemcpy(da.data, a.data(), bytes, cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(db.data, b.data(), bytes, cudaMemcpyHostToDevice));
        const float ms = kernel_ms([&] { launch(da.data, db.data, dc.data, n); }, repeats);
        CUDA_CHECK(cudaMemcpy(c.data(), dc.data, bytes, cudaMemcpyDeviceToHost));
        for (int r = 0; r < n; ++r)
            for (int col = 0; col < n; ++col) {
                const double ar = r % 7 - 3, bc = col % 13 - 6;
                const double expected = ar * sum_bk + n * ar * bc + sum_product + bc * sum_ak;
                if (double(c[size_t(r) * n + col]) != expected)
                    throw std::runtime_error("matrix mismatch at row " + std::to_string(r) + ", column " + std::to_string(col));
            }
        if (n <= 129) {
            for (int r = 0; r < n; ++r)
                for (int col = 0; col < n; ++col) {
                    double expected = 0;
                    for (int k = 0; k < n; ++k) expected += double(a[size_t(r) * n + k]) * b[size_t(k) * n + col];
                    if (double(c[size_t(r) * n + col]) != expected)
                        throw std::runtime_error("CPU dot-product mismatch");
                }
        }
        std::printf("PASS %s N=%d repeats=%d kernel_ms=%.6f GFLOP_s=%.3f checked=%zu\n",
                    name, n, repeats, ms, 2.0 * n * n * n / (ms * 1.0e6), count);
        return 0;
    });
}

