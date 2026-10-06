#pragma once
#include "../hw1/cuda_helpers.cuh"
template<class Launch>
int matrix_sums_main(int argc, char** argv, const std::vector<const char*>& names, Launch launch) {
    return checked_main([&] {
        const int n = int_arg(argc, argv, 1, 2048, 1, 16384);
        const int repeats = int_arg(argc, argv, 2, 10, 1, 10000);
        const size_t count = size_t(n) * n, bytes = count * sizeof(float);
        std::vector<float> a(count), output(n), row_reference(n, 0), column_reference(n, 0);
        for (int row = 0; row < n; ++row)
            for (int col = 0; col < n; ++col) {
                const float value = float((row % 17 - 8) + (col % 13 - 6));
                a[size_t(row) * n + col] = value;
                row_reference[row] += value;
                column_reference[col] += value;
            }
        DeviceBuffer<float> da(count), dout(n);
        CUDA_CHECK(cudaMemcpy(da.data, a.data(), bytes, cudaMemcpyHostToDevice));
        for (size_t method = 0; method < names.size(); ++method) {
            const float ms = kernel_ms([&] { launch(da.data, dout.data, n, int(method)); }, repeats);
            CUDA_CHECK(cudaMemcpy(output.data(), dout.data, size_t(n) * sizeof(float), cudaMemcpyDeviceToHost));
            // The column algorithm is always the final entry.
            const auto& reference = method + 1 == names.size() ? column_reference : row_reference;
            for (int i = 0; i < n; ++i)
                if (output[i] != reference[i]) throw std::runtime_error(std::string(names[method]) + " mismatch at " + std::to_string(i));
            std::printf("PASS %s N=%d repeats=%d kernel_ms=%.6f effective_GB_s=%.3f checked=%d\n",
                        names[method], n, repeats, ms, gb_per_second(bytes + size_t(n) * sizeof(float), ms), n);
        }
        return 0;
    });
}

