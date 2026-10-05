#pragma once
#include "../hw6/cuda_utils.cuh"

// Original exercises use COLUMN-major storage: row varies fastest.
#define INDX(row, col, ld) (size_t(col) * (ld) + (row))
using TransposeLaunch = void (*)(int, const double*, double*, dim3, dim3);
inline int transpose_benchmark(int argc, char** argv, const char* variant, TransposeLaunch launch) {
    Args args(argc, argv, {"size", "iterations"});
    int n = int(args.number("size", 2048, 8192));
    int iterations = int(args.number("iterations", 20, 10000));
    size_t count = size_t(n) * n, bytes = count * sizeof(double);
    std::vector<double> input(count), reference(count), output(count);
    for (size_t i = 0; i < count; ++i) input[i] = double(i); // Unique exact values catch misplaced elements.
    auto cpu_start = Clock::now();
    for (int col = 0; col < n; ++col)
        for (int row = 0; row < n; ++row) reference[INDX(row, col, n)] = input[INDX(col, row, n)];
    double cpu_ms = elapsed_ms(cpu_start);
    CudaBuffer<double> d_input(count), d_output(count);
    CUDA_CHECK(cudaMemcpy(d_input.get(), input.data(), bytes, cudaMemcpyHostToDevice));
    dim3 block(32, 32), grid((n + 31) / 32, (n + 31) / 32);
    launch(n, d_input.get(), d_output.get(), grid, block);
    CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    EventTimer timer;
    timer.start();
    for (int r = 0; r < iterations; ++r) launch(n, d_input.get(), d_output.get(), grid, block);
    CUDA_CHECK(cudaGetLastError());
    double ms = timer.stop() / iterations;
    CUDA_CHECK(cudaMemcpy(output.data(), d_output.get(), bytes, cudaMemcpyDeviceToHost));
    for (size_t i = 0; i < count; ++i) if (output[i] != reference[i])
        throw std::runtime_error("transpose mismatch at linear index " + std::to_string(i));
    CUDA_CHECK(cudaMemcpyAsync(d_output.get(), d_input.get(), bytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
    timer.start();
    for (int r = 0; r < iterations; ++r)
        CUDA_CHECK(cudaMemcpyAsync(d_output.get(), d_input.get(), bytes, cudaMemcpyDeviceToDevice));
    double copy_ms = timer.stop() / iterations;
    CUDA_CHECK(cudaMemcpy(output.data(), d_output.get(), bytes, cudaMemcpyDeviceToHost));
    if (output != input) throw std::runtime_error("D2D proxy mismatch");
    printf("PASS transpose variant=%s size=%d iterations=%d cpu_ms=%.6f kernel_ms=%.6f effective_GBps=%.3f\n",
        variant, n, iterations, cpu_ms, ms, 2.0 * bytes / (ms * 1e6));
    printf("D2D_copy_ms=%.6f D2D_effective_GBps=%.3f transpose_to_copy_ratio=%.3f\n",
        copy_ms, 2.0 * bytes / (copy_ms * 1e6), copy_ms / ms);
    return 0;
}
