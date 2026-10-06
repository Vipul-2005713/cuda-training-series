#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <cerrno>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

// Check the returned status rather than an unrelated later last-error value.
inline void check_cuda(cudaError_t status, const char* operation, int line) {
    if (status != cudaSuccess)
        throw std::runtime_error(std::string(operation) + " at line " +
                                 std::to_string(line) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(call) check_cuda((call), #call, __LINE__)

inline int int_arg(int argc, char** argv, int position, int fallback,
                   int minimum = 1, int maximum = 67108864) {
    if (position >= argc) return fallback;
    char* end = nullptr;
    errno = 0;
    const long long value = std::strtoll(argv[position], &end, 10);
    if (errno || !end || *end || end == argv[position] || value < minimum || value > maximum)
        throw std::invalid_argument("invalid integer argument " + std::to_string(position));
    return static_cast<int>(value);
}

template<class T> struct DeviceBuffer {
    T* data = nullptr;
    explicit DeviceBuffer(size_t n) { CUDA_CHECK(cudaMalloc(&data, n * sizeof(T))); }
    ~DeviceBuffer() { if (data) cudaFree(data); }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
};

struct Events {
    cudaEvent_t start = nullptr, stop = nullptr;
    Events() { CUDA_CHECK(cudaEventCreate(&start)); CUDA_CHECK(cudaEventCreate(&stop)); }
    ~Events() { if (start) cudaEventDestroy(start); if (stop) cudaEventDestroy(stop); }
};

// One untimed warm-up; average of repeat launches excludes transfers and CPU work.
template<class Launch> float kernel_ms(Launch launch, int repeats) {
    launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    Events events;
    CUDA_CHECK(cudaEventRecord(events.start));
    for (int i = 0; i < repeats; ++i) { launch(); CUDA_CHECK(cudaGetLastError()); }
    CUDA_CHECK(cudaEventRecord(events.stop));
    CUDA_CHECK(cudaEventSynchronize(events.stop));
    float elapsed = 0;
    CUDA_CHECK(cudaEventElapsedTime(&elapsed, events.start, events.stop));
    return elapsed / repeats;
}

// Reset is needed for atomic reductions and is excluded from the measured interval.
template<class Reset, class Launch>
float reset_kernel_ms(Reset reset, Launch launch, int repeats) {
    reset(); launch(); CUDA_CHECK(cudaGetLastError()); CUDA_CHECK(cudaDeviceSynchronize());
    Events events;
    float total = 0;
    for (int i = 0; i < repeats; ++i) {
        reset();
        CUDA_CHECK(cudaEventRecord(events.start));
        launch(); CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(events.stop));
        CUDA_CHECK(cudaEventSynchronize(events.stop));
        float elapsed = 0;
        CUDA_CHECK(cudaEventElapsedTime(&elapsed, events.start, events.stop));
        total += elapsed;
    }
    return total / repeats;
}

template<class Work> int checked_main(Work work) {
    try { return work(); }
    catch (const std::exception& e) { std::fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
}
inline double gb_per_second(size_t bytes, float milliseconds) {
    return static_cast<double>(bytes) / (milliseconds * 1.0e6);
}

