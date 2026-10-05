#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

inline void cuda_check(cudaError_t status, const char* call, int line) {
    if (status != cudaSuccess) throw std::runtime_error(std::string(call) +
        " at line " + std::to_string(line) + ": " + cudaGetErrorString(status));
}
#define CUDA_CHECK(call) cuda_check((call), #call, __LINE__)

// Shared utilities for HW6--9. CUDA resources are released on failure too.
template<class T> class CudaBuffer {
    T* data_ = nullptr;
public:
    explicit CudaBuffer(size_t n, bool managed = false) {
        if (n == 0 || n > std::numeric_limits<size_t>::max() / sizeof(T))
            throw std::runtime_error("invalid allocation size");
        if (managed) CUDA_CHECK(cudaMallocManaged(&data_, n * sizeof(T)));
        else CUDA_CHECK(cudaMalloc(&data_, n * sizeof(T)));
    }
    ~CudaBuffer() { if (data_) cudaFree(data_); }
    CudaBuffer(const CudaBuffer&) = delete;
    CudaBuffer& operator=(const CudaBuffer&) = delete;
    T* get() const { return data_; }
};
template<class T> class PinnedBuffer {
    T* data_ = nullptr;
public:
    explicit PinnedBuffer(size_t n) { CUDA_CHECK(cudaMallocHost(&data_, n * sizeof(T))); }
    ~PinnedBuffer() { if (data_) cudaFreeHost(data_); }
    PinnedBuffer(const PinnedBuffer&) = delete;
    T* get() const { return data_; }
};
class EventTimer {
    cudaEvent_t start_ = nullptr, stop_ = nullptr;
public:
    EventTimer() {
        CUDA_CHECK(cudaEventCreate(&start_));
        try { CUDA_CHECK(cudaEventCreate(&stop_)); }
        catch (...) { cudaEventDestroy(start_); throw; }
    }
    ~EventTimer() { cudaEventDestroy(start_); cudaEventDestroy(stop_); }
    void start(cudaStream_t stream = nullptr) { CUDA_CHECK(cudaEventRecord(start_, stream)); }
    float stop(cudaStream_t stream = nullptr) {
        CUDA_CHECK(cudaEventRecord(stop_, stream));
        CUDA_CHECK(cudaEventSynchronize(stop_));
        float ms; CUDA_CHECK(cudaEventElapsedTime(&ms, start_, stop_)); return ms;
    }
};
class Stream {
    cudaStream_t stream_ = nullptr;
public:
    Stream() { CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, cudaStreamNonBlocking)); }
    ~Stream() { cudaStreamDestroy(stream_); }
    Stream(const Stream&) = delete;
    cudaStream_t get() const { return stream_; }
};
using Clock = std::chrono::steady_clock;
inline double elapsed_ms(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}
class Args {
    std::map<std::string, std::string> values_;
public:
    Args(int argc, char** argv, std::initializer_list<const char*> allowed) {
        for (int i = 1; i < argc; i += 2) {
            std::string key = argv[i];
            if (key.size() < 3 || key.substr(0, 2) != "--" || i + 1 == argc)
                throw std::runtime_error("options require --name value pairs");
            key = key.substr(2);
            bool valid = false;
            for (const char* option : allowed) if (key == option) valid = true;
            if (!valid || values_.count(key)) throw std::runtime_error("invalid/duplicate option: " + key);
            values_[key] = argv[i + 1];
        }
    }
    std::string str(const char* key, const char* fallback) const {
        auto it = values_.find(key); return it == values_.end() ? fallback : it->second;
    }
    size_t number(const char* key, size_t fallback, size_t max = 2147483647ULL) const {
        auto it = values_.find(key); if (it == values_.end()) return fallback;
        const std::string& value = it->second;
        if (value.empty() || value.find_first_not_of("0123456789") != std::string::npos)
            throw std::runtime_error(std::string("invalid numeric option: ") + key);
        size_t consumed = 0; unsigned long long n = std::stoull(value, &consumed);
        if (!n || n > max || consumed != value.size())
            throw std::runtime_error(std::string("option out of range: ") + key);
        return static_cast<size_t>(n);
    }
};
inline cudaDeviceProp device_info() {
    cudaDeviceProp p{}; int device; CUDA_CHECK(cudaGetDevice(&device));
    CUDA_CHECK(cudaGetDeviceProperties(&p, device)); return p;
}
