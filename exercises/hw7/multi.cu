#include "gaussian.cuh"

struct Job {
    int device;
    float *x = nullptr, *y = nullptr;
    explicit Job(int d, size_t n) : device(d) {
        CUDA_CHECK(cudaSetDevice(device));
        CUDA_CHECK(cudaMalloc(&x, n * sizeof(float)));
        try { CUDA_CHECK(cudaMalloc(&y, n * sizeof(float))); }
        catch (...) { cudaFree(x); throw; }
    }
    ~Job() { cudaSetDevice(device); cudaFree(x); cudaFree(y); }
};
double benchmark(int devices, size_t n, int repeats) {
    std::vector<std::unique_ptr<Job>> jobs;
    std::vector<float> host(n);
    for (int j = 0; j < 4; ++j) {
        jobs.emplace_back(new Job(j % devices, n));
        for (size_t i = 0; i < n; ++i) host[i] = float((i + j) % 4096) / 4096;
        CUDA_CHECK(cudaMemcpy(jobs.back()->x, host.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    }
    auto launch_all = [&]() {
        for (const auto& job : jobs) {
            CUDA_CHECK(cudaSetDevice(job->device));
            gaussian_pdf<<<unsigned((n + 255) / 256), 256>>>(job->x, job->y, n);
            CUDA_CHECK(cudaGetLastError());
        }
        // cudaDeviceSynchronize only fences the CURRENT device, so fence every GPU.
        for (int d = 0; d < devices; ++d) {
            CUDA_CHECK(cudaSetDevice(d)); CUDA_CHECK(cudaDeviceSynchronize());
        }
    };
    launch_all();
    auto start = Clock::now();
    for (int r = 0; r < repeats; ++r) launch_all();
    double ms = elapsed_ms(start) / repeats;
    for (int j = 0; j < 4; ++j) {
        CUDA_CHECK(cudaSetDevice(jobs[j]->device));
        CUDA_CHECK(cudaMemcpy(host.data(), jobs[j]->y, n * sizeof(float), cudaMemcpyDeviceToHost));
        verify_gaussian(host.data(), n, j);
    }
    printf("PASS multi devices=%d jobs=4 n_per_job=%zu repeats=%d compute_wall_ms=%.6f\n", devices, n, repeats, ms);
    return ms;
}
int main(int argc, char** argv) try {
    Args args(argc, argv, {"n", "repeats"});
    size_t n = args.number("n", 1024 * 1024, 64ULL * 1024 * 1024);
    int repeats = int(args.number("repeats", 5, 1000));
    int devices; CUDA_CHECK(cudaGetDeviceCount(&devices));
    if (!devices) throw std::runtime_error("no CUDA devices");
    double single = benchmark(1, n, repeats);
    if (devices < 4) printf("SKIP four-GPU experiment: requires 4 CUDA GPUs; found %d\n", devices);
    else printf("four_gpu_speedup=%.3f\n", single / benchmark(4, n, repeats));
    return 0;
} catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
