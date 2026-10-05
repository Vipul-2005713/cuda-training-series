#include "gaussian.cuh"

int main(int argc, char** argv) try {
    Args args(argc, argv, {"n", "chunks", "streams", "repeats"});
    size_t n = args.number("n", 8 * 1024 * 1024, 128ULL * 1024 * 1024);
    int chunks = int(args.number("chunks", 32, 65536));
    int count = int(args.number("streams", 8, 128));
    int repeats = int(args.number("repeats", 5, 1000));
    PinnedBuffer<float> input(n), baseline(n), output(n);
    CudaBuffer<float> dx(n), dy(n);
    std::vector<std::unique_ptr<Stream>> streams;
    for (int i = 0; i < count; ++i) streams.emplace_back(new Stream);
    for (size_t i = 0; i < n; ++i) input.get()[i] = float(i % 4096) / 4096;
    const size_t bytes = n * sizeof(float);
    auto serial = [&]() {
        CUDA_CHECK(cudaMemcpy(dx.get(), input.get(), bytes, cudaMemcpyHostToDevice));
        gaussian_pdf<<<unsigned((n + 255) / 256), 256>>>(dx.get(), dy.get(), n);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(baseline.get(), dy.get(), bytes, cudaMemcpyDeviceToHost));
    };
    auto streamed = [&]() {
        // Integer partition boundaries include every element, including a tail chunk.
        for (int c = 0; c < chunks; ++c) {
            size_t begin = n * c / chunks, end = n * (c + 1) / chunks, length = end - begin;
            if (!length) continue;
            cudaStream_t s = streams[c % count]->get();
            CUDA_CHECK(cudaMemcpyAsync(dx.get() + begin, input.get() + begin, length * sizeof(float), cudaMemcpyHostToDevice, s));
            gaussian_pdf<<<unsigned((length + 255) / 256), 256, 0, s>>>(dx.get() + begin, dy.get() + begin, length);
            CUDA_CHECK(cudaGetLastError());
            CUDA_CHECK(cudaMemcpyAsync(output.get() + begin, dy.get() + begin, length * sizeof(float), cudaMemcpyDeviceToHost, s));
        }
        // Fence ALL streams before stopping the host wall-clock timer or reading output.
        CUDA_CHECK(cudaDeviceSynchronize());
    };
    serial(); streamed(); // Initialized warm-up for each complete schedule.
    double serial_ms = 0, streams_ms = 0;
    for (int r = 0; r < repeats; ++r) {
        auto start = Clock::now(); serial(); serial_ms += elapsed_ms(start);
        start = Clock::now(); streamed(); streams_ms += elapsed_ms(start);
    }
    verify_gaussian(baseline.get(), n);
    verify_gaussian(output.get(), n);
    for (size_t i = 0; i < n; ++i) if (output.get()[i] != baseline.get()[i])
        throw std::runtime_error("streamed versus sequential mismatch at " + std::to_string(i));
    auto prop = device_info();
    printf("PASS overlap n=%zu chunks=%d streams=%d repeats=%d asyncEngineCount=%d\n", n, chunks, count, repeats, prop.asyncEngineCount);
    printf("sequential_ms=%.6f streams_ms=%.6f speedup=%.3f (host wall clock; H2D+kernel+D2H included)\n",
        serial_ms / repeats, streams_ms / repeats, serial_ms / streams_ms);
    return 0;
} catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
