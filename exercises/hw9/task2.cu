#include "../hw6/cuda_utils.cuh"
#include <cooperative_groups.h>
#include <thrust/device_ptr.h>
#include <thrust/remove.h>
#include <thrust/system/cuda/execution_policy.h>
namespace cg = cooperative_groups;
constexpr unsigned threads = 256;

// Linear=false completes the original exercise: one grid barrier, with each
// survivor summing preceding tile totals. Linear=true adds a scalable extension.
template<bool Linear>
__global__ void compact(const int* input, int* output, unsigned* indices,
                        unsigned* totals, unsigned* count, unsigned n) {
    __shared__ unsigned scratch[threads];
    auto block = cg::this_thread_block();
    auto grid = cg::this_grid();
    unsigned lane = block.thread_rank();
    unsigned grid_stride = gridDim.x * threads;
    for (unsigned base = blockIdx.x * threads; base < n; base += grid_stride) {
        unsigned i = base + lane;
        unsigned value = i < n && input[i] != -1 ? 1 : 0;
        scratch[lane] = value;
        for (unsigned step = 1; step < threads; step <<= 1) {
            block.sync();
            if (lane >= step) value += scratch[lane - step];
            block.sync();
            scratch[lane] = value;
        }
        if (i < n) indices[i] = value;
        if (lane == threads - 1) totals[base / threads] = value;
        block.sync(); // Last readers finish before this block reuses scratch.
    }
    grid.sync(); // Every tile's inclusive scan and total must now be visible.
    if constexpr (Linear) {
        // Extra exercise: one thread computes tile offsets in O(number of tiles).
        if (grid.thread_rank() == 0) {
            unsigned sum = 0;
            for (unsigned tile = 0; tile < (n + threads - 1) / threads; ++tile) {
                unsigned next = totals[tile]; totals[tile] = sum; sum += next;
            }
            *count = sum;
        }
        grid.sync();
    }
    for (unsigned base = blockIdx.x * threads; base < n; base += grid_stride) {
        unsigned i = base + lane;
        if (i < n && (input[i] != -1 || (!Linear && i == n - 1))) {
            unsigned offset = 0, tile = base / threads;
            if constexpr (Linear) offset = totals[tile];
            else for (unsigned previous = 0; previous < tile; ++previous) offset += totals[previous];
            unsigned position = offset + indices[i];
            if (input[i] != -1) output[position - 1] = input[i];
            if (!Linear && i == n - 1) *count = position;
        }
    }
}

template<bool Linear>
void run_variant(unsigned n, int iterations, const cudaDeviceProp& prop) {
    unsigned tiles = (n + threads - 1) / threads;
    CudaBuffer<int> input(n), output(n), library(n);
    CudaBuffer<unsigned> indices(n), totals(tiles), count(1);
    Stream stream; EventTimer timer;
    int blocks_per_sm;
    CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&blocks_per_sm, compact<Linear>, threads, 0));
    if (blocks_per_sm <= 0) throw std::runtime_error("no resident cooperative blocks");
    unsigned blocks = std::min(tiles, unsigned(blocks_per_sm * prop.multiProcessorCount));
    int* in = input.get(); int* out = output.get(); unsigned* idx = indices.get();
    unsigned* tile_totals = totals.get(); unsigned* result_count = count.get();
    void* params[] = {&in, &out, &idx, &tile_totals, &result_count, &n};
    auto launch = [&]() {
        CUDA_CHECK(cudaLaunchCooperativeKernel((void*)compact<Linear>, dim3(blocks), dim3(threads), params, 0, stream.get()));
    };
    std::vector<int> host(n), expected, actual(n);
    float custom_ms = 0, library_ms = 0;
    // Verify stable ordering and exact length for no-removal, all-removal, mixed,
    // first-survivor-only and last-survivor-only inputs (including a partial tile).
    for (int pattern = 0; pattern < 5; ++pattern) {
        expected.clear();
        for (unsigned i = 0; i < n; ++i) {
            bool keep = pattern == 0 || (pattern == 2 && (i * 17U + 11U) % 7 < 3)
                || (pattern == 3 && i == 0) || (pattern == 4 && i == n - 1);
            host[i] = keep ? int(i) : -1;
            if (keep) expected.push_back(int(i));
        }
        CUDA_CHECK(cudaMemcpyAsync(input.get(), host.data(), n * sizeof(int), cudaMemcpyHostToDevice, stream.get()));
        launch(); // Warm-up and correctness run outside timing.
        CUDA_CHECK(cudaStreamSynchronize(stream.get()));
        unsigned got_count;
        CUDA_CHECK(cudaMemcpy(&got_count, count.get(), sizeof(unsigned), cudaMemcpyDeviceToHost));
        if (got_count != expected.size()) throw std::runtime_error("custom compacted length mismatch");
        if (got_count) CUDA_CHECK(cudaMemcpy(actual.data(), output.get(), got_count * sizeof(int), cudaMemcpyDeviceToHost));
        if (!std::equal(expected.begin(), expected.end(), actual.begin()))
            throw std::runtime_error("custom compacted values/order mismatch");
        if (pattern == 2) {
            timer.start(stream.get());
            for (int r = 0; r < iterations; ++r) launch();
            custom_ms = timer.stop(stream.get()) / iterations;
        }
        auto first = thrust::device_pointer_cast(library.get());
        int runs = pattern == 2 ? iterations + 1 : 1; // First run warms Thrust.
        for (int r = 0; r < runs; ++r) {
            CUDA_CHECK(cudaMemcpyAsync(library.get(), input.get(), n * sizeof(int), cudaMemcpyDeviceToDevice, stream.get()));
            timer.start(stream.get());
            auto end = thrust::remove(thrust::cuda::par.on(stream.get()), first, first + n, -1);
            float ms = timer.stop(stream.get());
            if (pattern == 2 && r > 0) library_ms += ms / iterations;
            if (size_t(end - first) != expected.size()) throw std::runtime_error("Thrust compacted length mismatch");
        }
        if (!expected.empty()) CUDA_CHECK(cudaMemcpy(actual.data(), library.get(), expected.size() * sizeof(int), cudaMemcpyDeviceToHost));
        if (!std::equal(expected.begin(), expected.end(), actual.begin()))
            throw std::runtime_error("Thrust compacted values/order mismatch");
    }
    printf("PASS compaction variant=%s n=%u iterations=%d SMs=%d blocks_per_SM=%d launched_blocks=%u patterns=5\n",
        Linear ? "linear_offsets" : "naive", n, iterations, prop.multiProcessorCount, blocks_per_sm, blocks);
    printf("custom_ms=%.6f thrust_ms=%.6f thrust_over_custom=%.3f\n", custom_ms, library_ms, library_ms / custom_ms);
}
int main(int argc, char** argv) try {
    Args args(argc, argv, {"n", "iterations", "mode"});
    unsigned n = unsigned(args.number("n", 256, 32ULL * 1024 * 1024));
    int iterations = int(args.number("iterations", 5, 1000));
    std::string mode = args.str("mode", "both");
    if (mode != "both" && mode != "naive" && mode != "linear")
        throw std::runtime_error("mode must be both, naive, or linear");
    auto prop = device_info();
    if (!prop.cooperativeLaunch) {
        printf("SKIP cooperative compaction: device cooperativeLaunch=0\n"); return 0;
    }
    if (mode == "both" || mode == "naive") run_variant<false>(n, iterations, prop);
    if (mode == "both" || mode == "linear") run_variant<true>(n, iterations, prop);
    return 0;
} catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
