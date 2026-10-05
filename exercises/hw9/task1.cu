#include "../hw6/cuda_utils.cuh"
#include <cooperative_groups.h>
namespace cg = cooperative_groups;
constexpr int threads = 256;

__device__ int reduce(cg::thread_group group, int* shared, int value) {
    int rank = group.thread_rank();
    for (int stride = group.size() / 2; stride > 0; stride /= 2) {
        shared[rank] = value; group.sync();
        if (rank < stride) value += shared[rank + stride];
        group.sync(); // No thread overwrites scratch until all readers finish.
    }
    return value;
}
template<int GroupSize>
__global__ void grouped_reduction(const int* input, int* output) {
    __shared__ int shared[threads];
    auto block = cg::this_thread_block(); // 1a: whole 256-thread block.
    if constexpr (GroupSize == 256) {
        int sum = reduce(block, shared, input[threadIdx.x]);
        if (block.thread_rank() == 0) output[0] = sum;
    } else {
        auto warp = cg::tiled_partition(block, 32); // 1b: runtime partition.
        if constexpr (GroupSize == 32) {
            int offset = (threadIdx.x / 32) * 32;
            int sum = reduce(warp, shared + offset, input[threadIdx.x]);
            if (warp.thread_rank() == 0) output[threadIdx.x / 32] = sum;
        } else {
            auto half_warp = cg::tiled_partition(warp, 16); // 1c: subdivide the existing tile.
            int offset = (threadIdx.x / 16) * 16;
            int sum = reduce(half_warp, shared + offset, input[threadIdx.x]);
            if (half_warp.thread_rank() == 0) output[threadIdx.x / 16] = sum;
        }
    }
}
template<int GroupSize> void run() {
    CudaBuffer<int> data(threads), output(threads / GroupSize);
    std::vector<int> host(threads, 1), result(threads / GroupSize);
    for (int pattern = 0; pattern < 2; ++pattern) {
        if (pattern) for (int i = 0; i < threads; ++i) host[i] = i % 13 - 6;
        CUDA_CHECK(cudaMemcpy(data.get(), host.data(), threads * sizeof(int), cudaMemcpyHostToDevice));
        grouped_reduction<GroupSize><<<1, threads>>>(data.get(), output.get());
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaMemcpy(result.data(), output.get(), result.size() * sizeof(int), cudaMemcpyDeviceToHost));
        for (size_t tile = 0; tile < result.size(); ++tile) {
            int expected = 0;
            for (int lane = 0; lane < GroupSize; ++lane) expected += host[tile * GroupSize + lane];
            if (result[tile] != expected) throw std::runtime_error("group reduction mismatch");
            if (!pattern) printf("group partial sum: %d\n", result[tile]);
        }
    }
    printf("PASS group_size=%d groups=%d patterns=ones,signed_values\n", GroupSize, threads / GroupSize);
}
int main(int argc, char** argv) try {
    Args args(argc, argv, {"group"}); std::string group = args.str("group", "all");
    if (group != "all" && group != "256" && group != "32" && group != "16")
        throw std::runtime_error("group must be all, 256, 32, or 16");
    if (group == "all" || group == "256") run<256>();
    if (group == "all" || group == "32") run<32>();
    if (group == "all" || group == "16") run<16>();
    return 0;
} catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
