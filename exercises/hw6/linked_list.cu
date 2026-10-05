#include "cuda_utils.cuh"
#include <memory>

struct list_elem { int key; list_elem* next; };
__host__ __device__ int element(list_elem* list, int index) {
    for (int i = 0; i < index; ++i) list = list->next;
    return list->key;
}
__global__ void gpu_read_list(list_elem* list, int* results, int n) {
    for (int i = 0; i < n; ++i) results[i] = element(list, i);
    printf("key = %d\n", element(list, 3));
}
int main() try {
    const int n = 5;
    if (!device_info().managedMemory) {
        printf("SKIP linked_list: device does not support managed memory\n"); return 0;
    }
    // The essential homework change is malloc -> cudaMallocManaged for EVERY node.
    std::vector<std::unique_ptr<CudaBuffer<list_elem>>> nodes;
    for (int i = 0; i < n; ++i) nodes.emplace_back(new CudaBuffer<list_elem>(1, true));
    for (int i = 0; i < n; ++i) {
        nodes[i]->get()->key = i;
        nodes[i]->get()->next = i + 1 < n ? nodes[i + 1]->get() : nullptr;
    }
    CudaBuffer<int> results(n, true);
    printf("key = %d\n", element(nodes[0]->get(), 3));
    gpu_read_list<<<1, 1>>>(nodes[0]->get(), results.get(), n);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize()); // CPU access must wait for GPU completion.
    for (int i = 0; i < n; ++i)
        if (results.get()[i] != i || element(nodes[0]->get(), i) != i)
            throw std::runtime_error("linked-list traversal mismatch");
    printf("PASS linked_list: all %d CPU/GPU keys match\n", n);
    return 0;
} catch (const std::exception& e) { fprintf(stderr, "FAIL: %s\n", e.what()); return 1; }
