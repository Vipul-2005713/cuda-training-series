// Completed explicit-graph exercise: kernel -> captured cuBLAS child -> kernel.
#include "cublas_graph_impl.cuh"
int main(int argc,char** argv) {
  try { return run_cublas_graph(argc,argv); }
  catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
