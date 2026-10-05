#pragma once
#include "graph_common.cuh"
#include <cublas_v2.h>
inline void check_blas(cublasStatus_t status,const char* call,int line) {
  if(status!=CUBLAS_STATUS_SUCCESS) throw std::runtime_error(std::string(call)+" at line "+std::to_string(line)+": cuBLAS status "+std::to_string(int(status)));
}
#define BLAS(call) check_blas((call),#call,__LINE__)
struct BlasHandle {
  cublasHandle_t h=nullptr;
  BlasHandle() { BLAS(cublasCreate(&h)); }
  ~BlasHandle() { if(h) cublasDestroy(h); }
};
__global__ void increment(float* y,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=1;
}
inline int run_cublas_graph(int argc,char** argv) {
  if(argc>3) throw std::runtime_error("Usage: executable [N=65536] [repetitions=100]");
  int n=argc>1?positive_arg(argv[1],10000000):65536;
  int reps=argc>2?positive_arg(argv[2],10000):100;
  std::vector<float> x(n),initial(n);
  for(int i=0;i<n;++i) { x[i]=float(i%17); initial[i]=float(i%13); }
  Buffer dx(n),dy(n); Stream stream; BlasHandle handle;
  BLAS(cublasSetStream(handle.h,stream.s)); BLAS(cublasSetPointerMode(handle.h,CUBLAS_POINTER_MODE_HOST));
  CUDA(cudaMemcpy(dx.p,x.data(),size_t(n)*sizeof(float),cudaMemcpyHostToDevice));
  auto reset=[&]() { CUDA(cudaMemcpyAsync(dy.p,initial.data(),size_t(n)*sizeof(float),cudaMemcpyHostToDevice,stream.s)); CUDA(cudaStreamSynchronize(stream.s)); };
  const int threads=256,blocks=(n+threads-1)/threads;
  const float alpha=5;
  auto direct_launch=[&]() {
    increment<<<blocks,threads,0,stream.s>>>(dy.p,n); CUDA(cudaGetLastError());
    BLAS(cublasSaxpy(handle.h,n,&alpha,dx.p,1,dy.p,1));
    increment<<<blocks,threads,0,stream.s>>>(dy.p,n); CUDA(cudaGetLastError());
  };
  // Initialize cuBLAS before capturing it, then reset the warm-up's arithmetic.
  reset(); direct_launch(); CUDA(cudaStreamSynchronize(stream.s));
  reset(); Measurement direct=measure(stream.s,reps,direct_launch);
  bool ok=verify("direct_cublas",dy.p,x,initial,reps,5,2);
  Graph parent,library;
  auto t=Clock::now();
  CUDA(cudaGraphCreate(&parent.g,0));
  cudaGraphNode_t first,library_node,last;
  void* args[]={&dy.p,&n};
  cudaKernelNodeParams params{};
  params.func=reinterpret_cast<void*>(increment); params.gridDim=dim3(blocks); params.blockDim=dim3(threads);
  params.kernelParams=args; params.sharedMemBytes=0;
  CUDA(cudaGraphAddKernelNode(&first,parent.g,nullptr,0,&params));
  CUDA(cudaStreamBeginCapture(stream.s,cudaStreamCaptureModeGlobal));
  BLAS(cublasSaxpy(handle.h,n,&alpha,dx.p,1,dy.p,1));
  CUDA(cudaStreamEndCapture(stream.s,&library.g));
  CUDA(cudaGraphAddChildGraphNode(&library_node,parent.g,&first,1,library.g));
  CUDA(cudaGraphAddKernelNode(&last,parent.g,&library_node,1,&params));
  double creation_ms=wall_ms(t);
  size_t parent_nodes=0,child_nodes=0;
  CUDA(cudaGraphGetNodes(parent.g,nullptr,&parent_nodes)); CUDA(cudaGraphGetNodes(library.g,nullptr,&child_nodes));
  t=Clock::now(); CUDA(cudaGraphInstantiate(&parent.exec,parent.g,nullptr,nullptr,0));
  double instantiate_ms=wall_ms(t);
  reset(); CUDA(cudaGraphLaunch(parent.exec,stream.s)); CUDA(cudaStreamSynchronize(stream.s));
  reset(); Measurement replay=measure(stream.s,reps,[&]() { CUDA(cudaGraphLaunch(parent.exec,stream.s)); });
  ok=verify("cublas_child_graph",dy.p,x,initial,reps,5,2) && ok;
  std::printf("N=%d repetitions=%d exact_recurrence=y0+repetitions*(5*x+2)\n",n,reps);
  std::printf("parent_graph_nodes=%zu child_graph_nodes=%zu creation_ms=%.6f instantiate_ms=%.6f\n",parent_nodes,child_nodes,creation_ms,instantiate_ms);
  print_measurement("direct_cublas",direct,reps); print_measurement("cublas_child_graph",replay,reps);
  std::printf("wall_speedup=%.6f graph_setup_plus_replay_ms=%.6f\n",direct.wall/replay.wall,creation_ms+instantiate_ms+replay.wall);
  return ok?0:1;
}
