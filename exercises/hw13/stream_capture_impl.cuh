#pragma once
#include "graph_common.cuh"
// Preserve A -> {B,C} -> D, using independent branch outputs to avoid the
// original concurrent B/C writes to y. The total update is still y += 8*x.
__global__ void kernel_a(const float* x,float* y,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]+=2*x[i];
}
__global__ void kernel_b(const float* x,const float* y,float* b,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) b[i]=y[i]+2*x[i];
}
__global__ void kernel_c(const float* x,const float* y,float* c,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) c[i]=y[i]+2*x[i];
}
__global__ void kernel_d(const float* x,float* y,const float* b,const float* c,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) y[i]=b[i]+c[i]-y[i]+2*x[i];
}
inline int run_stream_capture(int argc,char** argv,bool graph_enabled) {
  if(argc>3) throw std::runtime_error("Usage: executable [N=65536] [repetitions=100]");
  int n=argc>1?positive_arg(argv[1],10000000):65536;
  int reps=argc>2?positive_arg(argv[2],10000):100;
  std::vector<float> x(n),initial(n);
  for(int i=0;i<n;++i) { x[i]=float(i%17); initial[i]=float(i%13); }
  Buffer dx(n),dy(n),db(n),dc(n); Stream origin,branch;
  Event fork(cudaEventDisableTiming),join(cudaEventDisableTiming);
  CUDA(cudaMemcpy(dx.p,x.data(),size_t(n)*sizeof(float),cudaMemcpyHostToDevice));
  auto reset=[&]() { CUDA(cudaMemcpyAsync(dy.p,initial.data(),size_t(n)*sizeof(float),cudaMemcpyHostToDevice,origin.s)); CUDA(cudaStreamSynchronize(origin.s)); };
  const int threads=256,blocks=(n+threads-1)/threads;
  auto enqueue=[&]() {
    kernel_a<<<blocks,threads,0,origin.s>>>(dx.p,dy.p,n); CUDA(cudaGetLastError());
    CUDA(cudaEventRecord(fork.e,origin.s));
    kernel_b<<<blocks,threads,0,origin.s>>>(dx.p,dy.p,db.p,n); CUDA(cudaGetLastError());
    CUDA(cudaStreamWaitEvent(branch.s,fork.e,0));
    kernel_c<<<blocks,threads,0,branch.s>>>(dx.p,dy.p,dc.p,n); CUDA(cudaGetLastError());
    CUDA(cudaEventRecord(join.e,branch.s));
    CUDA(cudaStreamWaitEvent(origin.s,join.e,0));
    kernel_d<<<blocks,threads,0,origin.s>>>(dx.p,dy.p,db.p,dc.p,n); CUDA(cudaGetLastError());
  };
  reset(); enqueue(); CUDA(cudaStreamSynchronize(origin.s));
  reset(); Measurement direct=measure(origin.s,reps,enqueue);
  bool ok=verify("direct_streams",dy.p,x,initial,reps,8,0);
  std::printf("N=%d repetitions=%d exact_recurrence=y0+8*repetitions*x\n",n,reps);
  print_measurement("direct_streams",direct,reps);
  if(graph_enabled) {
    Graph graph;
    auto t=Clock::now();
    // EndCapture creates graph.g. Precreating a graph here would leak it.
    CUDA(cudaStreamBeginCapture(origin.s,cudaStreamCaptureModeGlobal));
    enqueue();
    CUDA(cudaStreamEndCapture(origin.s,&graph.g));
    double capture_ms=wall_ms(t);
    size_t nodes=0; CUDA(cudaGraphGetNodes(graph.g,nullptr,&nodes));
    t=Clock::now(); CUDA(cudaGraphInstantiate(&graph.exec,graph.g,nullptr,nullptr,0));
    double instantiate_ms=wall_ms(t);
    // First upload/launch is warmed separately; it is not steady-state execution.
    reset(); CUDA(cudaGraphLaunch(graph.exec,origin.s)); CUDA(cudaStreamSynchronize(origin.s));
    reset(); Measurement replay=measure(origin.s,reps,[&]() { CUDA(cudaGraphLaunch(graph.exec,origin.s)); });
    ok=verify("captured_graph",dy.p,x,initial,reps,8,0) && ok;
    std::printf("graph_nodes=%zu capture_ms=%.6f instantiate_ms=%.6f\n",nodes,capture_ms,instantiate_ms);
    print_measurement("captured_graph",replay,reps);
    std::printf("wall_speedup=%.6f graph_setup_plus_replay_ms=%.6f\n",direct.wall/replay.wall,capture_ms+instantiate_ms+replay.wall);
  }
  return ok?0:1;
}
