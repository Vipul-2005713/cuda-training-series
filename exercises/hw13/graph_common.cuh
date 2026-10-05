#pragma once
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <vector>
inline void check_cuda(cudaError_t status,const char* call,int line) {
  if(status!=cudaSuccess) throw std::runtime_error(std::string(call)+" at line "+std::to_string(line)+": "+cudaGetErrorString(status));
}
#define CUDA(call) check_cuda((call),#call,__LINE__)
using Clock=std::chrono::steady_clock;
inline double wall_ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
inline int positive_arg(char* s,int max) {
  char* end=nullptr; long long v=std::strtoll(s,&end,10);
  if(!*s || *end || v<1 || v>max) throw std::runtime_error("Argument outside supported positive range: "+std::string(s));
  return int(v);
}
struct Buffer {
  float* p=nullptr;
  explicit Buffer(size_t n) { CUDA(cudaMalloc(&p,n*sizeof(float))); }
  ~Buffer() { if(p) cudaFree(p); }
  Buffer(const Buffer&)=delete; Buffer& operator=(const Buffer&)=delete;
};
struct Stream {
  cudaStream_t s=nullptr;
  Stream() { CUDA(cudaStreamCreateWithFlags(&s,cudaStreamNonBlocking)); }
  ~Stream() { if(s) cudaStreamDestroy(s); }
  Stream(const Stream&)=delete; Stream& operator=(const Stream&)=delete;
};
struct Event {
  cudaEvent_t e=nullptr;
  explicit Event(unsigned flags=cudaEventDefault) { CUDA(cudaEventCreateWithFlags(&e,flags)); }
  ~Event() { if(e) cudaEventDestroy(e); }
  Event(const Event&)=delete; Event& operator=(const Event&)=delete;
};
struct Graph {
  cudaGraph_t g=nullptr;
  cudaGraphExec_t exec=nullptr;
  ~Graph() { if(exec) cudaGraphExecDestroy(exec); if(g) cudaGraphDestroy(g); }
};
struct Measurement { double wall; float gpu; };
template<class Launch>
Measurement measure(cudaStream_t stream,int reps,Launch launch) {
  Event start,stop;
  CUDA(cudaStreamSynchronize(stream));
  auto t=Clock::now(); CUDA(cudaEventRecord(start.e,stream));
  for(int i=0;i<reps;++i) launch();
  CUDA(cudaEventRecord(stop.e,stream)); CUDA(cudaEventSynchronize(stop.e));
  Measurement result{wall_ms(t),0}; CUDA(cudaEventElapsedTime(&result.gpu,start.e,stop.e));
  return result;
}
inline bool verify(const char* label,const float* device,const std::vector<float>& x,const std::vector<float>& initial,int reps,int scale,int offset) {
  std::vector<float> result(x.size()); CUDA(cudaMemcpy(result.data(),device,result.size()*sizeof(float),cudaMemcpyDeviceToHost));
  bool ok=true; double max_error=0;
  for(size_t i=0;i<x.size();++i) {
    double expected=initial[i]+double(reps)*(scale*x[i]+offset);
    double error=std::abs(result[i]-expected); max_error=std::max(max_error,error);
    if(!std::isfinite(result[i]) || error!=0) ok=false;
  }
  std::printf("%s %s all_elements=%zu max_abs_error=%.9g\n",label,ok?"PASS":"FAIL",x.size(),max_error);
  return ok;
}
inline void print_measurement(const char* label,Measurement m,int reps) {
  std::printf("%s total_wall_ms=%.6f total_gpu_event_ms=%.6f wall_us_per_iteration=%.6f\n",label,m.wall,m.gpu,m.wall*1000/reps);
}
