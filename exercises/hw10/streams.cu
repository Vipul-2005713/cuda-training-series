#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"%s:%d: %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e)); std::exit(1); } } while(0)
using Clock=std::chrono::steady_clock;
static double ms(Clock::time_point t) { return std::chrono::duration<double,std::milli>(Clock::now()-t).count(); }
static int argument(char* s, int limit) {
  char* end=nullptr; long long n=std::strtoll(s,&end,10);
  if(!*s || *end || n<1 || n>limit) { std::fprintf(stderr,"Invalid positive argument: %s\n",s); std::exit(2); }
  return static_cast<int>(n);
}
__host__ __device__ float gaussian(float x) {
  float in=x-11*0.01f, out=0;
  for(int j=0;j<22;++j) { out+=expf(-0.5f*in*in)/2.506628274794649f; in+=0.01f; }
  return out/22;
}
__global__ void gaussian_pdf(const float* x,float* y,int n) {
  int i=blockIdx.x*blockDim.x+threadIdx.x;
  if(i<n) y[i]=gaussian(x[i]);
}
struct Lane { int device; cudaStream_t stream; float *x,*y; };
int main(int argc,char** argv) {
  if(argc>5) { std::fprintf(stderr,"Usage: streams [N] [chunks] [streams_per_gpu] [requested_gpus]\n"); return 2; }
  const int n=argc>1?argument(argv[1],100000000):1048576;
  const int chunks=argc>2?argument(argv[2],65536):16;
  const int per_gpu=argc>3?argument(argv[3],64):4;
  const int requested=argc>4?argument(argv[4],64):1;
  int available=0; CUDA(cudaGetDeviceCount(&available));
  if(!available) { std::fprintf(stderr,"No CUDA devices\n"); return 1; }
  const int devices=std::min(requested,available);
  std::printf("N=%d chunks=%d streams_per_gpu=%d requested_gpus=%d available_gpus=%d used_gpus=%d\n",n,chunks,per_gpu,requested,available,devices);
  if(requested>available) std::printf("MULTI_GPU_LIMITATION: only %d GPU(s); executing the same device-aware path on available hardware.\n",available);
  float *hx=nullptr,*hy=nullptr,*reference=nullptr,*dx=nullptr,*dy=nullptr;
  const size_t bytes=size_t(n)*sizeof(float);
  CUDA(cudaHostAlloc(&hx,bytes,cudaHostAllocPortable));
  CUDA(cudaHostAlloc(&hy,bytes,cudaHostAllocPortable));
  CUDA(cudaHostAlloc(&reference,bytes,cudaHostAllocPortable));
  for(int i=0;i<n;++i) hx[i]=float((i*17LL)%1009)/1009;
  CUDA(cudaSetDevice(0)); CUDA(cudaMalloc(&dx,bytes)); CUDA(cudaMalloc(&dy,bytes));
  CUDA(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice));
  gaussian_pdf<<<(n+255)/256,256>>>(dx,dy,n); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  auto start=Clock::now();
  CUDA(cudaMemcpy(dx,hx,bytes,cudaMemcpyHostToDevice));
  gaussian_pdf<<<(n+255)/256,256>>>(dx,dy,n); CUDA(cudaGetLastError());
  CUDA(cudaMemcpy(reference,dy,bytes,cudaMemcpyDeviceToHost));
  double serial=ms(start);
  bool ok=true; double max_error=0;
  for(int i=0;i<n;++i) {
    double err=std::abs(double(reference[i])-gaussian(hx[i])); max_error=std::max(max_error,err);
    if(!std::isfinite(reference[i]) || err>3e-6) ok=false;
  }
  std::printf("serial_end_to_end_ms=%.6f CPU_max_abs_error=%.9g\n",serial,max_error);
  CUDA(cudaFree(dx)); CUDA(cudaFree(dy));
#ifdef USE_STREAMS
  const int lane_count=devices*per_gpu;
  const int capacity=(n+chunks-1)/chunks;
  std::vector<Lane> lanes(lane_count);
  for(int j=0;j<lane_count;++j) {
    Lane& l=lanes[j]; l.device=j%devices; CUDA(cudaSetDevice(l.device));
    CUDA(cudaStreamCreateWithFlags(&l.stream,cudaStreamNonBlocking));
    CUDA(cudaMalloc(&l.x,size_t(capacity)*sizeof(float))); CUDA(cudaMalloc(&l.y,size_t(capacity)*sizeof(float)));
    CUDA(cudaMemsetAsync(l.x,0,size_t(capacity)*sizeof(float),l.stream));
    gaussian_pdf<<<(capacity+255)/256,256,0,l.stream>>>(l.x,l.y,capacity); CUDA(cudaGetLastError());
  }
  for(auto& l:lanes) { CUDA(cudaSetDevice(l.device)); CUDA(cudaStreamSynchronize(l.stream)); }
#ifdef _OPENMP
  // Warm up the OpenMP team outside timing. Each lane owns a stream and scratch buffers.
#pragma omp parallel num_threads(lane_count)
  { CUDA(cudaSetDevice(omp_get_thread_num()%devices)); }
  std::printf("OpenMP=enabled max_threads=%d\n",omp_get_max_threads());
#else
  std::printf("OpenMP=disabled\n");
#endif
  start=Clock::now();
#ifdef _OPENMP
#pragma omp parallel for schedule(static) num_threads(lane_count)
#endif
  for(int j=0;j<lane_count;++j) {
    Lane& l=lanes[j]; CUDA(cudaSetDevice(l.device));
    // The stream orders buffer reuse, H2D -> kernel -> D2H -> next chunk.
    for(int chunk=j;chunk<chunks;chunk+=lane_count) {
      int begin=int(size_t(n)*chunk/chunks), end=int(size_t(n)*(chunk+1)/chunks), count=end-begin;
      if(!count) continue;
      CUDA(cudaMemcpyAsync(l.x,hx+begin,size_t(count)*sizeof(float),cudaMemcpyHostToDevice,l.stream));
      gaussian_pdf<<<(count+255)/256,256,0,l.stream>>>(l.x,l.y,count); CUDA(cudaGetLastError());
      CUDA(cudaMemcpyAsync(hy+begin,l.y,size_t(count)*sizeof(float),cudaMemcpyDeviceToHost,l.stream));
    }
  }
  // Synchronizing only the current device would miss work on the other devices.
  for(auto& l:lanes) { CUDA(cudaSetDevice(l.device)); CUDA(cudaStreamSynchronize(l.stream)); }
  double streamed=ms(start); max_error=0;
  for(int i=0;i<n;++i) {
    double err=std::abs(double(hy[i])-reference[i]); max_error=std::max(max_error,err);
    if(!std::isfinite(hy[i]) || err>3e-6) ok=false;
  }
  std::printf("streamed_end_to_end_ms=%.6f speedup=%.4f max_abs_error=%.9g\n",streamed,serial/streamed,max_error);
  for(auto& l:lanes) { CUDA(cudaSetDevice(l.device)); CUDA(cudaFree(l.x)); CUDA(cudaFree(l.y)); CUDA(cudaStreamDestroy(l.stream)); }
#endif
  CUDA(cudaFreeHost(hx)); CUDA(cudaFreeHost(hy)); CUDA(cudaFreeHost(reference));
  std::printf("%s: all %d elements checked\n",ok?"PASS":"FAIL",n);
  return ok?0:1;
}
