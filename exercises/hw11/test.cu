#include <cuda_runtime.h>
#ifndef NO_MPI
#include <mpi.h>
#endif
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <thread>
#include <vector>
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"%s:%d: %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e)); std::exit(1); } } while(0)
static size_t arg(char* s,size_t limit) {
  char* end=nullptr; long long v=std::strtoll(s,&end,10);
  if(!*s || *end || v<1 || static_cast<unsigned long long>(v)>limit) { std::fprintf(stderr,"Invalid positive argument: %s\n",s); std::exit(2); }
  return size_t(v);
}
__global__ void kernel(double* x,size_t n) {
  size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;
  if(i<n) x[i]=2*x[i];
}
int main(int argc,char** argv) {
  int rank=0,ranks=1;
#ifndef NO_MPI
  MPI_Init(&argc,&argv); MPI_Comm_size(MPI_COMM_WORLD,&ranks); MPI_Comm_rank(MPI_COMM_WORLD,&rank);
#endif
  size_t n=argc>1?arg(argv[1],1000000000ULL):1048576;
#ifdef NO_MPI
  ranks=argc>2?int(arg(argv[2],128)):1;
  if(argc>4) {
    char* end=nullptr; long value=std::strtol(argv[4],&end,10);
    if(!*argv[4] || *end || value<0 || value>=ranks) { std::fprintf(stderr,"Rank must be in [0,ranks).\n"); return 2; }
    rank=int(value);
  }
#endif
  int reps=argc>3?int(arg(argv[3],900)):100;
  if(n<size_t(ranks)) { std::fprintf(stderr,"N must be >= rank count\n"); return 2; }
  size_t begin=n*rank/ranks, end=n*(rank+1)/ranks, local=end-begin;
  // Exact powers of two avoid overflow, underflow, and tolerance ambiguity.
  std::vector<double> host(local,std::ldexp(1.0,-reps));
  double* x=nullptr; CUDA(cudaSetDevice(0)); CUDA(cudaMalloc(&x,local*sizeof(double)));
  CUDA(cudaMemcpy(x,host.data(),local*sizeof(double),cudaMemcpyHostToDevice));
  kernel<<<unsigned((local+255)/256),256>>>(x,local); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemcpy(x,host.data(),local*sizeof(double),cudaMemcpyHostToDevice));
#ifndef NO_MPI
  MPI_Barrier(MPI_COMM_WORLD);
#else
  // Optional common future Unix time supplied by the launcher. This lets all
  // processes finish CUDA initialization before starting the measured loop.
  if(argc>5) {
    size_t epoch_ms=arg(argv[5],9000000000000000ULL);
    auto target=std::chrono::system_clock::time_point(std::chrono::milliseconds(epoch_ms));
    if(std::chrono::system_clock::now()>target) std::fprintf(stderr,"WARNING: rank %d missed common start time; rerun with a longer launch delay.\n",rank);
    else std::this_thread::sleep_until(target);
  }
#endif
  auto start=std::chrono::steady_clock::now();
  // Deliberately preserve the lecture's launch + per-kernel synchronization workload.
  for(int i=0;i<reps;++i) { kernel<<<unsigned((local+255)/256),256>>>(x,local); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize()); }
  double elapsed=std::chrono::duration<double,std::milli>(std::chrono::steady_clock::now()-start).count();
  CUDA(cudaMemcpy(host.data(),x,local*sizeof(double),cudaMemcpyDeviceToHost));
  bool ok=true; for(double v:host) if(v!=1.0) { ok=false; break; }
  CUDA(cudaFree(x));
  std::printf("rank=%d ranks=%d N_total=%zu N_local=%zu reps=%d elapsed_ms=%.6f wall_ms_per_kernel=%.6f effective_local_GB_s=%.6f %s\n",rank,ranks,n,local,reps,elapsed,elapsed/reps,2.0*local*sizeof(double)*reps/(elapsed*1e6),ok?"PASS":"FAIL");
#ifndef NO_MPI
  int failed=ok?0:1,any_failed=0; MPI_Allreduce(&failed,&any_failed,1,MPI_INT,MPI_MAX,MPI_COMM_WORLD); MPI_Finalize(); return any_failed;
#else
  return ok?0:1;
#endif
}
