#include <cuda_runtime.h>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#define BLOCK_SIZE 512
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"%s:%d: %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e)); std::exit(1); } } while(0)
__device__ double ahs(size_t n) { return (n&1?1.0:-1.0)/double(n); }
__global__ void estimate_sum_ahs(size_t length,double* sum) {
  __shared__ double smem[BLOCK_SIZE];
  size_t idx=size_t(blockDim.x)*blockIdx.x+threadIdx.x;
#ifdef DEMO_ZERO_BUG
  smem[threadIdx.x]=idx<length?ahs(idx):0; // Reproducer: ahs(0) is -infinity.
#else
  // The series starts at n=1. Index zero stores its first term, never 1/0.
  smem[threadIdx.x]=idx<length?ahs(idx+1):0;
#endif
  for(int stride=blockDim.x/2;stride>0;stride/=2) {
    __syncthreads();
    if(threadIdx.x<stride) smem[threadIdx.x]+=smem[threadIdx.x+stride];
  }
  if(threadIdx.x==0) atomicAdd(sum,smem[0]);
}
int main(int argc,char** argv) {
  size_t n=1048576;
  if(argc>1) { char* end=nullptr; long long v=std::strtoll(argv[1],&end,10); if(!*argv[1] || *end || v<1 || v>100000000) { std::fprintf(stderr,"terms must be in [1,100000000]\n"); return 2; } n=size_t(v); }
  double* sum=nullptr; CUDA(cudaMalloc(&sum,sizeof(double))); CUDA(cudaMemset(sum,0,sizeof(double)));
  cudaEvent_t start,stop; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
  estimate_sum_ahs<<<unsigned((n+BLOCK_SIZE-1)/BLOCK_SIZE),BLOCK_SIZE>>>(n,sum); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaMemset(sum,0,sizeof(double))); CUDA(cudaEventRecord(start));
  estimate_sum_ahs<<<unsigned((n+BLOCK_SIZE-1)/BLOCK_SIZE),BLOCK_SIZE>>>(n,sum); CUDA(cudaGetLastError());
  CUDA(cudaEventRecord(stop)); CUDA(cudaEventSynchronize(stop));
  float elapsed=0; CUDA(cudaEventElapsedTime(&elapsed,start,stop));
  double actual=0; CUDA(cudaMemcpy(&actual,sum,sizeof(double),cudaMemcpyDeviceToHost));
  long double reference=0;
  for(size_t k=1;k<=n;++k) reference+=(k&1?1.0L:-1.0L)/static_cast<long double>(k);
  double cpu_error=std::abs(actual-double(reference));
  double limit_error=std::abs(actual-std::log(2.0)), truncation_bound=1.0/double(n+1);
  bool ok=std::isfinite(actual) && cpu_error<1e-10 && limit_error<=truncation_bound+1e-10;
  std::printf("terms=%zu estimate=%.15f cpu_reference=%.15f log2=%.15f cpu_abs_error=%.9g truncation_bound=%.9g kernel_ms=%.6f %s\n",n,actual,double(reference),std::log(2.0),cpu_error,truncation_bound,elapsed,ok?"PASS":"FAIL");
  CUDA(cudaFree(sum)); CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
  return ok?0:1;
}
