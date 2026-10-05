#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#define BLOCK_SIZE 32
#define CUDA(call) do { cudaError_t e=(call); if(e!=cudaSuccess) { fprintf(stderr,"%s:%d: %s: %s\n",__FILE__,__LINE__,#call,cudaGetErrorString(e)); std::exit(1); } } while(0)
struct Matrix { int width,height,stride; float* elements; };
__global__ void MatMulKernel(Matrix A,Matrix B,Matrix C) {
  int row=blockIdx.y*BLOCK_SIZE+threadIdx.y, col=blockIdx.x*BLOCK_SIZE+threadIdx.x;
  int ty=threadIdx.y, tx=threadIdx.x;
  __shared__ float As[BLOCK_SIZE][BLOCK_SIZE],Bs[BLOCK_SIZE][BLOCK_SIZE];
  float value=0;
  for(int tile=0;tile<(A.width+BLOCK_SIZE-1)/BLOCK_SIZE;++tile) {
    int ac=tile*BLOCK_SIZE+tx, br=tile*BLOCK_SIZE+ty;
    As[ty][tx]=(row<A.height && ac<A.width)?A.elements[row*A.stride+ac]:0;
    Bs[ty][tx]=(br<B.height && col<B.width)?B.elements[br*B.stride+col]:0;
    __syncthreads(); // Producer writes must finish before other warps consume this tile.
#ifdef DEMO_BOUNDS_BUG
    for(int e=0;e<=BLOCK_SIZE;++e) // Opt-in reproducer of the original off-by-one bug.
#else
    for(int e=0;e<BLOCK_SIZE;++e)
#endif
      value+=As[ty][e]*Bs[e][tx];
#ifndef DEMO_RACE_BUG
    __syncthreads(); // All readers finish before the next iteration overwrites shared memory.
#endif
  }
  if(row<C.height && col<C.width) C.elements[row*C.stride+col]=value;
}
int main(int argc,char** argv) {
  int n=128;
  if(argc>1) { char* end=nullptr; long v=std::strtol(argv[1],&end,10); if(!*argv[1] || *end || v<1 || v>4096) { std::fprintf(stderr,"side must be in [1,4096]\n"); return 2; } n=int(v); }
  size_t count=size_t(n)*n,bytes=count*sizeof(float);
  std::vector<float> a(count),b(count),c(count);
  for(size_t i=0;i<count;++i) { a[i]=float(int(i%7)-3)*0.25f; b[i]=float(int(i%11)-5)*0.125f; }
  Matrix A{n,n,n,nullptr},B{n,n,n,nullptr},C{n,n,n,nullptr};
  CUDA(cudaMalloc(&A.elements,bytes)); CUDA(cudaMalloc(&B.elements,bytes)); CUDA(cudaMalloc(&C.elements,bytes));
  CUDA(cudaMemcpy(A.elements,a.data(),bytes,cudaMemcpyHostToDevice)); CUDA(cudaMemcpy(B.elements,b.data(),bytes,cudaMemcpyHostToDevice));
  cudaEvent_t start,stop; CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&stop));
  dim3 block(BLOCK_SIZE,BLOCK_SIZE),grid((n+BLOCK_SIZE-1)/BLOCK_SIZE,(n+BLOCK_SIZE-1)/BLOCK_SIZE);
  MatMulKernel<<<grid,block>>>(A,B,C); CUDA(cudaGetLastError()); CUDA(cudaDeviceSynchronize());
  CUDA(cudaEventRecord(start));
  MatMulKernel<<<grid,block>>>(A,B,C); CUDA(cudaGetLastError());
  CUDA(cudaEventRecord(stop)); CUDA(cudaEventSynchronize(stop));
  float elapsed=0; CUDA(cudaEventElapsedTime(&elapsed,start,stop));
  CUDA(cudaMemcpy(c.data(),C.elements,bytes,cudaMemcpyDeviceToHost));
  bool ok=true; double max_error=0;
  for(int row=0;row<n;++row) for(int col=0;col<n;++col) {
    double expected=0; for(int k=0;k<n;++k) expected+=double(a[size_t(row)*n+k])*b[size_t(k)*n+col];
    double err=std::abs(c[size_t(row)*n+col]-expected); max_error=std::max(max_error,err);
    if(!std::isfinite(c[size_t(row)*n+col]) || err>1e-5) ok=false;
  }
  std::printf("side=%d kernel_ms=%.6f max_abs_error=%.9g %s: all %zu outputs checked\n",n,elapsed,max_error,ok?"PASS":"FAIL",count);
  CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(stop));
  CUDA(cudaFree(A.elements)); CUDA(cudaFree(B.elements)); CUDA(cudaFree(C.elements));
  return ok?0:1;
}
