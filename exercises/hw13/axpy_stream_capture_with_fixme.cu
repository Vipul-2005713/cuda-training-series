// Completed stream-capture exercise. The implementation is shared with the
// from-scratch exercise so both use the same race-free diamond and validation.
#include "stream_capture_impl.cuh"
int main(int argc,char** argv) {
  try { return run_stream_capture(argc,argv,true); }
  catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
