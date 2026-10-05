// Completed from-scratch exercise: capture -> instantiate -> replay -> destroy.
// See stream_capture_impl.cuh for the full two-stream fork/join implementation.
#include "stream_capture_impl.cuh"
int main(int argc,char** argv) {
  try { return run_stream_capture(argc,argv,true); }
  catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
