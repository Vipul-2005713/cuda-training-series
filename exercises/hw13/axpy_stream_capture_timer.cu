// Matching race-free direct-stream baseline; excludes graph construction/replay.
#include "stream_capture_impl.cuh"
int main(int argc,char** argv) {
  try { return run_stream_capture(argc,argv,false); }
  catch(const std::exception& e) { std::fprintf(stderr,"FAIL: %s\n",e.what()); return 1; }
}
