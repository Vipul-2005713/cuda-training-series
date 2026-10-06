#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source "$SCRIPT_DIR/ubuntu_env.sh"
if ! SANITIZER_EXECUTABLE=$(command -v compute-sanitizer); then
  echo "compute-sanitizer is missing; install it with the Linux CUDA Toolkit." >&2
  exit 2
fi
INJECTION_ARGS=()
# Ubuntu's toolkit package separates the executable from its injection libraries.
# Apply this only to that package layout, leaving other toolkit installs alone.
if [[ $(readlink -f -- "$SANITIZER_EXECUTABLE") == /usr/bin/compute-sanitizer &&
      -f /usr/lib/nvidia-cuda-toolkit/compute-sanitizer/libsanitizer-collection.so ]]; then
  INJECTION_ARGS=(--injection-path /usr/lib/nvidia-cuda-toolkit/compute-sanitizer)
  # The launcher injection library also depends on a sibling interceptor library.
  export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/usr/lib/nvidia-cuda-toolkit/compute-sanitizer"
fi
exec "$SANITIZER_EXECUTABLE" "${INJECTION_ARGS[@]}" "$@"
