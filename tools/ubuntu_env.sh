#!/usr/bin/env bash
# Source from an Ubuntu terminal: source tools/ubuntu_env.sh
# Scope the WSL driver preference to this shell; do not change system libraries.
if [[ $(uname -r) == *[Mm]icrosoft* && -f /usr/lib/wsl/lib/libcuda.so.1 ]]; then
  case "${LD_LIBRARY_PATH:-}" in
    /usr/lib/wsl/lib|/usr/lib/wsl/lib:*) ;;
    *) export LD_LIBRARY_PATH="/usr/lib/wsl/lib${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}" ;;
  esac
fi
