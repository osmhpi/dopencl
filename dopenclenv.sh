#!/bin/sh

# See https://unix.stackexchange.com/a/162893 for the trick used here to avoid extra colons
export PATH="${PATH:+${PATH}:}/opt/dopencl/bin"
export LIBRARY_PATH="${LIBRARY_PATH:+${LIBRARY_PATH}:}/opt/dopencl/lib"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:+${LD_LIBRARY_PATH}:}/opt/dopencl/lib"
export C_INCLUDE_PATH="${C_INCLUDE_PATH:+${C_INCLUDE_PATH}:}/opt/dopencl/include"
export CPLUS_INCLUDE_PATH="${CPLUS_INCLUDE_PATH:+${CPLUS_INCLUDE_PATH}:}/opt/dopencl/include"
